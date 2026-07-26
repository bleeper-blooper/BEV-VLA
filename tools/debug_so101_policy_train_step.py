import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_dataset_class():
    path = "mmdet3d/datasets/so101_cached_depth_lang_dataset.py"
    spec = importlib.util.spec_from_file_location("so101_cached_depth_lang_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SO101CachedDepthLangDataset


class RGBDEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img, depths):
        """
        img:    [B, N, 3, H, W]
        depths: [B, N, H, W]
        return: [B, N, D, Hf, Wf]
        """
        B, N, C, H, W = img.shape

        depth = depths.unsqueeze(2)          # [B, N, 1, H, W]
        rgbd = torch.cat([img, depth], dim=2) # [B, N, 4, H, W]

        rgbd = rgbd.reshape(B * N, 4, H, W)
        feat = self.net(rgbd)

        _, D, Hf, Wf = feat.shape
        feat = feat.reshape(B, N, D, Hf, Wf)

        return feat


class LanguageImageFuser(nn.Module):
    def __init__(self, hidden_dim=256, text_dim=384, num_heads=8):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens, text_tokens):
        """
        visual_tokens: [B, S, D]
        text_tokens:   [B, L, 384]
        return:        [B, S, D]
        """
        text = self.text_proj(text_tokens)

        fused, _ = self.cross_attn(
            query=visual_tokens,
            key=text,
            value=text,
        )

        x = self.norm1(visual_tokens + fused)
        x = self.norm2(x + self.ffn(x))

        return x


class PseudoBEVFormer(nn.Module):
    def __init__(self, hidden_dim=256, num_bev_tokens=64, num_heads=8):
        super().__init__()

        self.num_bev_tokens = num_bev_tokens
        self.bev_queries = nn.Parameter(torch.randn(num_bev_tokens, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens):
        """
        visual_tokens: [B, S, D]
        return:        [B, Q, D]
        """
        B = visual_tokens.shape[0]

        queries = self.bev_queries.unsqueeze(0).repeat(B, 1, 1)

        bev, _ = self.cross_attn(
            query=queries,
            key=visual_tokens,
            value=visual_tokens,
        )

        bev = self.norm1(queries + bev)
        bev = self.norm2(bev + self.ffn(bev))

        return bev


class CausalActionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        state_dim=6,
        action_dim=6,
        text_dim=384,
        num_layers=4,
        num_heads=8,
    ):
        super().__init__()

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

    def forward(self, bev_tokens, text_tokens, robot_state, action_prefix):
        """
        bev_tokens:     [B, Q, D]
        text_tokens:    [B, L, 384]
        robot_state:    [B, 6]
        action_prefix:  [B, T, 6]
        return:         [B, T, 6]
        """
        B, T, _ = action_prefix.shape
        device = action_prefix.device

        state_token = self.state_proj(robot_state).unsqueeze(1)  # [B, 1, D]
        text_memory = self.text_proj(text_tokens)                # [B, L, D]

        memory = torch.cat(
            [state_token, text_memory, bev_tokens],
            dim=1,
        )

        tgt = self.action_in_proj(action_prefix)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device) * float("-inf"),
            diagonal=1,
        )

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
        )

        pred_actions = self.action_out_proj(out)

        return pred_actions


class SO101DepthLangPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        text_dim=384,
        action_dim=6,
        state_dim=6,
        num_bev_tokens=64,
        num_heads=8,
    ):
        super().__init__()

        self.rgbd_encoder = RGBDEncoder(hidden_dim=hidden_dim)

        self.camera_embed = nn.Embedding(8, hidden_dim)

        self.lang_img_fuser = LanguageImageFuser(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        self.pseudo_bev = PseudoBEVFormer(
            hidden_dim=hidden_dim,
            num_bev_tokens=num_bev_tokens,
            num_heads=num_heads,
        )

        self.action_decoder = CausalActionDecoder(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            text_dim=text_dim,
            num_layers=4,
            num_heads=num_heads,
        )

    def forward(self, img, depths, text_tokens, robot_state, action_prefix):
        """
        img:           [B, N, 3, H, W]
        depths:        [B, N, H, W]
        text_tokens:   [B, L, 384]
        robot_state:   [B, 6]
        action_prefix: [B, T, 6]
        """
        B, N, _, _, _ = img.shape

        feat = self.rgbd_encoder(img, depths)  # [B, N, D, Hf, Wf]

        B, N, D, Hf, Wf = feat.shape

        # [B, N, D, Hf, Wf] -> [B, N, Hf*Wf, D]
        tokens = feat.flatten(3).permute(0, 1, 3, 2).contiguous()

        cam_ids = torch.arange(N, device=img.device)
        cam_emb = self.camera_embed(cam_ids)  # [N, D]
        tokens = tokens + cam_emb.view(1, N, 1, D)

        # [B, N, HW, D] -> [B, N*HW, D]
        visual_tokens = tokens.reshape(B, N * Hf * Wf, D)

        visual_tokens = self.lang_img_fuser(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )

        bev_tokens = self.pseudo_bev(visual_tokens)

        pred_actions = self.action_decoder(
            bev_tokens=bev_tokens,
            text_tokens=text_tokens,
            robot_state=robot_state,
            action_prefix=action_prefix,
        )

        return pred_actions


def move_batch_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v

    return out


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    SO101CachedDepthLangDataset = load_dataset_class()

    dataset = SO101CachedDepthLangDataset(
        ann_file="data/so101_depth_lang_cache/train_index.json",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    model = SO101DepthLangPolicy(
        hidden_dim=256,
        text_dim=384,
        action_dim=6,
        state_dim=6,
        num_bev_tokens=64,
        num_heads=8,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )

    model.train()

    pred_actions = model(
        img=batch["img"],
        depths=batch["depths"],
        text_tokens=batch["text_tokens"],
        robot_state=batch["robot_state"],
        action_prefix=batch["action_prefix"],
    )

    gt_actions = batch["gt_actions"]

    loss_l1 = F.smooth_l1_loss(pred_actions, gt_actions)
    loss_mse = F.mse_loss(pred_actions, gt_actions)
    loss = loss_l1 + 0.1 * loss_mse

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

    optimizer.step()

    print("pred_actions:", tuple(pred_actions.shape))
    print("gt_actions:  ", tuple(gt_actions.shape))
    print("loss_l1:", float(loss_l1.detach().cpu()))
    print("loss_mse:", float(loss_mse.detach().cpu()))
    print("loss:", float(loss.detach().cpu()))
    print("grad_norm:", float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else grad_norm)
    print("one training step ok")


if __name__ == "__main__":
    main()
