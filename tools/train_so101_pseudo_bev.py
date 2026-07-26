import os
import json
import time
import argparse
import importlib.util
from pathlib import Path

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


class ActionNormalizer:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x):
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": self.eps,
        }

    @staticmethod
    def from_state_dict(d):
        return ActionNormalizer(
            mean=d["mean"],
            std=d["std"],
            eps=d.get("eps", 1e-6),
        )


def compute_action_stats(dataset, max_samples=None):
    print("Computing action mean/std...")

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    total = None
    total_sq = None
    count = 0

    for i in range(n):
        sample = dataset[i]
        actions = sample["gt_actions"].float()  # [T, 6]

        if total is None:
            total = actions.sum(dim=0)
            total_sq = (actions ** 2).sum(dim=0)
        else:
            total += actions.sum(dim=0)
            total_sq += (actions ** 2).sum(dim=0)

        count += actions.shape[0]

    mean = total / count
    var = total_sq / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    print("action mean:", mean.tolist())
    print("action std: ", std.tolist())

    return ActionNormalizer(mean=mean, std=std)


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

        depth = depths.unsqueeze(2)            # [B, N, 1, H, W]
        rgbd = torch.cat([img, depth], dim=2)  # [B, N, 4, H, W]

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
        text_tokens:   [B, L, text_dim]
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


class LearnedTableBEV(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        bev_h=8,
        bev_w=8,
        num_heads=8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_bev_tokens = bev_h * bev_w

        self.bev_content = nn.Parameter(
            torch.randn(self.num_bev_tokens, hidden_dim) * 0.02
        )

        # Explicit canonical 2D table-grid coordinate embedding.
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

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

    def build_grid(self, device):
        ys = torch.linspace(-1.0, 1.0, self.bev_h, device=device)
        xs = torch.linspace(-1.0, 1.0, self.bev_w, device=device)

        yy, xx = torch.meshgrid(ys, xs)
        coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        coords = coords.reshape(self.num_bev_tokens, 2)

        return coords

    def forward(self, visual_tokens):
        """
        visual_tokens: [B, S, D]
        return:        [B, Q, D]
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device

        coords = self.build_grid(device)
        coord_emb = self.coord_mlp(coords)

        queries = self.bev_content + coord_emb
        queries = queries.unsqueeze(0).repeat(B, 1, 1)

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
        bev_tokens:    [B, Q, D]
        text_tokens:   [B, L, text_dim]
        robot_state:   [B, 6]
        action_prefix: [B, T, 6]
        return:        [B, T, 6]
        """
        B, T, _ = action_prefix.shape
        device = action_prefix.device

        state_token = self.state_proj(robot_state).unsqueeze(1)
        text_memory = self.text_proj(text_tokens)

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

        return self.action_out_proj(out)


class SO101PseudoBEVPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        text_dim=384,
        action_dim=6,
        state_dim=6,
        bev_h=8,
        bev_w=8,
        num_heads=8,
        max_cameras=8,
    ):
        super().__init__()

        self.rgbd_encoder = RGBDEncoder(hidden_dim=hidden_dim)

        self.camera_embed = nn.Embedding(max_cameras, hidden_dim)

        self.lang_img_fuser = LanguageImageFuser(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        self.table_bev = LearnedTableBEV(
            hidden_dim=hidden_dim,
            bev_h=bev_h,
            bev_w=bev_w,
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

        bev_tokens = self.table_bev(visual_tokens)

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


def make_normalized_prefix(gt_actions_norm):
    action_prefix = torch.zeros_like(gt_actions_norm)
    action_prefix[:, 1:] = gt_actions_norm[:, :-1]
    return action_prefix


def apply_camera_dropout(img, depths, drop_prob):
    """
    Randomly zeros full camera views during training.
    Ensures at least one view remains per sample.
    """
    if drop_prob <= 0:
        return img, depths

    B, N = img.shape[:2]

    if N <= 1:
        return img, depths

    device = img.device
    keep = torch.rand(B, N, device=device) > drop_prob

    # Ensure each sample keeps at least one camera.
    for b in range(B):
        if not keep[b].any():
            keep[b, torch.randint(0, N, (1,), device=device)] = True

    mask_img = keep.view(B, N, 1, 1, 1).float()
    mask_depth = keep.view(B, N, 1, 1).float()

    return img * mask_img, depths * mask_depth


def save_checkpoint(
    path,
    model,
    optimizer,
    normalizer,
    epoch,
    step,
    args,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def train(args):
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("device:", device)

    SO101CachedDepthLangDataset = load_dataset_class()

    dataset = SO101CachedDepthLangDataset(
        ann_file=args.ann_file,
    )

    print("num samples:", len(dataset))

    normalizer = compute_action_stats(
        dataset,
        max_samples=args.max_stat_samples,
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    model = SO101PseudoBEVPolicy(
        hidden_dim=args.hidden_dim,
        text_dim=args.text_dim,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        running_raw_l1 = 0.0
        t0 = time.time()

        for it, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            img = batch["img"]
            depths = batch["depths"]
            text_tokens = batch["text_tokens"]
            robot_state = batch["robot_state"]

            gt_actions_raw = batch["gt_actions"]
            gt_actions_norm = normalizer.normalize(gt_actions_raw)

            action_prefix_norm = make_normalized_prefix(gt_actions_norm)

            img, depths = apply_camera_dropout(
                img,
                depths,
                drop_prob=args.camera_dropout,
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                pred_actions_norm = model(
                    img=img,
                    depths=depths,
                    text_tokens=text_tokens,
                    robot_state=robot_state,
                    action_prefix=action_prefix_norm,
                )

                loss_l1 = F.smooth_l1_loss(pred_actions_norm, gt_actions_norm)
                loss_mse = F.mse_loss(pred_actions_norm, gt_actions_norm)
                loss = loss_l1 + args.mse_weight * loss_mse

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            else:
                grad_norm = torch.tensor(0.0)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred_actions_raw = normalizer.unnormalize(pred_actions_norm)
                raw_l1 = torch.mean(torch.abs(pred_actions_raw - gt_actions_raw))

            running_loss += float(loss.detach().cpu())
            running_raw_l1 += float(raw_l1.detach().cpu())
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = running_loss / (it + 1)
                avg_raw_l1 = running_raw_l1 / (it + 1)

                print(
                    f"epoch {epoch+1:03d}/{args.epochs:03d} "
                    f"iter {it+1:05d}/{len(loader):05d} "
                    f"step {global_step:06d} "
                    f"loss {avg_loss:.5f} "
                    f"raw_l1 {avg_raw_l1:.5f} "
                    f"grad {float(grad_norm):.3f}"
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        epoch_loss = running_loss / max(1, len(loader))
        epoch_raw_l1 = running_raw_l1 / max(1, len(loader))
        elapsed = time.time() - t0

        print(
            f"END epoch {epoch+1:03d}: "
            f"loss {epoch_loss:.5f} "
            f"raw_l1 {epoch_raw_l1:.5f} "
            f"time {elapsed:.1f}s"
        )

        ckpt_path = os.path.join(
            args.save_dir,
            f"pseudo_bev_epoch_{epoch+1:03d}.pth",
        )

        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            normalizer=normalizer,
            epoch=epoch + 1,
            step=global_step,
            args=args,
        )

        print("saved:", ckpt_path)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_path = os.path.join(args.save_dir, "pseudo_bev_latest.pth")

    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        epoch=epoch + 1,
        step=global_step,
        args=args,
    )

    print("saved latest:", final_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ann-file",
        default="data/so101_depth_lang_cache/train_index.json",
    )
    parser.add_argument(
        "--save-dir",
        default="work_dirs/so101_pseudo_bev",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=384)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--bev-h", type=int, default=8)
    parser.add_argument("--bev-w", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--camera-dropout", type=float, default=0.0)
    parser.add_argument("--max-stat-samples", type=int, default=5000)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
