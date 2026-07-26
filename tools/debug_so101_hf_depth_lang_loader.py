import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)
from huggingface_hub import hf_hub_download
import imageio.v3 as iio


class SO101HFDepthLangDataset(Dataset):
    def __init__(
        self,
        repo_id="whosricky/so101-megamix-v1",
        split="train",
        camera_keys=("observation.images.front",),
        image_size=(256, 320),
        pred_horizon=16,
        instruction="pick up the object and complete the task",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda",
        max_text_len=32,
        max_samples=32,
    ):
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)
        self.pred_horizon = pred_horizon
        self.instruction = instruction
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_text_len = max_text_len
        self.max_samples = max_samples

        self.dataset = load_dataset(repo_id, split=split)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(self.device)
        self.text_model.eval()

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(self.device)
        self.depth_model.eval()
        self.repo_id = repo_id
        self._video_cache = {}

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)

    def _to_chw_float(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        if not torch.is_tensor(img):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()

        if img.max() > 2.0:
            img = img / 255.0

        img = F.interpolate(
            img.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return img.contiguous()

    def _tensor_to_pil(self, img_chw):
        img = img_chw.detach().cpu().clamp(0.0, 1.0)
        img = (img * 255.0).byte()
        img = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img)

    def _download_video(self, video_key, episode_index):
        chunk_index = int(episode_index) // 1000
        file_index = int(episode_index)

        filename = (
            f"videos/{video_key}/"
            f"chunk-{chunk_index:03d}/"
            f"file-{file_index:03d}.mp4"
        )

        cache_key = (video_key, episode_index)
        if cache_key not in self._video_cache:
            self._video_cache[cache_key] = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
            )

        return self._video_cache[cache_key]


    def _read_video_frame(self, video_path, frame_index):
        frame = iio.imread(video_path, index=int(frame_index))

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        return Image.fromarray(frame.astype(np.uint8))

    @torch.no_grad()
    def _encode_text(self, text):
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.text_model(**inputs)

        return out.last_hidden_state[0].detach().cpu().float()

    @torch.no_grad()
    def _infer_depth(self, img_chw):
        pil = self._tensor_to_pil(img_chw)

        inputs = self.depth_processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.depth_model(**inputs)
        depth = out.predicted_depth

        depth = F.interpolate(
            depth.unsqueeze(1),
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
        )[0, 0]

        depth = depth.detach().cpu().float()
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-6)

        return depth.contiguous()

    def _fake_calib(self, num_cams):
        H, W = self.image_size

        camera2ego = torch.eye(4).float().repeat(num_cams, 1, 1)
        camera2lidar = torch.eye(4).float().repeat(num_cams, 1, 1)
        lidar2camera = torch.eye(4).float().repeat(num_cams, 1, 1)

        K = torch.eye(4).float().repeat(num_cams, 1, 1)
        K[:, 0, 0] = float(W)
        K[:, 1, 1] = float(W)
        K[:, 0, 2] = float(W) / 2.0
        K[:, 1, 2] = float(H) / 2.0

        lidar2image = K @ lidar2camera

        return {
            "camera2ego": camera2ego,
            "lidar2ego": torch.eye(4).float(),
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "camera_intrinsics": K,
            "camera2lidar": camera2lidar,
            "img_aug_matrix": torch.eye(4).float().repeat(num_cams, 1, 1),
            "lidar_aug_matrix": torch.eye(4).float(),
        }

    def _get_action_chunk(self, idx):
        actions = []
        start_row = self.dataset[idx]
        episode_index = int(start_row["episode_index"])

        last_action = torch.tensor(start_row["action"]).float()

        for j in range(self.pred_horizon):
            jj = min(idx + j, len(self.dataset) - 1)
            row = self.dataset[jj]

            if int(row["episode_index"]) != episode_index:
                action = last_action
            else:
                action = torch.tensor(row["action"]).float()
                last_action = action

            actions.append(action)

        return torch.stack(actions, dim=0)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        imgs = []
        depths = []

        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])

        for key in self.camera_keys:
            video_path = self._download_video(key, episode_index)
            pil_img = self._read_video_frame(video_path, frame_index)

            img = self._to_chw_float(pil_img)
            depth = self._infer_depth(img)

            imgs.append(img)
            depths.append(depth)

        img = torch.stack(imgs, dim=0)          # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)    # [N, H, W]

        robot_state = torch.tensor(row["observation.state"]).float()

        gt_actions = self._get_action_chunk(idx)  # [T, 6]

        action_prefix = torch.zeros_like(gt_actions)
        action_prefix[1:] = gt_actions[:-1]

        text_tokens = self._encode_text(self.instruction)

        calib = self._fake_calib(img.shape[0])

        return {
            "img": img,
            "depths": depths,
            "text_tokens": text_tokens,

            "robot_state": robot_state,
            "action_prefix": action_prefix,
            "gt_actions": gt_actions,

            "camera2ego": calib["camera2ego"],
            "lidar2ego": calib["lidar2ego"],
            "lidar2camera": calib["lidar2camera"],
            "lidar2image": calib["lidar2image"],
            "camera_intrinsics": calib["camera_intrinsics"],
            "camera2lidar": calib["camera2lidar"],
            "img_aug_matrix": calib["img_aug_matrix"],
            "lidar_aug_matrix": calib["lidar_aug_matrix"],

            "metas": {
                "idx": idx,
                "camera_keys": self.camera_keys,
                "instruction": self.instruction,
            },
        }


def main():
    dataset = SO101HFDepthLangDataset(
        repo_id="whosricky/so101-megamix-v1",
        split="train",
        camera_keys=("observation.images.front",),
        image_size=(256, 320),
        pred_horizon=16,
        max_samples=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nTensor shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:20s}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"{k:20s}: dict")
        else:
            print(f"{k:20s}: {type(v)}")

    print("\nExpected:")
    print("img:           [B, N, 3, H, W]")
    print("depths:        [B, N, H, W]")
    print("text_tokens:   [B, L, D]")
    print("robot_state:   [B, 6]")
    print("action_prefix: [B, T, 6]")
    print("gt_actions:    [B, T, 6]")


if __name__ == "__main__":
    main()