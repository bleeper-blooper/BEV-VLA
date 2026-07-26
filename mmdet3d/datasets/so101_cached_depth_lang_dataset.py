import os
import json
import torch
from torch.utils.data import Dataset

try:
    from mmdet.datasets import DATASETS
except Exception:
    DATASETS = None


def build_fake_calib(num_cams, image_size):
    H, W = image_size

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


class _SO101CachedDepthLangDataset(Dataset):
    def __init__(
        self,
        ann_file,
        test_mode=False,
        **kwargs,
    ):
        super().__init__()

        self.ann_file = ann_file
        self.root_dir = os.path.dirname(os.path.abspath(ann_file))
        self.test_mode = test_mode

        with open(ann_file, "r") as f:
            self.index = json.load(f)

        self.samples = self.index["samples"]
        self.image_size = tuple(self.index["image_size"])
        self.camera_keys = self.index["camera_keys"]

    def __len__(self):
        return len(self.samples)

    def _resolve_sample_path(self, p):
        if os.path.isabs(p):
            return p
        return os.path.join(self.root_dir, p)

    def __getitem__(self, idx):
        sample_path = self._resolve_sample_path(self.samples[idx])
        sample = torch.load(sample_path, map_location="cpu")

        img = sample["img"].float()                      # [N, 3, H, W]
        depths = sample["depths"].float()                # [N, H, W]
        text_tokens = sample["text_tokens"].float()      # [L, D]

        robot_state = sample["robot_state"].float()      # [6]
        action_prefix = sample["action_prefix"].float()  # [T, 6]
        gt_actions = sample["gt_actions"].float()        # [T, 6]

        num_cams = img.shape[0]
        calib = build_fake_calib(num_cams, self.image_size)

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

            # Compatibility placeholders.
            "points": [],
            "radar": [],

            "metas": {
                "dataset": "so101",
                "sample_path": sample_path,
                "dataset_index": sample["dataset_index"],
                "episode_index": sample["episode_index"],
                "frame_index": sample["frame_index"],
                "global_index": sample.get("global_index", sample["dataset_index"]),
                "task_index": sample["task_index"],
                "instruction": sample["instruction"],
                "camera_keys": sample["camera_keys"],
                "fake_calib": True,
            },
        }

    def evaluate(self, results, **kwargs):
        return {}


if DATASETS is not None:
    @DATASETS.register_module()
    class SO101CachedDepthLangDataset(_SO101CachedDepthLangDataset):
        pass
else:
    class SO101CachedDepthLangDataset(_SO101CachedDepthLangDataset):
        pass
