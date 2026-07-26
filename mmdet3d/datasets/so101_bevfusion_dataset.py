import torch
import torch.nn.functional as F
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None


@DATASETS.register_module()
class SO101BEVFusionDataset:
    def __init__(
        self,
        repo_id="whosricky/so101-megamix-v1",
        camera_keys=("observation.images.front",),
        image_size=(256, 320),
        pred_horizon=16,
        fps=30,
        fake_calib=True,
        test_mode=False,
        **kwargs,
    ):
        assert LeRobotDataset is not None, (
            "LeRobot is not installed. Install lerobot first, or use a preconverted local cache."
        )

        self.repo_id = repo_id
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)
        self.pred_horizon = pred_horizon
        self.fps = fps
        self.fake_calib = fake_calib
        self.test_mode = test_mode

        delta_timestamps = {
            "action": [i / fps for i in range(pred_horizon)],
        }

        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
        )

    def __len__(self):
        return len(self.dataset)

    def _to_chw_float(self, img):
        """
        Converts image to [3, H, W], float32, range [0, 1].
        Handles torch tensors from LeRobot.
        """
        if not torch.is_tensor(img):
            img = torch.tensor(img)

        # [H, W, C] -> [C, H, W]
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

    def _make_fake_calib(self, num_cams):
        """
        Fake tabletop calibration only for dataloader/model-shape testing.

        Later, replace this with real camera-to-robot/camera-to-base calibration.
        """
        H, W = self.image_size

        camera2ego = torch.eye(4).float().repeat(num_cams, 1, 1)
        camera2lidar = torch.eye(4).float().repeat(num_cams, 1, 1)
        lidar2camera = torch.eye(4).float().repeat(num_cams, 1, 1)

        # Simple pinhole intrinsics for resized images.
        K = torch.eye(4).float().repeat(num_cams, 1, 1)
        K[:, 0, 0] = float(W)
        K[:, 1, 1] = float(W)
        K[:, 0, 2] = float(W) / 2.0
        K[:, 1, 2] = float(H) / 2.0

        lidar2image = K @ lidar2camera

        img_aug_matrix = torch.eye(4).float().repeat(num_cams, 1, 1)
        lidar_aug_matrix = torch.eye(4).float()
        lidar2ego = torch.eye(4).float()

        return {
            "camera2ego": camera2ego,
            "lidar2ego": lidar2ego,
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "camera_intrinsics": K,
            "camera2lidar": camera2lidar,
            "img_aug_matrix": img_aug_matrix,
            "lidar_aug_matrix": lidar_aug_matrix,
        }

    def __getitem__(self, idx):
        item = self.dataset[idx]

        imgs = []
        for key in self.camera_keys:
            imgs.append(self._to_chw_float(item[key]))

        # [N, C, H, W]
        img = torch.stack(imgs, dim=0)

        num_cams = img.shape[0]
        H, W = self.image_size

        robot_state = item["observation.state"].float()

        gt_actions = item["action"].float()
        if gt_actions.ndim == 1:
            gt_actions = gt_actions.unsqueeze(0)

        # Guarantee [T, 6]
        if gt_actions.shape[0] < self.pred_horizon:
            pad = gt_actions[-1:].repeat(self.pred_horizon - gt_actions.shape[0], 1)
            gt_actions = torch.cat([gt_actions, pad], dim=0)
        else:
            gt_actions = gt_actions[: self.pred_horizon]

        # Teacher-forcing input: zero token + shifted ground-truth actions.
        action_prefix = torch.zeros_like(gt_actions)
        action_prefix[1:] = gt_actions[:-1]

        calib = self._make_fake_calib(num_cams)

        metas = {
            "dataset": "so101-megamix-v1",
            "idx": int(idx),
            "camera_keys": self.camera_keys,
            "image_size": self.image_size,
            "fake_calib": self.fake_calib,
        }

        # IMPORTANT:
        # BEVFusion.forward requires "depths", even if LSSTransform ignores it.
        depths = torch.zeros(num_cams, H, W).float()

        return {
            # BEVFusion camera input
            "img": DC(img, stack=True),

            # Camera geometry
            "camera2ego": DC(calib["camera2ego"], stack=True),
            "lidar2ego": DC(calib["lidar2ego"], stack=True),
            "lidar2camera": DC(calib["lidar2camera"], stack=True),
            "lidar2image": DC(calib["lidar2image"], stack=True),
            "camera_intrinsics": DC(calib["camera_intrinsics"], stack=True),
            "camera2lidar": DC(calib["camera2lidar"], stack=True),
            "img_aug_matrix": DC(calib["img_aug_matrix"], stack=True),
            "lidar_aug_matrix": DC(calib["lidar_aug_matrix"], stack=True),

            # Required by BEVFusion.forward, unused for camera-only LSSTransform
            "points": DC([], cpu_only=True),
            "radar": DC([], cpu_only=True),
            "depths": DC(depths, stack=True),

            # Robot policy fields
            "robot_state": DC(robot_state, stack=True),
            "action_prefix": DC(action_prefix, stack=True),
            "gt_actions": DC(gt_actions, stack=True),

            # Metadata
            "metas": DC(metas, cpu_only=True),
        }

    def evaluate(self, results, **kwargs):
        return {}