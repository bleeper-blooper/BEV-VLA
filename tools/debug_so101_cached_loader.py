import importlib.util
import torch
from torch.utils.data import DataLoader


def load_dataset_class():
    path = "mmdet3d/datasets/so101_cached_depth_lang_dataset.py"
    spec = importlib.util.spec_from_file_location("so101_cached_depth_lang_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SO101CachedDepthLangDataset


def main():
    SO101CachedDepthLangDataset = load_dataset_class()

    dataset = SO101CachedDepthLangDataset(
        ann_file="data/so101_depth_lang_cache/train_index.json",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Tensor shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:20s}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"{k:20s}: dict")
        else:
            print(f"{k:20s}: {type(v)}")

    print("\nExpected important shapes:")
    print("img:           [B, N, 3, H, W]")
    print("depths:        [B, N, H, W]")
    print("text_tokens:   [B, L, D]")
    print("robot_state:   [B, 6]")
    print("action_prefix: [B, T, 6]")
    print("gt_actions:    [B, T, 6]")


if __name__ == "__main__":
    main()
