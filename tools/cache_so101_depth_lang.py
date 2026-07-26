import os
import json
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


def parse_camera_keys(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def to_chw_float(img, image_size):
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
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return img.contiguous()


def tensor_to_pil(img_chw):
    img = img_chw.detach().cpu().clamp(0.0, 1.0)
    img = (img * 255.0).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def read_video_frame(video_path, frame_index):
    frame = iio.imread(video_path, index=int(frame_index))

    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    return Image.fromarray(frame.astype(np.uint8))


def discover_video_files(repo_id, camera_keys):
    repo_files = list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    video_files = {}

    for key in camera_keys:
        prefix = f"videos/{key}/"
        files = sorted(
            f for f in repo_files
            if f.startswith(prefix) and f.endswith(".mp4")
        )

        if len(files) == 0:
            raise RuntimeError(f"No video files found for camera key: {key}")

        print(f"{key}: found {len(files)} video file(s)")
        for f in files[:10]:
            print("  ", f)

        video_files[key] = files

    return video_files


def download_video(repo_id, video_key, episode_index, video_files, video_cache):
    files = video_files[video_key]

    # Observed SO101 case:
    # one huge video file per camera, so use the global frame index later.
    if len(files) == 1:
        filename = files[0]
    else:
        # Fallback for one-video-per-episode datasets.
        chunk_index = int(episode_index) // 1000
        file_index = int(episode_index)

        filename = (
            f"videos/{video_key}/"
            f"chunk-{chunk_index:03d}/"
            f"file-{file_index:03d}.mp4"
        )

        if filename not in files:
            raise RuntimeError(
                f"Expected video file not found: {filename}\n"
                f"Available examples for {video_key}: {files[:20]}"
            )

    if filename not in video_cache:
        video_cache[filename] = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )

    return video_cache[filename], filename


@torch.no_grad()
def encode_text(text, tokenizer, text_model, device, max_text_len):
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = text_model(**inputs)
    return out.last_hidden_state[0].detach().cpu().float()


@torch.no_grad()
def infer_depth(img_chw, depth_processor, depth_model, device, image_size):
    pil = tensor_to_pil(img_chw)

    inputs = depth_processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = depth_model(**inputs)
    depth = out.predicted_depth

    depth = F.interpolate(
        depth.unsqueeze(1),
        size=image_size,
        mode="bicubic",
        align_corners=False,
    )[0, 0]

    depth = depth.detach().cpu().float()

    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-6)

    return depth.contiguous()


def get_action_chunk(ds, idx, pred_horizon):
    actions = []
    start_row = ds[idx]
    episode_index = int(start_row["episode_index"])

    last_action = torch.tensor(start_row["action"]).float()

    for j in range(pred_horizon):
        jj = min(idx + j, len(ds) - 1)
        row = ds[jj]

        if int(row["episode_index"]) != episode_index:
            action = last_action
        else:
            action = torch.tensor(row["action"]).float()
            last_action = action

        actions.append(action)

    return torch.stack(actions, dim=0)


def make_instruction(task_index):
    return f"perform manipulation task {int(task_index)}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-id", default="whosricky/so101-megamix-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="data/so101_depth_lang_cache")
    parser.add_argument(
        "--camera-keys",
        default="observation.images.front",
        help="Comma-separated camera keys.",
    )

    parser.add_argument("--image-h", type=int, default=256)
    parser.add_argument("--image-w", type=int, default=320)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
    )

    args = parser.parse_args()

    image_size = (args.image_h, args.image_w)
    camera_keys = parse_camera_keys(args.camera_keys)

    os.makedirs(args.out_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("discovering video files...")
    video_files = discover_video_files(args.repo_id, camera_keys)

    print("loading dataset...")
    ds = load_dataset(args.repo_id, split=args.split)

    print("dataset columns:", ds.column_names)
    print("num rows:", len(ds))

    print("loading text model...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModel.from_pretrained(args.text_model).to(device)
    text_model.eval()

    print("loading depth model...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(device)
    depth_model.eval()

    video_cache = {}
    text_cache = {}
    sample_files = []

    end = min(len(ds), args.start + args.max_samples * args.stride)
    indices = list(range(args.start, end, args.stride))
    indices = indices[: args.max_samples]

    for idx in tqdm(indices, desc="caching samples"):
        row = ds[idx]

        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        global_index = int(row["index"])
        task_index = int(row["task_index"])

        imgs = []
        depths = []
        used_video_files = []

        for key in camera_keys:
            video_path, video_filename = download_video(
                repo_id=args.repo_id,
                video_key=key,
                episode_index=episode_index,
                video_files=video_files,
                video_cache=video_cache,
            )

            # If there is one large video per camera, use global frame index.
            # If there is one video per episode, use per-episode frame index.
            if len(video_files[key]) == 1:
                video_frame_index = global_index
            else:
                video_frame_index = frame_index

            pil_img = read_video_frame(video_path, video_frame_index)
            img = to_chw_float(pil_img, image_size)

            depth = infer_depth(
                img,
                depth_processor=depth_processor,
                depth_model=depth_model,
                device=device,
                image_size=image_size,
            )

            imgs.append(img)
            depths.append(depth)
            used_video_files.append(video_filename)

        img = torch.stack(imgs, dim=0)          # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)     # [N, H, W]

        robot_state = torch.tensor(row["observation.state"]).float()

        gt_actions = get_action_chunk(
            ds=ds,
            idx=idx,
            pred_horizon=args.pred_horizon,
        )

        action_prefix = torch.zeros_like(gt_actions)
        action_prefix[1:] = gt_actions[:-1]

        instruction = make_instruction(task_index)

        if task_index not in text_cache:
            text_cache[task_index] = encode_text(
                instruction,
                tokenizer=tokenizer,
                text_model=text_model,
                device=device,
                max_text_len=args.max_text_len,
            )

        text_tokens = text_cache[task_index]

        sample = {
            "img": img,
            "depths": depths,
            "text_tokens": text_tokens,

            "robot_state": robot_state,
            "action_prefix": action_prefix,
            "gt_actions": gt_actions,

            "episode_index": episode_index,
            "frame_index": frame_index,
            "global_index": global_index,
            "dataset_index": int(idx),
            "task_index": task_index,
            "instruction": instruction,
            "camera_keys": camera_keys,
            "video_files": used_video_files,
            "image_size": image_size,
        }

        sample_name = f"sample_{idx:09d}.pt"
        sample_path = os.path.join(sample_dir, sample_name)

        torch.save(sample, sample_path)

        rel_sample_path = os.path.relpath(sample_path, args.out_dir)
        sample_files.append(rel_sample_path)

    index = {
        "repo_id": args.repo_id,
        "split": args.split,
        "camera_keys": camera_keys,
        "image_size": list(image_size),
        "pred_horizon": args.pred_horizon,
        "max_text_len": args.max_text_len,
        "num_samples": len(sample_files),
        "samples": sample_files,
    }

    index_path = os.path.join(args.out_dir, f"{args.split}_index.json")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("wrote:", index_path)
    print("num samples:", len(sample_files))


if __name__ == "__main__":
    main()
