#!/usr/bin/env python3
"""Cache frozen MORAI vision tokens and actual future-ego trajectory targets.

The command is resumable at frame granularity.  It always uses evaluation RGB
transforms, deterministic LiDAR sampling, the current LiDAR sweep only, and the
recorded current ego speed used by MORAI vision-encoder training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parent
VISION_ROOT = REPOSITORY / "vision_encoder"
os.environ.setdefault("TORCH_HOME", str(VISION_ROOT / ".cache" / "torch"))
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

import numpy as np
import torch
from timm.data import create_carla_dataset, create_carla_loader
from timm.models import create_model


SCHEMA = "morai-vldrive-vision-cache-v1"
FEATURE_MODEL = "memfuser_baseline_e1d3_return_feature"
FEATURE_SHAPE = (106, 256)
TRAJECTORY_SHAPE = (5, 2)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    pending.replace(path)


def relative_path(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def make_dataset_and_loader(args, split):
    dataset = create_carla_dataset(
        "morai",
        root=str(args.dataset_root),
        split_file=str(args.split_file),
        split=split,
        with_lidar=True,
        multi_view=True,
        lidar_axis_policy="morai_flu_to_lmdrive_fru",
        sample_stride=1,
    )
    # Entries were selected above.  This switch only makes >40k point sampling
    # deterministic for training-split frames while caching frozen features.
    dataset.split = "validation"
    loader = create_carla_loader(
        dataset,
        input_size=(3, 224, 224),
        multi_view_input_size=(3, 128, 128),
        batch_size=args.batch_size,
        is_training=False,
        scale=[1.0, 1.0],
        color_jitter=0.0,
        interpolation="bilinear",
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return dataset, loader


def load_feature_model(checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("feature_model_name") != FEATURE_MODEL:
        raise ValueError("Checkpoint does not declare the expected feature model")
    if checkpoint.get("token_shape") != list(FEATURE_SHAPE):
        raise ValueError("Checkpoint token contract mismatch")
    model = create_model(FEATURE_MODEL)
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, str(result)


def record_is_valid(record_path, output):
    if not record_path.is_file():
        return False
    try:
        row = json.loads(record_path.read_text(encoding="utf-8"))
        for name in ("visual_feature", "future_trajectory"):
            ref = row[name]
            path = output / ref["path"]
            if not path.is_file() or sha256(path) != ref["sha256"]:
                return False
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            expected = FEATURE_SHAPE if name == "visual_feature" else TRAJECTORY_SHAPE
            if tuple(array.shape) != expected or not np.isfinite(array).all():
                return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def cache_split(args, model, device, split, completed):
    dataset, loader = make_dataset_and_loader(args, split)
    cursor = 0
    for inputs, targets in loader:
        batch_count = int(inputs["rgb_front"].shape[0])
        identities = dataset.route_frames[cursor:cursor + batch_count]
        cursor += batch_count
        missing = []
        for offset, (route_id, frame_id) in enumerate(identities):
            record = args.output / "records" / route_id / ("{:04d}.json".format(frame_id))
            if record_is_valid(record, args.output):
                completed.add((route_id, int(frame_id)))
            else:
                missing.append(offset)
        if not missing:
            continue

        selected = torch.tensor(missing, dtype=torch.long)
        model_inputs = {key: value.index_select(0, selected).to(device, non_blocking=True)
                        for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
            features = model(model_inputs)
        features = features.float().cpu().numpy()
        if tuple(features.shape[1:]) != FEATURE_SHAPE or not np.isfinite(features).all():
            raise RuntimeError("Invalid frozen feature output")

        for output_index, source_index in enumerate(missing):
            route_id, frame_id = identities[source_index]
            native = int(targets["native_sync_frame"][source_index])
            feature = features[output_index].astype(np.float16)
            trajectory = targets["waypoints"][source_index].numpy().astype(np.float32)
            if trajectory.shape != TRAJECTORY_SHAPE or not np.isfinite(trajectory).all():
                raise RuntimeError("Invalid actual future-ego trajectory")
            feature_path = args.output / "features" / route_id / ("{:04d}.npy".format(frame_id))
            target_path = args.output / "targets" / route_id / ("{:04d}.npy".format(frame_id))
            write_npy(feature_path, feature)
            write_npy(target_path, trajectory)
            row = {
                "schema_version": SCHEMA,
                "route_id": route_id,
                "frame_id": int(frame_id),
                "native_sync_frame": native,
                "split": "validation" if split in ("val", "validation") else split,
                "visual_feature": {
                    "path": relative_path(feature_path, args.output),
                    "shape": list(FEATURE_SHAPE),
                    "dtype": "float16",
                    "sha256": sha256(feature_path),
                },
                "future_trajectory": {
                    "path": relative_path(target_path, args.output),
                    "shape": list(TRAJECTORY_SHAPE),
                    "dtype": "float32",
                    "horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5],
                    "source": "future_synchronized_ego_positions",
                    "sha256": sha256(target_path),
                },
            }
            record_path = args.output / "records" / route_id / ("{:04d}.json".format(frame_id))
            record_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(record_path, row)
            completed.add((route_id, int(frame_id)))
            if len(completed) == 1 or len(completed) % args.log_interval == 0:
                print("cached frames={}".format(len(completed)), flush=True)
    if cursor != len(dataset):
        raise RuntimeError("Cache loader did not visit every {} frame".format(split))
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if min(args.batch_size, args.log_interval) < 1 or args.workers < 0:
        raise ValueError("Invalid batch-size/workers/log-interval")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.split_file = args.split_file.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    split_sha = sha256(args.split_file)
    checkpoint_sha = sha256(args.checkpoint)
    partial_meta = args.output / "cache_meta.partial.json"
    implementation_files = [
        Path(__file__).resolve(),
        VISION_ROOT / "timm" / "data" / "carla_dataset.py",
        VISION_ROOT / "timm" / "data" / "carla_loader.py",
        VISION_ROOT / "timm" / "data" / "transforms_carla_factory.py",
        VISION_ROOT / "timm" / "models" / "memfuser.py",
        VISION_ROOT / "timm" / "models" / "pointpillar.py",
    ]
    identity = {
        "schema_version": SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "split_sha256": split_sha,
        "dataset_index_sha256": sha256(args.dataset_root / "dataset_index.txt"),
        "implementation_sha256": {
            str(path.relative_to(REPOSITORY)): sha256(path) for path in implementation_files
        },
        "feature_model": FEATURE_MODEL,
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float16",
        "velocity_source": "current_ego_speed",
        "target_point_source": "recorded_global_path_x_command_y_command",
        "lidar_sweeps": 1,
        "lidar_axis_policy": "morai_flu_to_lmdrive_fru",
        "rgb_preprocessing": "lmdrive_eval_resize_crop_imagenet",
        "trajectory_source": "future_synchronized_ego_positions",
    }
    if (args.output / "cache_meta.json").exists():
        raise FileExistsError("Completed cache already exists: {}".format(args.output))
    if partial_meta.exists():
        if json.loads(partial_meta.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing partial cache belongs to a different checkpoint/split")
    else:
        write_json(partial_meta, identity)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to cache MORAI vision tokens")
    device = torch.device("cuda:0")
    model, strict_load = load_feature_model(args.checkpoint, device)
    completed = set()
    expected = 0
    for split in ("train", "validation"):
        expected += cache_split(args, model, device, split, completed)
    if len(completed) != expected:
        raise RuntimeError("Cache completeness mismatch: {} != {}".format(len(completed), expected))

    rows = []
    for path in sorted((args.output / "records").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row["route_id"], int(row["frame_id"])) in completed:
            rows.append(row)
    rows.sort(key=lambda row: (row["route_id"], int(row["frame_id"])))
    if len(rows) != expected:
        raise RuntimeError("Record manifest completeness mismatch")
    manifest = args.output / "manifest.jsonl"
    pending = manifest.with_suffix(manifest.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    pending.replace(manifest)
    final = dict(identity)
    final.update(
        records=expected,
        strict_load=strict_load,
        manifest_sha256=sha256(manifest),
    )
    write_json(args.output / "cache_meta.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
