#!/usr/bin/env python3
"""Validate a MORAI batch through LMDrive's real dataset and loader path."""
import argparse
import importlib
import os
import sys
import types

import torch


def _load_morai_data_modules(repo_root):
    """Import only MoraiDrive's timm.data package, avoiding model-only deps."""
    timm_root = os.path.join(repo_root, "MoraiDrive", "vision_encoder", "timm")
    timm = types.ModuleType("timm")
    timm.__path__ = [timm_root]
    sys.modules["timm"] = timm
    data_pkg = types.ModuleType("timm.data")
    data_pkg.__path__ = [os.path.join(timm_root, "data")]
    sys.modules["timm.data"] = data_pkg
    dataset_factory = importlib.import_module("timm.data.dataset_factory")
    morai_loader = importlib.import_module("timm.data.morai_loader")
    return dataset_factory, morai_loader


def _shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="directory containing dataset_index.txt and data/")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--multi-view-input-size", type=int, default=128)
    parser.add_argument("--index-file", default="dataset_index.txt")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for multi-frame validation")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    dataset_factory, morai_loader = _load_morai_data_modules(repo_root)
    dataset_root = os.path.abspath(args.dataset_root)

    dataset = dataset_factory.create_morai_dataset(
        "morai",
        root=dataset_root,
        with_lidar=True,
        with_seg=False,
        with_depth=False,
        multi_view=True,
        index_filename=args.index_file,
    )
    loader = morai_loader.create_morai_loader(
        dataset,
        input_size=(3, args.input_size, args.input_size),
        multi_view_input_size=(
            3, args.multi_view_input_size, args.multi_view_input_size
        ),
        batch_size=args.batch_size,
        is_training=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    print(f"dataset class: {type(dataset).__name__}")
    print(f"eligible route frames: {len(dataset.route_frames)}")
    print(f"dataset samples (all valid frames): {len(dataset)}")
    print(f"loader batches: {len(loader)}")

    failures = []
    expected_inputs = {
        "rgb_front", "rgb_left", "rgb_right", "rgb_rear", "rgb_center",
        "lidar", "num_points", "measurements", "velocity", "command",
        "target_point",
    }
    checked = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= args.batches:
            break
        checked += 1
        print(f"batch {batch_index} input shapes:")
        for key, value in inputs.items():
            print(f"  {key}: {_shape(value)} {getattr(value, 'dtype', '')}")
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                failures.append(f"batch {batch_index} input {key} has NaN/Inf")
        print(f"batch {batch_index} target shapes: {[ _shape(v) for v in targets ]}")

        missing = expected_inputs.difference(inputs)
        if missing:
            failures.append(f"batch {batch_index} missing inputs: {sorted(missing)}")
        actual_batch = inputs["rgb_front"].shape[0]
        if actual_batch != args.batch_size:
            failures.append(
                f"batch {batch_index} size is {actual_batch}, expected {args.batch_size}"
            )
        if tuple(inputs["lidar"].shape[1:]) != (40000, 4):
            failures.append(f"batch {batch_index} lidar shape is invalid")
        if tuple(inputs["target_point"].shape[1:]) != (2,):
            failures.append(f"batch {batch_index} target_point shape is invalid")
        for target_index, value in enumerate(targets):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                failures.append(
                    f"batch {batch_index} target {target_index} has NaN/Inf"
                )

    if checked == 0:
        failures.append("loader produced no batches")
    print("result:", "FAIL - " + "; ".join(failures) if failures else "PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
