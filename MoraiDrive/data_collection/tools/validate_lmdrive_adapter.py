#!/usr/bin/env python3
"""Load one MORAI frame through MoraiDrive's MoraiMVDetDataset adapter."""
import argparse
import glob
import importlib
import os
import sys
import types

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_dir")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    timm_root = os.path.join(repo_root, "MoraiDrive", "vision_encoder", "timm")
    # Load only timm.data, avoiding model-only torch_scatter during data validation.
    timm = types.ModuleType("timm")
    timm.__path__ = [timm_root]
    sys.modules["timm"] = timm
    data_pkg = types.ModuleType("timm.data")
    data_pkg.__path__ = [os.path.join(timm_root, "data")]
    sys.modules["timm.data"] = data_pkg
    dataset_module = importlib.import_module("timm.data.morai_dataset")

    dataset = dataset_module.MoraiMVDetDataset.__new__(
        dataset_module.MoraiMVDetDataset
    )
    dataset.root_path = ""
    dataset.waypoint_interval = 5
    dataset.with_waypoints = dataset.with_seg = dataset.with_depth = False
    dataset.with_lidar = dataset.multi_view = True
    dataset.seg_transform = dataset.depth_transform = dataset.lidar_transform = None

    def image_to_tensor(image):
        array = np.asarray(image).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    dataset.rgb_transform = image_to_tensor
    dataset.rgb_center_transform = image_to_tensor
    dataset.multi_view_transform = image_to_tensor
    data, targets = dataset._extract_data_item(
        os.path.abspath(args.route_dir), args.frame
    )
    expected = {
        "rgb_front": 3, "rgb_left": 3, "rgb_right": 3, "rgb_rear": 3,
        "lidar": 2, "target_point": 1,
    }
    failures = []
    for key, dimensions in expected.items():
        shape = tuple(data[key].shape)
        print(f"{key}: {shape}")
        if len(shape) != dimensions:
            failures.append(f"{key} 차원 오류")
    print(f"num_points: {data['num_points']}")
    print("target shapes:", [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                             for v in targets])
    if tuple(data["lidar"].shape) != (40000, 4):
        failures.append("LiDAR shape 오류")
    print("result:", "FAIL - " + "; ".join(failures) if failures else "PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
