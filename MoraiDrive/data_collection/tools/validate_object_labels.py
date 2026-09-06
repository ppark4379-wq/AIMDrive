#!/usr/bin/env python3
"""Validate MORAI actors_data and generated 3D bounding-box files."""
import argparse
import glob
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_dir")
    args = parser.parse_args()
    json_paths = sorted(glob.glob(os.path.join(args.route_dir, "actors_data", "*.json")))
    frames_with_actors = 0
    actors = 0
    valid_extent = 0
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            frame = json.load(f)
        if frame:
            frames_with_actors += 1
        for actor in frame.values():
            actors += 1
            if len(actor.get("box", [])) == 2 and all(v > 0 for v in actor["box"]):
                valid_extent += 1
    box_paths = sorted(glob.glob(os.path.join(args.route_dir, "3d_bbs", "*.npy")))
    boxes = 0
    for path in box_paths:
        data = np.load(path, allow_pickle=True).item()
        boxes += len(data.get("vehicles", [])) + len(data.get("pedestrians", []))
    print(f"frames: {len(json_paths)}")
    print(f"frames with actors: {frames_with_actors}")
    print(f"actor records: {actors}")
    print(f"actors with valid extents: {valid_extent}")
    print(f"3d_bbs records: {boxes}")
    failures = []
    if actors == 0:
        failures.append("객체 없음")
    if actors != valid_extent:
        failures.append("0 또는 누락된 객체 크기")
    if boxes != actors:
        failures.append("actors_data와 3d_bbs 개수 불일치")
    print("result:", "FAIL - " + "; ".join(failures) if failures else "PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
