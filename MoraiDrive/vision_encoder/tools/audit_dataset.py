#!/usr/bin/env python3
"""Audit a prepared MORAI dataset before starting expensive GPU training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_FILES = {
    "rgb_front": ".jpg",
    "rgb_left": ".jpg",
    "rgb_right": ".jpg",
    "rgb_rear": ".jpg",
    "lidar": ".npy",
    "3d_bbs": ".npy",
    "measurements_full": ".json",
}
REQUIRED_MEASUREMENT_KEYS = {
    "gps_x", "gps_y", "theta", "speed", "command", "x_command",
    "y_command", "actors_data", "is_junction", "is_red_light_present",
    "stop_sign",
}


def read_index(root: Path, filename: str):
    path = root / filename
    if not path.is_file():
        raise ValueError(f"missing index: {path}")
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected '<route> <frames>'")
        route = parts[0].replace("\\", "/").strip("/")
        if route.startswith("data/"):
            route = route[5:]
        frames = int(parts[1])
        if frames <= 0:
            raise ValueError(f"{path}:{lineno}: frame count must be positive")
        entries.append((route, frames))
    if not entries:
        raise ValueError(f"empty index: {path}")
    if len({route for route, _ in entries}) != len(entries):
        raise ValueError(f"duplicate route in {path}")
    return entries


def audit_route(root: Path, route: str, frames: int, errors: list[str], warnings: list[str]):
    route_dir = root / "data" / route
    measurements_all_path = route_dir / "measurements_all.json"
    if not (route_dir / "route.json").is_file():
        errors.append(f"{route}: route.json missing")
    try:
        measurements_all = json.loads(measurements_all_path.read_text(encoding="utf-8"))
        if len(measurements_all) < frames:
            errors.append(f"{route}: measurements_all has {len(measurements_all)} < {frames} frames")
    except (OSError, ValueError) as exc:
        errors.append(f"{route}: invalid measurements_all.json ({exc})")

    actor_count = box_count = 0
    for frame in range(frames):
        stem = f"{frame:04d}"
        for directory, suffix in REQUIRED_FILES.items():
            path = route_dir / directory / (stem + suffix)
            if not path.is_file():
                errors.append(f"{route}: missing {directory}/{stem}{suffix}")
        measurement_path = route_dir / "measurements_full" / (stem + ".json")
        box_path = route_dir / "3d_bbs" / (stem + ".npy")
        if not measurement_path.is_file() or not box_path.is_file():
            continue
        try:
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            missing = REQUIRED_MEASUREMENT_KEYS - measurement.keys()
            if missing:
                errors.append(f"{route}/{stem}: measurement keys missing: {sorted(missing)}")
                continue
            actors = measurement["actors_data"]
            actor_count += len(actors)
            for actor_id, actor in actors.items():
                extent = actor.get("box", [])
                if len(extent) != 2 or not all(float(value) > 0 for value in extent):
                    errors.append(f"{route}/{stem}: actor {actor_id} has invalid box extent")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{route}/{stem}: invalid measurement ({exc})")
        try:
            boxes = np.load(box_path, allow_pickle=True).item()
            if not isinstance(boxes, dict):
                raise ValueError("root object is not a dict")
            box_count += len(boxes.get("vehicles", [])) + len(boxes.get("pedestrians", []))
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"{route}/{stem}: invalid 3d_bbs ({exc})")
    if actor_count != box_count:
        warnings.append(f"{route}: actors_data={actor_count}, 3d_bbs={box_count}")
    return actor_count, box_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--train-index", default="dataset_index.txt")
    parser.add_argument("--val-index", default="dataset_index_test.txt")
    parser.add_argument("--strict-3d-bbs", action="store_true",
                        help="treat 3d_bbs corruption/count mismatches as fatal")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    train = read_index(root, args.train_index)
    val = read_index(root, args.val_index)
    train_routes = {route for route, _ in train}
    val_routes = {route for route, _ in val}
    overlap = train_routes & val_routes
    if overlap:
        raise SystemExit(f"FAIL: train/validation overlap: {sorted(overlap)}")

    errors = []
    warnings = []
    totals = Counter()
    for split, entries in (("train", train), ("validation", val)):
        for route, frames in entries:
            actors, boxes = audit_route(root, route, frames, errors, warnings)
            totals[f"{split}_routes"] += 1
            totals[f"{split}_frames"] += frames
            totals["actors"] += actors
            totals["3d_bbs"] += boxes

    print(", ".join(f"{key}={value}" for key, value in totals.items()))
    for warning in warnings[:50]:
        print("WARNING:", warning)
    if len(warnings) > 50:
        print(f"WARNING: ... and {len(warnings) - 50} more")
    if args.strict_3d_bbs and warnings:
        errors.extend(warnings)
    if errors:
        for error in errors[:50]:
            print("ERROR:", error)
        if len(errors) > 50:
            print(f"ERROR: ... and {len(errors) - 50} more")
        raise SystemExit(f"FAIL: {len(errors)} dataset contract error(s)")
    print("PASS: training dataset contract" +
          (f" ({len(warnings)} non-training 3d_bbs warning(s))" if warnings else ""))


if __name__ == "__main__":
    main()
