#!/usr/bin/env python3
"""완료된 MORAI route를 검사해 LMDrive train/validation index를 생성한다."""
import argparse
import json
import random
from pathlib import Path


FRAME_DIRS = (
    "rgb_front", "rgb_left", "rgb_right", "rgb_rear", "lidar",
    "measurements", "measurements_full", "actors_data", "affordances",
)


def contiguous_frame_count(route_dir: Path) -> int:
    stems = []
    for dirname in FRAME_DIRS:
        directory = route_dir / dirname
        if not directory.is_dir():
            return 0
        stems.append({path.stem for path in directory.iterdir() if path.is_file()})
    common = set.intersection(*stems) if stems else set()
    frame = 0
    while f"{frame:04d}" in common:
        frame += 1
    return frame


def index_line(route_dir: Path, frames: int) -> str:
    return f"data/{route_dir.name}/ {frames}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")

    root = Path(args.dataset_root).resolve()
    data_dir = root / "data"
    summary_path = root / "results" / "episode_summary.json"
    completed = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed = {
            Path(entry["route"].replace("\\", "/").rstrip("/")).name
            for entry in summary.get("episodes", [])
            if entry.get("status") == "completed"
        }
    routes = []
    for route_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if completed is not None and route_dir.name not in completed:
            continue
        if not (route_dir / "route.json").is_file():
            continue
        frames = contiguous_frame_count(route_dir)
        if frames:
            routes.append((route_dir, frames))

    shuffled = list(routes)
    random.Random(args.seed).shuffle(shuffled)
    val_count = 0
    if len(shuffled) >= 2 and args.val_ratio > 0:
        val_count = max(1, round(len(shuffled) * args.val_ratio))
        val_count = min(val_count, len(shuffled) - 1)
    validation = sorted(shuffled[:val_count], key=lambda item: item[0].name)
    train = sorted(shuffled[val_count:], key=lambda item: item[0].name)

    print(f"eligible routes={len(routes)}, train={len(train)}, validation={len(validation)}")
    for label, entries in (("train", train), ("validation", validation)):
        print(f"[{label}]")
        for route_dir, frames in entries:
            print(f"  {route_dir.name}: {frames}")

    if not args.dry_run:
        (root / "dataset_index.txt").write_text(
            "".join(index_line(route, frames) for route, frames in train),
            encoding="utf-8",
        )
        (root / "dataset_index_test.txt").write_text(
            "".join(index_line(route, frames) for route, frames in validation),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
