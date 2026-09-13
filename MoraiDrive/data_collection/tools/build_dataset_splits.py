#!/usr/bin/env python3
"""완료된 MORAI route를 검사해 LMDrive train/validation index를 생성한다."""
import argparse
import json
import random
from pathlib import Path


FRAME_DIRS = (
    "rgb_front", "rgb_left", "rgb_right", "rgb_rear", "lidar",
    "measurements", "measurements_full", "actors_data", "affordances",
    "3d_bbs",
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


def completed_routes(root: Path) -> dict:
    summary_path = root / "results" / "episode_summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    completed = {}
    for episode in summary.get("episodes", []):
        route = str(episode.get("route", "")).strip()
        if episode.get("status") == "completed" and route:
            # Writer output (data/foo/), legacy names (foo/) and Windows
            # separators all identify the same route directory.
            route_name = Path(route.replace("\\", "/").rstrip("/")).name
            completed[route_name] = int(episode.get("frames") or 0)
    return completed


def route_quality(route_dir: Path, max_route_error: float) -> tuple:
    """Return (ok, reason) for navigation-label quality.

    `contiguous_frame_count()` only checks file layout.  For training splits we
    also reject routes whose dense-route supervision is clearly unusable: the
    ego must stay close to the planned route, and route_index must move forward
    at least once.  This keeps off-lane captures and stationary warm-up captures
    out of LMDrive training indices.
    """
    measurement_dir = route_dir / "measurements_full"
    paths = sorted(measurement_dir.glob("*.json"))
    if not paths:
        return False, "measurements_full JSON이 없습니다"

    indices = []
    max_error = 0.0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            measurement = json.load(f)
        navigation = measurement.get("navigation") or {}
        if "route_index" not in navigation:
            return False, f"{path.name}: navigation.route_index 없음"
        distance = navigation.get("distance_to_link_m")
        if distance is None:
            return False, f"{path.name}: navigation.distance_to_link_m 없음"
        indices.append(int(navigation["route_index"]))
        max_error = max(max_error, float(distance))

    if any(b < a for a, b in zip(indices, indices[1:])):
        return False, "route_index가 역행함"
    if indices[-1] == indices[0]:
        return False, "route_index가 변하지 않음"
    if max_error > max_route_error:
        return False, f"route 오차 max={max_error:.2f}m > {max_route_error:.2f}m"
    return True, f"route_index {indices[0]}->{indices[-1]}, max_error={max_error:.2f}m"


def index_line(route_dir: Path, frames: int) -> str:
    return f"data/{route_dir.name}/ {frames}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-route-error", type=float, default=3.0,
                        help="exclude routes whose max route-label error exceeds this many metres")
    parser.add_argument("--skip-route-quality-check", action="store_true",
                        help="only check file layout/completion, not route_index/error quality")
    parser.add_argument("--allow-unsummarized", action="store_true",
                        help="include legacy routes not listed as completed in episode_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")

    root = Path(args.dataset_root).resolve()
    data_dir = root / "data"
    completed = completed_routes(root)
    routes = []
    for route_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if not (route_dir / "route.json").is_file():
            continue
        if not args.allow_unsummarized and route_dir.name not in completed:
            continue
        frames = contiguous_frame_count(route_dir)
        expected_frames = completed.get(route_dir.name)
        if expected_frames is not None and frames != expected_frames:
            print(
                f"skip {route_dir.name}: contiguous frames={frames}, "
                f"episode_summary frames={expected_frames}"
            )
            continue
        if frames < args.min_frames:
            continue
        if not args.skip_route_quality_check:
            ok, reason = route_quality(route_dir, args.max_route_error)
            if not ok:
                print(f"skip {route_dir.name}: {reason}")
                continue
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
