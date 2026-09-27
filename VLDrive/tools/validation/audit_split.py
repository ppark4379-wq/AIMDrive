#!/usr/bin/env python3
"""Audit a MORAI encoder split without requiring torch."""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.src.loaders.split_manifest import load_split_manifest, split_entries


SENSOR_FILES = (
    ("rgb_front", "jpg"),
    ("rgb_left", "jpg"),
    ("rgb_right", "jpg"),
    ("rgb_rear", "jpg"),
    ("lidar", "npy"),
    ("measurements_full", "json"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("split_file", type=Path)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    manifest = load_split_manifest(args.split_file)
    cache = {}
    report = {}
    missing = []
    bad_horizons = []

    for split in ("train", "validation"):
        entries = list(split_entries(manifest, split))
        commands = collections.Counter()
        traffic = collections.Counter()
        for relative, frame in entries:
            route = root / "data" / relative
            if relative not in cache:
                cache[relative] = json.loads(
                    (route / "measurements_all.json").read_text(encoding="utf-8")
                )
            measurements = cache[relative]
            current = measurements[frame]
            commands[int(current["command"])] += 1
            traffic["red" if current.get("is_red_light_present") else "not_red"] += 1
            for folder, extension in SENSOR_FILES:
                path = route / folder / ("{:04d}.{}".format(frame, extension))
                if not path.is_file():
                    missing.append(str(path))
            native = int(current["native_sync_frame"])
            for offset in manifest["future_offsets"]:
                future = measurements[frame + int(offset)]
                if int(future["native_sync_frame"]) - native != int(offset) * 5:
                    bad_horizons.append([relative, frame, int(offset)])
        report[split] = {
            "samples": len(entries),
            "first": entries[0],
            "last": entries[-1],
            "commands": dict(sorted(commands.items())),
            "traffic": dict(sorted(traffic.items())),
        }

    report["missing_files"] = len(missing)
    report["bad_future_horizons"] = len(bad_horizons)
    print(json.dumps(report, indent=2))
    if missing or bad_horizons:
        raise RuntimeError(
            "Split audit failed: missing={} bad_horizons={}".format(
                len(missing), len(bad_horizons)
            )
        )


if __name__ == "__main__":
    main()

