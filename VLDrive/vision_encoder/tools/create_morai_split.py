#!/usr/bin/env python3
"""Create and audit the deterministic 80/20 MORAI encoder split."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morai_data import build_chronological_split, expand_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--purge-frames", type=int, default=25)
    args = parser.parse_args()

    manifest = build_chronological_split(
        args.dataset_root,
        train_fraction=args.train_fraction,
        purge_frames=args.purge_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing split: {}".format(args.output))
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for route in manifest["routes"]:
        print(
            "{}: train={} purge={} validation={} last_anchor={}".format(
                route["path"],
                len(expand_range(route["train"])),
                len(expand_range(route["purge"])),
                len(expand_range(route["validation"])),
                route["last_anchor"],
            )
        )
    print("wrote {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()

