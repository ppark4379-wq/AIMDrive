#!/usr/bin/env python3
"""
Create explicit RUN-LEVEL train/val/test splits.

This script intentionally refuses random frame splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from aimdrive.data.stage6e_schema import ready_runs


def parse_ids(text):
    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default="data/stage6e_runs",
    )
    parser.add_argument(
        "--val-runs",
        default="",
        help="Comma-separated run IDs",
    )
    parser.add_argument(
        "--test-runs",
        default="",
        help="Comma-separated run IDs",
    )
    parser.add_argument(
        "--output",
        default="splits.json",
    )
    args = parser.parse_args()

    root = Path(
        args.runs_root
    ).expanduser().resolve()

    if REPO_ROOT not in root.parents:
        parser.error("Runs root must remain inside this repository")
    available, excluded = ready_runs(root)

    val = parse_ids(
        args.val_runs
    )
    test = parse_ids(
        args.test_runs
    )

    unknown = sorted(
        (
            set(val)
            | set(test)
        )
        - set(available)
    )

    if unknown:
        raise SystemExit(
            "Unknown/not-ready runs: {}".format(
                unknown
            )
        )

    overlap = (
        set(val)
        & set(test)
    )
    if overlap:
        raise SystemExit(
            "A run cannot be both val and test: {}".format(
                sorted(overlap)
            )
        )

    train = [
        run_id
        for run_id in available
        if (
            run_id not in val
            and run_id not in test
        )
    ]
    if not val and not test:
        train = []

    split = {
        "schema_version": (
            "morai-stage6e-run-split-v1"
        ),
        "policy": (
            "run-level only; never random frame split"
        ),
        "train": train,
        "val": val,
        "test": test,
        "ready_runs": available,
        "excluded_runs": excluded,
        "unassigned_runs": available if not val and not test else [],
        "split_status": "assigned" if val or test else "unassigned; collect independent runs before choosing validation",
    }

    output = (
        root
        / args.output
    )
    if root not in output.resolve().parents:
        parser.error("Output must stay in runs root")
    with output.open("x") as handle:
        json.dump(split, handle, indent=2)

    print(
        json.dumps(
            split,
            indent=2,
        )
    )
    print()
    print(
        "split file:",
        output,
    )


if __name__ == "__main__":
    main()
