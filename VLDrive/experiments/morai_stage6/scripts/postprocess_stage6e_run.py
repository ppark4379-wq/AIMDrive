#!/usr/bin/env python3
"""
Stage 6E-1 one-run postprocessor.

Pipeline:
    manifest.jsonl
      -> Stage 6D-1 v2 simulation-time trajectory labels
      -> training-only Ego-localized official Global Path
      -> manifest_ready.jsonl

Run this in the `lmdrive` environment, not the system ROS Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aimdrive.data.stage6e_runs import (
    summarize_manifest,
)
from aimdrive.ros_training.teacher_integrity import validate_manifest


def run_command(command):
    print()
    print("$", " ".join(
        str(x) for x in command
    ))
    subprocess.run(
        command,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-offline", action="store_true", help="Offline regression only; no automatic production approval")
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Run directory, e.g. "
            "data/stage6e_runs/run_0001"
        ),
    )
    parser.add_argument(
        "--global-path",
        default="data/competition/global_path.txt",
    )
    parser.add_argument(
        "--horizons",
        default="0.5,1.0,1.5,2.0,2.5",
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    run_dir = Path(
        args.run
    ).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root not in run_dir.parents:
        parser.error("Run directory must remain inside this repository")

    if not (
        run_dir
        / "manifest.jsonl"
    ).exists():
        raise SystemExit(
            "Missing run manifest: {}".format(
                run_dir
                / "manifest.jsonl"
            )
        )

    with (run_dir / "manifest.jsonl").open() as handle:
        validation = validate_manifest([json.loads(line) for line in handle if line.strip()])
    if validation["status"] != "pass":
        raise SystemExit("Teacher validation {}: {}. Raw data preserved; no labels written. "
                         "Run scripts/analyze_stage6e_integrity.py for evidence."
                         .format(validation["status"], validation["failures"]))
    from aimdrive.data.stage6e_validation import validate_run
    quality = validate_run(run_dir, allow_legacy=args.allow_legacy_offline)
    if quality["quality_status"] != "valid" and not (args.allow_legacy_offline and quality["quality_status"]=="warning"):
        raise SystemExit("Stage6E quality rejected before postprocessing: " + str(quality["reasons"]))
    outputs = ["manifest_labels.jsonl", "manifest_ready.jsonl", "labels_ego_teacher",
               "route_ego_teacher", "ego_teacher_label_stats.json", "route_stats.json"]
    existing = [name for name in outputs if (run_dir / name).exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing postprocess artifacts: " + str(existing))

    label_script = (
        repo_root
        / "scripts"
        / "build_stage6d1_ego_trajectory_labels.py"
    )
    route_script = (
        repo_root
        / "scripts"
        / "enrich_stage6e_training_route.py"
    )

    run_command(
        [
            sys.executable,
            "-u",
            str(label_script),
            "--dataset",
            str(run_dir),
            "--manifest",
            "manifest.jsonl",
            "--horizons",
            args.horizons,
            "--output-manifest",
            "manifest_labels.jsonl",
            "--label-dir",
            "labels_ego_teacher",
            "--stats-file",
            "ego_teacher_label_stats.json",
        ]
    )

    run_command(
        [
            sys.executable,
            "-u",
            str(route_script),
            "--dataset",
            str(run_dir),
            "--manifest",
            "manifest_labels.jsonl",
            "--path",
            str(
                Path(
                    args.global_path
                ).expanduser().resolve()
            ),
            "--output-manifest",
            "manifest_ready.jsonl",
            "--route-dir",
            "route_ego_teacher",
            "--stats-file",
            "route_stats.json",
            "--lookahead-m",
            str(
                args.lookahead_m
            ),
            "--spacing-m",
            str(
                args.spacing_m
            ),
        ]
    )

    label_stats = json.loads(
        (
            run_dir
            / "ego_teacher_label_stats.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    route_stats = json.loads(
        (
            run_dir
            / "route_stats.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    acquisition_stats = (
        summarize_manifest(
            run_dir
            / "manifest.jsonl"
        )
    )

    meta_path = (
        run_dir
        / "run_meta.json"
    )
    meta = (
        json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )
        if meta_path.exists()
        else {}
    )

    meta.update(
        {
            "postprocessed": True,
            "ready_manifest": (
                "manifest_ready.jsonl"
            ),
            "trajectory_labels": {
                "schema_version": (
                    "morai-stage6d1-ego-teacher-v2"
                ),
                "timebase": (
                    "ego_source_stamp_ns"
                ),
                "frame": (
                    "+x forward, +y left"
                ),
                "stats": label_stats,
            },
            "route_input": {
                "schema_version": (
                    "morai-stage6e-route-v1"
                ),
                "localization_source": (
                    "/Ego_topic training-only"
                ),
                "stats": route_stats,
            },
            "quality_summary": (
                acquisition_stats
            ),
        }
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print(
        "STAGE 6E-1 RUN POSTPROCESS: PASS"
    )
    print("=" * 108)
    print(
        "run              :",
        run_dir,
    )
    print(
        "ready manifest   :",
        run_dir
        / "manifest_ready.jsonl",
    )
    print(
        "trajectory labels:",
        label_stats.get(
            "written_labels"
        ),
    )
    print(
        "route distance   :",
        route_stats.get(
            "distance_to_path_m"
        ),
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
