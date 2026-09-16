#!/usr/bin/env python3
"""Publish sealed Sensor Record laps as direct-reference visual manifests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "vision_encoder"),
    str(PROJECT_ROOT),
]

from async_capture.journal import digest, write_json
from ssd_storage import require_ssd
from morai_pretraining.sensor_record_manifest import build_episode_manifest
from scenario_runner.scenarios.segment_modes.modes import (
    PRIMARY_SCENARIOS,
    get_scenario_zones,
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_counts(target, source):
    for key, value in source.items():
        if isinstance(value, int):
            target[key] += value


def required_training_coverage(scenario_name):
    """Return coverage required by the selected main scenario."""
    if scenario_name not in PRIMARY_SCENARIOS:
        raise ValueError("Unsupported scenario: " + str(scenario_name))
    zones = get_scenario_zones(scenario_name)
    required = ["waypoint_frames"]
    if any((
        zones.static,
        zones.dynamic,
        zones.roundabout,
        zones.highway,
        zones.tunnel,
    )):
        required.append("object_positive_frames")
    if scenario_name in ("full_loop", "full_loop_normal"):
        required.extend(("traffic_light_0", "traffic_light_1"))
    return tuple(required)


def select_training_laps(report):
    """Return only validator-approved laps while keeping old reports compatible."""
    laps = [
        lap for lap in report.get("laps", [])
        if lap.get("valid") is True and lap.get("training_usable", True) is not False
    ]
    if len(laps) < 2:
        raise ValueError(
            "At least two valid complete laps are required for episode-level "
            "train/val split"
        )
    return laps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-lap", type=int, action="append", default=[])
    args = parser.parse_args()
    report_path = args.validation_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require_ssd(report_path, reserve_bytes=0)
    require_ssd(output.parent, reserve_bytes=0)
    report = load(report_path)
    seal_path = report_path.parent / "COMPLETE.json"
    seal = load(seal_path)
    if not (report.get("schema") == "morai_sensor_record_campaign_validation_v1"
            and report.get("valid") is True
            and seal.get("status") == "passed"
            and seal.get("files", {}).get("report.json") == digest(report_path)):
        raise ValueError("A passed, sealed campaign validation is required")
    scenario_name = report.get("scenario_name", "full_loop")
    required = required_training_coverage(scenario_name)
    laps = select_training_laps(report)
    available = {int(lap["lap"]) for lap in laps}
    val_laps = set(args.val_lap or [max(available)])
    if not val_laps or not val_laps < available:
        raise ValueError("Validation laps must be a nonempty proper subset of captured laps")

    output.mkdir(parents=False, exist_ok=False)
    manifest_rows = {"train": [], "val": []}
    coverage = {"train": Counter(), "val": Counter()}
    contracts = set()
    validation_sha = digest(report_path)
    for lap in laps:
        number = int(lap["lap"])
        split = "val" if number in val_laps else "train"
        run = Path(lap["run"]).resolve()
        require_ssd(run, reserve_bytes=0)
        manifest = build_episode_manifest(run / "episode_000",
                                          campaign_validation_sha256=validation_sha,
                                          split=split,
                                          expected_evidence=lap.get("input_seals", {}))
        name = "lap_%03d_%s_manifest.json" % (number, split)
        path = output / name
        write_json(path, manifest)
        contracts.add(manifest["contract_sha256"])
        add_counts(coverage[split], manifest["report"]["counts"])
        manifest_rows[split].append({
            "lap": number,
            "scenario_name": scenario_name,
            "path": str(path),
            "sha256": digest(path),
            "frames": len(manifest["entries"]),
        })
    coverage_json = {split: dict(counts) for split, counts in coverage.items()}
    ready = bool(len(contracts) == 1 and all(
        all(counts.get(key, 0) > 0 for key in required)
        for counts in coverage.values()
    ))
    index = {
        "schema": "morai_visual_manifest_index_v1",
        "source_schema": "morai_sensor_record_campaign_validation_v1",
        "source_validation": str(report_path),
        "source_validation_sha256": validation_sha,
        "scenario_name": scenario_name,
        "required_coverage": list(required),
        "split_policy": "complete_lap_holdout_v1",
        "train_laps": sorted(available - val_laps),
        "val_laps": sorted(val_laps),
        "train": manifest_rows["train"],
        "val": manifest_rows["val"],
        "counts": {split: sum(row["frames"] for row in rows)
                   for split, rows in manifest_rows.items()},
        "coverage": coverage_json,
        "training_ready": ready,
        "physical_simultaneity_verified": False,
        "official_camera_exposure_timestamp_available": False,
        "training_publication_basis": "sealed Sensor Record internal-sync bundle plus empirical temporal audits",
        "validation_scope": (
            "held-out selected-scenario run on the same map/route; "
            "not independent-map generalization"
        ),
    }
    write_json(output / "manifest_index.json", index)
    write_json(output / "COMPLETE.json", {
        "schema": "morai_sensor_record_training_publication_v1",
        "status": "passed" if ready else "failed",
        "files": {path.name: digest(path) for path in output.glob("*.json")},
        "source_validation_sha256": validation_sha,
        "training_visible": ready,
        "physical_simultaneity_verified": False,
    })
    print(json.dumps({
        "training_ready": ready,
        "scenario_name": scenario_name,
        "manifest_index": str(output / "manifest_index.json"),
        "train_laps": index["train_laps"],
        "val_laps": index["val_laps"],
        "counts": index["counts"],
        "coverage": coverage_json,
    }, ensure_ascii=False, indent=2), flush=True)
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
