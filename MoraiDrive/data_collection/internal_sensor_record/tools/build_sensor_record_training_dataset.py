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
from dataset_layout import relative_path


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


def planned_split_laps(path, expected_sha256, available, requested_laps):
    """Load an immutable run-level split without reassigning adjacent frames."""
    path = Path(path).expanduser().resolve()
    if expected_sha256 and digest(path) != expected_sha256:
        raise ValueError("Static collection split plan changed after capture")
    plan = load(path)
    if (
        plan.get("schema") != "morai_static_collection_plan_v1"
        or plan.get("scenario") != "full_loop_static"
        or plan.get("ready") is not True
        or int(plan.get("target_runs", -1)) != int(requested_laps)
    ):
        raise ValueError("Invalid or incomplete static collection split plan")
    rows = list(plan.get("runs", []))
    by_lap = {int(row["run"]): row.get("split") for row in rows}
    if set(by_lap) != set(range(1, int(requested_laps) + 1)):
        raise ValueError("Static collection split plan has incomplete run numbering")
    if any(value not in ("train", "val", "test") for value in by_lap.values()):
        raise ValueError("Static collection split plan has an invalid split")
    result = {
        split: {number for number in available if by_lap[number] == split}
        for split in ("train", "val", "test")
    }
    if not result["train"] or not result["val"] or not result["test"]:
        raise ValueError(
            "Every planned split must retain at least one validator-approved lap"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-lap", type=int, action="append", default=[])
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--split-plan-sha256")
    args = parser.parse_args()
    report_path = args.validation_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require_ssd(report_path, reserve_bytes=0)
    require_ssd(output.parent, reserve_bytes=0)
    report = load(report_path)
    seal_path = report_path.parent / "COMPLETE.json"
    seal = load(seal_path)
    accepted_validation_schemas = {
        "morai_sensor_record_campaign_validation_v1",
        "morai_sensor_record_campaign_validation_subset_v1",
    }
    if not (report.get("schema") in accepted_validation_schemas
            and report.get("valid") is True
            and seal.get("status") == "passed"
            and seal.get("files", {}).get("report.json") == digest(report_path)):
        raise ValueError("A passed, sealed campaign validation is required")
    scenario_name = report.get("scenario_name", "full_loop")
    required = required_training_coverage(scenario_name)
    laps = select_training_laps(report)
    available = {int(lap["lap"]) for lap in laps}
    if args.split_plan and args.val_lap:
        raise ValueError("--split-plan and --val-lap are mutually exclusive")
    if args.split_plan:
        split_laps = planned_split_laps(
            args.split_plan, args.split_plan_sha256, available,
            report.get("requested_laps", len(laps)),
        )
        split_names = ("train", "val", "test")
        split_policy = "static_collection_plan_run_holdout_v1"
    else:
        val_laps = set(args.val_lap or [max(available)])
        if not val_laps or not val_laps < available:
            raise ValueError("Validation laps must be a nonempty proper subset of captured laps")
        split_laps = {"train": available - val_laps, "val": val_laps}
        split_names = ("train", "val")
        split_policy = "complete_lap_holdout_v1"

    output.mkdir(parents=False, exist_ok=False)
    manifest_rows = {split: [] for split in split_names}
    coverage = {split: Counter() for split in split_names}
    contracts = set()
    validation_sha = digest(report_path)
    portable = not Path(report.get("campaign", "")).is_absolute()
    for lap in laps:
        number = int(lap["lap"])
        split = next(
            name for name in split_names if number in split_laps[name]
        )
        raw_run = Path(lap["run"])
        run = ((report_path.parent / raw_run) if not raw_run.is_absolute()
               else raw_run).resolve()
        require_ssd(run, reserve_bytes=0)
        manifest = build_episode_manifest(run / "episode_000",
                                          campaign_validation_sha256=validation_sha,
                                          split=split,
                                          expected_evidence=lap.get("input_seals", {}))
        name = "lap_%03d_%s_manifest.json" % (number, split)
        path = output / name
        if portable:
            manifest["episode"] = relative_path(run / "episode_000", output)
            manifest["native_root"] = relative_path(run, output)
            route = Path(str(manifest.get("route", "")))
            if route.is_absolute():
                try:
                    route.relative_to(run)
                except ValueError:
                    pass
                else:
                    manifest["route"] = relative_path(route, output)
        write_json(path, manifest)
        contracts.add(manifest["contract_sha256"])
        add_counts(coverage[split], manifest["report"]["counts"])
        manifest_rows[split].append({
            "lap": number,
            "scenario_name": scenario_name,
            "path": path.name if portable else str(path),
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
        "source_schema": report["schema"],
        "source_validation": (relative_path(report_path, output)
                              if portable else str(report_path)),
        "source_validation_sha256": validation_sha,
        "scenario_name": scenario_name,
        "required_coverage": list(required),
        "split_policy": split_policy,
        "train_laps": sorted(split_laps["train"]),
        "val_laps": sorted(split_laps["val"]),
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
    if "test" in split_names:
        split_plan_path = args.split_plan.expanduser().resolve()
        index["test_laps"] = sorted(split_laps["test"])
        index["test"] = manifest_rows["test"]
        index["split_plan"] = (
            relative_path(split_plan_path, output)
            if portable else str(split_plan_path)
        )
        index["split_plan_sha256"] = digest(split_plan_path)
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
        "test_laps": index.get("test_laps", []),
        "counts": index["counts"],
        "coverage": coverage_json,
    }, ensure_ascii=False, indent=2), flush=True)
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
