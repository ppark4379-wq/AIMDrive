#!/usr/bin/env python3
"""Audit only object boxes that can overlap the supervised training BEV.

V1 required measured catalog geometry for every actor returned by the full
MORAI inventory, including actors hundreds of metres away.  This additive V2
keeps the complete inventory as evidence but only lets an actor gate object
supervision when its conservative observed box can touch the configured BEV.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from morai_pretraining.bev_geometry import (
    DEFAULT_TRAINING_BEV,
    conservative_training_bev_relation,
)
from morai_pretraining.sensor_record_manifest import dense_ego_timeline, interpolate_ego
from ssd_storage import require_ssd


DEFAULT_CATALOG = (ROOT / "evidence/native_mission_integration_v1/"
                   "catalog_review_v1/report.json")
EXPECTED_CATALOG_KIND = {"npc": "vehicle", "pedestrian": "pedestrian",
                         "obstacle": "static"}


def median_dimensions_wlh(case):
    rows = [view.get("dimensions_lwh_m") for view in (case or {}).get("views", [])]
    if len(rows) != 4 or any(not isinstance(row, list) or len(row) != 3 for row in rows):
        return None
    return [statistics.median(row[index] for row in rows) for index in (1, 0, 2)]


def check_object_geometry(item, catalog_cases, catalog_hash,
                          absolute_tolerance_m, relative_tolerance):
    model = str(item.get("model_name", "")).strip()
    kind = item.get("kind")
    dimensions = item.get("size_wlh_m")
    case = catalog_cases.get(model)
    check = {
        "instance_key": item.get("instance_key"),
        "actor_id": item.get("id"),
        "model_name": model,
        "kind": kind,
        "observed_size_wlh_m": dimensions,
        "catalog_report_sha256": catalog_hash,
    }
    reasons = []
    if (not isinstance(dimensions, list) or len(dimensions) != 3
            or not all(math.isfinite(float(value)) and float(value) > 0
                       for value in dimensions)):
        reasons.append("invalid_observed_dimensions")
    reference = median_dimensions_wlh(case)
    if case is None:
        reasons.append("model_absent_from_measured_catalog")
    elif case.get("kind") != EXPECTED_CATALOG_KIND.get(kind):
        reasons.append("catalog_kind_mismatch")
    elif case.get("visible_return_contradiction") is True:
        reasons.append("catalog_visible_return_contradiction")
    elif reference is None:
        reasons.append("catalog_lacks_four_views")
    elif not reasons:
        errors = [abs(float(value) - reference[index])
                  for index, value in enumerate(dimensions)]
        tolerances = [max(absolute_tolerance_m,
                          relative_tolerance * reference[index])
                      for index in range(3)]
        check.update(reference_size_wlh_m=reference,
                     absolute_error_m=errors, tolerance_m=tolerances)
        if any(error > tolerance for error, tolerance in zip(errors, tolerances)):
            reasons.append("observed_dimensions_differ_from_catalog")
    check["reasons"] = reasons
    check["api_box_geometry_candidate"] = not reasons
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt-timeline-dir", default="gt_timeline_v1")
    parser.add_argument("--actor-inventory-dir", default="actor_inventory_audit_v1")
    parser.add_argument("--catalog-report", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--absolute-tolerance-m", type=float, default=.03)
    parser.add_argument("--relative-tolerance", type=float, default=.02)
    parser.add_argument("--bev-forward-min-m", type=float,
                        default=DEFAULT_TRAINING_BEV["forward_min_m"])
    parser.add_argument("--bev-forward-max-m", type=float,
                        default=DEFAULT_TRAINING_BEV["forward_max_m"])
    parser.add_argument("--bev-left-min-m", type=float,
                        default=DEFAULT_TRAINING_BEV["left_min_m"])
    parser.add_argument("--bev-left-max-m", type=float,
                        default=DEFAULT_TRAINING_BEV["left_max_m"])
    parser.add_argument("--boundary-margin-m", type=float,
                        default=DEFAULT_TRAINING_BEV["boundary_margin_m"])
    args = parser.parse_args()
    for path in (args.episode, args.output):
        require_ssd(path, reserve_bytes=0)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.absolute_tolerance_m <= .2 or not 0 < args.relative_tolerance <= .1:
        raise ValueError("Invalid bounded geometry tolerances")

    gt_dir = args.episode / args.gt_timeline_dir
    actor_dir = args.episode / args.actor_inventory_dir
    quality_dir = args.episode / "quality"
    verify_sealed_directory(gt_dir)
    verify_sealed_directory(actor_dir)
    verify_sealed_directory(quality_dir, "QUALITY_COMPLETE.json")
    actor = json.loads((actor_dir / "report.json").read_text())
    frames = json.loads((gt_dir / "frames.json").read_text())
    active = json.loads((quality_dir / "active_signals.json").read_text())
    ego_times, ego_values = dense_ego_timeline(active)
    catalog = json.loads(args.catalog_report.read_text())
    catalog_cases = {row["model"]: row for row in catalog.get("cases", [])}
    catalog_hash = digest(args.catalog_report)
    bev = {
        "forward_min_m": args.bev_forward_min_m,
        "forward_max_m": args.bev_forward_max_m,
        "left_min_m": args.bev_left_min_m,
        "left_max_m": args.bev_left_max_m,
        "boundary_margin_m": args.boundary_margin_m,
    }

    observations = defaultdict(list)
    observed_models, relevant_models, outside_models, observed_instances = set(), set(), set(), set()
    output, status, relation_status = [], Counter(), Counter()
    relevant_actor_observations = outside_actor_observations = 0
    for frame in frames:
        reasons = []
        objects = frame["object_gt"].get("objects", [])
        if frame["object_gt"].get("source") != "grpc_full_inventory":
            reasons.append("not_full_grpc_inventory")
        if frame["object_gt"].get("transport_timing_candidate") is not True:
            reasons.append("object_time_association_failed")
        if frame["object_gt"].get("model_identity_candidate") is not True:
            reasons.append("actor_model_identity_failed")
        ego = interpolate_ego(ego_times, ego_values, int(frame["lidar_source_ns"]))
        checks = []
        for item in objects:
            relation = conservative_training_bev_relation(item, ego, **bev)
            model = str(item.get("model_name", "")).strip()
            key = item.get("instance_key")
            dimensions = item.get("size_wlh_m")
            if model:
                observed_models.add(model)
            if key:
                observed_instances.add(key)
            if isinstance(dimensions, list) and len(dimensions) == 3:
                observations[(model, key)].append([float(value) for value in dimensions])
            relation_status[relation["bev_relation"]] += 1
            if relation["training_bev_relevant"]:
                relevant_actor_observations += 1
                if model:
                    relevant_models.add(model)
                check = check_object_geometry(
                    item, catalog_cases, catalog_hash,
                    args.absolute_tolerance_m, args.relative_tolerance)
                reasons.extend("%s:%s" % (model or "<unknown>", reason)
                               for reason in check["reasons"])
            else:
                outside_actor_observations += 1
                if model:
                    outside_models.add(model)
                check = {
                    "instance_key": key,
                    "actor_id": item.get("id"),
                    "model_name": model,
                    "kind": item.get("kind"),
                    "observed_size_wlh_m": dimensions,
                    "catalog_report_sha256": catalog_hash,
                    "reasons": [],
                    "api_box_geometry_candidate": None,
                }
            check.update(relation)
            checks.append(check)
        reasons = sorted(set(reasons))
        candidate = not reasons
        status["candidate" if candidate else "+".join(reasons)] += 1
        output.append({
            "index": frame["index"],
            "lidar_source_ns": frame["lidar_source_ns"],
            "object_count": len(objects),
            "training_bev_relevant_object_count": sum(
                row["training_bev_relevant"] for row in checks),
            "outside_training_bev_object_count": sum(
                not row["training_bev_relevant"] for row in checks),
            "objects": checks,
            "reasons": reasons,
            "api_box_geometry_candidate": candidate,
            "physical_sensor_gt_atomicity_verified": False,
            "training_visible": False,
        })

    per_instance = []
    for (model, key), rows in sorted(observations.items()):
        ranges = [max(row[i] for row in rows) - min(row[i] for row in rows)
                  for i in range(3)]
        per_instance.append({
            "model_name": model,
            "instance_key": key,
            "observations": len(rows),
            "observed_size_range_wlh_m": ranges,
        })
    report = {
        "schema": "morai_sensor_record_object_geometry_audit_v2",
        "frames": len(frames),
        "frame_status": dict(status),
        "api_box_geometry_candidate_frames": sum(
            row["api_box_geometry_candidate"] for row in output),
        "raw_manifest_sha256": actor.get("raw_manifest_sha256"),
        "observed_models": sorted(observed_models),
        "training_bev_relevant_models": sorted(relevant_models),
        "outside_training_bev_models": sorted(outside_models),
        "observed_actor_instances": len(observed_instances),
        "actor_observations": {
            "training_bev_relevant": relevant_actor_observations,
            "definitely_outside": outside_actor_observations,
            "relation_status": dict(relation_status),
        },
        "training_bev": bev,
        "spatial_policy": (
            "Conservative observed W/L bounding circle plus boundary margin; "
            "unknown or ambiguous boxes remain fail-closed and relevant"
        ),
        "per_instance_dimension_stability": per_instance,
        "full_actor_model_identity_complete": actor.get("model_identity_complete") is True,
        "catalog_report": str(args.catalog_report.resolve()),
        "catalog_report_sha256": catalog_hash,
        "catalog_measurement_sha256": catalog.get("measurement_sha256"),
        "structural_valid": (
            len(output) == len(frames) and actor.get("structural_valid") is True),
        "complete_relevant_geometry_scope": bool(output and all(
            row["api_box_geometry_candidate"] for row in output)),
        "physical_sensor_gt_atomicity_verified": False,
        "object_training_label_valid": False,
        "training_visible": False,
        "limitation": (
            "Spatial relevance is proven against the configured BEV from observed gRPC "
            "box centres and dimensions. Relevant actors still require measured per-model "
            "geometry; this does not prove camera exposure atomicity or occluded-box tightness."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "frames.json", output)
    write_json(args.output / "COMPLETE.json", {
        "schema": report["schema"],
        "files": {name: digest(args.output / name)
                  for name in ("report.json", "frames.json")},
        "gt_timeline_seal_sha256": digest(gt_dir / "COMPLETE.json"),
        "actor_inventory_seal_sha256": digest(actor_dir / "COMPLETE.json"),
        "quality_seal_sha256": digest(quality_dir / "QUALITY_COMPLETE.json"),
        "catalog_report_sha256": catalog_hash,
        "script_sha256": digest(Path(__file__)),
        "source_timestamps_modified": False,
        "training_visible": False,
    })
    print(json.dumps(report), flush=True)
    if not report["structural_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
