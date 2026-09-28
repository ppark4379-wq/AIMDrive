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
from ssd_storage import require_ssd, SSD_ROOT


DEFAULT_CATALOG = (ROOT / "evidence/native_mission_integration_v1/"
                   "catalog_review_v1/report.json")
EXPECTED_CATALOG_KIND = {"npc": "vehicle", "pedestrian": "pedestrian",
                         "obstacle": "static"}


def median_dimensions_wlh(case):
    rows = [view.get("dimensions_lwh_m") for view in (case or {}).get("views", [])]
    if len(rows) != 4 or any(not isinstance(row, list) or len(row) != 3 for row in rows):
        return None
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
           for row in rows for v in row):
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
    elif ("actual_pose_audit" in case and
          (case.get("actual_pose_audit") or {}).get("consistent") is not True):
        reasons.append("catalog_actual_pose_inconsistent")
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
    check["catalog_actual_pose_checked"] = bool(case and
        (case.get("actual_pose_audit") or {}).get("consistent") is True)
    return check


def main(*, geometry_version=2):
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
    parser.add_argument("--geometry-registry", type=Path,
                        help="V5: per-model box-source registry built from native LiDAR")
    args = parser.parse_args()
    registry = None
    if geometry_version == 5:
        from morai_pretraining.fused_geometry import REGISTRY_SCHEMA
        if args.geometry_registry is None:
            parser.error("V5 requires --geometry-registry")
        registry = json.loads(args.geometry_registry.read_text())
        if registry.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("Unsupported object geometry registry")
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
    scale_evidence = pivot_evidence = None
    if geometry_version >= 3:
        from morai_pretraining.scaled_object_geometry import load_scale_evidence
        scale_evidence = load_scale_evidence(
            args.episode, active, actor.get("raw_manifest_sha256"),
            SSD_ROOT, ROOT.parents[1])
    ros_evidence = None
    if geometry_version >= 4:
        from morai_pretraining.center_geometry import load_pivot_evidence
        pivot_evidence = load_pivot_evidence(
            args.episode, active, actor.get("raw_manifest_sha256"), SSD_ROOT)
    if geometry_version == 5:
        from morai_pretraining.fused_geometry import load_ros_object_sizes
        ros_evidence = load_ros_object_sizes(
            args.episode, active, actor.get("raw_manifest_sha256"), SSD_ROOT)
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
                if geometry_version >= 4:
                    from morai_pretraining.center_geometry import check_center_object_geometry
                    check = check_center_object_geometry(
                        item, catalog_cases, catalog_hash,
                        args.absolute_tolerance_m, args.relative_tolerance,
                        scale_evidence, pivot_evidence,
                        frame["object_gt"].get("source_record_ids", []),
                        check_object_geometry)
                if geometry_version == 5:
                    from morai_pretraining.fused_geometry import check_fused_object_geometry
                    ros = ros_evidence["actors"].get(str(item.get("id")))
                    if ros is not None and ros["kind"] != item.get("kind"):
                        ros = None  # ids are compared only within one actor type
                    match = scale_evidence.get("instances", {}).get(key, {})
                    scale = ([1.0, 1.0, 1.0] if item.get("kind") != "obstacle"
                             else match.get("scale_xyz") if match.get("valid") is True else None)
                    pivot = pivot_evidence.get("instances", {}).get(key, {})
                    check = check_fused_object_geometry(
                        item, check, registry, ros, scale,
                        pivot["offset_xyz_m"][:2] if pivot.get("valid") is True else None)
                elif geometry_version == 3:
                    from morai_pretraining.scaled_object_geometry import check_scaled_object_geometry
                    check = check_scaled_object_geometry(
                        item, catalog_cases, catalog_hash,
                        args.absolute_tolerance_m, args.relative_tolerance,
                        scale_evidence, frame["object_gt"].get("source_record_ids", []),
                        check_object_geometry)
                else:
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
        "schema": "morai_sensor_record_object_geometry_audit_v%d" % geometry_version,
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
    files = ["report.json", "frames.json"]
    if geometry_version >= 3:
        report["scale_policy"] = ("proportional_to_measured_catalog_scale_v1" if geometry_version == 3
                                  else "unity_y_up_equal_horizontal_scale_v1")
        report["scale_evidence"] = {
            "file": "scale_evidence.json",
            "preset_resolved": scale_evidence["preset_resolved"],
            "matched_instances": sum(row["valid"] for row in scale_evidence["instances"].values()),
            "unresolved_instances": sum(not row["valid"] for row in scale_evidence["instances"].values()),
            "anisotropic_relative_scale_supported": geometry_version == 4,
        }
        files.append("scale_evidence.json")
    if geometry_version >= 4:
        from morai_pretraining.center_geometry import DEFAULT_CENTER_POLICY
        report["center_policy"] = dict(DEFAULT_CENTER_POLICY)
        report["center_source_counts"] = dict(Counter(
            (check.get("geometry_sources") or {}).get("center")
            or (check.get("center_evidence") or {}).get("center_source")
            or "unattributed"
            for row in output for check in row["objects"]
            if check.get("training_center_world_xy") is not None))
        report["pivot_evidence"] = {
            "file": "pivot_evidence.json",
            "instances": len(pivot_evidence["instances"]),
            "unstable_instances": sum(not row["valid"] for row in pivot_evidence["instances"].values()),
        }
        report["limitation"] += (
            " V4 additionally requires the catalog visible returns to support the box and the "
            "observed pivot-to-centre offset to match the catalog; a pivot-reported box is "
            "re-centred horizontally from the catalog offset.")
        files.append("pivot_evidence.json")
    if geometry_version == 5:
        report["geometry_registry"] = {
            "path": str(args.geometry_registry.resolve()),
            "sha256": digest(args.geometry_registry),
            "approved_models": len(registry.get("approved_models", [])),
        }
        report["registry_application_counts"] = dict(Counter(
            "%s:%s" % ((check.get("registry") or {}).get("applied"), (check.get("registry") or {}).get("reason"))
            for row in output for check in row["objects"] if check.get("training_bev_relevant")))
        report["limitation"] += (
            " V5 replaces the catalog decision wherever the LiDAR registry has sufficient evidence "
            "and trains on its selected MORAI-reported centre and size.")
        files.append("ros_object_evidence.json")
    args.output.mkdir(parents=True, exist_ok=False)
    if scale_evidence is not None:
        write_json(args.output / "scale_evidence.json", scale_evidence)
    if pivot_evidence is not None:
        write_json(args.output / "pivot_evidence.json", pivot_evidence)
    if ros_evidence is not None:
        write_json(args.output / "ros_object_evidence.json", ros_evidence)
    write_json(args.output / "report.json", report)
    write_json(args.output / "frames.json", output)
    write_json(args.output / "COMPLETE.json", {
        "schema": report["schema"],
        "files": {name: digest(args.output / name)
                  for name in files},
        "gt_timeline_seal_sha256": digest(gt_dir / "COMPLETE.json"),
        "actor_inventory_seal_sha256": digest(actor_dir / "COMPLETE.json"),
        "quality_seal_sha256": digest(quality_dir / "QUALITY_COMPLETE.json"),
        "catalog_report_sha256": catalog_hash,
        "script_sha256": digest(Path(__file__)),
        **({"scale_module_sha256": digest(ROOT / "morai_pretraining/scaled_object_geometry.py")}
           if geometry_version >= 3 else {}),
        **({"center_module_sha256": digest(ROOT / "morai_pretraining/center_geometry.py")}
           if geometry_version >= 4 else {}),
        **({"fused_module_sha256": digest(ROOT / "morai_pretraining/fused_geometry.py"),
            "geometry_registry_sha256": digest(args.geometry_registry)}
           if geometry_version == 5 else {}),
        "source_timestamps_modified": False,
        "training_visible": False,
    })
    print(json.dumps(report), flush=True)
    if not report["structural_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
