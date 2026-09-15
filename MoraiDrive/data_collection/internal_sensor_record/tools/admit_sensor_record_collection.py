#!/usr/bin/env python3
"""Create a sealed, fail-closed offline admission manifest for one capture."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, write_json
from async_capture.native_session import same_recorded_native_source
from async_capture.sensor_record_admission import (
    build_frame_admission, reason_counts, summarize_live_streams,
    SENSOR_SCENE_STATE_CONTRACT, verify_capture_contract, verify_sealed_directory,
)
from ssd_storage import SSD_ROOT


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--capture-contract", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--review-dir", default="review_closeout")
    parser.add_argument("--gt-timeline-dir", default="gt_timeline_v1")
    parser.add_argument("--actor-inventory-dir", default="actor_inventory_audit_v1")
    parser.add_argument("--object-geometry-dir", default="object_geometry_audit_v1")
    parser.add_argument("--route-signal-dir", default="route_signal_audit_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    evidence_dirs = {
        "native_audit": args.episode / "native_audit",
        "quality": args.episode / "quality",
        "native_associations": args.episode / "native_associations",
        "motion_time": args.episode / "motion_time_v1",
        "review": args.episode / args.review_dir,
        "gt_timeline": args.episode / args.gt_timeline_dir,
        "traffic_audit": args.episode / "traffic_audit_v1",
        "actor_inventory_audit": args.episode / args.actor_inventory_dir,
        "object_geometry_audit": args.episode / args.object_geometry_dir,
        "route_signal_audit": args.episode / args.route_signal_dir,
    }
    evidence_seals = {}
    seal_contents = {}
    for key, folder in evidence_dirs.items():
        seal_name = "QUALITY_COMPLETE.json" if key == "quality" else "COMPLETE.json"
        seal_contents[key] = verify_sealed_directory(folder, seal_name)
        evidence_seals[key] = digest(folder / seal_name)

    plan = load(args.run_plan)
    contract = load(args.capture_contract)
    contract_check = verify_capture_contract(contract, plan, args.run_plan)
    native = load(evidence_dirs["native_audit"] / "report.json")
    motion = load(evidence_dirs["motion_time"] / "report.json")
    review_report = load(evidence_dirs["review"] / "report.json")
    review_frames = load(evidence_dirs["review"] / "frame_ledger.json")
    gt_report = load(evidence_dirs["gt_timeline"] / "report.json")
    gt_frames = load(evidence_dirs["gt_timeline"] / "frames.json")
    traffic_audit = load(evidence_dirs["traffic_audit"] / "report.json")
    actor_audit = load(evidence_dirs["actor_inventory_audit"] / "report.json")
    object_geometry = load(evidence_dirs["object_geometry_audit"] / "report.json")
    route_signal_audit = load(evidence_dirs["route_signal_audit"] / "report.json")
    index = load(evidence_dirs["quality"] / "INDEX_COMPLETE.json")
    active = load(evidence_dirs["quality"] / "active_signals.json")
    outcome_path=args.episode/'outcome.json'
    scenario_outcome=(load(outcome_path) if outcome_path.is_file() else
                      dict(status='missing',training_visible=False))

    raw_ids = {native.get("raw_manifest_sha256"), motion.get("raw_manifest_sha256"),
               review_report.get("raw_manifest_sha256"), index.get("raw_manifest_sha256")}
    if len(raw_ids) != 1 or None in raw_ids:
        raise ValueError("Mixed raw recordings in admission evidence")
    if gt_report.get("frames") != len(native.get("bundles", [])):
        raise ValueError("GT report frame count differs from native capture")
    if (traffic_audit.get("raw_manifest_sha256") not in raw_ids
            or traffic_audit.get("structural_valid") is not True):
        raise ValueError("Direct traffic audit is invalid or linked to different raw evidence")
    if (actor_audit.get("raw_manifest_sha256") not in raw_ids
            or actor_audit.get("structural_valid") is not True):
        raise ValueError("Actor inventory audit is invalid or linked to different raw evidence")
    if (object_geometry.get("raw_manifest_sha256") not in raw_ids
            or object_geometry.get("structural_valid") is not True):
        raise ValueError("Object geometry audit is invalid or linked to different raw evidence")
    if route_signal_audit.get("raw_manifest_sha256") not in raw_ids:
        raise ValueError("Route signal audit is linked to different raw evidence")
    if not same_recorded_native_source(
            args.episode,native.get("source_root"),
            contract.get("native_session", {}).get("native_root"),SSD_ROOT):
        raise ValueError("Native audit source differs from the pre-capture session")
    native_report_sha = digest(evidence_dirs["native_audit"] / "report.json")
    if (seal_contents["gt_timeline"].get("raw_manifest_sha256") not in raw_ids
            or seal_contents["gt_timeline"].get("native_audit_sha256") != native_report_sha):
        raise ValueError("GT timeline is linked to different native/raw evidence")
    if (seal_contents["native_associations"].get("raw_manifest_sha256") not in raw_ids
            or seal_contents["native_associations"].get("native_audit_sha256") != native_report_sha):
        raise ValueError("Teacher associations are linked to different native/raw evidence")
    if (seal_contents["object_geometry_audit"].get("gt_timeline_seal_sha256")
            != evidence_seals["gt_timeline"]
            or seal_contents["object_geometry_audit"].get("actor_inventory_seal_sha256")
            != evidence_seals["actor_inventory_audit"]):
        raise ValueError("Object geometry audit is linked to different GT/actor evidence")
    expected_review_seals = review_report.get("input_seals", {})
    for key, review_key in (("native_audit", "native_audit"), ("quality", "quality"),
                            ("native_associations", "native_associations"),
                            ("motion_time", "motion_time_v1"),
                            ("gt_timeline", args.gt_timeline_dir),
                            ("traffic_audit", "traffic_audit_v1"),
                            ("actor_inventory_audit", args.actor_inventory_dir),
                            ("object_geometry_audit", args.object_geometry_dir),
                            ("route_signal_audit", args.route_signal_dir)):
        if expected_review_seals.get(review_key) != evidence_seals[key]:
            raise ValueError("Review summary is linked to different " + key + " evidence")
    native_times = [int(row["filename_simulation_time_hypothesis_ns"])
                    for row in native["bundles"]]
    if not native_times:
        raise ValueError("No native sensor frames")
    live = summarize_live_streams(active, native_times[0], native_times[-1])
    frames, counts = build_frame_admission(
        contract_valid=contract_check["valid"],
        native_report=native,
        review_frames=review_frames,
        gt_frames=gt_frames,
        motion_report=motion,
        live_streams=live,
        runtime_streams_valid=live["valid"],
    )
    actor_frames=load(evidence_dirs["actor_inventory_audit"] / "frame_brackets.json")
    object_geometry_frames=load(evidence_dirs["object_geometry_audit"] / "frames.json")
    route_frames=load(evidence_dirs["route_signal_audit"] / "frames.json")
    if (len(actor_frames)!=len(frames) or len(object_geometry_frames)!=len(frames)
            or len(route_frames)!=len(frames)):
        raise ValueError("Actor/signal audit frame count differs from admission ledger")
    for row,actor_row,geometry_row,signal_row,gt in zip(
            frames,actor_frames,object_geometry_frames,route_frames,gt_frames):
        if (row['index']!=actor_row['index'] or row['index']!=geometry_row['index']
                or row['index']!=signal_row['index']):
            raise ValueError('Actor/signal audit frame order differs')
        row['full_actor_inventory_transport_candidate']=bool(actor_row['transport_candidate'])
        row['object_model_identity_candidate']=bool(
            actor_row.get('model_identity_candidate')
            and gt['object_gt'].get('model_identity_candidate'))
        row['object_api_box_geometry_candidate']=bool(
            geometry_row.get('api_box_geometry_candidate'))
        row['route_signal_binding_candidate']=bool(signal_row['valid'])
        row['route_signal_training_label_candidate']=(int(signal_row['training_label'])
            if signal_row['valid'] else -1)
        # These remain candidates until object geometry and the MORAI
        # sensor/GT scene-time contract are approved for publication.
        row['object_training_label_valid']=False
        row['traffic_light_training_label_valid']=False
    counts['full_actor_inventory_transport_candidate']=sum(
        row['full_actor_inventory_transport_candidate'] for row in frames)
    counts['object_model_identity_candidate']=sum(
        row['object_model_identity_candidate'] for row in frames)
    counts['object_api_box_geometry_candidate']=sum(
        row['object_api_box_geometry_candidate'] for row in frames)
    counts['route_signal_binding_candidate']=sum(row['route_signal_binding_candidate'] for row in frames)
    report = {
        "schema": "morai_sensor_record_admission_v1",
        "capture_contract_valid": contract_check["valid"],
        "runtime_streams_valid": live["valid"],
        "sensor_capture_status": "admitted" if counts["sensor_bundle_valid"] == len(frames) else "rejected_or_partial",
        "teacher_transport_status": "admitted" if counts["ego_teacher_time_valid"] == len(frames) else "rejected_or_partial",
        "frames": len(frames),
        "counts": counts,
        "reason_counts": reason_counts(frames),
        "runtime_stream_summary": live,
        "capture_contract_checks": contract_check,
        "input_seals": evidence_seals,
        "capture_contract_sha256": digest(args.capture_contract),
        "run_plan_sha256": digest(args.run_plan),
        "raw_manifest_sha256": next(iter(raw_ids)),
        "source_timestamps_modified": False,
        "sensor_scene_state_contract": SENSOR_SCENE_STATE_CONTRACT,
        "physical_exposure_millisecond_bound_verified": False,
        "object_label_policy": ("transport, model identity, and measured API-box geometry are "
                                "separate candidates; sensor/GT scene-time approval remains required"),
        "object_geometry_evidence": {
            "observed_models": object_geometry.get("observed_models", []),
            "observed_actor_instances": object_geometry.get("observed_actor_instances", 0),
            "api_box_geometry_candidate_frames": object_geometry.get(
                "api_box_geometry_candidate_frames", 0),
            "complete_observed_geometry_scope": object_geometry.get(
                "complete_observed_geometry_scope", False),
            "physical_sensor_gt_atomicity_verified": False,
        },
        "object_inventory_evidence": {
            "snapshots": actor_audit.get("snapshots", 0),
            "maximum_actors_in_snapshot": actor_audit.get("maximum_actors_in_snapshot", 0),
            "lifecycle_evidence_present": actor_audit.get("lifecycle_evidence_present", False),
            "sensor_frames_transport_candidate": actor_audit.get("sensor_frames_transport_candidate", 0),
            "actor_instance_provenance_present": actor_audit.get(
                "actor_instance_provenance_present", False),
            "unique_non_ego_actor_instances": actor_audit.get(
                "unique_non_ego_actor_instances", 0),
            "identified_non_ego_actor_instances": actor_audit.get(
                "identified_non_ego_actor_instances", 0),
            "model_identity_complete": actor_audit.get("model_identity_complete", False),
            "sensor_frames_model_identity_candidate": actor_audit.get(
                "sensor_frames_model_identity_candidate", 0),
        },
        "traffic_light_label_policy": "exact-route binding candidate; publication still requires the sensor/GT scene contract",
        "traffic_light_timing_evidence": {
            "direct_grpc_query_snapshots": gt_report.get("direct_light_query_snapshots", 0),
            "association_methods": gt_report.get("light_association_methods", {}),
            "frame_status": gt_report.get("light_frame_status", {}),
            "audit": {
                "valid_queries": traffic_audit.get("valid_queries", 0),
                "failed_queries": traffic_audit.get("failed_queries", 0),
                "failed_attempts": traffic_audit.get("failed_attempts", 0),
                "recovered_after_retry": traffic_audit.get("recovered_after_retry", 0),
                "structural_valid": traffic_audit.get("structural_valid", False),
                "complete_route_query_coverage": traffic_audit.get("complete_route_query_coverage", False),
                "detected_color_transitions": traffic_audit.get("detected_color_transitions", 0),
            },
        },
        "route_signal_binding_evidence": {
            "stopline_lookahead_frames": route_signal_audit.get("stopline_lookahead_frames", 0),
            "bound_pure_color_frames": route_signal_audit.get("bound_pure_color_frames", 0),
            "red_frames": route_signal_audit.get("red_frames", 0),
            "green_frames": route_signal_audit.get("green_frames", 0),
            "both_binary_classes_observed": route_signal_audit.get("both_binary_classes_observed", False),
        },
        "scenario_outcome": {
            "status": scenario_outcome.get("status"),
            "reason": scenario_outcome.get("reason"),
            "full_mission_completed": scenario_outcome.get("full_mission_completed"),
            "collision_summary": scenario_outcome.get("collision_summary"),
            "cleanup_errors": scenario_outcome.get("cleanup_errors", []),
            "sha256": digest(outcome_path) if outcome_path.is_file() else None,
            "gates_sensor_capture": False,
        },
        "publication_policy": "offline dataset builder must consume dataset_build_candidate and produce a separate sealed manifest",
        "training_visible": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "frames.json", frames)
    write_json(args.output / "COMPLETE.json", {
        "schema": "morai_sensor_record_admission_v1",
        "files": {name: digest(args.output / name) for name in ("report.json", "frames.json")},
        "capture_contract_sha256": digest(args.capture_contract),
        "script_sha256": digest(Path(__file__)),
        "training_visible": False,
    })
    print(json.dumps(report), flush=True)
    if not contract_check["valid"] or not live["valid"] or not counts["dataset_build_candidate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
