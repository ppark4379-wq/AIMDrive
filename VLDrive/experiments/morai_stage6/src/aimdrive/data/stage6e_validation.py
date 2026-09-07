"""Run validation shared by recorder, postprocessing, registry and splits."""
import json
import math
from pathlib import Path

import numpy as np

from aimdrive.ros_training.collection_quality import validate_rows
from aimdrive.ros_training.teacher_integrity import stats


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def artifact_path(run, relative):
    path = (Path(run)/relative).resolve()
    if Path(run).resolve() not in path.parents:
        raise ValueError("Artifact escapes its run: " + str(relative))
    return path


def validate_run(run, *, require_features=False, allow_legacy=False, check_artifacts=True):
    run = Path(run).resolve()
    rows = read_rows(run/"manifest.jsonl")
    report = validate_rows(rows)
    from .stage6e_runs import summarize_manifest
    motion_summary = summarize_manifest(run/"manifest.jsonl")
    report["gps_displacement"] = motion_summary["gps"]
    report["ego_motion_summary"] = motion_summary["ego_teacher"]
    reasons = report["reasons"]
    warnings = []
    meta = json.loads((run/"run_meta.json").read_text()) if (run/"run_meta.json").exists() else {}
    counts = json.loads((run/"stats.json").read_text()) if (run/"stats.json").exists() else {}
    if counts.get("writer_error") or meta.get("writer_error"):
        reasons.append("recorder_writer_error")
    if meta.get("acquisition_failure"):
        reasons.append("acquisition_failure: " + str(meta["acquisition_failure"]))
    bridge_log = run/"bridge_health.jsonl"
    if bridge_log.exists():
        health_rows = read_rows(bridge_log)
        recording = [r for r in health_rows if r.get("phase")=="recording"]
        if not recording:
            reasons.append("missing_recording_bridge_health")
        else:
            instances = {r.get("bridge", {}).get("instance_id") for r in recording}
            if len(instances)!=1 or None in instances:
                reasons.append("bridge_instance_changed")
            if any(r.get("decision", {}).get("state") in
                   ("BRIDGE_NOT_PATCHED", "INVALID_EGO_TEACHER", "SENSOR_MISSING") for r in recording):
                reasons.append("recording_health_failure")
            queue_stats = [s["queue_delay_ms"] for r in recording for s in r.get("bridge", {}).get("streams", {}).values()
                           if s.get("queue_delay_ms")]
            report["bridge"] = dict(identity={k: recording[0]["bridge"].get(k) for k in
                                              ("implementation", "code_sha256", "instance_id", "node", "pid", "port", "hostname", "boot_id")},
                                     max_queue_delay_ms=max((s["max"] for s in queue_stats), default=None),
                                     window_median_queue_delay_ms=stats(s["median"] for s in queue_stats),
                                     window_p95_queue_delay_ms=stats(s["p95"] for s in queue_stats),
                                     note="Median/p95 summaries describe per-stream five-second windows; raw windows are in bridge_health.jsonl.",
                                     first_clients=recording[0]["bridge"].get("clients", []),
                                     last_clients=recording[-1]["bridge"].get("clients", []))
            report["callbacks"] = recording[-1].get("decision", {}).get("callbacks", {})
    elif allow_legacy:
        warnings.append("legacy_offline: no production bridge attestation; excluded from automatic training registry")
    else:
        reasons.append("missing_production_bridge_health")
    report["counts"] = {k: counts.get(k, 0) for k in
                        ("frames_written", "frames_skipped_missing", "writer_queue_drops", "frames_discarded_writer_error")}
    report["counts"]["stale_samples_rejected"] = counts.get("stale_samples_rejected", {})
    report["counts"]["missing_sensor_count"] = counts.get("missing_sensor_count", {})
    if counts.get("writer_queue_drops", 0):
        warnings.append("writer_queue_drops")
    ready = read_rows(run/"manifest_ready.jsonl") if (run/"manifest_ready.jsonl").exists() else []
    features = read_rows(run/"manifest_features.jsonl") if (run/"manifest_features.jsonl").exists() else []
    if require_features and (not ready or len(features)!=len(ready)):
        reasons.append("missing_or_incomplete_feature_cache")
    if (run/"manifest_training.jsonl").exists():
        from .stage6e_schema import training_sample
        try:
            training = read_rows(run/"manifest_training.jsonl")
            if len(training)!=len(features) or any(sample!=training_sample(row,run.name) for sample,row in zip(training,features)):
                reasons.append("training_manifest_contract_or_completeness_mismatch")
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append("training_manifest_invalid: " + str(exc))
    if check_artifacts:
        try:
            from PIL import Image
            for row in rows:
                for role in ("front", "left", "right", "rear"):
                    with Image.open(artifact_path(run, row["cameras"][role]["path"])) as im:
                        im.verify()
                lidar = np.load(artifact_path(run, row["lidar"]["path"]), allow_pickle=False)
                if lidar.ndim!=2 or lidar.shape[1]!=4 or lidar.dtype!=np.float32 or not np.isfinite(lidar).all():
                    raise ValueError("invalid_lidar_array")
            raw_by_frame = {r["frame_index"]: r for r in rows}
            from aimdrive.ros_training.teacher_integrity import manifest_streams
            teacher = manifest_streams(rows).get("ego_teacher", [])
            origin = teacher[0]["source_stamp_ns"] if teacher else 0
            teacher_time = np.array([(s["source_stamp_ns"]-origin)/1e9 for s in teacher])
            teacher_xy = np.array([s["position_xyz"][:2] for s in teacher])
            if ready:
                expected_ids = {r["frame_index"] for r in rows
                                if r["ego_teacher"]["source_stamp_ns"]+2_500_000_000 <= teacher[-1]["source_stamp_ns"]}
                actual_ids = [r["frame_index"] for r in ready]
                if set(actual_ids)!=expected_ids or len(actual_ids)!=len(set(actual_ids)):
                    raise ValueError("incomplete_or_duplicate_ready_manifest")
            for row in ready:
                if row["frame_index"] not in raw_by_frame:
                    raise ValueError("ready_frame_outside_run")
                if row["labels"].get("runtime_model_input") or row["route_input"].get("runtime_localization_dependency"):
                    raise ValueError("privileged_inference_dependency")
                with np.load(artifact_path(run, row["labels"]["path"]), allow_pickle=False) as label:
                    trajectory = label["trajectory_xy_forward_left"]
                    if trajectory.shape != (5, 2) or not np.isfinite(trajectory).all():
                        raise ValueError("invalid_trajectory_array")
                    horizons = label["horizons_s"]
                    if not np.allclose(horizons, [.5,1,1.5,2,2.5]):
                        raise ValueError("unexpected_future_horizons")
                    anchor_teacher = raw_by_frame[row["frame_index"]]["ego_teacher"]
                    targets = (anchor_teacher["source_stamp_ns"]-origin)/1e9+horizons
                    if not len(teacher_time) or targets[-1]>teacher_time[-1] or targets[0]<teacher_time[0]:
                        raise ValueError("future_horizon_crosses_run_boundary")
                    future = np.column_stack([np.interp(targets,teacher_time,teacher_xy[:,axis]) for axis in (0,1)])
                    delta = future-np.array(anchor_teacher["position_xyz"][:2])
                    theta = math.radians(anchor_teacher["heading_deg"])
                    expected = delta@np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
                    # float32 saved targets can round millimetres; metre-scale
                    # corruption or another run's label must never pass.
                    if not np.allclose(trajectory,expected,rtol=1e-5,atol=.002):
                        raise ValueError("saved_trajectory_disagrees_with_raw_teacher")
                with np.load(artifact_path(run, row["route_input"]["path"]), allow_pickle=False) as route:
                    if route["ego_frame_xy"].shape!=(41, 2) or not np.isfinite(route["ego_frame_xy"]).all():
                        raise ValueError("invalid_route_array")
            ready_ids = {r["frame_index"] for r in ready}
            feature_ids = [r["frame_index"] for r in features]
            if features and (set(feature_ids)!=ready_ids or len(feature_ids)!=len(set(feature_ids))):
                raise ValueError("feature_frame_identity_mismatch")
            for row in features:
                ref = row["visual_feature"]
                feature = np.load(artifact_path(run, ref["path"]), allow_pickle=False)
                if feature.shape!=(106,256) or feature.dtype not in (np.float16,np.float32) or not np.isfinite(feature).all():
                    raise ValueError("invalid_visual_feature")
                if ref["velocity_mps"]!=0 or ref["lidar_rotation_deg"]!=0 or ref["trainable_encoder"]:
                    raise ValueError("visual_encoder_contract_changed")
        except (OSError, ValueError, KeyError) as exc:
            reasons.append("artifact_validation: " + str(exc))
    report.update(schema_version="stage6e-quality-v2", frames=len(rows), ready_frames=len(ready),
                  feature_frames=len(features), reasons=sorted(set(reasons)), warnings=warnings,
                  quality_status="rejected" if reasons else "warning" if warnings else "valid")
    return report
