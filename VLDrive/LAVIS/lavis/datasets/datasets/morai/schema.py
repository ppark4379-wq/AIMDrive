"""Compact training manifest; privileged fields never enter model_inputs."""
import json
from pathlib import Path

from .validation import read_rows, validate_run

SCHEMA = "stage6e-training-sample-v1"


def training_sample(row, run_id):
    feature = row["visual_feature"]
    route = row["route_input"]
    labels = row["labels"]
    teacher = row["ego_teacher"]
    if feature["shape"] != [106,256] or feature["velocity_mps"] != 0 or feature["lidar_rotation_deg"] != 0:
        raise ValueError("Visual feature contract mismatch")
    if not route["localization_teacher_only"] or route["runtime_localization_dependency"] or labels["runtime_model_input"]:
        raise ValueError("Privileged boundary mismatch")
    anchor = row["lidar"]["receive_monotonic_ns"]
    if any(ref["receive_monotonic_ns"] > anchor for ref in
           list(row["cameras"].values()) + [row["imu"], teacher, row["gps"]]):
        raise ValueError("Noncausal sample")
    return dict(schema_version=SCHEMA, sample_id="%s/%08d" % (run_id,row["frame_index"]), run_id=run_id,
                anchor_receive_monotonic_ns=anchor,
                model_inputs=dict(visual_feature={k:feature[k] for k in ("path","shape","dtype")},
                                  route=dict(path=route["path"], key="ego_frame_xy", shape=route["shape"])),
                training_targets=dict(future_trajectory=dict(path=labels["path"], key="trajectory_xy_forward_left",
                                                            shape=[5,2], horizons_s=labels["horizons_s"])),
                privileged_metadata=dict(ego_teacher=teacher, route_progress_m=route["route_s_m"],
                                         route_localization="/Ego_topic training only"),
                sensor_refs=dict(cameras=row["cameras"], lidar=row["lidar"], imu=row["imu"], gps=row["gps"]),
                diagnostics=dict(sync_clock="receive_monotonic_ns", teacher_timebase="ego_source_stamp_ns",
                                 visual_velocity_source=feature["velocity_source"],
                                 checkpoint_sha256=feature["checkpoint_sha256"]))


def publish_training_manifest(run, allow_legacy=False):
    run = Path(run)
    quality = validate_run(run, require_features=True, allow_legacy=allow_legacy)
    if quality["quality_status"] != "valid" and not (allow_legacy and quality["quality_status"]=="warning"):
        raise ValueError("Dataset cannot be promoted: " + str(quality["reasons"]))
    rows = read_rows(run/"manifest_features.jsonl")
    output = run/"manifest_training.jsonl"
    if output.exists():
        raise ValueError("Refusing existing training manifest")
    partial = output.with_suffix(output.suffix+".partial")
    with partial.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(training_sample(row, run.name), separators=(",", ":"))+"\n")
    partial.rename(output)
    return quality


def ready_runs(root):
    """Recheck quality and completeness; an old registry line is not approval."""
    available, excluded = [], {}
    for run in sorted(Path(root).glob("run_[0-9][0-9][0-9][0-9]")):
        if not (run/"manifest_training.jsonl").exists():
            excluded[run.name] = ["training_manifest_missing"]
            continue
        try:
            quality = validate_run(run, require_features=True, check_artifacts=False)
            if quality["quality_status"] == "valid":
                available.append(run.name)
            else:
                excluded[run.name] = quality["reasons"] + quality["warnings"]
        except (ValueError, OSError, KeyError) as exc:
            excluded[run.name] = [str(exc)]
    return available, excluded
