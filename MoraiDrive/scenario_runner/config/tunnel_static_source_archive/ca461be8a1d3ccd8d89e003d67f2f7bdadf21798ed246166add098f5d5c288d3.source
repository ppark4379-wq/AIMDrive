"""Shared strict trial classification and production source pinning."""
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from .screening import screening_provenance

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE_ROOT.parent

def atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return copy.deepcopy(default)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def source_snapshot(golden, recovery_helper, scene_helper=None):
    files = [Path(__file__), PACKAGE_ROOT/"tools/__init__.py",
        PACKAGE_ROOT/"scenarios/full_loop_parts/tunnel_static_parts/trial.py",
        PACKAGE_ROOT/"tools/run_tunnel_static_trial.sh",
        PACKAGE_ROOT/"scenarios/segment_modes/tunnel_static_trial.py", PACKAGE_ROOT/"scenarios/segment_modes/scenario.py",
        PACKAGE_ROOT/"scenarios/segment_modes/tunnel_static_startup.py",
        PACKAGE_ROOT/"scenarios/segment_modes/expert.py", PACKAGE_ROOT/"adapters/expert_process.py",
        PACKAGE_ROOT/"adapters/expert_controller.py", PACKAGE_ROOT/"adapters/morai_grpc.py",
        PACKAGE_ROOT/"core/full_loop_drive_base.py", PACKAGE_ROOT/"config/tunnel_static.yaml",
        PACKAGE_ROOT/"config/static_obstacle_footprints_v1.json",
        PACKAGE_ROOT/"data/R_KR_PR_K-city_2025/link_set.json", PACKAGE_ROOT/"config/global.yaml"]
    files.extend(sorted((PACKAGE_ROOT/"scenarios/full_loop_parts/tunnel_static_parts").glob("*.py")))
    files.extend(sorted((PROJECT_ROOT/"data_collection/internal_sensor_record/evidence/object_geometry_registry_default3d_v2").glob("*.json")))
    files.extend(sorted((PROJECT_ROOT/"data_collection/internal_sensor_record/evidence/object_geometry_registry_unit3d_v1").glob("*.json")))
    files.extend((PROJECT_ROOT/"expert/src/expert/src/network/ros_manager.py", Path(recovery_helper)))
    if Path(recovery_helper).suffix == ".sh" and Path(recovery_helper).with_suffix(".py").is_file():
        files.append(Path(recovery_helper).with_suffix(".py"))
    if scene_helper:
        files.append(Path(scene_helper))
        if Path(scene_helper).with_suffix(".py").is_file():
            files.append(Path(scene_helper).with_suffix(".py"))
    files.append(PACKAGE_ROOT/"scenarios/full_loop_parts/tunnel_static_parts/safety.py")
    local = PACKAGE_ROOT/"config/local.yaml"
    if local.exists():
        files.append(local)
    return {"screening": screening_provenance(),
            "runtime_files": {str(path.resolve()): digest(path) for path in files},
            "golden_candidates": {kind: {"path": str(path), "sha256": digest(path)} for kind, path in golden.items()}}


def physical_motion(trace_path):
    first = None
    samples, moved = 0, False
    try:
        with Path(trace_path).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                x, y, speed = float(row["x"]), float(row["y"]), float(row.get("speed", 0.))
                if not all(math.isfinite(value) for value in (x, y, speed)):
                    return False
                if first is None:
                    first = (x, y)
                samples += 1
                moved |= math.dist(first, (x, y)) >= .5
                if moved and samples >= 3:
                    return True
    except (OSError, ValueError, KeyError):
        pass
    return False


def classify_trial(directory):
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json", {})
    drive = read_json(directory/"drive_trial.json", {})
    planner = read_json(directory/"planner_diagnostics.json", {})
    fixture = manifest.get("fixture_result", {})
    expert_entered = int(planner.get("calls", 0)) > 0
    moved = physical_motion(directory/"ego_trace.csv")
    required = ("completed", "collision_free", "returned_to_original_lane", "fixture_stationary")
    complete = drive.get("status") == "passed" and all(drive.get(key) is True for key in required)
    conditions = drive.get("completion_conditions", {})
    strict_pass = (manifest.get("status") == "passed" and manifest.get("expert_verified") is True
        and manifest.get("live_avoidance_verified") is True and fixture.get("status") == "valid_driving_trial"
        and fixture.get("cleanup", {}).get("verified") is True and complete
        and bool(conditions) and all(value is True for value in conditions.values())
        and drive.get("ego_stopped_and_parked") is True and planner.get("validated_path_seen") is True)
    screen = fixture.get("measured_route_screen", {})
    geometry_rejection = (not moved and not complete and fixture.get("cleanup", {}).get("verified") is True
        and screen.get("path_found") is False and screen.get("status") in (
            "no_path_found", "geometry_rejected", "search_timeout", "invalid_candidate"))
    metadata_failure = fixture.get("native_metadata_failure", {})
    if not isinstance(metadata_failure, dict):
        metadata_failure = {}
    previous_sequence, current_sequence = (metadata_failure.get(key) for key in (
        "previous_source_sequence", "current_source_sequence"))
    previous_stamp, current_stamp = (metadata_failure.get(key) for key in (
        "previous_source_stamp_ns", "current_source_stamp_ns"))
    numbers_valid = all(isinstance(value, int) and not isinstance(value, bool) and value>=0
                        for value in (previous_sequence, current_sequence, previous_stamp, current_stamp))
    same_publisher = (isinstance(metadata_failure.get("previous_publisher"), str)
        and bool(metadata_failure["previous_publisher"])
        and metadata_failure["previous_publisher"] == metadata_failure.get("current_publisher"))
    sequence_regressed = numbers_valid and current_sequence<=previous_sequence
    stamp_regressed = numbers_valid and current_stamp>0 and previous_stamp>0 and current_stamp<=previous_stamp
    metadata_ordering = (not expert_entered and not moved and not complete
        and fixture.get("cleanup", {}).get("verified") is True
        and metadata_failure.get("schema") == "aimdrive_native_static_metadata_failure_v1"
        and metadata_failure.get("stage") == "preflight_static_binding" and metadata_failure.get("actor_kind") == "obstacle"
        and metadata_failure.get("retryable_same_publisher_ordering") is True and same_publisher
        and ((metadata_failure.get("reason") == "source_sequence_regression" and sequence_regressed)
            or (metadata_failure.get("reason") == "source_stamp_regression" and stamp_regressed)
            or (metadata_failure.get("reason") == "source_sequence_and_stamp_regression" and sequence_regressed and stamp_regressed)))
    return {"actual_driving": expert_entered and (moved or complete), "expert_entered": expert_entered,
            "physical_motion_observed": moved, "strict_pass": bool(strict_pass),
            "status": manifest.get("status", "no_manifest"), "planner_calls": int(planner.get("calls", 0)),
            "trajectory_checks": int(drive.get("trajectory_checks", 0)),
            "failure_reason": manifest.get("failure_reason") or drive.get("failure_reason") or "",
            "fixture_cleanup_verified": fixture.get("cleanup", {}).get("verified") is True,
            "completion_conditions": conditions,
            "recoverable_geometry_rejection": geometry_rejection,
            "retryable_native_metadata_ordering": bool(metadata_ordering),
            "native_metadata_failure": metadata_failure or None,
            "screening_status": screen.get("status"),
            "requested_pattern": drive.get("requested_pattern"),
            "realized_pattern": drive.get("realized_pattern"),
            "pattern_classification": drive.get("pattern_classification")}
