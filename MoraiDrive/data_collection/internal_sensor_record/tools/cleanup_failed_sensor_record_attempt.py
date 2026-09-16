#!/usr/bin/env python3
"""Safely remove only bulky payloads owned by one failed collection attempt."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssd_storage import SSD_ROOT, require_ssd


BASE = SSD_ROOT / "sensor_record_auto_v1"
RUN_NAME = re.compile(r"^[A-Za-z0-9_]{1,80}$")
NATIVE_ROLES = (
    "CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4",
    "LIDAR_5", "GPS_6", "IMU_7",
)
PRE_READINESS_EXIT = "Scenario ROS capture exited before readiness"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _assert_direct_child(path, parent, label):
    path = Path(path)
    parent = Path(parent)
    if path.parent != parent or path.is_symlink():
        raise ValueError("unsafe %s path: %s" % (label, path))


def _assert_real_tree(path, root, label):
    path = Path(path)
    root = Path(root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("unsafe %s path: %s" % (label, path)) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("refusing symlink in %s path: %s" % (label, current))


def _measure(path):
    path = Path(path)
    if not path.exists():
        return {"files": 0, "bytes": 0}
    if path.is_symlink():
        raise ValueError("refusing to inspect symlink payload: " + str(path))
    files = 0
    size = 0
    for directory, names, filenames in os.walk(str(path), followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError("refusing symlink directory: " + str(directory_path))
        for name in names:
            if (directory_path / name).is_symlink():
                raise ValueError("refusing symlink payload: " + str(directory_path / name))
        for name in filenames:
            item = directory_path / name
            if item.is_symlink():
                raise ValueError("refusing symlink payload: " + str(item))
            files += 1
            size += item.stat().st_size
    return {"files": files, "bytes": size}


def _clear_contents(path):
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir() or path.is_symlink():
        raise ValueError("payload target is not a real directory: " + str(path))
    for child in path.iterdir():
        if child.is_symlink():
            raise ValueError("refusing symlink payload: " + str(child))
        if child.is_dir():
            shutil.rmtree(str(child))
        else:
            child.unlink()


def _pre_record_failure_evidence(run, native, base, capture_result):
    """Prove that the Windows Sensor Record owner was never invoked.

    The supervisor prepares an isolated native session before launching the
    Scenario Runner. It creates the capture contract, coordination directory,
    and raw/native capture files only after the child publishes ready.json.
    Keep this exception deliberately narrow: any partial capture evidence must
    remain fail-closed.
    """
    run = Path(run)
    native = Path(native)
    base = Path(base)
    error = str(capture_result.get("error") or "")
    if PRE_READINESS_EXIT not in error:
        return None
    if capture_result.get("scenario_exit_code") in (None, 0):
        return None

    episode = run / "episode_000"
    forbidden = {
        "ready": episode / "ready.json",
        "coordination": episode / "sensor_record_coordination",
        "raw": episode / "raw",
        "capture_contract": run / "capture_contract.json",
        "native_setup": run / "native_setup.json",
        "native_ui_result": run / "native_ui_result.json",
    }
    if any(path.exists() for path in forbidden.values()):
        return None

    plan_path = base / (run.name + "_native_session_plan.json")
    apply_path = base / (run.name + "_native_session_apply.json")
    session_path = native / "SESSION.json"
    documents = (plan_path, apply_path, session_path)
    if any(not path.is_file() or path.is_symlink() for path in documents):
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        applied = json.loads(apply_path.read_text(encoding="utf-8-sig"))
        session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    applied_plan = applied.get("plan") if isinstance(applied, dict) else None
    session_plan = session.get("plan") if isinstance(session, dict) else None
    if (not isinstance(plan, dict)
            or not isinstance(applied, dict)
            or not isinstance(session, dict)
            or not isinstance(applied_plan, dict)
            or not isinstance(session_plan, dict)
            or plan.get("session") != run.name
            or applied.get("applied") is not True
            or applied_plan.get("session") != run.name
            or session.get("applied") is not True
            or session_plan.get("session") != run.name):
        return None

    expected_names = set(NATIVE_ROLES) | {"SESSION.json"}
    if not native.is_dir() or native.is_symlink():
        return None
    if {path.name for path in native.iterdir()} != expected_names:
        return None
    for role in NATIVE_ROLES:
        directory = native / role
        if (not directory.is_dir() or directory.is_symlink()
                or any(directory.iterdir())):
            return None

    return {
        "stage": "scenario_exited_before_readiness",
        "sensor_record_owner_invoked": False,
        "scenario_exit_code": capture_result.get("scenario_exit_code"),
        "capture_error": error,
        "capture_result": str(run / "NATIVE_CAPTURE_RESULT.json"),
        "native_session_plan": str(plan_path),
        "native_session_apply": str(apply_path),
        "native_session_record": str(session_path),
        "absent_capture_paths": {
            label: str(path) for label, path in forbidden.items()
        },
    }


def cleanup_failed_attempt(run_name, *, base=BASE, failure_kind,
                           failure_reason, validated_invalid=False,
                           evidence=None, require_storage=True):
    """Clean known bulky payload directories after a proven terminal failure.

    The run directory, diagnostics, manifests, native role directories, and
    native SESSION.json are preserved. The operation is deliberately
    fail-closed: a confirmed Sensor Record OFF marker is mandatory after
    recording starts. A narrowly proven exit before readiness is recorded as
    not_started and has no payload to delete.
    """
    if not RUN_NAME.fullmatch(str(run_name)):
        raise ValueError("invalid run name: " + repr(run_name))
    base = Path(base).expanduser().resolve()
    if base.name != "sensor_record_auto_v1":
        raise ValueError("unexpected Sensor Record base: " + str(base))
    if require_storage:
        require_ssd(base, reserve_bytes=0)
    run = base / str(run_name)
    native_parent = base / "native_sessions"
    native = native_parent / str(run_name)
    _assert_direct_child(run, base, "run")
    _assert_direct_child(native, native_parent, "native session")
    _assert_real_tree(run, base, "run")
    _assert_real_tree(native, base, "native session")
    if not run.is_dir():
        raise FileNotFoundError("failed run directory is missing: " + str(run))

    marker_path = run / "FAILED_ATTEMPT.json"
    if marker_path.is_symlink():
        raise ValueError("failed-attempt marker must not be a symlink")
    if marker_path.is_file():
        marker = _load(marker_path)
        if (marker.get("schema") == "morai_sensor_record_failed_attempt_v1"
                and marker.get("cleanup_status") == "completed"
                and marker.get("failure_kind") == failure_kind
                and marker.get("run_name") == str(run_name)
                and marker.get("training_visible") is False):
            return marker

    capture_result = None
    if failure_kind == "capture":
        capture_result = _load(run / "NATIVE_CAPTURE_RESULT.json")
        if capture_result.get("error") is None:
            raise RuntimeError("capture result does not prove failure")
    elif failure_kind == "validation":
        if validated_invalid is not True:
            raise RuntimeError("validation failure was not authorized by validator")
    else:
        raise ValueError("unknown failure kind: " + str(failure_kind))

    stopped_path = (
        run / "episode_000" / "sensor_record_coordination"
        / "recording_stopped.json"
    )
    stopped = None
    pre_record_evidence = None
    if stopped_path.is_file():
        stopped = _load(stopped_path)
        if not (stopped.get("status") == "stopped"
                and stopped.get("record_on") is False
                and stopped.get("error") is None):
            raise RuntimeError(
                "Sensor Record OFF is not proven for " + str(run_name))
        recording_state = "stopped"
    elif failure_kind == "capture":
        pre_record_evidence = _pre_record_failure_evidence(
            run, native, base, capture_result)
        if pre_record_evidence is None:
            raise RuntimeError(
                "Sensor Record start/stop state is not proven for "
                + str(run_name))
        recording_state = "not_started"
    else:
        raise RuntimeError("Sensor Record OFF is not proven for " + str(run_name))

    targets = []
    raw_chunks = None
    if recording_state == "stopped":
        setup_path = run / "native_setup.json"
        if not setup_path.is_file():
            raise RuntimeError("native session ownership evidence is missing")
        recorded_native = Path(_load(setup_path).get("native_root", ""))
        if recorded_native.expanduser().resolve() != native:
            raise RuntimeError("native session ownership mismatch")
        if native.exists() and (not native.is_dir() or native.is_symlink()):
            raise RuntimeError("unsafe native session path: " + str(native))

        raw_chunks = run / "episode_000" / "raw" / "chunks"
        targets = [raw_chunks] + [native / role for role in NATIVE_ROLES]
        for target in targets:
            expected_parent = (
                raw_chunks.parent if target == raw_chunks else native)
            _assert_direct_child(target, expected_parent, "payload")
            _assert_real_tree(target, base, "payload")

    measurements = []
    total_files = 0
    total_bytes = 0
    for target in targets:
        measured = _measure(target)
        total_files += measured["files"]
        total_bytes += measured["bytes"]
        measurements.append({
            "path": str(target),
            "files": measured["files"],
            "bytes": measured["bytes"],
        })

    common = {
        "schema": "morai_sensor_record_failed_attempt_v1",
        "run_name": str(run_name),
        "failure_kind": failure_kind,
        "failure_reason": str(failure_reason),
        "training_visible": False,
        "recording_state": recording_state,
        "recording_stopped": str(stopped_path) if stopped is not None else None,
        "pre_record_failure_evidence": pre_record_evidence,
        "payload_targets": measurements,
        "deleted_files": total_files,
        "deleted_bytes": total_bytes,
        "evidence": json.loads(json.dumps(evidence or {})),
    }
    _write_json_atomic(marker_path, dict(
        common, cleanup_status="started", started_wall_ns=time.time_ns()))
    for target in targets:
        _clear_contents(target)
    if recording_state == "stopped":
        raw_chunks.mkdir(parents=True, exist_ok=True)
        for role in NATIVE_ROLES:
            (native / role).mkdir(parents=True, exist_ok=True)
    completed = dict(
        common, cleanup_status="completed", completed_wall_ns=time.time_ns())
    _write_json_atomic(marker_path, completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--failure-status", required=True)
    args = parser.parse_args()
    result = cleanup_failed_attempt(
        args.run_name,
        failure_kind="capture",
        failure_reason=args.failure_status,
    )
    print(json.dumps({
        "run_name": result["run_name"],
        "cleanup_status": result["cleanup_status"],
        "deleted_files": result["deleted_files"],
        "deleted_bytes": result["deleted_bytes"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
