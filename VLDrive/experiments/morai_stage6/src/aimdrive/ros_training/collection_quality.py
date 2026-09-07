"""Shared preflight/run quality policy; no ROS or encoder dependency.

Policy thresholds are collection guards, not MORAI specifications. Observed
healthy rates were ~46 Hz Ego/IMU, ~20 Hz GPS, ~9 Hz LiDAR and >10 Hz/camera.
The low rate floors allow slower simulation while rejecting serious starvation.
250 ms selected age is retained; live freshness allows two such intervals.
"""
import math
import numpy as np

from .bridge_health import check_bridge
from aimdrive._lavis_compat import load_module
_quality = load_module("lavis.datasets.datasets.morai.quality")
ego_kinematics = _quality.ego_kinematics
validate_rows = _quality.validate_rows
from .teacher_integrity import clock_summary, manifest_streams, stats, unique_receive, validate_manifest

ROLES = ("front", "left", "right", "rear", "lidar", "gps", "imu", "ego_teacher")
MIN_RATES = dict(front=3, left=3, right=3, rear=3, lidar=3, gps=3, imu=5, ego_teacher=5)






def callback_summary(callbacks, now_ns):
    result = {}
    for role in ROLES:
        samples = [s for s in callbacks.get(role, []) if s["receive_monotonic_ns"] >= now_ns-5_000_000_000]
        summary = clock_summary(samples)
        summary["receive_age_ms"] = (now_ns-samples[-1]["receive_monotonic_ns"])/1e6 if samples else None
        result[role] = summary
    return result


def evaluate_preflight(rows, callbacks, health, *, now_ns, publishers=None,
                       health_callerid=None, expected_instance=None, bridge_options=None):
    failures = check_bridge(health, now_ns=now_ns, publishers=publishers,
                            health_callerid=health_callerid, expected_instance=expected_instance,
                            **(bridge_options or {}))
    summary = callback_summary(callbacks, now_ns)
    for role, s in summary.items():
        if s.get("samples", 0)<2 or s["receive_age_ms"] is None or s["receive_age_ms"]>500:
            failures.append("SENSOR_MISSING: " + role)
        elif s.get("receive_duration_s", 0)>=3 and s.get("receive_rate_hz", 0)<MIN_RATES[role]:
            failures.append("SENSOR_MISSING: low callback rate " + role)
        ids = s.get("callerids", [])
        if ids and set(ids) != {health.get("node")}:
            failures.append("BRIDGE_NOT_PATCHED: actual callback caller mismatch " + role)
    # A partial first snapshot is a normal subscriber warmup, not an invalid run.
    complete = [r for r in rows if all(r.get(k) for k in ("lidar", "imu", "gps", "ego_teacher"))
                and {"front", "left", "right", "rear"} <= set(r.get("cameras", {}))]
    quality = validate_rows(complete)
    pending = {"insufficient_independent_motion_evidence", "insufficient_future_horizon", "empty_or_insufficient_teacher"}
    hard = [r for r in quality["reasons"] if r not in pending]
    if hard and len(complete)>=3:
        failures.extend("INVALID_EGO_TEACHER: " + r for r in hard)
    state = "READY"
    if any(r.startswith("BRIDGE_NOT_PATCHED") for r in failures):
        state = "BRIDGE_NOT_PATCHED"
    elif any(r.startswith("INVALID_EGO_TEACHER") for r in failures):
        state = "INVALID_EGO_TEACHER"
    elif failures:
        state = "SENSOR_MISSING"
    elif quality["quality_status"] != "valid":
        state = "WAITING_FOR_MOTION"
    return dict(state=state, reasons=failures or quality["reasons"], callbacks=summary,
                ego_kinematics=quality["ego_kinematics"],
                moving_windows=quality["teacher_validation"]["motion"].get("moving_windows", 0))
