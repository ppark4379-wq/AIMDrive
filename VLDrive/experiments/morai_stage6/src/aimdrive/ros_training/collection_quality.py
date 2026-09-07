"""Shared preflight/run quality policy; no ROS or encoder dependency.

Policy thresholds are collection guards, not MORAI specifications. Observed
healthy rates were ~46 Hz Ego/IMU, ~20 Hz GPS, ~9 Hz LiDAR and >10 Hz/camera.
The low rate floors allow slower simulation while rejecting serious starvation.
250 ms selected age is retained; live freshness allows two such intervals.
"""
import math
import numpy as np

from .bridge_health import check_bridge
from .teacher_integrity import clock_summary, manifest_streams, stats, unique_receive, validate_manifest

ROLES = ("front", "left", "right", "rear", "lidar", "gps", "imu", "ego_teacher")
MIN_RATES = dict(front=3, left=3, right=3, rear=3, lidar=3, gps=3, imu=5, ego_teacher=5)


def ego_kinematics(samples):
    """Compare source-time FD with source velocity, never wall-time simulation speed.

Use ~0.5 s nonoverlapping source windows to reduce quantization/acceleration
noise. A factor-two disagreement in >30% of >=5 moving windows is severe;
ordinary braking, curved chords, and slight frame timing jitter are tolerated.
Both FD clocks are reported explicitly, so slow simulation is not called a bug.
"""
    samples = unique_receive(samples)
    raw = [math.hypot(*s["velocity_xyz"][:2]) for s in samples]
    source_fd, receive_fd, ratios, comparisons = [], [], [], []
    start = 0
    for i in range(1, len(samples)):
        a, b = samples[start], samples[i]
        sa, sb = a.get("source_stamp_ns"), b.get("source_stamp_ns")
        if sa is None or sb is None or sb <= sa:
            continue
        dt = (sb-sa)/1e9
        if dt < .5:
            continue
        dr = (b["receive_monotonic_ns"]-a["receive_monotonic_ns"])/1e9
        distance = math.dist(a["position_xyz"][:2], b["position_xyz"][:2])
        fd = distance/dt
        mean_raw = float(np.mean(raw[start:i+1]))
        source_fd.append(fd)
        if dr > 0:
            receive_fd.append(distance/dr)
        if max(fd, mean_raw) > 2:
            comparisons.append(min(fd, mean_raw) < .5*max(fd, mean_raw))
            if fd > .01:
                ratios.append(mean_raw/fd)
        start = i
    position = np.array([s["position_xyz"][:2] for s in samples], dtype=float)
    heading = np.unwrap(np.radians([s["heading_deg"] for s in samples]))
    return dict(raw_velocity_mps=stats(raw), fd_source_velocity_mps=stats(source_fd),
                fd_receive_velocity_mps=stats(receive_fd), raw_over_fd_source_ratio=stats(ratios),
                moving_windows=len(comparisons), severe_mismatch_windows=sum(comparisons),
                severe_mismatch=len(comparisons)>=5 and sum(comparisons)/len(comparisons)>.3,
                cumulative_path_m=float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()) if len(samples)>1 else 0,
                position_span_m=float(np.linalg.norm(np.ptp(position, axis=0))) if len(samples)>1 else 0,
                raw_velocity_span_mps=float(np.ptp(raw)) if raw else 0,
                heading_span_deg=float(np.degrees(np.ptp(heading))) if len(heading)>1 else 0)


def validate_rows(rows, max_age_ms=250, future_horizon_s=2.5):
    base = validate_manifest(rows, max_age_ms)
    reasons = list(base["failures"])
    streams = manifest_streams(rows)
    if any(not r.get("gps") or not r.get("cameras", {}).get("rear") for r in rows):
        reasons.append("sensor_missing: GPS and Camera-4 required for Stage6E collection")
    ego = streams.get("ego_teacher", [])
    kin = ego_kinematics(ego)
    if kin["severe_mismatch"]:
        reasons.append("ego_fd_raw_severe_mismatch")
    if ego and all(s.get("seq") is not None for s in ego) and len({s["seq"] for s in ego}) < 2:
        reasons.append("ego_sequence_not_progressing")
    source = [s["source_stamp_ns"] for s in ego if s.get("source_stamp_ns") is not None]
    if not source or (max(source)-min(source))/1e9 <= future_horizon_s:
        reasons.append("insufficient_future_horizon")
    # GPS remains diagnostic, but reusing an old GPS must not certify a teacher.
    for r in rows:
        if r.get("gps") and r.get("lidar"):
            age = (r["lidar"]["receive_monotonic_ns"]-r["gps"]["receive_monotonic_ns"])/1e6
            if age < 0 or age > max_age_ms:
                reasons.append("future_or_stale_gps")
                break
    if base["status"] == "unverified":
        reasons.append("insufficient_independent_motion_evidence")
    return dict(quality_status="rejected" if reasons else "valid", reasons=sorted(set(reasons)),
                teacher_validation=base, ego_kinematics=kin,
                policy=dict(max_selected_receive_age_ms=max_age_ms, min_callback_rates_hz=MIN_RATES,
                            future_horizon_s=future_horizon_s, fd_raw_ratio_tolerance=[.5, 2.0]))


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
