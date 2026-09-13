"""Bounded bridge telemetry and a local-host production compatibility contract.

This is operational attestation, not authentication against a malicious ROS peer.
The recorder also checks actual sensor caller IDs and the ROS publisher graph.
"""
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import threading
import time

from .teacher_integrity import stats

IMPLEMENTATION = "stage6e-binary-json-v1"
HEALTH_TOPIC = "/stage6e/bridge_health"
REQUIRED_STREAMS = {"Camera-1", "Camera-2", "Camera-3", "Camera-4",
                    "/lidar3D", "/gps", "/imu", "/Ego_topic"}
SENSOR_TOPICS = {"/image_jpeg/compressed", "/lidar3D", "/gps", "/imu", "/Ego_topic"}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def implementation_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("rosbridge_ingress.py", "bridge_health.py", "production_bridge.py")}


class BridgeMetrics:
    """Keep at most 2,048 small observations per stream; never retain payloads."""
    def __init__(self, window_s=5):
        self.window_ns = int(window_s * 1e9)
        self.lock = threading.Lock()
        self.streams = defaultdict(lambda: deque(maxlen=2048))
        self.errors = deque(maxlen=2048)
        self.total_errors = 0
        self.total_messages = 0

    def observe(self, row):
        with self.lock:
            self.total_messages += 1
            if row.get("errors", 0) or row.get("residual", 0):
                self.errors.append(row["end_ns"])
                self.total_errors += 1
            if row.get("op") != "publish":
                return
            key = row.get("frame_id") if row.get("topic") == "/image_jpeg/compressed" else row.get("topic")
            if key in REQUIRED_STREAMS:
                self.streams[key].append(row)

    def snapshot(self, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        cutoff = now - self.window_ns
        with self.lock:
            streams = {k: [dict(r) for r in v if cutoff <= r["end_ns"] <= now]
                       for k, v in self.streams.items()}
            errors = sum(t >= cutoff for t in self.errors)
            total_errors, total_messages = self.total_errors, self.total_messages
        result = {}
        for key, rows in streams.items():
            if not rows:
                continue
            times = [r["receive_ns"] for r in rows]
            duration = (max(times)-min(times))/1e9
            result[key] = dict(samples=len(rows), receive_rate_hz=(len(rows)-1)/duration if duration > 0 else 0,
                               receive_age_ms=(now-max(times))/1e6,
                               queue_delay_ms=stats(r["queue_ms"] for r in rows),
                               processing_ms=stats(r["processing_ms"] for r in rows),
                               client_ids=sorted({str(r["client_id"]) for r in rows}),
                               last_receive_monotonic_ns=max(times))
        return dict(streams=result, recent_errors=errors, total_errors=total_errors,
                    total_messages=total_messages, window_s=self.window_ns/1e9)


def check_bridge(health, *, now_ns=None, expected_boot=None, expected_hashes=None,
                 health_callerid=None, publishers=None, expected_instance=None,
                 require_streams=True, max_queue_ms=250):
    now = time.monotonic_ns() if now_ns is None else now_ns
    failures = []
    if not health or health.get("implementation") != IMPLEMENTATION or not health.get("binary_json_patch"):
        return ["BRIDGE_NOT_PATCHED: production health heartbeat absent or incompatible"]
    if health.get("boot_id") != (boot_id() if expected_boot is None else expected_boot):
        failures.append("BRIDGE_NOT_PATCHED: different host/boot; receive clocks cannot be compared")
    if health.get("code_sha256") != (implementation_hashes() if expected_hashes is None else expected_hashes):
        failures.append("BRIDGE_NOT_PATCHED: running implementation differs from recorder code")
    age = (now-health.get("monotonic_ns", 0))/1e9
    if age < 0 or age > 2.5:
        failures.append("BRIDGE_NOT_PATCHED: stale health heartbeat")
    if not health.get("listening"):
        failures.append("BRIDGE_NOT_PATCHED: bridge port is not listening")
    if health_callerid is not None and health_callerid != health.get("node"):
        failures.append("BRIDGE_NOT_PATCHED: health publisher identity mismatch")
    if expected_instance and health.get("instance_id") != expected_instance:
        failures.append("BRIDGE_NOT_PATCHED: bridge restarted during run")
    if publishers is not None:
        for topic in SENSOR_TOPICS:
            if publishers.get(topic) and set(publishers[topic]) != {health.get("node")}:
                failures.append("BRIDGE_NOT_PATCHED: unexpected publisher for " + topic)
    if health.get("recent_errors", 0):
        failures.append("INVALID_EGO_TEACHER: bridge JSON or payload processing errors")
    for client in health.get("clients", []):
        if client.get("oldest_pending_ms", 0) > max_queue_ms:
            failures.append("INVALID_EGO_TEACHER: bridge pending queue exceeds policy")
    if require_streams:
        for key in sorted(REQUIRED_STREAMS):
            s = health.get("streams", {}).get(key, {})
            if s.get("samples", 0) < 2 or s.get("receive_age_ms", 1e9) > 500:
                failures.append("SENSOR_MISSING: bridge " + key)
            elif (s.get("queue_delay_ms") or {}).get("max", 1e9) > max_queue_ms:
                failures.append("INVALID_EGO_TEACHER: bridge queue " + key)
            if len(s.get("client_ids", [])) > 1:
                failures.append("INVALID_EGO_TEACHER: multiple live producers for " + key)
    return failures
