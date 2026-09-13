"""Operational guard for the already running binary-JSON bridge; never starts it."""
from pathlib import Path

STREAMS = {"Camera-1", "Camera-2", "Camera-3", "Camera-4", "/lidar3D", "/gps", "/imu", "/Ego_topic"}


def validate_bridge(health, now_ns, caller_id, *, expected_boot_id=None,
                    max_stream_age_ms=500, required_streams=None,
                    expected_camera_ingress_policy=None,
                    expected_lidar_ingress_policy=None):
    failures = []
    required_streams = STREAMS if required_streams is None else set(required_streams)
    if not required_streams or not required_streams <= STREAMS:
        raise ValueError("Bridge required_streams must be a nonempty STREAMS subset")
    if health.get("implementation") != "stage6e-binary-json-v1" or not health.get("binary_json_patch"):
        failures.append("missing_binary_json_patch")
    if health.get("boot_id") != (expected_boot_id or Path("/proc/sys/kernel/random/boot_id").read_text().strip()):
        failures.append("bridge_receive_clock_is_not_local")
    if not 0 <= now_ns - health.get("monotonic_ns", 0) <= 2500000000:
        failures.append("stale_bridge_health")
    if caller_id != health.get("node") or not health.get("listening"):
        failures.append("bridge_publisher_or_listener_mismatch")
    if not health.get("instance_id"):
        failures.append("missing_bridge_instance")
    if (expected_camera_ingress_policy is not None
            and health.get('camera_ingress_policy') != expected_camera_ingress_policy):
        failures.append('bridge_camera_ingress_policy_mismatch')
    if (expected_lidar_ingress_policy is not None
            and health.get('lidar_ingress_policy') != expected_lidar_ingress_policy):
        failures.append('bridge_lidar_ingress_policy_mismatch')
    if health.get("recent_errors", 0):
        failures.append("bridge_payload_errors")
    if any(c.get("oldest_pending_ms", 0) > 250 for c in health.get("clients", [])):
        failures.append("bridge_queue_over_250ms")
    for key in required_streams:
        stream = health.get("streams", {}).get(key, {})
        if stream.get("samples", 0) < 2 or stream.get("receive_age_ms", 1e9) > max_stream_age_ms:
            failures.append("bridge_missing_stream:" + key)
        if (stream.get("queue_delay_ms") or {}).get("max", 1e9) > 250:
            failures.append("bridge_delayed_stream:" + key)
        if len(stream.get("client_ids", [])) != 1:
            failures.append("bridge_producer_count:" + key)
        if (expected_camera_ingress_policy == 'drop_before_ros'
                and key.startswith('Camera-')
                and stream.get('dropped_before_ros') != stream.get('samples')):
            failures.append('bridge_camera_ingress_not_fully_dropped:' + key)
        if (expected_lidar_ingress_policy == 'drop_before_ros'
                and key == '/lidar3D'
                and stream.get('dropped_before_ros') != stream.get('samples')):
            failures.append('bridge_lidar_ingress_not_fully_dropped')
    return failures
