"""Fast, opt-in identification of MORAI camera publish envelopes.

MORAI sends JSON in binary WebSocket frames and may place the envelope/header
on either side of the large image payload.  An explicitly configured bridge
can inspect only those boundaries and drop duplicate ROS camera publications
before rosbridge JSON deserialization.
Native Sensor Record remains MORAI-owned and is verified independently.
"""
import re


_OP = re.compile(br'"op"\s*:\s*"publish"')
_TOPIC = re.compile(br'"topic"\s*:\s*"/image_jpeg/compressed"')
_LIDAR_TOPIC = re.compile(br'"topic"\s*:\s*"/lidar3D"')
_FRAME = re.compile(br'"frame_id"\s*:\s*"(Camera-[1-4])"')
_PREFIX_BYTES = 16 * 1024
UNKNOWN_CAMERA_FRAME = "camera-topic"


def camera_publish_frame(message, *, bson_only=False):
    """Identify a camera publish without scanning the base64 image body.

    MORAI does not use a stable JSON key order.  Some builds put ``header``
    before ``data`` and others put it after the large base64 payload.  The
    operation and topic must be present near the start; the optional frame id
    may be present near either end.  A camera-topic sentinel still permits the
    explicitly configured ingress drop when the frame id is unavailable.
    """
    if bson_only or not isinstance(message, (bytes, bytearray, memoryview, str)):
        return None
    if isinstance(message, str):
        prefix = message[:_PREFIX_BYTES].encode("utf-8", "strict")
        suffix = message[-_PREFIX_BYTES:].encode("utf-8", "strict")
    else:
        prefix = bytes(message[:_PREFIX_BYTES])
        suffix = bytes(message[-_PREFIX_BYTES:])
    if not prefix.lstrip().startswith(b"{"):
        return None
    boundary = prefix if suffix == prefix else prefix + suffix
    if not _OP.search(boundary) or not _TOPIC.search(boundary):
        return None
    match = _FRAME.search(boundary)
    return (match.group(1).decode("ascii") if match
            else UNKNOWN_CAMERA_FRAME)


def dropped_publish_stream(message, *, drop_camera=False, drop_lidar=False,
                           bson_only=False):
    """Return ``(topic, metrics stream)`` for an explicitly dropped publish.

    This inspects only the JSON envelope boundaries.  In particular it never
    scans or deserializes the large camera/base64 or LiDAR array body.  Native
    Sensor Record is a separate MORAI output and is unaffected by this ROS
    transport policy.
    """
    if (not drop_camera and not drop_lidar) or bson_only or not isinstance(
            message, (bytes, bytearray, memoryview, str)):
        return None
    if isinstance(message, str):
        prefix = message[:_PREFIX_BYTES].encode("utf-8", "strict")
        suffix = message[-_PREFIX_BYTES:].encode("utf-8", "strict")
    else:
        prefix = bytes(message[:_PREFIX_BYTES])
        suffix = bytes(message[-_PREFIX_BYTES:])
    if not prefix.lstrip().startswith(b"{"):
        return None
    boundary = prefix if suffix == prefix else prefix + suffix
    if not _OP.search(boundary):
        return None
    if drop_camera and _TOPIC.search(boundary):
        match = _FRAME.search(boundary)
        return ("/image_jpeg/compressed",
                match.group(1).decode("ascii") if match
                else UNKNOWN_CAMERA_FRAME)
    if drop_lidar and _LIDAR_TOPIC.search(boundary):
        return "/lidar3D", "/lidar3D"
    return None
