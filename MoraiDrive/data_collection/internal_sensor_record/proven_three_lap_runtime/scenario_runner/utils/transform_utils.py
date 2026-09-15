import math
import os
import sys
from typing import List, Sequence, Tuple


def add_grpc_paths(grpc_src: str):
    api_path = os.path.join(grpc_src, "api")
    proto_path = os.path.join(grpc_src, "proto")

    for path in [grpc_src, api_path, proto_path]:
        if path not in sys.path:
            sys.path.append(path)


def distance_2d(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def polyline_length(points: List[Sequence[float]]) -> float:
    total = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        total += distance_2d(p0, p1)
    return total


def nearest_point_index(points: List[Sequence[float]], x: float, y: float) -> int:
    if len(points) == 0:
        raise ValueError("Empty points")

    best_idx = 0
    best_dist = float("inf")
    target = (x, y)
    for i, point in enumerate(points):
        d = distance_2d(point, target)
        if d < best_dist:
            best_idx = i
            best_dist = d
    return best_idx


def project_distance_on_polyline(points: List[Sequence[float]], x: float, y: float) -> float:
    if len(points) < 2:
        raise ValueError("Polyline must have at least 2 points")

    best_s = 0.0
    best_dist = float("inf")
    cumulative = 0.0

    for p0, p1 in zip(points[:-1], points[1:]):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-9:
            continue

        t = ((x - p0[0]) * dx + (y - p0[1]) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = p0[0] + t * dx
        proj_y = p0[1] + t * dy
        d = distance_2d((x, y), (proj_x, proj_y))
        seg_len = math.sqrt(seg_len_sq)

        if d < best_dist:
            best_dist = d
            best_s = cumulative + t * seg_len

        cumulative += seg_len

    return best_s


def interpolate_on_polyline(
    points: List[Sequence[float]],
    offset_m: float,
) -> Tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError("Link must have at least 2 points")

    remain = max(0.0, float(offset_m))

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg_len = distance_2d(p0, p1)

        if seg_len < 1e-6:
            continue

        if remain <= seg_len:
            ratio = remain / seg_len
            x = p0[0] + ratio * (p1[0] - p0[0])
            y = p0[1] + ratio * (p1[1] - p0[1])
            z = p0[2] + ratio * (p1[2] - p0[2]) if len(p0) >= 3 and len(p1) >= 3 else 0.0

            yaw_rad = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            yaw_deg = math.degrees(yaw_rad)
            return x, y, z, yaw_deg

        remain -= seg_len

    p0 = points[-2]
    p1 = points[-1]
    yaw_rad = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    yaw_deg = math.degrees(yaw_rad)
    z = p1[2] if len(p1) >= 3 else 0.0
    return p1[0], p1[1], z, yaw_deg


def make_transform(x: float, y: float, z: float, yaw_deg: float, grpc_src: str):
    add_grpc_paths(grpc_src)
    from proto.morai.common.type_pb2 import Transform

    tf = Transform()
    tf.location.x = float(x)
    tf.location.y = float(y)
    tf.location.z = float(z)

    tf.rotation.x = 0.0
    tf.rotation.y = 0.0
    tf.rotation.z = float(yaw_deg)
    return tf


def get_polyline_end_point(points):
    """
    link points의 마지막 점 반환.
    return: x, y, z
    """
    if len(points) == 0:
        raise ValueError("Empty points")
    p = points[-1]
    z = p[2] if len(p) >= 3 else 0.0
    return float(p[0]), float(p[1]), float(z)


def dist_xy(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)
