"""
Byte-level MORAI packet parsers.

parse_ego_info / parse_object_info are UNVERIFIED PLACEHOLDERS. MORAI's
UDP packet layout differs across SDK/competition-kit versions, and
guessing the wrong struct format silently produces garbage numbers
instead of an error -- so these raise NotImplementedError until you fill
them in from your actual MORAI packet spec (ctypes Structure definition
or protocol doc).

parse_camera_jpeg / parse_lidar_points assume a common convention (JPEG
bytes / raw float32 xyzI array) -- confirm against MORAI's actual output
format before trusting them either.
"""
from typing import Dict, Any

import numpy as np
import cv2


def parse_ego_info(raw: bytes) -> Dict[str, Any]:
    """
    TODO: replace with MORAI's real Ego/EgoVehicleStatus packet layout.
    Must return at least: x, y, theta (rad), speed (m/s),
    optionally x_command/y_command if MORAI also broadcasts route target.
    """
    raise NotImplementedError(
        "parse_ego_info: MORAI UDP ego-status 패킷 spec을 채워넣어야 합니다"
    )


def parse_object_info(raw: bytes) -> Dict[str, Any]:
    """
    TODO: replace with MORAI's real Object List packet layout.
    Should return {actor_id: {"loc": [...], "ori": [...], "box": [...],
    "vel": [...], "tpe": int}, ...} -- one entry per surrounding actor.
    """
    raise NotImplementedError(
        "parse_object_info: MORAI UDP 주변 객체 리스트 패킷 spec을 채워넣어야 합니다"
    )


def parse_camera_jpeg(raw: bytes) -> np.ndarray:
    """Assumes `raw` is a JPEG-encoded frame. Confirm MORAI's actual encoding."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        raise ValueError("camera payload could not be decoded as JPEG")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_lidar_points(raw: bytes) -> np.ndarray:
    """Assumes `raw` is a flat float32 array of [x, y, z, intensity, ...]."""
    points = np.frombuffer(raw, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(
            f"lidar payload size {points.size} is not a multiple of 4 "
            "(x,y,z,intensity) -- check MORAI's actual point field count"
        )
    return points.reshape(-1, 4).copy()
