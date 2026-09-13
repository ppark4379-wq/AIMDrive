"""
Pure-Python helpers for the MORAI ROS training recorder.

No rospy/morai_msgs imports are allowed here. This keeps the canonical
dataset contract testable outside the ROS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional
import math

import numpy as np


CURRENT_CAMERA_FRAME_MAP = {
    "Camera-1": "front",
    "Camera-2": "left",
    "Camera-3": "right",
    "Camera-4": "rear",
}

MANDATORY_CAMERA_ROLES = (
    "front",
    "left",
    "right",
)


@dataclass(frozen=True)
class BufferedSample:
    source_stamp_ns: Optional[int]
    receive_monotonic_ns: int
    receive_wall_ns: int
    frame_id: str
    payload: Any
    seq: Optional[int] = None
    callerid: Optional[str] = None


def latest_not_after_receive(
    samples: Iterable[BufferedSample],
    anchor_receive_monotonic_ns: int,
) -> Optional[BufferedSample]:
    """
    Select the latest sample whose PC monotonic receive time is <= anchor.

    This is the shared causal policy intended for:
      ROS training acquisition and UDP competition inference.
    """
    best = None

    for sample in samples:
        t = sample.receive_monotonic_ns

        if t > anchor_receive_monotonic_ns:
            continue

        if (
            best is None
            or t > best.receive_monotonic_ns
        ):
            best = sample

    return best


def decode_current_lidar_xyzi(
    data: bytes,
    *,
    width: int,
    height: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
    fields,
) -> np.ndarray:
    """
    Decode the currently observed MORAI PointCloud2 layout:

      x          FLOAT32 offset 0
      y          FLOAT32 offset 4
      z          FLOAT32 offset 8
      intensity  FLOAT32 offset 12
      point_step 16

    The function intentionally validates the layout instead of silently
    accepting a different future packet contract.
    """
    expected = [
        ("x", 0, 7, 1),
        ("y", 4, 7, 1),
        ("z", 8, 7, 1),
        ("intensity", 12, 7, 1),
    ]

    observed = [
        (
            str(field["name"]),
            int(field["offset"]),
            int(field["datatype"]),
            int(field["count"]),
        )
        for field in fields
    ]

    if observed != expected:
        raise ValueError(
            "Unexpected PointCloud2 field contract: "
            f"{observed!r}"
        )

    if point_step != 16:
        raise ValueError(
            f"Expected point_step=16, got {point_step}"
        )

    if height < 1 or width < 0:
        raise ValueError(
            "Invalid PointCloud2 width/height"
        )

    packed_row_step = width * point_step

    if row_step != packed_row_step:
        raise ValueError(
            "Row padding is not supported by the current "
            f"contract: row_step={row_step}, "
            f"expected={packed_row_step}"
        )

    expected_bytes = row_step * height

    if len(data) != expected_bytes:
        raise ValueError(
            f"PointCloud2 data bytes={len(data)}, "
            f"expected={expected_bytes}"
        )

    dtype = np.dtype(
        ">f4" if is_bigendian else "<f4"
    )

    array = np.frombuffer(
        data,
        dtype=dtype,
    ).reshape(
        height * width,
        4,
    )

    # Canonical dataset is always native float32 and owns its memory.
    return np.asarray(
        array,
        dtype=np.float32,
    ).copy()


def quaternion_norm(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    return math.sqrt(
        x * x
        + y * y
        + z * z
        + w * w
    )


def quaternion_is_valid(
    x: float,
    y: float,
    z: float,
    w: float,
    tolerance: float = 1e-3,
) -> bool:
    norm = quaternion_norm(
        x,
        y,
        z,
        w,
    )
    return (
        math.isfinite(norm)
        and abs(norm - 1.0) <= tolerance
    )
