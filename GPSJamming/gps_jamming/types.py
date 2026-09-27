"""LiDAR contracts. Times are integer nanoseconds unless explicitly raw."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LidarCluster:
    cluster_id: int  # Local to one frame; never an object tracking ID.
    points: np.ndarray
    centroid: np.ndarray
    dimensions: np.ndarray  # Axis-aligned XYZ extents, metres.
    point_count: int


@dataclass
class LidarFrame:
    frame_id: int
    raw_timestamp: int
    timestamp: int | None  # Absolute Unix ns; None until an anchor is verified.
    receive_monotonic_time: int  # First packet receipt, monotonic ns.
    coordinate_frame: str
    complete: bool
    points: np.ndarray  # Nx4 XYZ metres + original intensity.
    receive_end_monotonic_time: int
    clusters: list[LidarCluster] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    packet_count: int = 0
    duplicate_count: int = 0
    clock_epoch: int = 0
    timestamp_status: str = "unmapped"
    diagnostics: dict = field(default_factory=dict)
