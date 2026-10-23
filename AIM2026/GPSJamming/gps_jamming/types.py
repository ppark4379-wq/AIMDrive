"""LiDAR contracts. Times are integer nanoseconds unless explicitly raw."""
from __future__ import annotations
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


@dataclass
class CameraTrack:
    track_id: int
    bbox: np.ndarray  # xyxy pixels at calibration image resolution.
    class_id: int
    confidence: float

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=float).copy()
        if (self.bbox.shape != (4,) or not np.isfinite(self.bbox).all()
                or np.any(self.bbox[2:] <= self.bbox[:2])):
            raise ValueError("bbox must be finite xyxy with positive width and height")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class ProjectedCluster:
    cluster_id: int
    pixels: np.ndarray  # Visible Nx2 image points.
    point_indices: np.ndarray  # Indices into original cluster.points.
    depths: np.ndarray  # Camera optical Z, metres.


@dataclass
class AssociatedCluster:
    track_id: int
    class_id: int
    confidence: float
    cluster_id: int
    cluster_points: np.ndarray
    centroid: np.ndarray
    coordinate_frame: str
    inside_points: int
    visible_points: int
    overlap_ratio: float


@dataclass
class AssociationResult:
    frame_id: int
    timestamp: int | None
    matches: list[AssociatedCluster]
    projections: list[ProjectedCluster]
    unmatched_cluster_ids: list[int]
    unmatched_track_ids: list[int]
