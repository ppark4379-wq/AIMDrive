"""Camera intrinsics, distortion, and sensor extrinsics."""
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraCalibration:
    intrinsic: np.ndarray
    extrinsic: np.ndarray  # p_camera_optical = R @ p_source + t (metres).
    image_size: tuple[int, int]  # width, height
    source_frame: str
    distortion: np.ndarray | None = None  # k1,k2,p1,p2,k3.

    def __post_init__(self):
        self.intrinsic = np.asarray(self.intrinsic, dtype=float).copy()
        self.extrinsic = np.asarray(self.extrinsic, dtype=float).copy()
        k, t = self.intrinsic, self.extrinsic
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.isclose(k[1, 0], 0)):
            raise ValueError("intrinsic must be a finite pinhole 3x3 K with positive focal lengths")
        if (t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6)):
            raise ValueError("extrinsic must be a rigid source-to-camera-optical 4x4 transform")
        if (len(self.image_size) != 2 or any(
                not isinstance(v, (int, np.integer)) or v <= 0 for v in self.image_size)):
            raise ValueError("image_size must be [width, height] positive integers")
        if not isinstance(self.source_frame, str) or not self.source_frame.strip():
            raise ValueError("source_frame must name the input point coordinate frame")
        self.distortion = np.zeros(5) if self.distortion is None else np.asarray(self.distortion, dtype=float).copy()
        if self.distortion.shape != (5,) or not np.isfinite(self.distortion).all():
            raise ValueError("distortion must be [k1, k2, p1, p2, k3], or null for rectified images")

    @classmethod
    def from_config(cls, config):
        for key in ("intrinsic", "extrinsic", "image_size", "source_frame"):
            if config.get(key) is None:
                raise ValueError(f"Missing verified calibration: {key}")
        return cls(**{key: config[key] for key in
                      ("intrinsic", "extrinsic", "image_size", "source_frame")},
                   distortion=config.get("distortion"))
