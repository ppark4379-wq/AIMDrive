"""
Saves each frame's point cloud as lidar/%04d.npy, shape (N, 4) float32
[x, y, z, intensity].

Note: LMDrive's original CARLA pipeline splits LiDAR into lidar/ +
lidar_odd/ (two staggered half-scans it concatenates for 360-degree
coverage, since CARLA's semi-scan design needs that). If MORAI's LiDAR
already outputs a full 360-degree scan per tick, don't replicate that --
just write one lidar/%04d.npy per frame, and make sure the custom
training Dataset you write later reads only `lidar/`, not `lidar_odd/`.
"""
import os
import numpy as np
from .atomic_io import save_numpy


class LidarWriter:
    def __init__(self, route_dir: str, subdir="lidar"):
        self.dir = os.path.join(route_dir, subdir)
        os.makedirs(self.dir, exist_ok=True)

    def write(self, frame_id: int, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 4, points.shape
        path = os.path.join(self.dir, f"{frame_id:04d}.npy")
        save_numpy(path, points.astype(np.float32))
        return path
