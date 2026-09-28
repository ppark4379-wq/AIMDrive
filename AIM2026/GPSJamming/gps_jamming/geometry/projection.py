"""LiDAR point projection into camera images."""
import numpy as np
from ..types import ProjectedCluster


def project_cluster(cluster, calibration, *, min_depth=0.1):
    """Keep finite points in front of the camera and inside the calibrated image."""
    if not np.isfinite(min_depth) or min_depth <= 0:
        raise ValueError("min_depth must be positive")
    points = np.asarray(cluster.points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Expected Nx3 XYZ or Nx4 XYZI cluster points")
    indices = np.flatnonzero(np.isfinite(points[:, :3]).all(axis=1))
    t = calibration.extrinsic
    camera = points[indices, :3] @ t[:3, :3].T + t[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > min_depth)
    camera, indices = camera[valid], indices[valid]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, y = (camera[:, :2] / camera[:, 2:3]).T
        k1, k2, p1, p2, k3 = calibration.distortion
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2*k3))
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        pixels = (np.column_stack((xd, yd, np.ones(len(x)))) @ calibration.intrinsic.T)[:, :2]
    width, height = calibration.image_size
    visible = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0)
               & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    return ProjectedCluster(cluster.cluster_id, pixels[visible], indices[visible], camera[visible, 2])
