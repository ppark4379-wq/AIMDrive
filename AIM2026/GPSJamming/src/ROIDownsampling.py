"""Sensor-to-vehicle transform, ROI and deterministic voxel centroids."""
import numpy as np
from time import perf_counter

def transform_xyzi(points, translation_m, rotation_rpy_deg):
    """Map sensor to parent: Rz(yaw) @ Ry(pitch) @ Rx(roll), then translation."""
    translation = np.asarray(translation_m, dtype=np.float64)
    angles = np.asarray(rotation_rpy_deg, dtype=np.float64)
    if (translation.shape != (3,) or angles.shape != (3,)
            or not np.isfinite(translation).all() or not np.isfinite(angles).all()):
        raise ValueError("Expected finite XYZ translation and roll/pitch/yaw triples")
    roll, pitch, yaw = np.deg2rad(angles)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    result = np.array(points, dtype=np.float32, copy=True)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("Expected Nx4 XYZI points")
    result[:, :3] = result[:, :3] @ (rz @ ry @ rx).T + translation
    return result

def ROI(points, roi_min, roi_max, voxel_size=0.15, timings=None, intermediates=None):
    """Optionally expose the already filtered, pre-voxel points for visualization."""
    started = perf_counter() if timings is not None else 0
    points = np.asarray(points, dtype=np.float32)
    lo, hi = np.asarray(roi_min), np.asarray(roi_max)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("Expected Nx4 XYZI points")
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
        raise ValueError("ROI must ontain three ordered finite bounds")
    if not np.isfinite([*lo, *hi, voxel_size]).all() or voxel_size < 0:
        raise ValueError("Invalid ROI or voxel size")
    mask = np.isfinite(points).all(axis=1)
    mask &= ((points[:, :3] >= lo) & (points[:, :3] <= hi)).all(axis=1)
    points = points[mask]
    if intermediates is not None:
        intermediates['roi'] = points
    if timings is not None:
        timings['roi'] = (perf_counter() - started) * 1000
        started = perf_counter()
    if not len(points) or voxel_size == 0:
        return points.copy()
    keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    result = np.zeros((len(counts), 4), dtype=np.float64)
    np.add.at(result, inverse, points)
    result = (result / counts[:, None]).astype(np.float32)
    if timings is not None:
        timings['voxel'] = (perf_counter() - started) * 1000
    return result
