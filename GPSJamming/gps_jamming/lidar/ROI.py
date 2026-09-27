"""Deterministic ROI filtering and voxel centroids; intensity is not clipped."""
import numpy as np
from time import perf_counter


def ROI(points, roi_min, roi_max, voxel_size=0.15, timings=None):
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
