"""LiDAR foreground endpoint selection used by offline camera checks."""
import numpy as np


def foreground_endpoints(points):
    scan = points.reshape(-1, 16, 4)
    ranges = np.linalg.norm(scan[:, :, :3], axis=-1)
    near = (ranges > 2.0) & (ranges < 55.0) & (scan[:, :, 2] > -0.8)
    selected = np.zeros(ranges.shape, dtype=bool)
    for shift in (-1, 1):
        other = np.roll(ranges, shift, axis=0)
        selected |= near & ((other < 0.2) | (other - ranges > 0.8))
    return scan[selected, :3]
