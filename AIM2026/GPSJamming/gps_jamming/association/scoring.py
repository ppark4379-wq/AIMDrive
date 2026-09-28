"""Projected point and bounding box overlap scoring."""
import numpy as np


def bbox_overlap(pixels, bbox):
    """Return (inside count, inside / visible count); xyxy bounds are inclusive."""
    if len(pixels) == 0:
        return 0, 0.0
    count = int(np.count_nonzero(np.all((pixels >= bbox[:2]) & (pixels <= bbox[2:]), axis=1)))
    return count, count / len(pixels)
