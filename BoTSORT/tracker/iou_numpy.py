"""Portable float64 IoU, matching cython_bbox's inclusive (+1) coordinates."""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    query_boxes = np.asarray(query_boxes, dtype=np.float64).reshape(-1, 4)
    overlaps = np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
    if not overlaps.size:
        return overlaps
    widths = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
    widths -= np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    widths += 1.0
    heights = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
    heights -= np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    heights += 1.0
    intersection = np.maximum(widths, 0) * np.maximum(heights, 0)
    area_a = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area_b = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    union = area_a[:, None] + area_b[None, :] - intersection
    np.divide(intersection, union, out=overlaps, where=(widths > 0) & (heights > 0))
    return overlaps
