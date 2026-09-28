"""LiDAR cluster to camera track association."""
import numpy as np
from .scoring import bbox_overlap


def match_clusters(projections, tracks, *, min_inside_points=3, min_overlap_ratio=0.5):
    """Independent cluster choices; a track may receive multiple clusters.

    Ties prefer higher camera confidence, then smaller bbox, then lower track ID.
    Returns (projection index, track index, inside count, overlap ratio).
    """
    if not isinstance(min_inside_points, int) or min_inside_points < 1:
        raise ValueError("min_inside_points must be a positive integer")
    if not np.isfinite(min_overlap_ratio) or not 0 <= min_overlap_ratio <= 1:
        raise ValueError("min_overlap_ratio must be in [0, 1]")
    ids = [track.track_id for track in tracks]
    if len(set(ids)) != len(ids):
        raise ValueError("Camera track IDs must be unique within a frame")
    matches = []
    for i, projection in enumerate(projections):
        candidates = []
        for j, track in enumerate(tracks):
            count, ratio = bbox_overlap(projection.pixels, track.bbox)
            if count >= min_inside_points and ratio >= min_overlap_ratio:
                area = float(np.prod(track.bbox[2:] - track.bbox[:2]))
                candidates.append(((ratio, track.confidence, -area, -track.track_id), j, count))
        if candidates:
            key, j, count = max(candidates)
            matches.append((i, j, count, key[0]))
    return matches
