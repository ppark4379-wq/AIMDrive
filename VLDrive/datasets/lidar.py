"""LiDAR preprocessing shared by both datasets."""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def lidar_to_raw_features(lidar, max_points=40000, rotation_deg=-90, clip_abs=None, rng=None):
    """Remove ego points, shuffle, zero-pad to max_points and rotate into the model frame.

    Returns (points [max_points, 4] float32, number of valid points).  The
    pre-training dataset of LMDrive additionally zeroed coordinates beyond
    +-100 m (clip_abs=100); the instruction dataset did not.

    Pass rng (a numpy RandomState) to make the point shuffle reproducible; the
    MORAI validation split does this so evaluation is repeatable.
    """
    ego_box = (lidar[:, 0] > -1.2) & (lidar[:, 0] < 1.2) & (lidar[:, 1] > -1.2) & (lidar[:, 1] < 1.2)
    lidar_xyzr = np.delete(lidar, np.argwhere(ego_box), axis=0)

    order = np.arange(len(lidar_xyzr))
    (rng if rng is not None else np.random).shuffle(order)
    lidar_xyzr = lidar_xyzr[order]

    points = np.zeros((max_points, 4), dtype=np.float32)
    num_points = min(max_points, len(lidar_xyzr))
    points[:num_points, :4] = lidar_xyzr[:num_points]
    points[np.isinf(points)] = 0
    points[np.isnan(points)] = 0
    if clip_abs is not None:
        points[points > clip_abs] = 0
        points[points < -clip_abs] = 0
    return rotate_lidar(points, rotation_deg).astype(np.float32), num_points


def replace_non_finite(data, info):
    """Zero NaN/inf entries in-place, logging where they came from."""
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if np.isnan(value).any():
                logger.warning("NaN in %s (%s)", key, info)
                value[np.isnan(value)] = 0
            if np.isinf(value).any():
                logger.warning("inf in %s (%s)", key, info)
                value[np.isinf(value)] = 0
        elif isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                logger.warning("NaN in %s (%s)", key, info)
                value[torch.isnan(value)] = 0
            if torch.isinf(value).any():
                logger.warning("inf in %s (%s)", key, info)
                value[torch.isinf(value)] = 0
    return data
