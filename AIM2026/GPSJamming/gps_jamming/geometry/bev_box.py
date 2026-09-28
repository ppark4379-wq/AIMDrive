"""Minimum-area BEV rectangle of the observed cluster, in its source frame."""
import cv2
import numpy as np


def fit_bev_box(points):
    xyz = np.asarray(points, dtype=float)[:, :3]
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        return None
    # Subtract an origin before float32 conversion for numerical precision.
    origin = xyz[:, :2].mean(axis=0)
    rect = cv2.minAreaRect(np.asarray(xyz[:, :2]-origin, dtype=np.float32))
    corners = cv2.boxPoints(rect).astype(float)+origin
    edges = np.roll(corners, -1, axis=0)-corners
    lengths = np.linalg.norm(edges, axis=1)
    axis = edges[int(np.argmax(lengths))]
    length, width = float(lengths.max()), float(lengths.min())
    yaw = float((np.arctan2(axis[1], axis[0])+np.pi/2) % np.pi-np.pi/2)
    center = np.asarray(rect[0], dtype=float)+origin
    return {'center_xy_m': center.tolist(),
            'center_xyz_m': [*center.tolist(), float((xyz[:, 2].min()+xyz[:, 2].max())/2)],
            'length_m': length, 'width_m': width,
            'yaw_rad': yaw, 'corners_xy_m': corners.tolist(),
            'degenerate': width < 1e-3, 'method': 'cluster_bev_minimum_area_rectangle',
            'note': 'Observed cluster extent; yaw is an unoriented axis modulo pi, not heading.'}
