"""Sensor coordinate transformations."""
import numpy as np


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
