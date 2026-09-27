"""Height- and tilt-constrained RANSAC road plane in a Z-up LiDAR frame."""
import numpy as np


def remove_ground(points, expected_z=-1.48, seed_band=0.45,
                  distance_threshold=0.12, max_tilt_deg=12.0,
                  iterations=80, min_inliers=40, seed=0):
    if (seed_band <= 0 or distance_threshold <= 0 or iterations < 1
            or min_inliers < 3 or not 0 <= max_tilt_deg < 90):
        raise ValueError("Invalid ground-removal parameters")
    xyz = points[:, :3].astype(np.float64)
    candidates = xyz[np.abs(xyz[:, 2] - expected_z) <= seed_band]
    if len(candidates) < min_inliers:
        return points.copy(), {"ground_points": 0, "ground_status": "insufficient_seeds"}
    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    cos_tilt = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(iterations):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < cos_tilt:
            continue
        offset = -normal @ a
        if abs(-offset / normal[2] - expected_z) > seed_band:
            continue
        count = np.count_nonzero(np.abs(candidates @ normal + offset) <= distance_threshold)
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None or best_count < min_inliers:
        return points.copy(), {"ground_points": 0, "ground_status": "no_supported_plane"}
    normal, offset = best
    # Refit the supported plane rather than keeping a noisy three-point sample.
    support = candidates[np.abs(candidates @ normal + offset) <= distance_threshold]
    center = support.mean(axis=0)
    _, _, basis = np.linalg.svd(support - center, full_matrices=False)
    refined = basis[-1]
    if refined[2] < 0:
        refined *= -1
    refined_offset = -refined @ center
    if (refined[2] >= cos_tilt and
            abs(-refined_offset / refined[2] - expected_z) <= seed_band):
        normal, offset = refined, refined_offset
    ground = np.abs(xyz @ normal + offset) <= distance_threshold
    return points[~ground].copy(), {"ground_points": int(ground.sum()),
        "ground_status": "plane_found", "ground_plane": [*normal.tolist(), float(offset)]}
