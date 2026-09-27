"""Euclidean connected components using a KD-tree (not DBSCAN)."""
import numpy as np
from scipy.spatial import cKDTree


def cluster_indices(points, tolerance=0.6, min_points=8, max_points=20000):
    if tolerance <= 0 or min_points < 1 or max_points < min_points:
        raise ValueError("Invalid clustering thresholds")
    if not len(points):
        return []
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    tree = cKDTree(xyz)
    seen = np.zeros(len(points), dtype=bool)
    result = []
    for start in range(len(points)):
        if seen[start]:
            continue
        seen[start] = True
        component = [start]
        cursor = 0
        while cursor < len(component):
            neighbors = np.asarray(tree.query_ball_point(xyz[component[cursor]], tolerance), dtype=int)
            fresh = neighbors[~seen[neighbors]]
            seen[fresh] = True
            component.extend(fresh.tolist())
            cursor += 1
        if min_points <= len(component) <= max_points:
            result.append(np.array(sorted(component), dtype=int))
    return result
