"""Exact connectivity, ordering and memory-guard regression tests."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.lidar.clustering import cluster_indices


def reference(points, tolerance, minimum, maximum):
    # Independent small-cloud oracle using direct distances rather than a KD-tree.
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    adjacency = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2) <= tolerance
    seen = set()
    result = []
    for start in range(len(xyz)):
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        for index in component:
            for neighbor in np.flatnonzero(adjacency[index]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.append(int(neighbor))
        if minimum <= len(component) <= maximum:
            result.append(sorted(component))
    return result


class ClusteringTests(unittest.TestCase):
    def test_matches_distance_oracle_and_dense_fallback(self):
        rng = np.random.default_rng(5)
        points = np.column_stack((rng.uniform(-2, 2, (150, 3)), rng.uniform(0, 255, 150)))
        for tolerance, minimum, maximum in [(0.3, 1, 200), (0.8, 2, 20), (10, 1, 149)]:
            expected = reference(points, tolerance, minimum, maximum)
            for limit in [1_000_000, 0]:
                with patch('gps_jamming.lidar.clustering._MAX_GRAPH_EDGES', limit):
                    actual = cluster_indices(points, tolerance, minimum, maximum)
                self.assertEqual([c.tolist() for c in actual], expected)

    def test_boundary_duplicates_chains_and_order(self):
        points = np.array([[10, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 5],
                           [2, 0, 0, 9], [10, 0, 0, 2], [30, 0, 0, 0]], dtype=float)
        actual = cluster_indices(points, tolerance=1, min_points=1, max_points=3)
        self.assertEqual([c.tolist() for c in actual], [[0, 4], [1, 2, 3], [5]])
        self.assertEqual(cluster_indices(np.empty((0, 4))), [])


if __name__ == '__main__':
    unittest.main()
