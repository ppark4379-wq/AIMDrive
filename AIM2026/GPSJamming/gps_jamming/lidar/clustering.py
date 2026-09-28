"""Euclidean connected components using a KD-tree (not DBSCAN)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Avoid materializing an unbounded O(N^2) graph for very dense clouds.
_MAX_GRAPH_EDGES = 1_000_000


def cluster_indices(points, tolerance=0.6, min_points=8, max_points=20000):
    if tolerance <= 0 or min_points < 1 or max_points < min_points:
        raise ValueError("Invalid clustering thresholds")
    if not len(points):
        return []
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    tree = cKDTree(xyz)
    # Self-query counts each non-self pair twice, including duplicate coordinates.
    edge_count = (int(tree.count_neighbors(tree, tolerance)) - len(xyz)) // 2
    if edge_count > _MAX_GRAPH_EDGES:
        return _cluster_bfs(tree, xyz, tolerance, min_points, max_points)
    pairs = tree.query_pairs(tolerance, eps=0, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8),
                        (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz))).tocsr()
    # Each unordered pair is stored once; directed=False treats it as an undirected edge.
    _, labels = connected_components(graph, directed=False, return_labels=True)
    counts = np.bincount(labels)
    order = np.argsort(labels, kind="stable")
    ends = np.cumsum(counts)
    starts = ends - counts
    result = [order[start:end] for start, end, count in zip(starts, ends, counts)
              if min_points <= count <= max_points]
    # Preserve both the original within-cluster order and frame-local cluster IDs.
    result.sort(key=lambda indices: int(indices[0]))
    return result


def _cluster_bfs(tree, xyz, tolerance, min_points, max_points):
    """Bound memory for dense neighborhoods using the original per-point traversal."""
    seen = np.zeros(len(xyz), dtype=bool)
    result = []
    for start in range(len(xyz)):
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
