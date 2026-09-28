"""Cluster features in the unchanged LiDAR coordinate frame."""
import numpy as np
from ..types import LidarCluster


def build_clusters(points, components):
    clusters = []
    for cluster_id, indices in enumerate(components):
        subset = points[indices].copy()
        xyz = subset[:, :3]
        clusters.append(LidarCluster(cluster_id, subset, xyz.mean(axis=0),
                                     np.ptp(xyz, axis=0), len(subset)))
    return clusters
