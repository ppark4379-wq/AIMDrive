"""Build results preserving full cluster points and their coordinate frame."""
from ..types import AssociatedCluster, AssociationResult


def build_result(frame, tracks, projections, assignments):
    matches = []
    for i, j, count, ratio in assignments:
        cluster, track = frame.clusters[i], tracks[j]
        matches.append(AssociatedCluster(
            track.track_id, track.class_id, track.confidence, cluster.cluster_id,
            cluster.points, cluster.centroid, frame.coordinate_frame,
            count, len(projections[i].pixels), ratio))
    matched_clusters = {i for i, _, _, _ in assignments}
    matched_tracks = {j for _, j, _, _ in assignments}
    return AssociationResult(
        frame.frame_id, frame.timestamp, matches, projections,
        [c.cluster_id for i, c in enumerate(frame.clusters) if i not in matched_clusters],
        [t.track_id for j, t in enumerate(tracks) if j not in matched_tracks])
