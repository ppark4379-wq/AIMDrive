"""Per-track BEV OBB centers in the unchanged point coordinate frame."""
import numpy as np
from ..geometry.bev_box import fit_bev_box


def estimate_positions(tracks, result):
    """Choose by bbox-supported depth; fit the full selected cluster in BEV.

    Observed extents do not recover the unseen physical body of an object.
    """
    if result is None:
        return {}
    boxes = {t.track_id: t.bbox for t in tracks}
    projections = {p.cluster_id: p for p in result.projections}
    candidates = {}
    for match in result.matches:
        projection = projections.get(match.cluster_id)
        bbox = boxes.get(match.track_id)
        if projection is None or bbox is None:
            continue
        inside = np.all((projection.pixels >= bbox[:2]) & (projection.pixels <= bbox[2:]), axis=1)
        indices = projection.point_indices[inside]
        if len(indices) < 3:
            continue
        xyz = np.asarray(match.cluster_points[indices, :3], dtype=float)
        if not np.isfinite(xyz).all():
            continue
        box = fit_bev_box(match.cluster_points)
        if box is None:
            continue
        position = np.asarray(box['center_xyz_m'])
        depth = float(np.median(projection.depths[inside]))
        if not np.isfinite(depth) or depth <= 0:
            continue
        candidates.setdefault(match.track_id, []).append({
            'xyz_m': position.tolist(),
            'coordinate_frame': match.coordinate_frame,
            'planar_distance_m': float(np.linalg.norm(position[:2])),
            'distance_3d_m': float(np.linalg.norm(position)),
            'cluster_id': match.cluster_id,
            'support_points': len(indices),
            'median_camera_depth_m': depth,
            'support_extent_xyz_m': np.ptp(xyz, axis=0).tolist(),
            'overlap_ratio': match.overlap_ratio,
            'method': 'cluster_bev_obb_center',
            'bev_obb': box,
            'cluster_centroid_xyz_m': match.centroid.tolist(),
            'lidar_frame_id': result.frame_id,
            'timestamp_ns': result.timestamp,
        })
    positions = {}
    for track_id, rows in candidates.items():
        selected = min(rows, key=lambda r: (r['median_camera_depth_m'], -r['support_points'], r['cluster_id']))
        positions[track_id] = {**selected, 'candidate_count': len(rows),
                               'selection': 'nearest_median_camera_depth',
                               'ambiguous': len(rows) > 1,
                               'candidate_cluster_ids': [r['cluster_id'] for r in rows]}
    return positions
