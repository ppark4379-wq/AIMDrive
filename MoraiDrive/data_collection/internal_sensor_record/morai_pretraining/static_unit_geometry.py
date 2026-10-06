"""Apply measured full 3D bounds for explicitly validated unit static assets.

The immutable API GT stays intact. These fields describe a derived training box.
Profiles use actor-local forward/left/up, while manifests use width/length/height.
Normalized timelines carry yaw only; tilted or unstable raw actor poses fail closed.
"""
import math
import numpy as np


def apply_static_unit_profile(item, entry, scale, pivot):
    def fail(reason):
        return None, reason
    p = entry.get('geometry_3d') or {}
    if item.get('kind') != 'obstacle':
        return fail('unit_profile_requires_static_obstacle')
    if scale is None or len(scale) != 3 or not np.allclose(scale, [1, 1, 1], rtol=0, atol=1e-6):
        return fail('unit_profile_requires_verified_unit_scale')
    if entry.get('approved') is not True or p.get('approved') is not True:
        return fail('static_3d_profile_not_approved')
    if p.get('validation_scope') != 'static_unit_visible_surface_multiview_3d':
        return fail('static_3d_validation_scope_missing')
    if not p.get('evidence_sha256') or p.get('camera_review_pass') is not True or p.get('lidar_pass') is not True:
        return fail('static_3d_validation_evidence_missing')
    if not pivot or pivot.get('valid') is not True:
        return fail('static_3d_pivot_evidence_missing')
    try:
        ranges = np.asarray(pivot['offset_range_xyz_m'], float)
        if ranges.shape != (3,) or not np.isfinite(ranges).all() or max(ranges) > .01:
            return fail('static_3d_pivot_offset_unstable')
        tilt = float(pivot['max_actor_roll_pitch_deg'])
        if not math.isfinite(tilt) or tilt > .5:
            return fail('static_3d_normalized_timeline_requires_upright_actor')
        if float(pivot['max_box_pivot_yaw_difference_deg']) > .5:
            return fail('static_3d_box_and_actor_heading_differ')
        offset, observed = np.asarray(p['center_offset_flu_m'], float), np.asarray(pivot['offset_xyz_m'], float)
        expected = np.asarray(p['raw_offset_flu_m'], float)
        size, raw_size = np.asarray(p['size_lwh_m'], float), np.asarray(p['raw_size_lwh_m'], float)
        position = np.asarray(item['position'], float)
        actual_raw_size = np.asarray(item['size_wlh_m'], float)[[1, 0, 2]]
        vectors = (offset, observed, expected, size, raw_size, position, actual_raw_size)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors) or min(size) <= 0:
            return fail('static_3d_nonfinite_or_invalid_geometry')
        if max(abs(observed - expected)) > .05:
            return fail('static_3d_api_pivot_contract_changed')
        if np.any(abs(actual_raw_size - raw_size) > np.maximum(.03, raw_size * .02)):
            return fail('static_3d_api_size_contract_changed')
        delta = offset - observed
        heading = math.radians(float(item['heading_deg']))
        if not math.isfinite(heading):
            return fail('static_3d_invalid_heading')
        c, s = math.cos(heading), math.sin(heading)
        center = position + [c*delta[0]-s*delta[1], s*delta[0]+c*delta[1], delta[2]]
    except (KeyError, TypeError, ValueError, IndexError):
        return fail('static_3d_incomplete_pose_or_profile')
    wlh = size[[1, 0, 2]].tolist()
    return {'training_center_world_xyz': center.tolist(),
            'training_center_world_xy': center[:2].tolist(),
            'training_size_wlh_m': wlh, 'training_size_wl_m': wlh[:2],
            'geometry_sources': {'center': 'static_unit_surface_3d', 'size': 'static_unit_surface_3d'},
            'geometry_validation_scope': p['validation_scope']}, None
