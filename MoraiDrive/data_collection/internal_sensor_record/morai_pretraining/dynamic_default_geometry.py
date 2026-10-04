"""Apply reviewed native-scale dynamic surface envelopes using recorded full actor poses.

Held motion-pose geometry does not certify sensor exposure timing.
The immutable API GT stays intact. These fields describe a derived training box.
Profiles use actor-local forward/left/up, while manifests use width/length/height.
Full actor transforms preserve roll/pitch. Legacy yaw-only timelines are accepted
only for upright actors; a tilted legacy frame requires timeline regeneration.
"""
import math
import numpy as np


def rotation_matrix(rpy_degrees):
    r,p,y=np.radians(rpy_degrees)
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                     [-sp,cp*sr,cp*cr]])


def apply_dynamic_default_profile(item, entry, scale, pivot):
    def fail(reason):
        return None, reason
    p = entry.get('geometry_3d') or {}
    if item.get('kind') not in ('npc', 'pedestrian'):
        return fail('dynamic_profile_requires_npc_or_pedestrian')
    expected_kind = 'vehicle' if item['kind'] == 'npc' else 'pedestrian'
    if p.get('kind') != expected_kind:
        return fail('dynamic_3d_profile_kind_mismatch')
    if p.get('default_scale_contract') != 'morai_native_actor_spawn_no_scale':
        return fail('dynamic_3d_default_scale_contract_missing')
    if p.get('motion_pose_review_pass') is not True:
        return fail('dynamic_3d_motion_pose_evidence_missing')
    # Native vehicle/pedestrian spawn messages have no scale property.
    # An explicit non-default scale is still rejected.
    if scale is None:
        scale = [1, 1, 1]
    if scale is None or len(scale) != 3 or not np.allclose(scale, [1, 1, 1], rtol=0, atol=1e-6):
        return fail('unit_profile_requires_verified_unit_scale')
    if entry.get('approved') is not True or p.get('approved') is not True:
        return fail('dynamic_3d_profile_not_approved')
    if p.get('validation_scope') != 'dynamic_default_held_motion_surface_multiview_3d':
        return fail('dynamic_3d_validation_scope_missing')
    if not p.get('evidence_sha256') or p.get('camera_review_pass') is not True or p.get('lidar_pass') is not True:
        return fail('dynamic_3d_validation_evidence_missing')
    if not pivot or pivot.get('valid') is not True:
        return fail('dynamic_3d_pivot_evidence_missing')
    # World-speed estimates are noisy finite differences and do not identify
    # the animation selected by MORAI's native pedestrian velocity parameter.
    if item['kind'] == 'pedestrian':
        try:
            speeds = np.asarray(p['reviewed_motion_speeds_kmh'], float)
            observed_speeds = np.asarray(pivot['pedestrian_velocity_parameters_kmh'], float)
            if observed_speeds.ndim != 1 or observed_speeds.size == 0 or speeds.ndim != 1 or speeds.size == 0 or not np.isfinite(speeds).all() or not np.isfinite(observed_speeds).all():
                return fail('dynamic_3d_motion_speed_evidence_missing')
            if any(v > .1 and not np.any(abs(speeds-v) <= .1) for v in observed_speeds):
                return fail('dynamic_3d_pedestrian_speed_not_reviewed')
        except (KeyError, TypeError, ValueError):
            return fail('dynamic_3d_motion_speed_evidence_missing')
    try:
        ranges = np.asarray(pivot['offset_range_xyz_m'], float)
        if ranges.shape != (3,) or not np.isfinite(ranges).all() or max(ranges) > .01:
            return fail('dynamic_3d_pivot_offset_unstable')
        tilt = float(pivot['max_actor_roll_pitch_deg'])
        if not math.isfinite(tilt):
            return fail('dynamic_3d_invalid_actor_tilt')
        if float(pivot['max_box_pivot_yaw_difference_deg']) > .5:
            return fail('dynamic_3d_box_and_actor_heading_differ')
        offset, observed = np.asarray(p['center_offset_flu_m'], float), np.asarray(pivot['offset_xyz_m'], float)
        expected = np.asarray(p['raw_offset_flu_m'], float)
        size, raw_size = np.asarray(p['size_lwh_m'], float), np.asarray(p['raw_size_lwh_m'], float)
        position = np.asarray(item['position'], float)
        actual_raw_size = np.asarray(item['size_wlh_m'], float)[[1, 0, 2]]
        vectors = (offset, observed, expected, size, raw_size, position, actual_raw_size)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors) or min(size) <= 0:
            return fail('dynamic_3d_nonfinite_or_invalid_geometry')
        if max(abs(observed - expected)) > .05:
            return fail('dynamic_3d_api_pivot_contract_changed')
        if np.any(abs(actual_raw_size - raw_size) > np.maximum(.03, raw_size * .02)):
            return fail('dynamic_3d_api_size_contract_changed')
        pose_fields=('actor_pivot_world_xyz','actor_rotation_rpy_deg','rotation_rpy_deg')
        if any(k in item for k in pose_fields):
            frame_pivot,rpy,box_rpy=[np.asarray(item[k],float) for k in pose_fields]
            if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (frame_pivot,rpy,box_rpy)):
                return fail('dynamic_3d_invalid_frame_actor_pose')
            rotation=rotation_matrix(rpy);box_rotation=rotation_matrix(box_rpy)
            angle=math.degrees(math.acos(float(np.clip((np.trace(rotation.T@box_rotation)-1)/2,-1,1))))
            if angle>.5:return fail('dynamic_3d_box_and_actor_rotations_differ')
            if max(abs((position-frame_pivot)@rotation-expected))>.05:
                return fail('dynamic_3d_frame_api_pivot_contract_changed')
            center=frame_pivot+rotation@offset
            pose_source='recorded_full_actor_transform'
        else:
            if tilt>1.0:return fail('dynamic_3d_normalized_timeline_requires_upright_actor')
            delta=offset-observed
            heading=float(item['heading_deg'])
            if not math.isfinite(heading):return fail('dynamic_3d_invalid_heading')
            rpy=np.array([0.,0.,heading]);rotation=rotation_matrix(rpy)
            center=position+rotation@delta
            pose_source='legacy_upright_yaw_only'
    except (KeyError, TypeError, ValueError, IndexError):
        return fail('dynamic_3d_incomplete_pose_or_profile')
    wlh = size[[1, 0, 2]].tolist()
    return {'training_center_world_xyz': center.tolist(),
            'training_center_world_xy': center[:2].tolist(),
            'training_rotation_rpy_deg': rpy.tolist(),
            'geometry_pose_source': pose_source,
            'training_size_wlh_m': wlh, 'training_size_wl_m': wlh[:2],
            'geometry_sources': {'center': 'dynamic_default_surface_3d', 'size': 'dynamic_default_surface_3d'},
            'geometry_validation_scope': p['validation_scope']}, None
