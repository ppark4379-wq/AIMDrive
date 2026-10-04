"""Reproducible, additive map-basis correction of sealed full-pose manifests.

This corrects reference frames, not timestamps, dimensions, or exposure timing.
Unsupported geometry and actors newly touching the BEV fail closed.
"""
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .morai_map_geometry import validate_map_basis
from .morai_transfer_targets import local_flu, render_object_target

SCHEMA = 'morai_map_basis_derivation_v1'


def rotate_waypoints(waypoints, angle_degrees):
    a = math.radians(angle_degrees)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    points = np.asarray(waypoints, dtype=float)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid source waypoint array')
    return (points @ rotation.T).tolist()


def correct_entry(before, timeline, geometry, basis, signal, pivots=None):
    """Every output field is derived from an admitted entry and sealed evidence.

    Invert the old audited center transform to recover its measured model-local
    offset, then reconstruct about the original actor pivot in projected grid.
    Never rotate the absolute position, global velocity, or sensor extrinsics.
    """
    geo = validate_map_basis(basis) if isinstance(basis, dict) else basis
    if (before['frame_id'] != timeline['index'] or geometry['index'] != timeline['index']
            or before['sensor_time_ns'] != timeline['lidar_source_ns']
            or geometry['lidar_source_ns'] != timeline['lidar_source_ns']):
        raise ValueError('Mixed frame/time evidence')
    out = copy.deepcopy(before)
    ego = out['ego_pose']
    angle = geo.convergence_degrees([ego['x'], ego['y'], 0.])
    ego['yaw'] += math.radians(angle)
    out['waypoints'] = rotate_waypoints(before['waypoints'], angle)
    raw = {o['instance_key']: o for o in timeline['object_gt']['objects']}
    audited = {o['instance_key']: o for o in geometry['objects']}
    represented = {o['instance_key'] for o in before['objects']}
    reasons = []
    for obj in out['objects']:
        key = obj['instance_key']
        src, audit = raw[key], audited[key]
        if (src['model_name'] != obj['model_name'] or not obj['geometry_valid']
                or audit.get('api_box_geometry_candidate') is not True):
            raise ValueError('Unapproved actor identity/geometry')
        pivot = np.asarray(src['actor_pivot_world_xyz'], float)
        rpy = np.asarray(src['actor_rotation_rpy_deg'], float)
        r = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
        old_center = np.array([obj[k] for k in 'xyz'])
        if not np.allclose(old_center, audit.get('training_center_world_xyz', [*audit['training_center_world_xy'], src['position'][2]]), atol=1e-8, rtol=0):
            raise ValueError('Source manifest center differs from audited geometry')
        source = audit.get('geometry_sources', {}).get('center')
        if source == 'dynamic_default_surface_3d':
            if audit.get('geometry_pose_source') != 'recorded_full_actor_transform':
                raise ValueError('Full dynamic actor pose required')
        elif source == 'static_unit_surface_3d':
            # Recover the actual fitted offset, including a nonzero API
            # center-to-pivot offset recorded by the sealed static audit.
            pivot_evidence = (pivots or {}).get(key, {})
            if (pivot_evidence.get('valid') is not True
                    or max(pivot_evidence['offset_range_xyz_m']) > .01):
                raise ValueError('Missing stable static pivot evidence')
            observed = np.asarray(pivot_evidence['offset_xyz_m'], float)
            r = Rotation.from_euler('z', src['heading_deg'], degrees=True).as_matrix()
        elif source == 'grpc':
            if not np.allclose(old_center, src['position'], atol=1e-8, rtol=0):
                raise ValueError('Declared gRPC center was moved')
        else:
            raise ValueError('Unsupported source center recipe: ' + str(source))
        local_offset = r.T @ (old_center - pivot)
        if source == 'static_unit_surface_3d':
            local_offset = r.T @ (old_center - np.asarray(src['position'])) + observed
        center = pivot + geo.orientation(r, pivot, source_frame='true_enu') @ local_offset
        if source == 'grpc':
            center = old_center  # Already a projected world center, not a local offset.
        obj.update(zip('xyz', center.tolist()))
        box_rpy = list(obj['rotation_rpy_deg'])
        box_rpy[2] += geo.convergence_degrees(pivot)
        obj['rotation_rpy_deg'] = box_rpy
        obj['heading_deg'] = box_rpy[2]
    # Re-prove exclusion after changing ego basis. A missing actor must never
    # turn a possibly occupied cell into supervised background.
    for key, src in raw.items():
        if key in represented:
            continue
        audit = audited.get(key, {})
        if audit.get('bev_exclusion_proven') is not True:
            reasons.append('unrepresented_actor_without_exclusion')
            continue
        forward, left = local_flu([src['position'][:2]], ego)[0]
        distance = math.hypot(max(-20-forward, 0., forward-30), max(-25-left, 0., left-25))
        if distance <= audit['enclosing_radius_m'] + audit['boundary_margin_m']:
            reasons.append('previously_excluded_actor_near_corrected_bev')
    if reasons:
        out['object_valid'] = False
        out['object_positive'] = False
    else:
        target, valid, reason = render_object_target({'ego_raw': ego}, out['objects'])
        if not valid:
            raise ValueError(reason)
        out['object_positive'] = bool(np.any(target[:, 0] > 0))
    # The original route/light binding already passed identity, lane and timing
    # checks. Preserve it only if the triangle bound proves heading still <20°.
    # No previously masked light becomes supervised by this transformation.
    light_masked = False
    if out['traffic_light'] != -1:
        if (signal.get('index') != before['frame_id']
                or signal.get('sensor_source_ns') != before['sensor_time_ns']
                or signal.get('training_label') != before['traffic_light']):
            raise ValueError('Changed route/light source binding')
        error = float(signal.get('heading_error_deg', math.inf))
        if not math.isfinite(error) or error + abs(angle) > 20.:
            out['traffic_light'] = -1
            light_masked = True
    return out, {'object_mask_reasons': sorted(set(reasons)), 'light_masked': light_masked}


def verify_map_derivation(doc, resolver, parent, episode):
    from .morai_sensor_record_dataset import (digest, read_json, verify_seal,
                                               verify_geometry_derivation)
    evidence = doc['map_basis_reprocessing']
    if evidence.get('schema') != SCHEMA:
        raise ValueError('Unsupported map basis derivation')
    geo = validate_map_basis(doc['map_basis'])
    source_path = resolver(evidence['source_manifest'], parent)
    if digest(source_path) != evidence['source_manifest_sha256']:
        raise ValueError('Changed map basis source manifest')
    source = read_json(source_path)
    if source.get('map_basis') or source.get('map_basis_reprocessing'):
        raise ValueError('Cannot apply map correction twice')
    if resolver(source['episode'], source_path.parent) != episode:
        raise ValueError('Changed source episode')
    verify_geometry_derivation(source, resolver, source_path.parent, episode)
    # Only these document-level fields may differ; immutable sensor contracts,
    # split, timing and all evidence references remain exactly as recorded.
    allowed = {'entries', 'map_basis', 'map_basis_reprocessing'}
    if {k:v for k,v in doc.items() if k not in allowed} != {k:v for k,v in source.items() if k not in allowed}:
        raise ValueError('Map correction changed unrelated document fields')
    map_path = resolver(evidence['map_metadata'], parent)
    if digest(map_path) != evidence['map_metadata_sha256']:
        raise ValueError('Changed map coordinate evidence')
    import ast
    from pyproj import CRS
    map_doc = read_json(map_path)
    hashes = ast.literal_eval(map_doc['mgeo_file_hash'])
    if (hashes['node_set.json'] != source['navigation_contract']['route_source_resolution']['recorded_sha256']['node_set']
            or CRS(map_doc['global_coordinate_system']) != geo.crs
            or map_doc['local_origin_in_global'] != doc['map_basis']['origin_xyz']):
        raise ValueError('Map basis is not bound to the recorded map')
    first = source['entries'][0]
    gps_rel = first['asset_roles']['gps']
    from .morai_sensor_record_dataset import SensorRecordDataset
    gps = SensorRecordDataset.read_asset(resolver(source['native_root'], source_path.parent), first, gps_rel)
    fields = {k.strip().lower():float(v) for k,v in
              (line.split(':',1) for line in gps.decode('utf-8-sig').splitlines() if ':' in line)}
    if [fields['eastoffset'], fields['northoffset']] != doc['map_basis']['origin_xyz'][:2]:
        raise ValueError('Map origin disagrees with recorded GPS offsets')
    ev = source['geometry_reprocessing']
    timeline = {r['index']:r for r in read_json(resolver(ev['timeline_directory'])/'frames.json')}
    geometry = {r['index']:r for r in read_json(resolver(ev['geometry_directory'])/'frames.json')}
    pivots = read_json(resolver(ev['geometry_directory'])/'pivot_evidence.json')['instances']
    verify_seal(episode/'route_signal_audit_v1')
    if digest(episode/'route_signal_audit_v1/COMPLETE.json') != source['evidence_seal_sha256']['route_signal']:
        raise ValueError('Changed route signal evidence')
    signals = {r['index']:r for r in read_json(episode/'route_signal_audit_v1/frames.json')}
    if len(source['entries']) != len(doc['entries']):
        raise ValueError('Map derivation must preserve frame membership and ordering')
    for old, new in zip(source['entries'], doc['entries']):
        expected, _ = correct_entry(old, timeline[old['frame_id']], geometry[old['frame_id']], geo, signals[old['frame_id']], pivots)
        if new != expected:
            raise ValueError('Map derivation changed a field beyond its deterministic correction')
