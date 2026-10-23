"""Read-only LAV bbox adapter for recorded-input Expert connection checks.

VE boxes have no semantic class or persistent actor identity. Never populate
those fields from GT matching. All sizes below are full physical dimensions;
coordinate axes are bound to the checkpoint's projected-map contract.
"""
import math

import numpy as np

from .morai_object_metrics import _ground_truth, decode_object_boxes


def _pose(ego_pose):
    values = tuple(float(ego_pose[key]) for key in ('x', 'y', 'yaw'))
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Finite projected-map ego pose required')
    return values


def right_forward_to_world(center, heading, ego_pose):
    """right/forward -> map x/y; yaw is radians, CCW from map +x."""
    x, y, yaw = _pose(ego_pose)
    right, forward = map(float, center)
    hr, hf = map(float, heading)
    norm = math.hypot(hr, hf)
    if not all(math.isfinite(value) for value in (right, forward, hr, hf, norm)) or norm < 1e-8:
        raise ValueError('Invalid local bbox centre or heading')
    c, s = math.cos(yaw), math.sin(yaw)
    position = [x + c * forward + s * right, y + s * forward - c * right]
    hx, hy = c * hf + s * hr, s * hf - c * hr
    return position, math.degrees(math.atan2(hy, hx))


def _world_box(box, ego_pose, source, *, speed_mps=None, track_id=None,
               kind='unknown', is_static=None, coordinate_transform=None):
    if coordinate_transform is None:
        centre, heading = right_forward_to_world(box['center'], box['heading'], ego_pose)
    else:
        x,y,yaw = _pose(ego_pose)
        centre, heading = coordinate_transform.ve_box_to_enu(box['center'], box['heading'], (x,y), yaw)
    return dict(center_world=centre, heading_deg=heading,
                length_m=float(box['length']), width_m=float(box['width']),
                speed_mps=speed_mps, track_id=track_id, kind=kind,
                is_static=is_static, confidence=float(box.get('score', 1.)),
                source=source,
                center_right_forward_m=list(map(float, box['center'])),
                partial_bev=bool(box.get('partial', False)))


def decode_world_predictions(detection, ego_pose, coordinate_transform=None):
    """Decode at existing fixed confidence/NMS policy, without GT access."""
    boxes, counts = decode_object_boxes(detection)
    values = detection.detach().float().cpu().numpy() if hasattr(detection, 'detach') else np.asarray(detection)
    result = []
    invalid_speeds = 0
    for box in boxes:
        speed = float(values[box['cell_index'], 7]) * 8.
        if not math.isfinite(speed) or speed < 0.:
            speed = None
            invalid_speeds += 1
        record = _world_box(box, ego_pose, 've', speed_mps=speed, coordinate_transform=coordinate_transform)
        record['cell_index'] = int(box['cell_index'])
        # Cell IDs are not actor IDs; do not turn a shifting cell into a track.
        result.append(record)
    return dict(boxes=result, rejections=dict(counts, invalid_speed_values=invalid_speeds),
                object_class_predicted=False, persistent_tracks_available=False,
                velocity_vector_available=False,
                prediction_policy='lmdrive_lav_v1_raw_logits_fixed_0_5_nms_0_3')


def world_ground_truth(entry):
    """Same positive-area BEV membership as official object diagnostics."""
    if entry.get('object_valid') is not True:
        raise ValueError('Planning GT requires a supervised object frame')
    boxes, outside = _ground_truth(entry)
    result = []
    for box in boxes:
        obj = entry['objects'][box['object_index']]
        vx, vy = float(obj.get('vel_x', 0.)), float(obj.get('vel_y', 0.))
        if not math.isfinite(vx) or not math.isfinite(vy):
            raise ValueError('Invalid GT velocity')
        record = _world_box(box, entry['ego_pose'], 'gt',
                            speed_mps=math.hypot(vx, vy), track_id=box['instance_key'])
        record.update(object_index=int(box['object_index']),
                      model_name=box['model_name'])
        result.append(record)
    return dict(boxes=result, outside_bev=int(outside),
                semantic_policy='class-agnostic footprint comparison; GT class is not copied to VE')


def verify_adapter_roundtrip(entry):
    """Check converted GT against its sealed world centres and dimensions."""
    converted = world_ground_truth(entry)
    centre_errors, heading_errors, size_errors = [], [], []
    for box in converted['boxes']:
        obj = entry['objects'][box['object_index']]
        centre_errors.append(math.hypot(box['center_world'][0] - obj['x'],
                                        box['center_world'][1] - obj['y']))
        difference = math.radians(box['heading_deg'] - obj['heading_deg'])
        # An oriented footprint is invariant under a 180-degree heading flip.
        heading_errors.append(math.degrees(math.acos(min(1., abs(math.cos(difference))))))
        size_errors.extend([abs(box['length_m'] - obj['length']),
                            abs(box['width_m'] - obj['width'])])
    maximum = dict(center_error_m=max(centre_errors, default=0.),
                   heading_error_deg_mod180=max(heading_errors, default=0.),
                   size_error_m=max(size_errors, default=0.))
    if maximum['center_error_m'] > 1e-7 or maximum['size_error_m'] > 1e-9 or maximum['heading_error_deg_mod180'] > 1e-5:
        raise ValueError('BBox adapter does not preserve sealed world geometry: ' + str(maximum))
    return dict(status='PASS', included_boxes=len(converted['boxes']),
                outside_bev=converted['outside_bev'], **maximum)
