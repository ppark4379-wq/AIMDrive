"""Delay descriptive-target handoff until its front-camera footprint exits.

Uses calibrated horizontal FOV and a conservative obstacle bounding circle;
this is geometric image-boundary visibility, not an occlusion detector.
"""
import copy
import json
import math
from pathlib import Path

from .ego_position import ego_position, MAINTAIN_STATES


def camera_views(episode):
    from MoraiDrive.data_collection.internal_sensor_record.async_capture.camera_geometry import projection_transforms
    root = Path(__file__).resolve().parents[1] / 'internal_sensor_record/evidence/native_geometry_v1'
    contract = json.loads((Path(episode).parent/'sensor_contract.json').read_text(encoding='utf-8'))
    transforms, _ = projection_transforms(contract, root/'persisted_sensor_settings.json', root/'camera_projection_evidence.json')
    result = {}
    for side in ('front',):
        tf = transforms[side]; q = tf['rotation']
        yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
        result[side] = (tf['translation']['x'], tf['translation']['y'], yaw,
                        math.radians(contract['camera_fov_degrees'][side]/2))
    return result


def in_side_view(obj, camera):
    f, l = obj.get('forward_m'), obj.get('left_m')
    w, length = obj.get('actor_width_m'), obj.get('actor_length_m')
    if ego_position(f, l) is None or any(not isinstance(v, (int,float)) or not math.isfinite(v) or v <= 0 for v in (w,length)):
        return None
    x,y,yaw,half = camera
    dx,dy = f-x,l-y
    along = dx*math.cos(yaw)+dy*math.sin(yaw)
    across = -dx*math.sin(yaw)+dy*math.cos(yaw)
    radius = math.hypot(w,length)/2
    return (along >= -radius and
            along*math.sin(half)-abs(across)*math.cos(half) >= -radius)


def hold_side_targets(rules, indexed, cameras):
    held = None
    prior_index = None
    release_until = None
    delayed = 0
    for rule in rules:
        index = rule['index']
        if (prior_index is None or index != prior_index+1 or rule.get('zone_id') != 'static_obstacle'
                or not rule.get('instruction')):
            held = None
            release_until = None
        prior_index = index
        state = rule.get('instruction_state')
        evidence = rule.setdefault('evidence', {})
        if held is not None and state in MAINTAIN_STATES | {'avoid','avoid_continue'}:
            ref, side, template_id = held
            candidates = [o for o in indexed.get((index,str(ref['actor_id'])), [])
                          if o.get('instance_key') == ref.get('instance_key')]
            visible = in_side_view(candidates[0], cameras[side]) if len(candidates)==1 else None
            current = evidence.get('target_actor_ref')
            if visible is True and current != ref:
                evidence['side_view_deferred_rule'] = dict(state=state, instruction_id=rule.get('instruction_id'),
                    target_actor_ref=copy.deepcopy(current), variables=copy.deepcopy(rule.get('variables', {})))
                evidence.update(side_view_hold=side, side_view_hold_method='calibrated_horizontal_fov_bounding_circle',
                                target_actor_ref=copy.deepcopy(ref))
                rule.update(instruction_state='approach_maintain',instruction_id=template_id,
                            related_actors=[ref['actor_id']])
                rule['variables'].update(target_actor_id=ref['actor_id'],target_kind='obstacle',
                    target_distance_m=math.hypot(candidates[0]['forward_m'], candidates[0]['left_m']))
                rule['variables'].pop('avoid_direction',None)
                delayed += 1
                continue
            if visible is False:
                evidence['side_view_released_actor'] = copy.deepcopy(ref)
                held = None
                if state == 'avoid_continue':
                    release_until = index+4
            elif visible is None:
                raise ValueError(f'Missing side-view geometry at frame {index}')
        elif state not in MAINTAIN_STATES:
            held = None
        if release_until is not None:
            if state in ('avoid','avoid_continue') and index <= release_until:
                rule.update(instruction_state='avoid',instruction_id='follow_avoid' if rule.get('label')=='follow' else 'static_avoid')
                evidence['side_view_delayed_avoid_start'] = True
            else:
                release_until = None
        if held is None and state in MAINTAIN_STATES:
            ref = evidence.get('target_actor_ref')
            original = evidence.get('rule_target_actor_ref')
            # Only hold a descriptive bypass target, not the actual conflict target.
            if ref and original and ref != original:
                matches = indexed.get((index,str(ref['actor_id'])), [])
                if len(matches)==1:
                    if in_side_view(matches[0],cameras['front']) is True:
                        held=(copy.deepcopy(ref),'front',rule['instruction_id'])
    return delayed
