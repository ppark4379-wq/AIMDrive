"""Bind highway target words to one fresh current actor's recorded body geometry.

This binds position words and independently checks the shared current-target
choice. Existing exclusions and separately causal traffic response actors stay.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import math

SCHEMA = 'current_highway_position_v1'


@lru_cache(maxsize=1)
def _catalog():
    from .renderer import load_catalog
    return load_catalog()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def body_position(actor):
    """Same width-aware lateral guard for KEEP, FOLLOW, response and maneuvers."""
    forward, left, width = (actor.get(k) for k in ('forward_m', 'left_m', 'actor_width_m'))
    if not all(_finite(v) for v in (forward, left, width)) or width <= 0:
        return None
    guard = 1.0 + width / 2.0 + .25
    side = 'left' if left > guard else 'right' if left < -guard else None
    if forward > 0:
        return 'front_' + side if side else 'front'
    if forward < 0:
        return side + '_rear' if side else 'rear'
    return side  # A coincident centre has no independently supported direction.


def _target(rule):
    value = (rule.get('variables') or {}).get('target_actor_id')
    return None if value is None else str(value)


def current_target_position(frame, objects, rule):
    """Return (position, reasons, receipt); never choose a different actor."""
    target, reasons = _target(rule), []
    if target is None:
        return None, ['current_highway_target_missing'], {}
    if (frame is None or frame.get('zone_id') != 'highway' or frame.get('valid') is not True
            or frame.get('dataset_build_candidate') is not True
            or frame.get('object_inventory_usable') is not True
            or not _finite(frame.get('lidar_source_ns'))):
        return None, ['current_highway_position_frame_unusable'], {}
    actors = [a for a in objects if a.get('index') == frame['index']
              and str(a.get('actor_id')) == target]
    if len(actors) != 1:
        return None, ['current_highway_position_target_nonunique_or_missing'], {}
    actor = actors[0]
    if (not actor.get('instance_key') or actor.get('kind') != 'npc'
            or actor.get('model_identity_verified') is not True
            or actor.get('semantic_class_valid') is not True
            or actor.get('semantic_class') != 'vehicle'):
        reasons.append('current_highway_position_identity_unverified')
    if (actor.get('lidar_source_ns') != frame['lidar_source_ns']
            or any(actor[k] != frame['lidar_source_ns'] for k in ('sensor_time_ns',) if k in actor)
            or rule.get('index') != frame['index']):
        reasons.append('current_highway_position_timestamp_disagrees')
    if (actor.get('rule_geometry_usable') is not True
            or not all(_finite(actor.get(k)) for k in
                       ('world_x_m', 'world_y_m', 'actor_width_m', 'actor_length_m', 'distance_to_ego_m'))
            or actor.get('actor_width_m', 0) <= 0 or actor.get('actor_length_m', 0) <= 0):
        reasons.append('current_highway_position_geometry_unverified')
    for marker in ('target_actor_ref', 'stale_highway_current_target'):
        old = (rule.get('evidence') or {}).get(marker)
        if old is not None and (str(old.get('actor_id')), old.get('instance_key')) != (target, actor.get('instance_key')):
            reasons.append('current_highway_position_actor_reference_disagrees')
        if old and 'lidar_source_ns' in old and old['lidar_source_ns'] != frame['lidar_source_ns']:
            reasons.append('current_highway_position_actor_reference_stale')
    position = body_position(actor)
    if position is None:
        reasons.append('current_highway_position_body_coordinates_unverified')
    receipt = dict(schema=SCHEMA, actor_id=actor['actor_id'], instance_key=actor.get('instance_key'),
                   index=frame['index'], lidar_source_ns=frame['lidar_source_ns'],
                   model_name=actor.get('model_name'), forward_m=actor.get('forward_m'),
                   left_m=actor.get('left_m'), actor_width_m=actor.get('actor_width_m'),
                   actor_length_m=actor.get('actor_length_m'), world_x_m=actor.get('world_x_m'),
                   world_y_m=actor.get('world_y_m'), position=position,
                   uses_future_ego=False, uses_neighbor_sentence=False,
                   source='current_sealed_gt_identity_and_geometry_audit')
    return position, sorted(set(reasons)), receipt


def _exclude(rule, reasons):
    rule['action_valid'] = False
    rule['observation_valid'] = False
    rule.setdefault('unknown_reasons', [])[:] = sorted(set(rule.get('unknown_reasons', []) + reasons))
    rule.setdefault('evidence', {})['current_highway_position_rejection'] = list(reasons)


def bind_current_highway_positions(frames, objects, rules, *, render=False):
    """Update only current target position slots; never restore excluded text."""
    grouped = defaultdict(list)
    for actor in objects:
        grouped[actor.get('index')].append(actor)
    indexed = defaultdict(list)
    for frame in frames:
        indexed[frame.get('index')].append(frame)
    counts = Counter()
    for rule in rules:
        if rule.get('zone_id') != 'highway':
            continue
        fs = indexed[rule.get('index')]
        frame = fs[0] if len(fs) == 1 else None
        from .highway_rules import lane_dependent_action_reasons
        lane_reasons = lane_dependent_action_reasons(frame, rule)
        if lane_reasons:
            evidence = rule.setdefault('evidence', {})
            if 'current_highway_lane_association_rejection' not in evidence:
                rule.setdefault('instruction_history', []).append({
                    'pass': 'current_highway_lane_association',
                    'instruction': rule.get('instruction'),
                    'instruction_id': rule.get('instruction_id'), 'reasons': lane_reasons})
            evidence['current_highway_lane_association_rejection'] = deepcopy(
                frame.get('highway_lane_association_evidence'))
            rule['action_valid'] = False
            # The current actor/observations remain intact; only the action's
            # lane association is unsupported. Never raise an old False mask.
            rule['instruction'] = None
            rule['instruction_consistency'] = 'excluded_current_highway_lane_association'
            rule.setdefault('unknown_reasons', [])[:] = sorted(set(
                rule.get('unknown_reasons', []) + lane_reasons))
            counts['excluded_ambiguous_lane_action'] += 1
            # Still validate/bind the current observation below. A rejected
            # action is not permission to retain stale actor position slots.
        from .highway_target_selection import validate_current_highway_target_selection, target_selection_applies
        if target_selection_applies(rule):
            choice_valid, choice_reasons, choice_receipt = validate_current_highway_target_selection(
                frame, grouped[rule.get('index')], rule)
            if not choice_valid:
                rule['action_valid'] = False
                rule['instruction'] = None
                rule['instruction_consistency'] = 'excluded_current_highway_target_selection'
                rule.setdefault('unknown_reasons', [])[:] = sorted(set(
                    rule.get('unknown_reasons', []) + choice_reasons))
                rule.setdefault('evidence', {})['current_highway_target_selection_rejection'] = choice_reasons
                counts['excluded_current_target_selection'] += 1
            else:
                rule.setdefault('evidence', {})['current_highway_target_selection'] = choice_receipt
        if _target(rule) is None:
            continue
        position, reasons, receipt = current_target_position(fs[0] if len(fs) == 1 else None,
                                                             grouped[rule.get('index')], rule)
        if reasons:
            _exclude(rule, reasons)
            counts['excluded_unverified_current_target'] += 1
            continue
        variables = rule.setdefault('variables', {})
        if variables.get('target_position') != position or variables.get('target_position_text') != position:
            counts['position_words_changed'] += 1
        variables.update(target_position=position, target_position_text=position)
        rule.setdefault('evidence', {})['current_highway_position'] = receipt
        # A late rerender repairs slots only for a sentence that still exists.
        # Prior nulls, False masks and reasons are never raised or cleared.
        if render and rule.get('instruction'):
            from .renderer import render_instruction
            identifier = rule.get('instruction_id')
            catalog = _catalog()
            if identifier not in catalog['highway']:
                _exclude(rule, ['current_highway_position_template_unverified'])
                counts['excluded_unverified_template'] += 1
                continue
            text, slots, missing = render_instruction(catalog, 'highway', identifier, variables)
            if missing:
                _exclude(rule, ['current_highway_position_render_unverified'])
                counts['excluded_unverified_render'] += 1
                continue
            rule.update(instruction=text, instruction_variables=slots)
            counts['rerendered_current_slots'] += 1
        counts['bound_current_target'] += 1
    return dict(sorted(counts.items()))


def validate_current_highway_position(frames, objects, rule):
    """Recompute exact source target and final words, independently of a marker."""
    if rule.get('zone_id') != 'highway':
        return True, []
    current = [f for f in frames if f.get('index') == rule.get('index')]
    from .highway_rules import lane_dependent_action_reasons
    lane_reasons = lane_dependent_action_reasons(current[0] if len(current) == 1 else None, rule)
    if lane_reasons:
        return False, lane_reasons
    from .highway_target_selection import validate_current_highway_target_selection
    choice_valid, choice_reasons, _ = validate_current_highway_target_selection(
        current[0] if len(current) == 1 else None, objects or [], rule)
    if not choice_valid:
        return False, choice_reasons
    if _target(rule) is None:
        slots = rule.get('instruction_variables') or {}
        return (False, ['current_highway_target_missing']) if slots.get('position') else (True, [])
    if objects is None:
        return False, ['current_highway_position_inventory_missing']
    position, reasons, receipt = current_target_position(current[0] if len(current) == 1 else None, objects, rule)
    variables = rule.get('variables') or {}
    if variables.get('target_position') != position or variables.get('target_position_text') != position:
        reasons.append('current_highway_position_variables_disagree')
    marker = (rule.get('evidence') or {}).get('current_highway_position')
    if marker is not None and marker != receipt:
        reasons.append('current_highway_position_receipt_disagrees')
    if not reasons and rule.get('instruction'):
        from .renderer import render_instruction
        identifier, catalog = rule.get('instruction_id'), _catalog()
        if identifier not in catalog['highway']:
            reasons.append('current_highway_position_template_unverified')
        else:
            text, slots, missing = render_instruction(catalog, 'highway', identifier, variables)
            if missing or rule['instruction'] != text or rule.get('instruction_variables') != slots:
                reasons.append('current_highway_position_rendered_slots_disagree')
    return not reasons, sorted(set(reasons))
