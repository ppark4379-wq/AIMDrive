"""Render static obstacle positions from current-frame ego geometry.

Route coordinates determine maneuver feasibility. Maintain sentences describe
the nearest eligible obstacle ahead of the ego, with 1 m switch hysteresis.
This final rendering pass must run after temporal instruction postprocessing.
"""
from collections import Counter
import math

from .renderer import load_catalog, render_instruction

MAINTAIN_STATES = {'approach_maintain', 'keep_path'}


def select_maintain_targets(rules, indexed, cameras=None):
    """Retain maneuver targets; select a separate descriptive maintain target."""
    held = None
    previous_index = None
    changed = 0
    for rule in rules:
        if (rule.get('zone_id') != 'static_obstacle'
                or rule.get('instruction_state') not in MAINTAIN_STATES):
            held = previous_index = None
            continue
        if previous_index is None or rule['index'] != previous_index + 1:
            held = None
        previous_index = rule['index']
        evidence = rule.setdefault('evidence', {})
        candidates = []
        for actor in evidence.get('relevant_obstacle_ids', []):
            matches = indexed.get((rule['index'], str(actor)), [])
            if len(matches) != 1:
                continue
            obj = matches[0]
            if cameras is not None:
                from .side_view_hold import in_side_view
                if in_side_view(obj, cameras['front']) is not True:
                    continue
            forward, left = obj.get('forward_m'), obj.get('left_m')
            if ego_position(forward, left) is not None and forward > 0:
                candidates.append((math.hypot(forward, left), str(actor), obj))
        if not candidates:
            held = None
            continue
        candidates.sort(key=lambda c: (c[0], c[1]))
        selected = candidates[0]
        prior = next((c for c in candidates if (c[1], c[2].get('instance_key')) == held), None)
        if prior is not None and prior[0] <= selected[0] + 1.0:
            selected = prior
        distance, _, obj = selected
        held = (str(obj['actor_id']), obj.get('instance_key'))
        variables = rule.setdefault('variables', {})
        ref = dict(actor_id=obj['actor_id'], instance_key=obj.get('instance_key'))
        if evidence.get('target_actor_ref') != ref:
            changed += 1
        evidence.setdefault('rule_target_actor_ref', evidence.get('target_actor_ref'))
        evidence['target_actor_ref'] = ref
        evidence['description_target_policy'] = 'nearest_ego_front_1m_hysteresis'
        evidence['description_target_distance_m'] = distance
        rule['related_actors'] = [obj['actor_id']]
        variables.update(target_actor_id=obj['actor_id'], target_kind='obstacle', target_distance_m=distance)
    return changed


def ego_position(forward, left):
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in (forward, left)):
        return None
    if left > 0.3:
        return "left"
    if left < -0.3:
        return "right"
    return "front" if forward >= 0 else "rear"


def update_static_positions(rules, objects, catalog=None, cameras=None):
    """Use exact current-frame actor identities; never fall back to route side."""
    catalog = load_catalog() if catalog is None else catalog
    indexed = {}
    for obj in objects:
        indexed.setdefault((obj['index'], str(obj.get('actor_id'))), []).append(obj)
    counts = Counter(targets_changed=select_maintain_targets(rules, indexed, cameras))
    if cameras is not None:
        from .side_view_hold import hold_side_targets
        counts['side_view_delayed_frames'] = hold_side_targets(rules, indexed, cameras)
    for rule in rules:
        if rule.get('zone_id') != 'static_obstacle':
            continue
        template = catalog['static_obstacle'].get(rule.get('instruction_id'), {})
        if 'position' not in template.get('required_variables', []):
            continue
        evidence = rule.setdefault('evidence', {})
        refs = evidence.get('corridor_pair_actor_refs') if rule.get('instruction_state') == 'pass_between' else None
        refs = refs or [evidence.get('target_actor_ref') or
                        {'actor_id': rule.get('variables', {}).get('target_actor_id')}]
        coordinates = []
        for ref in refs:
            matches = indexed.get((rule['index'], str(ref.get('actor_id'))), [])
            if ref.get('instance_key'):
                matches = [obj for obj in matches if obj.get('instance_key') == ref['instance_key']]
            if len(matches) != 1:
                break
            obj = matches[0]
            forward, left = obj.get('forward_m'), obj.get('left_m')
            if ego_position(forward, left) is None:
                break
            coordinates.append((forward, left))
        side = None
        if len(coordinates) == len(refs):
            side = ego_position(*(sum(c[i] for c in coordinates) / len(coordinates) for i in (0, 1)))
        if rule.get('instruction_state') == 'pass_between':
            # A pair's centroid must not turn a left/right pair into a one-sided label.
            side = None
            if len(refs) == len(coordinates) == 2 and refs[0] != refs[1]:
                positions = {ego_position(*c) for c in coordinates}
                side = '_'.join(p for p in ('front', 'left', 'right', 'rear') if p in positions)
                evidence['pair_actor_positions'] = [dict(actor_id=ref.get('actor_id'),
                    position=ego_position(*c)) for ref,c in zip(refs,coordinates)]
        variables = rule.setdefault('variables', {})
        evidence.setdefault('route_target_side', variables.get('target_side'))
        evidence['position_reference'] = 'current_ego_geometry'
        evidence['position_actor_coordinates'] = [dict(actor_id=ref.get('actor_id'),
            forward_m=c[0], left_m=c[1]) for ref, c in zip(refs, coordinates)]
        variables['target_side'] = variables['target_position'] = side
        variables.pop('target_position_text', None)
        text, values, missing = render_instruction(catalog, 'static_obstacle', rule['instruction_id'], variables)
        before = rule.get('instruction')
        # Preserve prior exclusions, including direction contradictions.
        if before is not None:
            rule['instruction'] = text
        if 'instruction_dropped' in rule:
            rule['instruction_dropped'] = text
        rule['instruction_variables'] = values
        if missing:
            rule['instruction_mapping_status'] = 'missing_variables'
            rule['instruction_missing_variables'] = missing
            counts['missing_geometry'] += 1
        else:
            rule.pop('instruction_missing_variables', None)
            counts['resolved'] += 1
        counts['changed_sentences'] += before != rule.get('instruction')
    return dict(counts)
