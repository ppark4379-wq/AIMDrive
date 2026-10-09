"""Choose one highway description target from the current sealed inventory.

The choice is independent of the raw lane-change state and previous sentences.
Actual hazard-response policies keep their separately established causal actor.
"""
import math

SCHEMA = 'current_highway_target_selection_v1'
LEAD_ROUTE_RANGE_M = 80.0
LEAD_NEARBY_RANGE_M = 85.0
# Description-only candidates use the same current verified 85 m inventory
# cohort as lead/nearby geometry admission. A shared MGeo node may briefly
# change a nearest-link lane number; existence/position do not require claiming
# that the actor is the current-lane lead. This does not enlarge hazard range.
APPROACH_RANGE_M = LEAD_NEARBY_RANGE_M


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _id(value):
    # Native integer 17 and string "17" denote the same ID. Other strings
    # remain exact ("017" is not silently rewritten), and booleans are not IDs.
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def _verified(actor, frame, counts, limit):
    identifier = _id(actor.get('actor_id'))
    return (identifier is not None and counts.get(identifier) == 1
            and _finite(actor.get('index'))
            and actor.get('index') == frame.get('index')
            and actor.get('kind') == 'npc' and actor.get('instance_key')
            and actor.get('lidar_source_ns') == frame.get('lidar_source_ns')
            and all(actor[k] == frame['lidar_source_ns'] for k in ('sensor_time_ns',) if k in actor)
            and actor.get('model_identity_verified') is True
            and actor.get('semantic_class_valid') is True
            and actor.get('semantic_class') == 'vehicle'
            and actor.get('rule_geometry_usable') is True
            and all(_finite(actor.get(k)) for k in
                    ('forward_m', 'left_m', 'actor_width_m', 'distance_to_ego_m'))
            and actor['actor_width_m'] > 0 and actor['forward_m'] > 0
            and 0 < actor['distance_to_ego_m'] <= limit)


def select_current_highway_target(frame, objects):
    """Return (actor, role, reasons, receipt); unknown declared leads fail closed.

    This is target selection, not physical safety or a stopping-cause proof.
    Existing final position validation additionally checks world coordinates,
    length and source-admission evidence. Never substitute another vehicle for
    a declared lead whose identity/lane/geometry cannot be verified.
    """
    receipt = dict(schema=SCHEMA, index=(frame or {}).get('index'),
                   lidar_source_ns=(frame or {}).get('lidar_source_ns'),
                   source='current_sealed_inventory_and_current_route_lead',
                   uses_future_ego=False, uses_neighbor_sentence=False,
                   uses_previous_target=False, physical_front_safety_certified=False,
                   id_policy='native_integer_or_exact_string; integer17_equals_string17',
                   selected_actor_ref=None, role=None, basis=None,
                   eligible_approach_actor_ids=[],
                   description_range_m=APPROACH_RANGE_M,
                   description_range_basis='same_verified_current_85m_inventory_cohort',
                   approach_role_does_not_certify_same_lane_lead=True)
    if (frame is None or frame.get('valid') is not True
            or frame.get('object_inventory_usable') is not True
            or frame.get('dataset_build_candidate') is False
            or frame.get('zone_id') not in (None, 'highway')
            or not all(_finite(frame.get(k)) for k in ('index', 'lidar_source_ns'))):
        return None, None, ['current_highway_target_selection_frame_unusable'], receipt
    if frame.get('highway_nearby_geometry_unverified') is True:
        return None, None, ['relevant_vehicle_geometry_unverified'], receipt
    current = [a for a in objects if a.get('index') == frame['index']]
    counts = {}
    for actor in current:
        identifier = _id(actor.get('actor_id'))
        if identifier is not None:
            counts[identifier] = counts.get(identifier, 0) + 1
    declared = frame.get('lead_actor_id')
    if declared is not None:
        identifier = _id(declared)
        leads = [a for a in current if identifier is not None and _id(a.get('actor_id')) == identifier]
        if (len(leads) != 1 or not _verified(leads[0], frame, counts, LEAD_NEARBY_RANGE_M)
                or not _finite(frame.get('lead_distance_m'))
                or not 0 < frame['lead_distance_m'] <= LEAD_ROUTE_RANGE_M
                or frame.get('ego_current_lane_id') is None
                or leads[0].get('highway_lane_id') != frame['ego_current_lane_id']):
            receipt['basis'] = 'declared_current_lead_unverified'
            return None, None, ['current_highway_lead_unverified'], receipt
        actor, role = leads[0], 'lead_vehicle'
        receipt['basis'] = 'verified_current_same_lane_lead'
    else:
        eligible = []
        for candidate in current:
            if candidate.get('kind') != 'npc':
                continue
            distance, forward = candidate.get('distance_to_ego_m'), candidate.get('forward_m')
            # Missing location cannot establish that an unknown NPC is safely
            # outside this selection cohort. Do not silently skip it.
            relevant = (not _finite(distance) or distance <= APPROACH_RANGE_M) and (
                not _finite(forward) or forward > 0)
            if not relevant:
                continue
            if not _verified(candidate, frame, counts, APPROACH_RANGE_M):
                receipt['basis'] = 'current_forward_approach_unverified'
                return None, None, ['current_highway_approach_unverified'], receipt
            eligible.append(candidate)
        eligible.sort(key=lambda a: (a['distance_to_ego_m'], _id(a['actor_id']), str(a['instance_key'])))
        receipt['eligible_approach_actor_ids'] = [_id(a['actor_id']) for a in eligible]
        if not eligible:
            receipt['basis'] = 'no_current_forward_target'
            return None, None, [], receipt
        actor, role = eligible[0], 'approaching_vehicle'
        receipt['basis'] = 'nearest_verified_current_forward_npc'
    receipt.update(selected_actor_ref={'actor_id': actor['actor_id'], 'instance_key': actor['instance_key']},
                   role=role, forward_m=actor['forward_m'], left_m=actor['left_m'],
                   distance_to_ego_m=actor['distance_to_ego_m'],
                   current_route_lead_distance_m=frame.get('lead_distance_m') if role == 'lead_vehicle' else None)
    return actor, role, [], receipt


def target_selection_applies(rule):
    """KEEP descriptions share the policy; causal response actors are preserved."""
    return (rule.get('zone_id') == 'highway' and (
        (rule.get('label_candidate') == 'KEEP_LANE' and rule.get('instruction_state') in
         ('normal', 'approach_keep', 'keep_completed'))
        or 'current_highway_target_selection' in (rule.get('evidence') or {})))


def validate_current_highway_target_selection(frame, objects, rule):
    """Recompute the selected ID; neither receipt presence nor text is a proof."""
    if not target_selection_applies(rule):
        return True, [], {}
    actor, role, reasons, receipt = select_current_highway_target(frame, objects)
    if frame is None or rule.get('index') != frame.get('index'):
        reasons.append('current_highway_target_selection_index_disagrees')
    variables = rule.get('variables') or {}
    expected_id = _id(actor.get('actor_id')) if actor is not None else None
    if _id(variables.get('target_actor_id')) != expected_id:
        reasons.append('current_highway_selected_target_disagrees')
    if not reasons and variables.get('response_reason') != role:
        reasons.append('current_highway_selected_role_disagrees')
    marker = (rule.get('evidence') or {}).get('current_highway_target_selection')
    if marker is not None and marker != receipt:
        reasons.append('current_highway_target_selection_receipt_disagrees')
    return not reasons, sorted(set(reasons)), receipt
