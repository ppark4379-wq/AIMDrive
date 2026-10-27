"""Continuous adjacent-lane traffic and measured merge-scenario evidence."""
import json
import math
from pathlib import Path

from ...utils.transform_utils import (polyline_length, interpolate_on_polyline,
                                     project_distance_on_polyline)


def enabled(context):
    return context.cfg['highway_npc']['managed_traffic'].get('flowing_traffic', {}).get('enabled', False)


def prepare(context, plans):
    if not enabled(context):
        return
    cfg = context.cfg['highway_npc']['managed_traffic']
    if cfg.get('merge_sync', {}).get('enabled', False):
        raise ValueError('Continuous traffic cannot use timed departure')
    settings = cfg['flowing_traffic']
    for key in ('recycle_after_merge_m', 'observation_radius_m', 'respawn_clearance_m', 'minimum_ego_distance_m'):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError('Invalid continuous traffic setting: ' + key)
    lane = next(lane for lane in cfg['lanes'] if lane.get('merge_adjacent'))
    route = lane['route']
    points = [p for link in route for p in context.map_loader.get_link_points(link)]
    departure_points = context.map_loader.get_link_points(settings['departure_link'])
    departure_distance = float(settings['departure_distance_from_link_start_m'])
    if not math.isfinite(departure_distance) or not 0 <= departure_distance <= polyline_length(departure_points):
        raise ValueError('Highway departure point is outside the approach curve')
    dx, dy, _, _ = interpolate_on_polyline(departure_points, departure_distance)
    departure_s = project_distance_on_polyline(context.route_points, dx, dy)
    rx, ry, _, _ = interpolate_on_polyline(context.route_points, departure_s)
    if math.hypot(dx-rx, dy-ry) > 5.:
        raise ValueError('Highway departure point is outside the active Ego route')
    target_link = context.map_loader.get_link(cfg['ego_merge_link'])['left_lane_change_dst_link_idx']
    target_points = context.map_loader.get_link_points(target_link)
    target_length = polyline_length(target_points)
    merge = None
    min_y, max_y = min(t[1] for t in target_points), max(t[1] for t in target_points)
    for index, p in enumerate(context.route_points):
        if not min_y <= p[1] <= max_y:
            continue
        s = project_distance_on_polyline(target_points, *p[:2])
        x, y, _, _ = interpolate_on_polyline(target_points, s)
        if 20 < s < target_length - 10 and math.hypot(p[0] - x, p[1] - y) <= 1.75:
            merge = (polyline_length(context.route_points[:index + 1]),
                     project_distance_on_polyline(points, x, y))
            break
    if merge is None:
        raise ValueError('Continuous traffic requires an identifiable Expert merge entry')
    if departure_s >= merge[0]:
        raise ValueError('Highway departure must precede the merge')
    slot = min((p for p in plans if p['merge_adjacent']), key=lambda p: p['offset_m'])
    context.highway_flow = dict(settings=settings, points=points, route=route,
        slot=dict(slot), merge_ego_s=merge[0], merge_lane_s=merge[1], generation=0,
        departure_ego_s=departure_s,
        result=dict(status='awaiting_merge', collision_seen=False, collision_data_available=False,
                    departure=dict(phase='waiting_at_upstream_slots', ego_s_m=departure_s,
                                   link=settings['departure_link'], link_offset_m=departure_distance),
                    telemetry_gap_during_merge=False,
                    entry=None, completed=False, recycle_count=0, state_errors=0))
    _save(context)


def _save(context):
    flow = context.highway_flow
    path = Path(context.global_cfg['paths']['runtime_root']) / 'highway_merge_evaluation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flow['result'], indent=2), encoding='utf-8')


def _sample(state, points):
    p = state.transform.location
    v = state.velocity
    x, y = float(p.x), float(p.y)
    speed = math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2) / 3.6
    if not all(math.isfinite(value) for value in (x, y, speed)):
        raise ValueError('Non-finite traffic state')
    s = project_distance_on_polyline(points, x, y)
    px, py, _, _ = interpolate_on_polyline(points, s)
    return dict(x=x, y=y, s=s, speed_mps=speed, lane_distance_m=math.hypot(x-px, y-py))


def _recycle(context, item):
    from .highway_traffic import route_waypoints
    flow = context.highway_flow
    slot = flow['slot']
    # Only a far downstream, scenario-owned actor is removed. The same model
    # returns at the upstream slot; live model identities stay distinct.
    if not context.grpc.destroy_vehicle_by_id(item['actor_id']):
        raise RuntimeError('Cannot recycle highway actor: ' + item['actor_id'])
    context.scenario_actors.remove(item['actor'])
    flow['generation'] += 1
    actor = context.grpc.spawn_vehicle(
        transform=context.grpc.make_transform(slot['x'], slot['y'], slot['z'], slot['yaw_deg']),
        model_name=item['model'], label=item['label'], velocity=0., multi_ego=False, pause=True,
        request_id=f'{item["label"]}_{context.run_instance_id}_flow_{flow["generation"]}',
        recover_ambiguous=True)
    if actor is None:
        raise RuntimeError('Cannot replenish distinct adjacent highway model: ' + item['model'])
    context.scenario_actors.append(actor)
    if not context.grpc.set_vehicle_route(actor, slot['route'],
            route_waypoint_indices=route_waypoints(context, slot),
            decision_range=float(context.cfg.get('npc_decision_range_m', 30.)), label=item['label']):
        raise RuntimeError('Cannot assign replenished highway route')
    if not context.grpc.set_vehicle_speed_limit(actor, item['speed_kmh'] / 3.6, enabled=True):
        raise RuntimeError('Cannot assign replenished highway speed')
    ai = context.grpc.set_vehicle_ai(actor, True)
    if not actor.set_pause(False) or not context.grpc.set_vehicle_velocity(actor, item['speed_kmh']):
        raise RuntimeError('Cannot start replenished highway vehicle')
    item.update({key: value for key, value in slot.items() if key != 'speed_kmh'})
    item.update(actor=actor, actor_id=str(actor.get_id()), recycle_generation=flow['generation'],
                motion_contract=dict(ai_enable_acknowledged=bool(ai), unpause_acknowledged=True,
                                     velocity_acknowledged=True))
    flow['result']['recycle_count'] += 1
    from .highway_traffic import _persist
    _persist(context, 'flowing')


def update(context, elapsed, ego_state, current_s):
    if not enabled(context) or context.highway_traffic_cleared or not context.highway_spawn_points_active:
        return
    flow = context.highway_flow
    result, settings = flow['result'], flow['settings']
    if result['completed']:
        return
    states = context.grpc.get_all_vehicle_states()
    try:
        samples = {str(state.actor_info.id.value): _sample(state, flow['points']) for state in states}
        adjacent = [(item, samples[item['actor_id']]) for item in context.highway_npcs if item['merge_adjacent']]
        ex, ey = float(ego_state['x']), float(ego_state['y'])
        if not math.isfinite(ex) or not math.isfinite(ey):
            raise ValueError('Invalid Ego position')
    except (AttributeError, KeyError, TypeError, ValueError):
        result['state_errors'] += 1
        if current_s >= flow['merge_ego_s'] - 30:
            result['telemetry_gap_during_merge'] = True
        result['status'] = 'unverified_state_missing'
        _save(context)
        return  # No destructive action without a complete spatial snapshot.
    moving = [(item, sample) for item, sample in adjacent
              if sample['speed_mps'] >= 1. and sample['lane_distance_m'] <= 1.75
              and abs(sample['s'] - flow['merge_lane_s']) <= settings['observation_radius_m']]
    result['latest'] = dict(elapsed_s=elapsed, moving_distinct_models=sorted({i['model'] for i, _ in moving}),
        minimum_center_distance_m=min(math.hypot(s['x']-ex, s['y']-ey) for _, s in adjacent))
    if 'collision_objects' in ego_state:
        result['collision_data_available'] = True
        result['collision_seen'] |= bool(ego_state['collision_objects'])
    elif current_s >= flow['merge_ego_s'] - 30:
        result['telemetry_gap_during_merge'] = True
    ego_lane_s = project_distance_on_polyline(flow['points'], ex, ey)
    lx, ly, _, _ = interpolate_on_polyline(flow['points'], ego_lane_s)
    in_lane = math.hypot(ex-lx, ey-ly) <= 1.75
    if result['entry'] is None and current_s >= flow['merge_ego_s'] and in_lane:
        gaps = [s['s'] - ego_lane_s for _, s in moving]
        result['entry'] = dict(result['latest'], traffic_valid=len(moving) >= 4,
            front_center_gap_m=min((g for g in gaps if g >= 0), default=None),
            rear_center_gap_m=min((-g for g in gaps if g < 0), default=None))
        print(f'[HighwayMerge] entry: moving distinct models={len(moving)}, required=4')
    if current_s >= flow['merge_ego_s'] + 35:
        result['completed'] = True
        if not result['entry'] or not result['entry']['traffic_valid']:
            result['status'] = 'invalid_test_insufficient_moving_traffic'
        elif result['collision_seen']:
            result['status'] = 'failed_collision'
        elif not in_lane:
            result['status'] = 'failed_merge_not_completed'
        elif not result['collision_data_available']:
            result['status'] = 'unverified_collision_data_missing'
        elif result['telemetry_gap_during_merge']:
            result['status'] = 'unverified_telemetry_gap'
        else:
            result['status'] = 'passed_observed_merge'
        print('[HighwayMerge] ' + result['status'])
    else:
        slot = flow['slot']
        spawn_clear = (math.hypot(slot['x']-ex, slot['y']-ey) >= settings['minimum_ego_distance_m']
                       and all(math.hypot(s['x']-slot['x'], s['y']-slot['y']) >= settings['respawn_clearance_m']
                               for s in samples.values()))
        if spawn_clear:
            for item, sample in sorted(adjacent, key=lambda pair: -pair[1]['s']):
                if (sample['s'] >= flow['merge_lane_s'] + settings['recycle_after_merge_m']
                        and math.hypot(sample['x']-ex, sample['y']-ey) >= settings['minimum_ego_distance_m']):
                    _recycle(context, item)
                    break  # One replacement per snapshot; never reuse a stale clearance check.
    _save(context)
