"""Explicit, unique-model highway traffic prepared before Ego starts driving."""
import json
import math
import random
import copy
from pathlib import Path

from ...utils.transform_utils import (interpolate_on_polyline, polyline_length,
                                     project_distance_on_polyline)


EXCLUDED_MODELS = frozenset(('SE_CT797F_2009', 'RandomVehicle', 'AR_DJMini3pro_2022'))


def enabled(context):
    return bool(context.cfg.get('highway_npc', {}).get('managed_traffic', {}).get('enabled', False))


def model_candidates(context):
    config = context.cfg['highway_npc']['managed_traffic']
    excluded = EXCLUDED_MODELS | set(config.get('excluded_models', []))
    available = set(context.vehicle_models()) - excluded
    rng = random.Random(int(context.run_seed) ^ 0x4857444956455253)
    groups = []
    used = set()
    for names in config['model_groups'].values():
        group = []
        for name in names:
            if name in available and name not in used:
                group.append(name)
                used.add(name)
        rng.shuffle(group)
        groups.append(group)
    # Interleave body types so the merge-adjacent lane also gets a varied mix.
    candidates = []
    while any(groups):
        for group in groups:
            if group:
                candidates.append(group.pop())
    return candidates


def selected_lanes(context):
    config = context.cfg['highway_npc']['managed_traffic']
    lanes = copy.deepcopy(config['lanes'])
    if config.get('randomize_adjacent_count', False):
        count = random.Random(int(context.run_seed) ^ 0x48574E).choice([4, 5, 6])
        adjacent = next(lane for lane in lanes if lane.get('merge_adjacent'))
        if len(adjacent['offsets_m']) != 6:
            raise ValueError('Random adjacent traffic requires six candidate slots')
        adjacent['offsets_m'] = adjacent['offsets_m'][:count]
        # Transfer unused slots to background traffic; keep the total fixed.
        for lane in [lane for lane in lanes if not lane.get('merge_adjacent')][:6-count]:
            lane['offsets_m'] = sorted(lane['offsets_m'] + [lane['additional_upstream_offset_m']])
    return lanes


def build_plan(context):
    config = context.cfg['highway_npc']['managed_traffic']
    lanes = selected_lanes(context)
    reference = context.map_loader.get_link(config['ego_merge_link'])
    adjacent = reference.get('left_lane_change_dst_link_idx')
    if not adjacent or not reference.get('can_move_left_lane'):
        raise ValueError('Highway merge reference has no legal left lane')
    plans = []
    adjacent_count = 0
    for lane in lanes:
        route = list(lane['route'])
        if set(route) & set(config.get('forbidden_route_links', [])):
            raise ValueError('Highway NPC route enters the separate Ego on-ramp')
        segments = []
        distance = 0.
        for i, link_id in enumerate(route):
            link = context.map_loader.get_link(link_id)
            points = context.map_loader.get_link_points(link_id)
            length = polyline_length(points)
            if length <= 0:
                raise ValueError('Empty highway lane geometry: ' + link_id)
            if i:
                previous = context.map_loader.get_link(route[i - 1])
                if (previous['to_node_idx'] != link['from_node_idx']
                        or math.dist(previous['points'][-1][:3], points[0][:3]) > 1.):
                    raise ValueError('Disconnected highway lane route: ' + link_id)
            segments.append((distance, distance + length, i, points))
            distance += length
        offsets = list(map(float, lane['offsets_m']))
        if not offsets or any(not math.isfinite(x) for x in offsets):
            raise ValueError('Highway offsets must be finite and non-empty')
        origin_link = lane.get('offset_origin_link', route[0])
        if origin_link not in route:
            raise ValueError('Highway offset origin is outside the lane route')
        origin_s = segments[route.index(origin_link)][0]
        absolute_offsets = [origin_s + offset for offset in offsets]
        if absolute_offsets[0] < 25 or absolute_offsets[-1] > distance - 40:
            raise ValueError('Highway traffic is too close to a lane endpoint')
        minimum_gap = 20. if lane.get('merge_adjacent', False) else 30.
        if any(b - a < minimum_gap for a, b in zip(offsets, offsets[1:])):
            raise ValueError(f'Highway traffic requires at least {minimum_gap:g}m center spacing')
        speed = float(lane['speed_kmh'])
        if not math.isfinite(speed) or not 0 < speed <= 100:
            raise ValueError('Highway traffic speed must be in (0, 100] km/h')
        is_adjacent = bool(lane.get('merge_adjacent', False))
        if is_adjacent:
            if adjacent not in route:
                raise ValueError('Merge-adjacent traffic must follow the actual left lane')
            adjacent_count += len(offsets)
        for offset, absolute_s in zip(offsets, absolute_offsets):
            start, _, index, points = next(row for row in segments if absolute_s < row[1])
            x, y, z, yaw = interpolate_on_polyline(points, absolute_s - start)
            waypoint = min(range(len(points)), key=lambda j: math.hypot(points[j][0] - x, points[j][1] - y))
            plans.append(dict(lane=lane['name'], merge_adjacent=is_adjacent,
                              route=route[index:], waypoint=waypoint, offset_m=offset,
                              offset_origin_link=origin_link, route_distance_m=absolute_s,
                              spawn_link_s_m=absolute_s - start,
                              x=x, y=y, z=z, yaw_deg=yaw, speed_kmh=speed))
    # Different destination lanes can share a one-lane upstream approach.
    # Check that their spawn slots do not overlap on that shared link.
    for index, first in enumerate(plans):
        for second in plans[index + 1:]:
            minimum_gap = 20. if first['merge_adjacent'] and second['merge_adjacent'] else 30.
            if (first['route'][0] == second['route'][0]
                    and abs(first['spawn_link_s_m'] - second['spawn_link_s_m']) < minimum_gap - 1e-6):
                raise ValueError(f'Highway shared approach requires at least {minimum_gap:g}m center spacing')
            # Distinct map links can still converge at the fork. Link IDs alone
            # do not guarantee enough physical room for side-by-side vehicles.
            if math.hypot(first['x'] - second['x'], first['y'] - second['y']) < 5.:
                raise ValueError('Highway spawn positions require at least 5m physical separation: '
                                 f'{first["lane"]}@{first["offset_m"]}, '
                                 f'{second["lane"]}@{second["offset_m"]}')
    # The old default was 12 moving vehicles plus one independent stopped car.
    minimum = max(15, int(config.get('previous_total_vehicles', 13)) + 2)
    if len(plans) < minimum or adjacent_count < 4:
        raise ValueError('Highway requires at least old total + 2 vehicles and 4 adjacent-lane vehicles')
    if len(plans) > int(context.cfg['highway_npc']['collection_maximum_total_vehicles']):
        raise ValueError('Managed highway traffic exceeds configured cap')
    speed_range = config.get('speed_range_kmh')
    if speed_range is not None:
        if (len(speed_range) != 2 or any(type(v) is not int for v in speed_range)
                or not 20 <= speed_range[0] <= speed_range[1] <= 100):
            raise ValueError('Highway speed range must contain integer bounds within 20..100 km/h')
        rng = random.Random(int(context.run_seed) ^ 0x48575350454544)
        # Independent from model/count RNGs. Shuffle speed choices so a run
        # covers varied speeds instead of accidentally assigning one speed.
        choices = list(range(speed_range[0], speed_range[1] + 1))
        rng.shuffle(choices)
        for index, plan in enumerate(plans):
            plan['speed_kmh'] = float(choices[index % len(choices)])
    candidates = model_candidates(context)
    if len(candidates) < len(plans):
        raise RuntimeError(f'Need {len(plans)} distinct highway models; only {len(candidates)} available')
    return plans, candidates


def _prepare_merge_sync(context, plans):
    cfg = context.cfg['highway_npc']['managed_traffic'].get('merge_sync', {})
    context.highway_merge_sync = None
    if not cfg.get('enabled', False):
        return
    values = {key: float(cfg[key]) for key in (
        'lane_entry_distance_m', 'arm_distance_m', 'minimum_ego_speed_mps',
        'lead_pass_before_ego_s', 'npc_speed_ratio', 'minimum_npc_speed_kmh',
        'maximum_npc_speed_kmh')}
    if (any(not math.isfinite(v) or v <= 0 for v in values.values())
            or not values['minimum_npc_speed_kmh'] <= values['maximum_npc_speed_kmh'] <= 60):
        raise ValueError('Invalid highway merge synchronization settings')
    reference = context.map_loader.get_link(
        context.cfg['highway_npc']['managed_traffic']['ego_merge_link'])
    lane_points = context.map_loader.get_link_points(reference['left_lane_change_dst_link_idx'])
    lane_length = polyline_length(lane_points)
    # Find the first actual entry into the target lane on the Expert path,
    # rather than using the old traffic activation checkpoint.
    target = None
    ego_s = 0.
    min_y, max_y = min(p[1] for p in lane_points), max(p[1] for p in lane_points)
    for index, point in enumerate(context.route_points):
        if index:
            ego_s += math.dist(context.route_points[index - 1][:2], point[:2])
        if not (min_y - 2 <= point[1] <= max_y + 2):
            continue
        lane_s = project_distance_on_polyline(lane_points, *point[:2])
        x, y, _, _ = interpolate_on_polyline(lane_points, lane_s)
        if 20 < lane_s < lane_length - 10 and math.hypot(point[0] - x, point[1] - y) <= values['lane_entry_distance_m']:
            target = (ego_s, x, y)
            break
    if target is None:
        raise ValueError('Expert route has no identifiable highway left-lane merge entry')
    adjacent = [p for p in plans if p['merge_adjacent']]
    for plan in adjacent:
        route_points = [p for link in plan['route'] for p in context.map_loader.get_link_points(link)]
        plan['merge_travel_m'] = project_distance_on_polyline(route_points, target[1], target[2]) - plan['spawn_link_s_m']
    if any(p['merge_travel_m'] <= 0 for p in adjacent):
        raise ValueError('Synchronized adjacent NPC must start upstream of the merge')
    context.highway_merge_sync = dict(values, merge_ego_s_m=target[0],
        lead_travel_m=min(p['merge_travel_m'] for p in adjacent), phase='waiting_for_ego')


def _persist(context, phase):
    output = Path(context.global_cfg['paths']['runtime_root']) / 'highway_traffic.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [{key: value for key, value in item.items() if key != 'actor'}
               for item in context.highway_npcs]
    output.write_text(json.dumps(dict(phase=phase, seed=context.run_seed,
        count=len(records), distinct_models=len({x['model'] for x in records}),
        adjacent_lane_count=sum(x['merge_adjacent'] for x in records),
        excluded_models=sorted(EXCLUDED_MODELS),
        merge_sync=getattr(context, 'highway_merge_sync', None), vehicles=records), indent=2), encoding='utf-8')


def route_waypoints(context, plan):
    """Keep the spawn waypoint and explicitly target the end of the final link."""
    route = plan['route']
    return {route[0]: plan['waypoint'],
            route[-1]: len(context.map_loader.get_link_points(route[-1])) - 1}


def prepare(context):
    from . import highway_competition
    if highway_competition.enabled(context):
        return highway_competition.prepare(context)
    plans, candidates = build_plan(context)
    _prepare_merge_sync(context, plans)
    from . import highway_flow
    highway_flow.prepare(context, plans)
    context.highway_fixed_model = context.cfg['highway_npc'].get('model', 'DEFAULT_SEDAN')
    context.highway_npcs = []
    context.highway_spawn_points = []
    context.highway_spawn_points_active = False
    for index, plan in enumerate(plans):
        label = f'highway_{plan["lane"]}_{index}'
        actor = None
        while candidates:
            model = candidates.pop(0)
            actor = context.grpc.spawn_vehicle(
                transform=context.grpc.make_transform(plan['x'], plan['y'], plan['z'], plan['yaw_deg']),
                model_name=model, label=label, velocity=0., multi_ego=False, pause=True,
                request_id=f'{label}_{context.run_instance_id}_{model}', recover_ambiguous=True)
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f'Cannot fill all highway slots with distinct models: slot={index}, '
                               f'lane={plan["lane"]}, link={plan["route"][0]}, '
                               f'position=({plan["x"]:.3f}, {plan["y"]:.3f}, {plan["z"]:.3f})')
        # Register immediately so any setup failure is cleaned by the owner.
        context.scenario_actors.append(actor)
        pause_ok = actor.set_pause(True)
        route_ok = context.grpc.set_vehicle_route(actor, plan['route'],
            route_waypoint_indices=route_waypoints(context, plan),
            decision_range=float(context.cfg.get('npc_decision_range_m', 30.)), label=label)
        limit_ok = context.grpc.set_vehicle_speed_limit(actor, plan['speed_kmh'] / 3.6, enabled=True)
        if not (pause_ok and route_ok and limit_ok):
            raise RuntimeError(f'Cannot prepare highway vehicle: {label}')
        context.highway_npcs.append(dict(plan, actor=actor, actor_id=str(actor.get_id()),
                                         model=model, label=label))
    context.highway_npc_planned_count = len(context.highway_npcs)
    _persist(context, 'prepared_paused')
    print(f'[HighwayTraffic] prepared {len(context.highway_npcs)} distinct moving NPCs; '
          f'adjacent={sum(p["merge_adjacent"] for p in plans)}; '
          f'departure={"approach_curve_trigger" if highway_flow.enabled(context) else "activation_trigger"}')
    print(f'[HighwayTraffic] target speeds km/h: {[p["speed_kmh"] for p in context.highway_npcs]}')


def activate(context, current_s, ego_state=None, elapsed=None):
    from . import highway_competition
    if highway_competition.enabled(context):
        # Released only by the pre-warm before Ego starts, never by Ego's position; its own manager removes NPCs.
        return bool(context.highway_spawn_points_active)
    if context.highway_spawn_points_active:
        return True
    config = context.cfg['highway_npc']
    threshold = max(0., context.highway_activation_s - float(config.get('background_warmup_distance_m', 100.)))
    from . import highway_flow
    if highway_flow.enabled(context):
        threshold = context.highway_flow['departure_ego_s']
    if current_s < threshold or context.highway_traffic_cleared:
        return False
    sync = getattr(context, 'highway_merge_sync', None)
    if sync:
        if not ego_state:
            return False
        speed = float(ego_state.get('speed_mps', ego_state.get('speed', 0.)))
        if not math.isfinite(speed) or speed < sync['minimum_ego_speed_mps']:
            return False
        remaining = sync['merge_ego_s_m'] - current_s
        if remaining > sync['arm_distance_m']:
            return False
        speed_kmh = min(sync['maximum_npc_speed_kmh'], max(
            sync['minimum_npc_speed_kmh'], speed * 3.6 * sync['npc_speed_ratio']))
        ego_eta = max(0., remaining) / speed
        lead_eta = sync['lead_travel_m'] / (speed_kmh / 3.6)
        if ego_eta > lead_eta + sync['lead_pass_before_ego_s']:
            return False
        sync.update(phase='released', release_elapsed_s=elapsed,
                    release_ego_s_m=current_s, ego_speed_mps=speed,
                    ego_eta_s=ego_eta, lead_npc_eta_s=lead_eta, npc_speed_kmh=speed_kmh)
        # Latch the launch once. Subsequent Ego braking is part of the merge
        # test; do not teleport, pause, or accelerate traffic to chase Ego.
        for item in context.highway_npcs:
            if item['merge_adjacent']:
                item['speed_kmh'] = speed_kmh
                item['predicted_pass_relative_to_ego_s'] = item['merge_travel_m'] / (speed_kmh / 3.6) - ego_eta
                if not context.grpc.set_vehicle_speed_limit(item['actor'], speed_kmh / 3.6, enabled=True):
                    raise RuntimeError('Cannot set synchronized highway NPC speed limit')
        print(f'[HighwayTraffic] MERGE_SYNC ego_eta={ego_eta:.1f}s '
              f'lead_npc_eta={lead_eta:.1f}s adjacent_speed={speed_kmh:.1f}km/h')
    context.highway_spawn_point_activation_started = True
    for item in context.highway_npcs:
        actor = item['actor']
        # Same contract as other route-following NPCs: limit=m/s, velocity=km/h.
        pause_ok = actor.set_pause(False)
        ai_ok = context.grpc.set_vehicle_ai(actor, True)
        velocity_ok = context.grpc.set_vehicle_velocity(actor, item['speed_kmh'])
        # Match roundabout NPCs: a rejected SetAI acknowledgement does not
        # establish that the actor's native route-following AI is disabled.
        # Route/speed-limit were required during prepare; unpause and velocity
        # still must succeed. Preserve the acknowledgement for live diagnosis.
        item['motion_contract'] = dict(unpause_acknowledged=bool(pause_ok),
            ai_enable_acknowledged=bool(ai_ok), velocity_acknowledged=bool(velocity_ok))
        if not (pause_ok and velocity_ok):
            raise RuntimeError(f'Cannot start highway vehicle: {item["label"]}; '
                               f'unpause={pause_ok}, velocity={velocity_ok}, ai_ack={ai_ok}')
        if not ai_ok:
            print(f'[HighwayTraffic] {item["label"]}: SetAI unacknowledged; '
                  'using assigned route and native AI (unpause/velocity accepted)')
    context.highway_spawn_points_active = True
    if highway_flow.enabled(context):
        context.highway_flow['result']['departure'].update(
            phase='moving', actual_ego_s_m=current_s, elapsed_s=elapsed)
        highway_flow._save(context)
        print(f'[HighwayTraffic] APPROACH_DEPARTURE ego_s={current_s:.1f}m threshold={threshold:.1f}m')
    context.spawn_highway_merge_static_vehicle()
    _persist(context, 'active')
    print(f'[HighwayTraffic] ACTIVE count={len(context.highway_npcs)}, '
          f'unique_models={len({p["model"] for p in context.highway_npcs})}')
    return True
