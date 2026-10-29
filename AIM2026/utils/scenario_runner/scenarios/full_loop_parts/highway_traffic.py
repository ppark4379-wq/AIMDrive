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
    if config.get('required_total_vehicles') is not None and len(plans) != config['required_total_vehicles']:
        raise ValueError('Managed highway traffic must match the required total')
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
    fixed_model = config.get('fixed_model')
    if fixed_model:
        if fixed_model in EXCLUDED_MODELS | set(config.get('excluded_models', [])) or fixed_model not in context.vehicle_models():
            raise RuntimeError(f'Fixed highway NPC model is unavailable or excluded: {fixed_model}')
        candidates = [fixed_model] * len(plans)
    else:
        candidates = model_candidates(context)
    if config.get('guaranteed_large_vehicles') and not fixed_model:
        candidates = _assign_dense_models(context, plans, candidates)
    if len(candidates) < len(plans):
        raise RuntimeError(f'Need {len(plans)} distinct highway models; only {len(candidates)} available')
    if fixed_model == '2016_Hyundai_Universe':
        _fit_large_vehicle_slots(context, plans, _universe_box)
    elif fixed_model == '2013_Hyundai_Xcient(P520)':
        _fit_large_vehicle_slots(context, plans, _xcient_box)
    elif config.get('guaranteed_large_vehicles'):
        # Conservative envelope covering both supported large vehicles.
        def footprint(plan):
            return dict(_universe_box(plan), width_m=3.146)
        _fit_large_vehicle_slots(context, plans, footprint)
    remove_count = config.get('remove_passenger_vehicles', 0)
    removed_slots = []
    if remove_count:
        if remove_count != 4 or not config.get('fixed_dense_heavy'):
            raise ValueError('Passenger reduction requires the fixed dense heavy fleet')
        for lane in ('middle', 'left', 'merge_adjacent', 'middle'):
            eligible = [i for i, p in enumerate(plans) if p['lane'] == lane
                        and p['required_model'] not in ('2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)')]
            index = max(eligible, key=lambda i: plans[i]['route_distance_m'])
            removed_slots.append(plans.pop(index))
            candidates.pop(index)
    extra_buses = config.get('extra_buses', 0)
    if extra_buses:
        if type(extra_buses) is not int or not 0 < extra_buses <= 6:
            raise ValueError('Extra dense buses must be an integer from 0 to 6')
        bus = '2016_Hyundai_Universe'
        if bus not in context.vehicle_models() or bus in set(config.get('excluded_models', [])):
            raise RuntimeError('Additional Universe bus model is unavailable or excluded')
        if len(plans) + extra_buses > int(context.cfg['highway_npc']['collection_maximum_total_vehicles']):
            raise ValueError('Extra buses exceed configured highway vehicle cap')
        candidates = candidates[:len(plans)]
        for number in range(extra_buses):
            lane = ('merge_adjacent', 'middle', 'left')[number % 3]
            vacant = next((p for p in removed_slots if p['lane'] == lane), None)
            if vacant is not None:
                removed_slots.remove(vacant)
                tail, minimum_shift = vacant, 0
            else:
                tail = max((p for p in plans if p['lane'] == lane), key=lambda p: p['route_distance_m'])
                minimum_shift = 40
            extra = dict(tail, required_model=bus, additional_bus=True)
            # Fit only the new vehicle; do not move/reassign the original population.
            _fit_large_vehicle_slots(context, [extra],
                lambda p: dict(_universe_box(p), width_m=3.146), placed=list(plans), minimum_shift=minimum_shift)
            plans.append(extra)
            candidates.append(bus)
    if config.get('downstream_relocation_count'):
        _redistribute_downstream(context, plans, config['downstream_relocation_count'])
    return plans, candidates


def _redistribute_downstream(context, plans, count):
    """Keep two adjacent buses and move five northern-approach NPCs onto the southbound mainline."""
    from .tunnel_static_parts.layout import boxes_overlap
    if count != 5:
        raise ValueError('The dense downstream layout relocates exactly five approach NPCs')
    buses = [p for p in plans if p['merge_adjacent'] and p['required_model'] == '2016_Hyundai_Universe']
    if len(buses) != 3:
        raise ValueError('Expected three adjacent buses before redistribution')
    moved_bus = max(buses, key=lambda p: p['route_distance_m'])
    upstream = sorted((p for p in plans if p['offset_m'] <= -80), key=lambda p: p['offset_m'])[:count]
    if len(upstream) != count:
        raise ValueError('Not enough northern approach NPCs to relocate')
    moves = [(moved_bus, 'left', 'adjacent_bus_limit')] + [(p, p['lane'], 'upstream_to_downstream') for p in upstream]
    moving_ids = {id(p) for p, _, _ in moves}
    placed = [p for p in plans if id(p) not in moving_ids]
    lanes = {lane['name']: lane for lane in selected_lanes(context)}
    def footprint(p):
        return dict(_universe_box(p), width_m=3.146)
    for plan, destination, reason in moves:
        lane = lanes[destination]
        route = lane['route']
        lengths = [polyline_length(context.map_loader.get_link_points(link)) for link in route]
        origin = sum(lengths[:route.index(lane.get('offset_origin_link', route[0]))])
        old = dict(lane=plan['lane'], offset_m=plan['offset_m'], x=plan['x'], y=plan['y'])
        for offset in range(0, int(sum(lengths)-origin-40), 2):
            absolute = origin + offset
            distance = absolute
            for index, length in enumerate(lengths):
                if distance < length:
                    break
                distance -= length
            points = context.map_loader.get_link_points(route[index])
            x, y, z, yaw = interpolate_on_polyline(points, distance)
            if not -135 <= yaw <= -45:
                continue
            candidate = dict(plan, lane=destination, merge_adjacent=bool(lane.get('merge_adjacent')),
                route=route[index:], offset_origin_link=lane.get('offset_origin_link', route[0]),
                offset_m=float(offset), route_distance_m=absolute, spawn_link_s_m=distance,
                x=x, y=y, z=z, yaw_deg=yaw,
                waypoint=min(range(len(points)), key=lambda j: math.hypot(points[j][0]-x, points[j][1]-y)))
            gap = 20. if candidate['merge_adjacent'] else 30.
            if any(p['lane'] == destination and abs(p['route_distance_m']-absolute) < gap for p in placed):
                continue
            if any(boxes_overlap(footprint(candidate), footprint(p), margin_m=.25) for p in placed):
                continue
            candidate.pop('spawn_adjustment_m', None)
            candidate.pop('original_offset_m', None)
            plan.update(candidate, relocation=dict(reason=reason, original=old))
            placed.append(plan)
            break
        else:
            raise RuntimeError('No safe downstream spawn slot for ' + destination)


def _assign_dense_models(context, plans, candidates):
    """Reserve one merge slot and three background slots for buses/trucks."""
    large = [model for model in candidates if model in (
        '2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)')]
    if not large:
        raise RuntimeError('Dense traffic requires an available Universe bus or Xcient P520 truck')
    groups = context.cfg['highway_npc']['managed_traffic']['model_groups']
    heavy = set(groups['truck']) | set(groups['bus'])
    passenger = [model for model in candidates if model not in heavy]
    if len(passenger) < len(plans) - 4:
        raise RuntimeError('Dense traffic requires 16 available passenger vehicle models')
    rng = random.Random(int(context.run_seed) ^ 0x44454E5345)
    fixed_heavy = context.cfg['highway_npc']['managed_traffic'].get('fixed_dense_heavy', False)
    if fixed_heavy and set(large) != {'2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)'}:
        raise RuntimeError('Fixed heavy fleet requires both Universe and Xcient models')
    reserved = {'merge_adjacent': 1, 'middle': 2, 'left': 1}
    assigned = []
    for plan in plans:
        lane = plan['lane']
        if reserved.get(lane, 0):
            model = (('2013_Hyundai_Xcient(P520)' if lane == 'middle' else '2016_Hyundai_Universe')
                     if fixed_heavy else rng.choice(large))
            reserved[lane] -= 1
        else:
            model = passenger.pop(0)
        plan['required_model'] = model
        assigned.append(model)
    if any(reserved.values()):
        raise ValueError('Dense traffic is missing required large-vehicle lane slots')
    return assigned


def _universe_box(plan):
    # Measured from paused Universe GT: half extents (1.4363, 6.0153),
    # bbox centre 2.775m ahead of the spawn pivot. Yaw points forward.
    yaw = math.radians(plan['yaw_deg'])
    return dict(center_xyz=[plan['x'] + 2.775 * math.cos(yaw),
                            plan['y'] + 2.775 * math.sin(yaw), plan['z']],
                heading_deg=plan['yaw_deg'], width_m=2.873, length_m=12.031)


def _xcient_box(plan):
    # Paused P520 GT from highway-20261028-214114-869117:
    # half extents (1.57286, 3.47366), centre 2.6865m ahead of spawn pivot.
    yaw = math.radians(plan['yaw_deg'])
    return dict(center_xyz=[plan['x'] + 2.687 * math.cos(yaw),
                            plan['y'] + 2.687 * math.sin(yaw), plan['z']],
                heading_deg=plan['yaw_deg'], width_m=3.146, length_m=6.948)


def _fit_large_vehicle_slots(context, plans, footprint, placed=None, minimum_shift=0):
    """Move overlapping large-vehicle slots along their own routes before spawn."""
    from .tunnel_static_parts.layout import boxes_overlap
    placed = [] if placed is None else placed
    for plan in plans:
        original = dict(plan)
        placed_before = len(placed)
        for shift in range(minimum_shift, 81, 2):
            distance = original['spawn_link_s_m'] + shift
            remaining = sum(polyline_length(context.map_loader.get_link_points(link))
                            for link in original['route'])
            if distance > remaining - 40:
                break
            for index, link in enumerate(original['route']):
                points = context.map_loader.get_link_points(link)
                length = polyline_length(points)
                if distance < length:
                    break
                distance -= length
            x, y, z, yaw = interpolate_on_polyline(points, distance)
            candidate = dict(original, x=x, y=y, z=z, yaw_deg=yaw,
                route=original['route'][index:], spawn_link_s_m=distance,
                route_distance_m=original['route_distance_m'] + shift,
                offset_m=original['offset_m'] + shift,
                waypoint=min(range(len(points)),
                             key=lambda j: math.hypot(points[j][0]-x, points[j][1]-y)))
            gap = 20. if plan['merge_adjacent'] else 30.
            if any(other['lane'] == plan['lane'] and
                   abs(candidate['route_distance_m'] - other['route_distance_m']) < gap
                   for other in placed):
                continue
            if any(boxes_overlap(footprint(candidate), footprint(other), margin_m=.25)
                   for other in placed):
                continue
            if shift:
                candidate['spawn_adjustment_m'] = shift
                candidate['original_offset_m'] = original['offset_m']
            plan.update(candidate)
            placed.append(plan)
            break
        if len(placed) == placed_before:
            raise RuntimeError('No collision-free large-vehicle spawn slot within 80m')
        if len(placed) == 0 or placed[-1] is not plan:
            raise RuntimeError('Large-vehicle spawn adjustment reached route endpoint')


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
        merge_sync=getattr(context, 'highway_merge_sync', None),
        lane_removals=getattr(context, 'highway_lane_removals', []), vehicles=records), indent=2), encoding='utf-8')


def route_waypoints(context, plan):
    """Keep the spawn waypoint and explicitly target the end of the final link."""
    route = plan['route']
    return {route[0]: plan['waypoint'],
            route[-1]: len(context.map_loader.get_link_points(route[-1])) - 1}


def update_lane_lifecycle(context, elapsed):
    """Keep dense NPCs on their assigned lane; delete lane-drop traffic before the taper."""
    if context.highway_traffic_cleared or not context.highway_spawn_points_active:
        return
    from . import highway_competition
    states = highway_competition._states(context)
    if not states:
        return
    cache = getattr(context, '_dense_lane_geometry', None)
    if cache is None:
        cache = {}
        for lane in selected_lanes(context):
            points = [p for link in lane['route'] for p in context.map_loader.get_link_points(link)]
            cumulative = [0.]
            for a, b in zip(points, points[1:]):
                cumulative.append(cumulative[-1] + math.dist(a[:2], b[:2]))
            cache[lane['name']] = dict(lane, xy=[p[:2] for p in points], cum=cumulative)
        context._dense_lane_geometry = cache
    for item in list(context.highway_npcs):
        sample = states.get(item['actor_id'])
        if sample is None:
            continue
        lane = cache[item['lane']]
        hint, error = highway_competition._local_lateral(
            lane, item.get('lane_hint'), sample['x'], sample['y'])
        item['lane_hint'] = hint
        remaining = lane['cum'][-1] - lane['cum'][hint]
        reason = None
        if lane.get('retire_at_route_end') and remaining <= float(lane.get('lane_end_margin_m', 30.)):
            reason = 'lane_end'
        elif error > 1.9:
            reason = 'lane_departure'
        if reason is None:
            continue
        if not context.grpc.destroy_vehicle_by_id(item['actor_id']):
            raise RuntimeError('Cannot remove highway NPC before lane change: ' + item['label'])
        context.highway_npcs.remove(item)
        if item['actor'] in context.scenario_actors:
            context.scenario_actors.remove(item['actor'])
        record = dict(actor_id=item['actor_id'], lane=item['lane'], reason=reason,
                      elapsed_s=elapsed, remaining_m=remaining, lateral_error_m=error)
        if not hasattr(context, 'highway_lane_removals'):
            context.highway_lane_removals = []
        context.highway_lane_removals.append(record)
        print(f'[HighwayTraffic] {reason.upper()}_REMOVED {item["label"]}', flush=True)
        _persist(context, 'lane_retirement')


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
    fixed_model = context.cfg['highway_npc']['managed_traffic'].get('fixed_model')
    for index, plan in enumerate(plans):
        label = f'highway_{plan["lane"]}_{index}'
        actor = None
        required_model = plan.get('required_model') or fixed_model
        slot_candidates = [required_model] if required_model else candidates
        while slot_candidates:
            model = slot_candidates.pop(0)
            actor = context.grpc.spawn_vehicle(
                transform=context.grpc.make_transform(plan['x'], plan['y'], plan['z'], plan['yaw_deg']),
                model_name=model, label=label, velocity=0., multi_ego=False, pause=True,
                request_id=f'{label}_{context.run_instance_id}_{model}', recover_ambiguous=True,
                lookup_before_create=False)
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f'Cannot fill highway slot (fixed_model={fixed_model!r}): slot={index}, '
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
            raise RuntimeError(
                f'Cannot prepare highway vehicle: {label}; model={model}; '
                f'pause_ok={pause_ok!r}, route_ok={route_ok!r}, limit_ok={limit_ok!r}')
        context.highway_npcs.append(dict(plan, actor=actor, actor_id=str(actor.get_id()),
                                         model=model, label=label))
    context.highway_npc_planned_count = len(context.highway_npcs)
    _persist(context, 'prepared_paused')
    print(f'[HighwayTraffic] prepared {len(context.highway_npcs)} moving NPCs; '
          f'unique_models={len({p["model"] for p in context.highway_npcs})}; '
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
