"""Competition highway traffic (``--highway-density competition``): NPCs that run independently of Ego.

Unlike ``dense`` this policy never ties traffic to Ego's position:

* every NPC is spawned paused, released *before* Ego is switched to AUTO, and Ego only starts once real movement of the
  fleet has been measured (pre-warm);
* one target speed per lane group (``lane_speeds_kmh``); the per-vehicle random speeds and the downstream-lead override
  of ``dense`` are not used and must be null in this policy;
* NPCs keep their assigned lane route, nothing brakes hard, cuts in or changes lanes;
* a lane that ends in a taper deletes its NPCs before the taper (``retire_at_route_end``); NPCs of the other lanes are
  only removed at their far route end, away from Ego;
* removed NPCs leave the population and are replaced, one at a time, at the lane's upstream supply slot when that slot
  is outside every no-spawn zone, far from Ego and clear of other vehicles. Shortages are logged with the reason;
* the manager keeps running for the whole highway section (it is independent of the merge evaluation);
* Ego-synchronised feed: fixed placements cannot follow Ego's real (slow, irregular) progress, so the manager feeds
  traffic from Ego's measured position and speed. Before the merge it schedules NPCs upstream so that they pass the merge
  entry around Ego's predicted arrival time (dense traffic when Ego enters the lane); afterwards it keeps the stretch
  140..260 m ahead of Ego populated (overtaking). Nothing is ever created closer than 140 m to Ego (100 m on the
  separate approach road); NPCs far ahead of or behind Ego are recycled to stay inside the cap.
"""
import bisect
import json
import math
import random
import time
from pathlib import Path

from ...utils.transform_utils import (interpolate_on_polyline, polyline_length,
                                     project_distance_on_polyline)
from . import highway_traffic

POLICY = 'competition'
_LANE_ORDER_XOR = 0x434F4D50_4C414E45
_FEED_XOR = 0x46454544_4D455247
_REPORT = 'highway_competition.json'
_OBSERVATION_LIMIT = 900
_EVENT_LIMIT = 300

DEFAULTS = dict(
    cap=25,
    minimum_spacing_m=30.,
    minimum_physical_separation_m=8.,
    prewarm=dict(minimum_s=4., verify_timeout_s=25., poll_s=0.5, minimum_travel_m=3., minimum_moving_fraction=0.9),
    respawn=dict(clearance_m=22., body_clearance_m=12., minimum_ego_distance_m=90., minimum_ego_progress_m=150.,
                 minimum_lane_interval_s=1.5, max_model_attempts=3),
    removal=dict(end_tolerance_m=5., lane_end_margin_m=30., terminal_minimum_ego_distance_m=60., failures_before_abort=3,
                 stall_s=4., stall_speed_mps=0.5, stall_grace_s=6.),
    lane_keeping=dict(max_lateral_error_m=1.9, violation_hold_s=1.),
    observation=dict(period_s=1., radius_m=150., corridor_half_width_m=15., route_lateral_m=25.),
    feed=dict(enabled=True, interval_s=0.3, min_distance_m=140., merge_min_distance_m=60., ahead_window_m=300.,
              target_ahead=20, speed_floor_mps=8., merge_before_s=15., merge_after_s=15., min_headway_s=1.6,
              merge_per_lane={}, lane_weights={}, recycle_far_ahead_m=600., recycle_far_behind_m=300., recycle_min_ego_distance_m=250.,
              reserve=10, route_lateral_m=25.),
    rear_reduction=dict(enabled=False, min_distance_m=80., keep_behind=2, keep_range_m=300., max_per_step=3),
    update_period_s=0.1,      # a 5 m end window must not be skipped: 75 km/h covers ~2 m per update
    persist_period_s=5.,
)


def _managed(context):
    return context.cfg.get('highway_npc', {}).get('managed_traffic', {})


def enabled(context):
    managed = _managed(context)
    return bool(managed.get('enabled', False)) and managed.get('policy') == POLICY


def has_state(context):
    return getattr(context, 'highway_competition', None) is not None


def _merged(name, given):
    values = dict(DEFAULTS[name])
    values.update(given or {})
    return values


def settings(context):
    """The competition section merged with defaults and validated."""
    managed = _managed(context)
    raw = managed.get('competition')
    if not isinstance(raw, dict):
        raise ValueError('Competition highway traffic needs highway_npc.managed_traffic.competition')
    # The dense speed policies would silently overwrite lane speeds; they must be switched off explicitly.
    for key in ('speed_range_kmh', 'downstream_lead_speed_kmh'):
        if managed.get(key) is not None:
            raise ValueError(f'Competition highway traffic ignores {key}; set it to null')
    if managed.get('flowing_traffic', {}).get('enabled') or managed.get('merge_sync', {}).get('enabled'):
        raise ValueError('Competition highway traffic cannot use flowing_traffic or merge_sync')
    cfg = dict(raw)
    for name in ('prewarm', 'respawn', 'removal', 'lane_keeping', 'observation', 'feed', 'rear_reduction'):
        cfg[name] = _merged(name, raw.get(name))
    for key in ('cap', 'minimum_spacing_m', 'minimum_physical_separation_m', 'update_period_s', 'persist_period_s'):
        cfg[key] = raw.get(key, DEFAULTS[key])
    cap = int(cfg['cap'])
    maximum = int(context.cfg['highway_npc'].get('collection_maximum_total_vehicles', cap))
    if not 1 <= cap <= maximum:
        raise ValueError(f'Competition highway cap must be within 1..{maximum}')
    cfg['cap'] = cap
    for key in ('minimum_spacing_m', 'minimum_physical_separation_m', 'update_period_s', 'persist_period_s'):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f'Invalid competition setting: {key}')
    speeds = raw.get('lane_speeds_kmh')
    if not isinstance(speeds, dict) or not speeds:
        raise ValueError('Competition highway traffic needs lane_speeds_kmh')
    cfg['lane_speeds_kmh'] = {str(name): float(value) for name, value in speeds.items()}
    for name, value in cfg['lane_speeds_kmh'].items():
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError(f'Lane speed must be within (0, 100] km/h: {name}')
    cfg['no_spawn_zones'] = list(raw.get('no_spawn_zones') or [])
    return cfg


# ----------------------------------------------------------------------------- geometry / zones

def _link_width(link):
    widths = [float(link[key]) for key in ('width_start', 'width_end') if link.get(key)]
    return sum(widths) / len(widths) if widths else 3.5


def zones(context, cfg):
    """No-spawn zones as lane bands: link centre line plus the real lane width and a vehicle-width margin."""
    result = []
    for index, zone in enumerate(cfg['no_spawn_zones']):
        links = list(zone.get('links') or ([zone['link']] if zone.get('link') else []))
        if not links:
            raise ValueError(f'No-spawn zone {index} needs link or links')
        lateral = float(zone.get('lateral_margin_m', 1.0))
        longitudinal = float(zone.get('longitudinal_margin_m', 5.))
        if lateral < 0 or longitudinal < 0 or not math.isfinite(lateral + longitudinal):
            raise ValueError(f'Invalid no-spawn margin: zone {index}')
        bands = []
        for link_id in links:
            link = context.map_loader.get_link(link_id)
            points = context.map_loader.get_link_points(link_id)
            length = polyline_length(points)
            start = float(zone.get('start_s_m', 0.))
            end = float(zone.get('end_s_m', length))
            if not 0 <= start < end <= length + 1e-6:
                raise ValueError(f'No-spawn zone range lies outside link {link_id}')
            bands.append(dict(link=link_id, points=points, start=start - longitudinal, end=end + longitudinal,
                              half_width=_link_width(link) / 2. + lateral))
        result.append(dict(name=str(zone.get('name', links[0])), bands=bands))
    return result


def zone_containing(zone_list, x, y):
    """Name of the no-spawn zone whose lane band holds (x, y), else None. Adjacent lanes stay outside the band."""
    for zone in zone_list:
        for band in zone['bands']:
            s = project_distance_on_polyline(band['points'], x, y)
            px, py, _, _ = interpolate_on_polyline(band['points'], s)
            if band['start'] <= s <= band['end'] and math.hypot(x - px, y - py) <= band['half_width']:
                return zone['name']
    return None


def _route_geometry(context, route, forbidden):
    if set(route) & set(forbidden):
        raise ValueError('Highway NPC route enters the separate Ego on-ramp')
    segments, points, distance = [], [], 0.
    for i, link_id in enumerate(route):
        link = context.map_loader.get_link(link_id)
        link_points = context.map_loader.get_link_points(link_id)
        length = polyline_length(link_points)
        if length <= 0:
            raise ValueError('Empty highway lane geometry: ' + link_id)
        if i:
            previous = context.map_loader.get_link(route[i - 1])
            if (previous['to_node_idx'] != link['from_node_idx']
                    or math.dist(previous['points'][-1][:3], link_points[0][:3]) > 1.):
                raise ValueError('Disconnected highway lane route: ' + link_id)
        segments.append((distance, distance + length, i, link_points))
        points.extend(link_points)
        distance += length
    return segments, points, distance


def _slot(context, route, segments, absolute_s):
    start, _, index, link_points = next(row for row in segments if absolute_s < row[1])
    x, y, z, yaw = interpolate_on_polyline(link_points, absolute_s - start)
    waypoint = min(range(len(link_points)), key=lambda j: math.hypot(link_points[j][0] - x, link_points[j][1] - y))
    return dict(route=route[index:], waypoint=waypoint, route_distance_m=absolute_s,
                spawn_link_s_m=absolute_s - start, x=x, y=y, z=z, yaw_deg=yaw)


def _local_lateral(lane, hint, x, y, window=80):
    """Distance from (x, y) to the lane route, searching only around the previous nearest vertex.

    A full ``project_distance_on_polyline`` over a 2700-point lane route costs ~3 ms; 25 NPCs at 10 Hz would take most
    of a core. Returns (nearest vertex index, lateral distance); falls back to a full search when the vehicle is far.
    """
    xy = lane['xy']
    last = len(xy) - 1
    spans = [(0, last)] if hint is None else [(max(0, hint - window), min(last, hint + window)), (0, last)]
    for low, high in spans:
        best = min(range(low, high + 1), key=lambda j: (xy[j][0] - x) ** 2 + (xy[j][1] - y) ** 2)
        lateral = min((_segment_distance(xy[j], xy[j + 1], x, y) for j in (best - 1, best) if 0 <= j < last),
                      default=math.hypot(xy[best][0] - x, xy[best][1] - y))
        at_edge = (best == low and low > 0) or (best == high and high < last)
        if high - low == last or (lateral <= 10. and not at_edge):
            return best, lateral
    return best, lateral


def _segment_distance(a, b, x, y):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = 0. if length == 0. else max(0., min(1., ((x - a[0]) * dx + (y - a[1]) * dy) / length))
    return math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy))


def _offsets(values, name, spacing):
    offsets = [float(v) for v in values or ()]
    if any(not math.isfinite(v) for v in offsets):
        raise ValueError(f'Competition offsets must be finite: {name}')
    if any(b - a < spacing - 1e-6 for a, b in zip(offsets, offsets[1:])):
        raise ValueError(f'Competition lane {name} requires at least {spacing:g}m centre spacing')
    return offsets


def build_plan(context):
    """Initial slots, per-lane state and the seeded model order. Raises before anything is spawned."""
    cfg = settings(context)
    managed = _managed(context)
    forbidden = list(managed.get('forbidden_route_links', []))
    zone_list = zones(context, cfg)
    lane_cfgs = cfg.get('lanes') or []
    if not lane_cfgs:
        raise ValueError('Competition highway traffic needs lanes')
    lanes, plans = {}, []
    for lane_cfg in lane_cfgs:
        name = str(lane_cfg['name'])
        if name in lanes:
            raise ValueError('Duplicate competition lane: ' + name)
        if name not in cfg['lane_speeds_kmh']:
            raise ValueError('Competition lane has no lane_speeds_kmh entry: ' + name)
        route = list(lane_cfg['route'])
        segments, points, total = _route_geometry(context, route, forbidden)
        origin = lane_cfg.get('offset_origin_link', route[0])
        if origin not in route:
            raise ValueError('Competition offset origin is outside the lane route')
        origin_s = segments[route.index(origin)][0]
        end_points = context.map_loader.get_link_points(route[-1])

        def place(offset, label):
            absolute = origin_s + float(offset)
            if absolute < 25 or absolute > total - 40:
                raise ValueError(f'Competition {label} is too close to a lane endpoint: {name}@{offset}')
            slot = _slot(context, route, segments, absolute)
            hit = zone_containing(zone_list, slot['x'], slot['y'])
            if hit:
                raise ValueError(f'Competition {label} lies in no-spawn zone {hit}: {name}@{offset}')
            return dict(slot, offset_m=float(offset), offset_origin_link=origin)

        initial = _offsets(lane_cfg.get('initial_offsets_m'), name, cfg['minimum_spacing_m'])
        if not initial:
            raise ValueError('Competition lane needs initial_offsets_m: ' + name)
        if lane_cfg.get('supply_offset_m') is None:
            raise ValueError('Competition lane needs supply_offset_m: ' + name)
        supply = place(lane_cfg['supply_offset_m'], 'supply slot')   # raises when the slot is inside a no-spawn zone
        cumulative = [0.]
        for a, b in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        lane = dict(name=name, route=route, points=points, xy=[(p[0], p[1]) for p in points],
                    cum=cumulative, total=cumulative[-1], segments=segments,
                    end_points=end_points, end_link=route[-1],
                    speed_kmh=cfg['lane_speeds_kmh'][name],
                    retire_at_route_end=bool(lane_cfg.get('retire_at_route_end', False)),
                    merge_adjacent=bool(lane_cfg.get('merge_adjacent', False)),
                    target_count=int(lane_cfg.get('target_count', len(initial))), supply=supply,
                    last_spawn_s=float('-inf'), spawned=0, supply_zone=None)
        if lane['target_count'] < 1:
            raise ValueError('Competition target_count must be positive: ' + name)
        lanes[name] = lane
        for offset in initial:
            plans.append(dict(place(offset, 'initial slot'), lane=name, source='initial'))
    unknown = set(cfg['lane_speeds_kmh']) - set(lanes)
    if unknown:
        raise ValueError('lane_speeds_kmh names no lane: ' + ', '.join(sorted(unknown)))
    # Lanes share the upstream approach until the fork: link ids alone do not give enough room for two vehicles.
    spacing = float(cfg['minimum_spacing_m'])
    separation = float(cfg['minimum_physical_separation_m'])
    for index, first in enumerate(plans):
        for second in plans[index + 1:]:
            if first['route'][0] == second['route'][0] and abs(first['spawn_link_s_m'] - second['spawn_link_s_m']) < spacing - 1e-6:
                raise ValueError(f'Competition shared approach requires at least {spacing:g}m centre spacing')
            if math.hypot(first['x'] - second['x'], first['y'] - second['y']) < separation:
                raise ValueError('Competition spawn positions require at least '
                                 f'{separation:g}m physical separation: {first["lane"]}@{first["offset_m"]}, '
                                 f'{second["lane"]}@{second["offset_m"]}')
    if len(plans) > cfg['cap']:
        raise ValueError('Competition initial population exceeds the configured cap')
    if sum(lane['target_count'] for lane in lanes.values()) > cfg['cap']:
        raise ValueError('Competition target counts exceed the configured cap')
    for plan in plans:
        plan['speed_kmh'] = lanes[plan['lane']]['speed_kmh']
        plan['retire_at_route_end'] = lanes[plan['lane']]['retire_at_route_end']
        plan['merge_adjacent'] = lanes[plan['lane']]['merge_adjacent']
    pool = highway_traffic.model_candidates(context)
    if len(pool) < min(cfg['cap'], 25):
        raise RuntimeError(f'Need {min(cfg["cap"], 25)} distinct highway models; only {len(pool)} available')
    return cfg, plans, lanes, zone_list, pool


# ----------------------------------------------------------------------------- vehicles

def _states(context):
    """Live vehicle states by actor id; empty when the snapshot is unusable (no destructive action then)."""
    try:
        raw = list(context.grpc.get_all_vehicle_states())
    except (AttributeError, TypeError):
        return {}
    result = {}
    for item in raw:
        try:
            location, velocity = item.transform.location, item.velocity
            x, y = float(location.x), float(location.y)
            speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2) / 3.6
        except (AttributeError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (x, y, speed)):
            result[str(item.actor_info.id.value)] = dict(x=x, y=y, speed_mps=speed)
    return result


def _candidates(state, context):
    """Models not used by an active NPC, in the seeded rotation order starting after the last one used."""
    active = {item['model'] for item in context.highway_npcs}
    pool = state['pool']
    count = int(state['cfg']['respawn']['max_model_attempts'])
    found = []
    for step in range(len(pool)):
        model = pool[(state['pool_cursor'] + step) % len(pool)]
        if model not in active:
            found.append(model)
            if len(found) == count:
                break
    if not found:                                       # more vehicles than distinct models: repeat the rotation
        found = [pool[(state['pool_cursor'] + step) % len(pool)] for step in range(min(count, len(pool)))]
    return found


def _used(state, model):
    state['pool_cursor'] = (state['pool'].index(model) + 1) % len(state['pool'])


def _create(context, state, lane, slot, label, source, elapsed, *, models, strict):
    """Spawn one paused, routed, speed-limited NPC owned by this scenario. Returns the item or None."""
    grpc = context.grpc
    actor, model = None, None
    for model in models:
        spawn_started = time.monotonic()
        actor = grpc.spawn_vehicle(
            transform=grpc.make_transform(slot['x'], slot['y'], slot['z'], slot['yaw_deg']),
            model_name=model, label=label, velocity=0., multi_ego=False, pause=True,
            request_id=f'{label}_{context.run_instance_id}_{model}', recover_ambiguous=True)
        if actor is not None:
            break
    if actor is None:
        if strict:
            raise RuntimeError(f'Cannot fill all highway slots with distinct models: lane={lane["name"]}, '
                               f'position=({slot["x"]:.3f}, {slot["y"]:.3f}, {slot["z"]:.3f})')
        return None
    context.scenario_actors.append(actor)          # registered at once: any later failure is cleaned by the owner
    ready = (actor.set_pause(True)
             and grpc.set_vehicle_route(actor, slot['route'], route_waypoint_indices=highway_traffic.route_waypoints(context, slot),
                                        decision_range=float(context.cfg.get('npc_decision_range_m', 30.)), label=label)
             and grpc.set_vehicle_speed_limit(actor, lane['speed_kmh'] / 3.6, enabled=True))
    if not ready:
        if strict:
            raise RuntimeError(f'Cannot prepare highway vehicle: {label}')
        grpc.destroy_vehicle_by_id(str(actor.get_id()))
        _forget(context, actor)
        return None
    record_spawn = getattr(context, 'record_live_spawn_contact_evidence', None)
    if callable(record_spawn):
        record_spawn(actor, (slot['x'], slot['y']), spawn_started)
    return dict(slot, lane=lane['name'], merge_adjacent=lane['merge_adjacent'], speed_kmh=lane['speed_kmh'],
                target_speed_kmh=lane['speed_kmh'], actual_speed_kmh=None, speed_samples=0, speed_sum_kmh=0.,
                retire_at_route_end=lane['retire_at_route_end'], actor=actor, actor_id=str(actor.get_id()),
                model=model, label=label, source=source, spawned_s=elapsed, spawn_xy=(slot['x'], slot['y']),
                max_lane_error_m=0., lane_violation=False, removal_failures=0)


def _forget(context, actor):
    try:
        context.scenario_actors.remove(actor)
    except ValueError:
        pass


def _release(context, item):
    actor, grpc = item['actor'], context.grpc
    pause_ok = actor.set_pause(False)
    ai_ok = grpc.set_vehicle_ai(actor, True)
    velocity_ok = grpc.set_vehicle_velocity(actor, item['speed_kmh'])   # km/h; the limit above is m/s
    item['motion_contract'] = dict(unpause_acknowledged=bool(pause_ok), ai_enable_acknowledged=bool(ai_ok),
                                   velocity_acknowledged=bool(velocity_ok))
    if not (pause_ok and velocity_ok):
        raise RuntimeError(f'Cannot start highway vehicle: {item["label"]}; '
                           f'unpause={pause_ok}, velocity={velocity_ok}, ai_ack={ai_ok}')


def prepare(context):
    """Spawn the whole initial fleet paused. Nothing moves until ``prewarm`` runs."""
    cfg, plans, lanes, zone_list, pool = build_plan(context)
    context.highway_fixed_model = context.cfg['highway_npc'].get('model', 'DEFAULT_SEDAN')
    context.highway_npcs = []
    context.highway_spawn_points = []
    context.highway_spawn_points_active = False
    rng = random.Random(int(context.run_seed) ^ _LANE_ORDER_XOR)
    order = sorted(lanes)
    rng.shuffle(order)
    state = dict(cfg=cfg, lanes=lanes, zones=zone_list, pool=list(pool), pool_cursor=0, lane_order=order,
                 prewarm=dict(phase='prepared'), next_update_s=0., next_persist_s=0., next_observation_s=0.,
                 supplied=0, removed=[], lane_violations=[], shortages={}, shortage_log=[], observations=[],
                 telemetry_gaps=0, initial_count=len(plans), spawn_failures=0, generation=0,
                 fed=0, fed_by_source={}, feed_blocked={}, next_feed_s=0., rng=random.Random(int(context.run_seed) ^ _FEED_XOR))
    context.highway_competition = state
    for index, plan in enumerate(plans):
        lane = lanes[plan['lane']]
        label = f'highway_{plan["lane"]}_{index}'
        item = _create(context, state, lane, plan, label, 'initial', 0., models=_candidates(state, context), strict=True)
        _used(state, item['model'])
        context.highway_npcs.append(item)
    context.highway_npc_planned_count = len(context.highway_npcs)
    _route_state(context, state)
    _persist(context, 'prepared_paused')
    print(f'[HighwayCompetition] prepared {len(context.highway_npcs)} paused NPCs; '
          f'lanes={ {n: l["speed_kmh"] for n, l in lanes.items()} } km/h; cap={cfg["cap"]}; '
          f'no_spawn_zones={[z["name"] for z in zone_list]}', flush=True)


def prewarm(context, *, sleep=time.sleep, clock=time.monotonic):
    """Release the fleet and wait until real movement is measured. Ego has not been switched to AUTO yet."""
    state = context.highway_competition
    if state['prewarm']['phase'] == 'verified':
        return state['prewarm']
    cfg = state['cfg']['prewarm']
    for item in context.highway_npcs:
        _release(context, item)
    context.highway_spawn_point_activation_started = True
    context.highway_spawn_points_active = True
    context.spawn_highway_merge_static_vehicle()
    started = clock()
    required = max(1, math.ceil(float(cfg['minimum_moving_fraction']) * len(context.highway_npcs)))
    while True:
        sleep(float(cfg['poll_s']))
        elapsed = clock() - started
        states = _states(context)
        moved = 0
        for item in context.highway_npcs:
            sample = states.get(item['actor_id'])
            if sample and math.hypot(sample['x'] - item['spawn_xy'][0], sample['y'] - item['spawn_xy'][1]) >= float(cfg['minimum_travel_m']):
                moved += 1
        if elapsed >= float(cfg['minimum_s']) and moved >= required:
            state['prewarm'] = dict(phase='verified', moved=moved, total=len(context.highway_npcs),
                                    duration_s=round(elapsed, 2), required=required)
            break
        if elapsed >= float(cfg['verify_timeout_s']):
            state['prewarm'] = dict(phase='failed', moved=moved, total=len(context.highway_npcs),
                                    duration_s=round(elapsed, 2), required=required)
            _persist(context, 'prewarm_failed')
            raise RuntimeError(f'Highway NPC movement was not verified before Ego start: moved={moved}/'
                               f'{len(context.highway_npcs)} (need {required}) after {elapsed:.1f}s')
    _persist(context, 'prewarmed')
    print(f'[HighwayCompetition] PREWARM verified moving={state["prewarm"]["moved"]}/{state["prewarm"]["total"]} '
          f'after {state["prewarm"]["duration_s"]}s; Ego may start', flush=True)
    return state['prewarm']


# ----------------------------------------------------------------------------- per-tick management

def _ego_position(ego_state):
    try:
        x, y = float(ego_state['x']), float(ego_state['y'])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _sample_vehicles(context, state, states, elapsed):
    keeping = state['cfg']['lane_keeping']
    for item in context.highway_npcs:
        sample = states.get(item['actor_id'])
        if sample is None:
            continue
        item['x_now'], item['y_now'] = sample['x'], sample['y']
        kmh = sample['speed_mps'] * 3.6
        item['actual_speed_kmh'] = kmh
        if sample['speed_mps'] >= 1.:
            item['speed_samples'] += 1
            item['speed_sum_kmh'] += kmh
        stall = state['cfg']['removal']
        if sample['speed_mps'] < float(stall['stall_speed_mps']) and elapsed - item['spawned_s'] >= float(stall['stall_grace_s']):
            item.setdefault('stalled_since_s', elapsed)
        else:
            item.pop('stalled_since_s', None)
        lane = state['lanes'][item['lane']]
        item['lane_hint'], error = _local_lateral(lane, item.get('lane_hint'), sample['x'], sample['y'])
        item['lane_error_m'] = error
        item['max_lane_error_m'] = max(item['max_lane_error_m'], error)
        if error > float(keeping['max_lateral_error_m']):
            item.setdefault('violation_since_s', elapsed)
            if not item['lane_violation'] and elapsed - item['violation_since_s'] >= float(keeping['violation_hold_s']):
                item['lane_violation'] = True
                event = dict(label=item['label'], lane=item['lane'], t=round(elapsed, 2),
                             lateral_error_m=round(error, 2), x=round(sample['x'], 2), y=round(sample['y'], 2))
                state['lane_violations'].append(event)
                print(f'[HighwayCompetition] LANE_VIOLATION {event}', flush=True)
        else:
            item.pop('violation_since_s', None)


def _remove(context, state, item, reason, ego, elapsed):
    if not context.grpc.destroy_vehicle_by_id(item['actor_id']):
        item['removal_failures'] += 1
        if reason == 'lane_end' and item['removal_failures'] >= int(state['cfg']['removal']['failures_before_abort']):
            item['actor'].set_pause(True)
            raise RuntimeError('Cannot remove highway NPC at lane end: ' + item['label'])
        return False
    _forget(context, item['actor'])
    context.highway_npcs.remove(item)
    distance = None if ego is None else math.hypot(item['x_now'] - ego[0], item['y_now'] - ego[1])
    record = dict(label=item['label'], lane=item['lane'], reason=reason, link=item['route'][-1], t=round(elapsed, 2),
                  ego_distance_m=None if distance is None else round(distance, 1),
                  target_speed_kmh=item['target_speed_kmh'],
                  mean_actual_speed_kmh=(round(item['speed_sum_kmh'] / item['speed_samples'], 1) if item['speed_samples'] else None))
    state['removed'].append(record)
    print(f'[HighwayCompetition] {reason.upper()}_REMOVED {item["label"]} link={item["route"][-1]}', flush=True)
    return True


def _remove_stalled(context, state, ego, elapsed):
    """An NPC that stopped on the highway (blocked, stuck AI) is removed: Ego would run into a vehicle that no
    competition scenario places there (a real run ended with Ego hitting such an NPC)."""
    limit = float(state['cfg']['removal']['stall_s'])
    for item in list(context.highway_npcs):
        since = item.get('stalled_since_s')
        if since is not None and elapsed - since >= limit and 'x_now' in item:
            print(f'[HighwayCompetition] STALLED {item["label"]} stopped for {elapsed - since:.1f}s', flush=True)
            _remove(context, state, item, 'stalled', ego, elapsed)


def _remove_finished(context, state, ego, elapsed):
    """Remove NPCs that reached (or passed) the end of their lane route.

    Progress is the arc length of the nearest lane vertex, so a vehicle that skipped a short end window between two
    slow updates is still removed. A lane-drop lane deletes ``lane_end_margin_m`` *before* its end: the NPC must never
    reach the taper, even when an update comes late.
    """
    removal = state['cfg']['removal']
    for item in list(context.highway_npcs):
        hint = item.get('lane_hint')
        if 'x_now' not in item or hint is None:
            continue                                    # no telemetry for this actor is not evidence of arrival
        lane = state['lanes'][item['lane']]
        remaining = lane['total'] - lane['cum'][hint]
        if lane['retire_at_route_end']:
            if remaining <= float(removal['lane_end_margin_m']):
                _remove(context, state, item, 'lane_end', ego, elapsed)    # mandatory: it must not enter the taper
        elif remaining <= float(removal['end_tolerance_m']) and (
                ego is None or math.hypot(item['x_now'] - ego[0], item['y_now'] - ego[1])
                >= float(removal['terminal_minimum_ego_distance_m'])):
            _remove(context, state, item, 'terminal', ego, elapsed)        # far route end, never in front of Ego


def _blocked(context, state, lane, states, ego, elapsed):
    respawn = state['cfg']['respawn']
    slot = lane['supply']
    if elapsed - lane['last_spawn_s'] < float(respawn['minimum_lane_interval_s']):
        return 'headway'
    if lane.get('supply_zone'):
        return 'no_spawn_zone:' + lane['supply_zone']      # static slot: decided once when the plan was built
    progress = state.get('ego_progress')
    if progress is None or float(progress) < float(respawn['minimum_ego_progress_m']):
        return 'ego_progress'                           # the start belongs to the initial burst
    if ego is None or math.hypot(slot['x'] - ego[0], slot['y'] - ego[1]) < float(respawn['minimum_ego_distance_m']):
        return 'ego_distance'
    if _clearance_blocked(context, state, lane['name'], slot['x'], slot['y'], states):
        return 'clearance'
    return None


def _clearance_blocked(context, state, lane_name, x, y, states):
    respawn = state['cfg']['respawn']
    owner = {item['actor_id']: item['lane'] for item in context.highway_npcs}
    for actor_id, vehicle in states.items():
        distance = math.hypot(x - vehicle['x'], y - vehicle['y'])
        # An NPC of another lane group drives in its own lane: only a body overlap blocks. Same-lane NPCs, Ego and
        # every other vehicle need the full following clearance.
        needed = respawn['body_clearance_m'] if owner.get(actor_id, lane_name) != lane_name else respawn['clearance_m']
        if distance < float(needed):
            return True
    return False


def _note_shortage(state, lane, deficit, reason, elapsed):
    entry = state['shortages'].setdefault(lane['name'], dict(count=0, reasons={}))
    entry['count'] += 1
    entry['reasons'][reason] = entry['reasons'].get(reason, 0) + 1
    entry.update(last_reason=reason, last_deficit=deficit, last_t=round(elapsed, 2))
    # One line per lane and reason per 10 s: a blocked slot is reported without flooding the log.
    key = (lane['name'], reason)
    if elapsed - state.setdefault('shortage_printed', {}).get(key, float('-inf')) >= 10.:
        state['shortage_printed'][key] = elapsed
        event = dict(t=round(elapsed, 2), lane=lane['name'], deficit=deficit, reason=reason)
        state['shortage_log'] = (state['shortage_log'] + [event])[-_EVENT_LIMIT:]
        print(f'[HighwayCompetition] SHORTAGE lane={lane["name"]} deficit={deficit} reason={reason}', flush=True)


def _resupply(context, state, states, ego, elapsed):
    if not state['cfg'].get('live_spawn_enabled', True):
        return
    cap = state['cfg']['cap']
    if state.get('route_state') and state['cfg']['feed']['enabled']:
        return                                          # the Ego-synchronised feed replaces the fork supply
    if len(context.highway_npcs) >= cap:
        return
    active = {}
    for item in context.highway_npcs:
        active[item['lane']] = active.get(item['lane'], 0) + 1
    deficits = [(state['lanes'][name]['target_count'] - active.get(name, 0), name) for name in state['lane_order']]
    for deficit, name in sorted((d for d in deficits if d[0] > 0), key=lambda d: -d[0]):
        lane = state['lanes'][name]
        reason = _blocked(context, state, lane, states, ego, elapsed)
        if reason in ('headway', 'ego_progress'):
            continue                                    # spacing the supply / the initial burst are not shortages
        if reason:
            _note_shortage(state, lane, deficit, reason, elapsed)
            continue
        state['generation'] += 1
        models = _candidates(state, context)
        label = f'highway_{name}_s{state["generation"]}'
        item = _create(context, state, lane, lane['supply'], label, 'supply', elapsed, models=models, strict=False) if models else None
        if item is None:
            state['spawn_failures'] += 1
            _note_shortage(state, lane, deficit, 'spawn_failed', elapsed)
            continue
        try:
            _release(context, item)
        except RuntimeError:
            context.grpc.destroy_vehicle_by_id(item['actor_id'])
            _forget(context, item['actor'])
            state['spawn_failures'] += 1
            _note_shortage(state, lane, deficit, 'release_failed', elapsed)
            continue
        _used(state, item['model'])
        context.highway_npcs.append(item)
        lane['last_spawn_s'] = elapsed
        lane['spawned'] += 1
        state['supplied'] += 1
        print(f'[HighwayCompetition] SUPPLY {label} model={item["model"]} speed={lane["speed_kmh"]:g}km/h', flush=True)
        return                                          # one spawn per snapshot: clearance was measured before it


# ----------------------------------------------------------------------------- Ego-synchronised feed

def _route_state(context, state):
    """Ego's mission route (cumulative length), the merge entry on it and where each lane passes that point."""
    cached = state.get('route_state')
    if cached:
        return cached
    points = getattr(context, 'route_points', None)
    if not points or len(points) < 3:
        return None                                     # no route (yet): the fixed layout and the fork supply carry on
    xy = [(p[0], p[1]) for p in points]
    cumulative = [0.]
    for a, b in zip(xy, xy[1:]):
        cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    info = dict(xy=xy, cum=cumulative, total=cumulative[-1], hint=None, merge=None, lane_p={})
    state['route_state'] = info
    try:
        link = context.map_loader.get_link(_managed(context)['ego_merge_link'])
        target = link['left_lane_change_dst_link_idx']
        target_points = context.map_loader.get_link_points(target)
        length = polyline_length(target_points)
        low, high = min(p[1] for p in target_points) - 2., max(p[1] for p in target_points) + 2.
        for index in range(0, len(xy), 2):
            x, y = xy[index]
            if not low <= y <= high:
                continue
            s = project_distance_on_polyline(target_points, x, y)
            lx, ly, _, _ = interpolate_on_polyline(target_points, s)
            if 20 < s < length - 10 and math.hypot(x - lx, y - ly) <= 1.75:
                info['merge'] = dict(s=cumulative[index], x=x, y=y)
                break
    except (KeyError, TypeError, AttributeError, ValueError):
        info['merge'] = None
    if info['merge']:
        for name, lane in state['lanes'].items():
            index, lateral = _local_lateral(lane, None, info['merge']['x'], info['merge']['y'])
            if lateral <= 12.:
                info['lane_p'][name] = lane['cum'][index]       # where this lane passes the merge entry
    return info


def _route_at(info, s):
    s = max(0., min(float(s), info['total']))
    i = min(max(bisect.bisect_right(info['cum'], s) - 1, 0), len(info['xy']) - 2)
    a, b = info['xy'][i], info['xy'][i + 1]
    span = info['cum'][i + 1] - info['cum'][i]
    f = 0. if span == 0. else (s - info['cum'][i]) / span
    return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])


def _lane_speed_estimate(context, state, lane):
    """The lane's speed as measured on the road (queues and slow links keep it below the target)."""
    items = [i for i in context.highway_npcs if i['lane'] == lane['name'] and i['speed_samples'] >= 20]
    if not items:
        return lane['speed_kmh'] / 3.6 * 0.9
    mean = sum(i['speed_sum_kmh'] for i in items) / sum(i['speed_samples'] for i in items)
    return max(3., mean / 3.6)


def _track_route(context, state, info, ego, ego_state, elapsed):
    """Ego's position/speed on its route and every NPC's position relative to it."""
    feed = state['cfg']['feed']
    index, _ = _local_lateral(dict(xy=info['xy']), info['hint'], ego[0], ego[1], window=400)
    info['hint'] = index
    ego_s = info['cum'][index]
    speed = 0.
    if isinstance(ego_state, dict):
        for key in ('speed_mps', 'speed'):
            try:
                speed = float(ego_state[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
    route = dict(xy=info['xy'])
    for item in context.highway_npcs:
        item['route_rel'] = None
        if 'x_now' not in item or math.hypot(item['x_now'] - ego[0], item['y_now'] - ego[1]) > 700.:
            continue
        hint = item.get('route_hint')
        j, lateral = _local_lateral(route, index if hint is None else hint, item['x_now'], item['y_now'],
                                    window=1000 if hint is None else 150)
        item['route_hint'] = j
        if lateral <= float(feed['route_lateral_m']):
            item['route_rel'] = info['cum'][j] - ego_s
    return ego_s, speed


def _recycle_far(context, state, ego, elapsed):
    feed = state['cfg']['feed']
    for item in list(context.highway_npcs):
        rel = item.get('route_rel')
        if rel is None or 'x_now' not in item:
            continue
        full = len(context.highway_npcs) >= state['cfg']['cap'] - 2
        # NPCs ahead are traffic Ego has not met yet: they stay (and keep driving) unless the cap is full and they are
        # far ahead. Only NPCs well behind Ego are always recycled.
        far_ahead = full and rel > float(feed['recycle_far_ahead_m'])
        if far_ahead or rel < -float(feed['recycle_far_behind_m']):
            if math.hypot(item['x_now'] - ego[0], item['y_now'] - ego[1]) >= float(feed['recycle_min_ego_distance_m']):
                _remove(context, state, item, 'recycled', ego, elapsed)


def _refused(state, source, reason):
    key = f'{source}:{reason}'
    state['feed_blocked'][key] = state['feed_blocked'].get(key, 0) + 1
    return None


def _reduce_rear(context, state, info, ego, elapsed, ego_s):
    """Fewer rear NPCs: only from the moment Ego enters the highway lane (the merge entry, not a clock time) the rear
    population is reduced, and only beyond ``min_distance_m`` (80 m) of Ego: upstream NPCs that have not reached the
    merge entry yet and NPCs behind Ego beyond the ``keep_behind`` nearest. Before the lane entry nothing differs from
    the normal traffic."""
    cfg = state['cfg']['rear_reduction']
    merge = info['merge']
    if not cfg['enabled'] or not merge or ego_s < merge['s']:
        return
    if 'rear_started' not in state:
        state['rear_started'] = dict(t=round(elapsed, 2), ego_route_s_m=round(ego_s, 1))
        print(f'[HighwayCompetition] REAR_REDUCTION from the lane entry at ego_s={ego_s:.0f}m', flush=True)
    minimum = float(cfg['min_distance_m'])
    removable = []
    behind = []
    for item in context.highway_npcs:
        if 'x_now' not in item:
            continue
        distance = math.hypot(item['x_now'] - ego[0], item['y_now'] - ego[1])
        if distance < minimum:
            continue
        lane = state['lanes'][item['lane']]
        point = info['lane_p'].get(item['lane'])
        rel = item.get('route_rel')
        if rel is None and point is not None and item.get('lane_hint') is not None \
                and lane['cum'][item['lane_hint']] < point - 20.:
            removable.append(item)                      # still upstream of the merge entry: it would arrive behind Ego
        elif rel is not None and rel < 0:
            behind.append((-rel, item))
    behind.sort(key=lambda pair: pair[0])
    removable += [item for rank, (gap, item) in enumerate(behind) if rank >= int(cfg['keep_behind']) or gap > float(cfg['keep_range_m'])]
    for item in removable[:int(cfg['max_per_step'])]:
        if _remove(context, state, item, 'rear_thinned', ego, elapsed):
            state['rear_removed'] = state.get('rear_removed', 0) + 1


def _try_feed(context, state, lane, absolute, states, ego, elapsed, min_distance, source):
    if absolute < 25. or absolute > lane['total'] - (90. if lane['retire_at_route_end'] else 45.):
        return _refused(state, source, 'lane_bounds')
    slot = _slot(context, lane['route'], lane['segments'], absolute)
    if math.hypot(slot['x'] - ego[0], slot['y'] - ego[1]) < min_distance:
        return _refused(state, source, 'ego_distance')
    if zone_containing(state['zones'], slot['x'], slot['y']):
        return _refused(state, source, 'no_spawn_zone')
    if _clearance_blocked(context, state, lane['name'], slot['x'], slot['y'], states):
        return _refused(state, source, 'clearance')
    models = _candidates(state, context)
    if not models:
        return _refused(state, source, 'no_model')
    state['generation'] += 1
    label = f'highway_{lane["name"]}_f{state["generation"]}'
    item = _create(context, state, lane, dict(slot, offset_m=absolute), label, source, elapsed, models=models, strict=False)
    if item is None:
        state['spawn_failures'] += 1
        return None
    try:
        _release(context, item)
    except RuntimeError:
        context.grpc.destroy_vehicle_by_id(item['actor_id'])
        _forget(context, item['actor'])
        state['spawn_failures'] += 1
        return None
    _used(state, item['model'])
    item['spawn_ego_distance_m'] = round(math.hypot(slot['x'] - ego[0], slot['y'] - ego[1]), 1)
    context.highway_npcs.append(item)
    lane['spawned'] += 1
    state['fed'] += 1
    state['fed_by_source'][source] = state['fed_by_source'].get(source, 0) + 1
    return item


def _feed(context, state, states, ego, ego_state, elapsed, current_s):
    """Keep traffic around Ego where it will actually be: at the merge entry, then ahead of it."""
    if not state['cfg'].get('live_spawn_enabled', True):
        return
    feed = state['cfg']['feed']
    if not feed['enabled']:
        return
    info = _route_state(context, state)
    if info is None:
        return
    if elapsed < state['next_feed_s']:
        return
    state['next_feed_s'] = elapsed + float(feed['interval_s'])
    ego_s, ego_speed = _track_route(context, state, info, ego, ego_state, elapsed)
    state['ego_route_s'] = ego_s
    _recycle_far(context, state, ego, elapsed)
    _reduce_rear(context, state, info, ego, elapsed, ego_s)
    if len(context.highway_npcs) >= state['cfg']['cap']:
        return
    rng = state['rng']
    merge = info['merge']
    if merge and ego_s < merge['s'] and info['lane_p']:
        eta = (merge['s'] - ego_s) / max(ego_speed, float(feed['speed_floor_mps']))
        window = (eta - float(feed['merge_before_s']), eta + float(feed['merge_after_s']))
        wanted = dict(feed['merge_per_lane'])
        arrivals = {}
        for item in context.highway_npcs:
            lane = state['lanes'][item['lane']]
            if item['lane'] in info['lane_p'] and item.get('lane_hint') is not None:
                tau = (info['lane_p'][item['lane']] - lane['cum'][item['lane_hint']]) / _lane_speed_estimate(context, state, lane)
                arrivals.setdefault(item['lane'], []).append(tau)
        order = sorted((n for n in info['lane_p']),
                       key=lambda n: sum(1 for t in arrivals.get(n, []) if window[0] <= t <= window[1]) - int(wanted.get(n, 6)))
        for name in order:
            have = [t for t in arrivals.get(name, []) if window[0] <= t <= window[1]]
            if len(have) >= int(wanted.get(name, 6)):
                continue
            lane = state['lanes'][name]
            speed = _lane_speed_estimate(context, state, lane)
            for _ in range(8):
                tau = window[0] + rng.random() * (window[1] - window[0])
                headway = float(feed['min_headway_s']) * (0.7 + 1.6 * rng.random())      # irregular gaps, not a grid
                if tau < 0 or any(abs(tau - t) < headway for t in arrivals.get(name, [])):
                    continue
                if _try_feed(context, state, lane, info['lane_p'][name] - speed * tau, states, ego, elapsed,
                             float(feed['merge_min_distance_m']), 'feed_merge'):
                    return
        return
    # past the merge entry (or no merge on the route): keep the stretch ahead of Ego populated
    ahead = [i for i in context.highway_npcs if i.get('route_rel') is not None and 0 < i['route_rel'] <= float(feed['ahead_window_m'])]
    if len(ahead) >= int(feed['target_ahead']):
        return
    # Weighted random lane order (the middle lane gets the largest share) instead of "emptiest lane first", which
    # filled every lane evenly.
    weights = dict(feed['lane_weights'])
    order = sorted(state['lanes'], key=lambda n: -rng.random() ** (1. / max(0.05, float(weights.get(n, 1.)))))
    for name in order:
        lane = state['lanes'][name]
        for _ in range(6):
            distance = float(feed['min_distance_m']) + rng.random() * (float(feed['ahead_window_m']) - float(feed['min_distance_m']))
            x, y = _route_at(info, ego_s + distance)
            index, lateral = _local_lateral(lane, None, x, y)
            if lateral > float(feed['route_lateral_m']):
                _refused(state, 'feed_ahead', 'no_lane_there')
                continue
            if _try_feed(context, state, lane, lane['cum'][index], states, ego, elapsed, float(feed['min_distance_m']), 'feed_ahead'):
                return


def _observe(context, state, ego, ego_state, elapsed, current_s):
    """Fleet around Ego: counts and nearest gaps ahead/behind along Ego's heading, per-lane population."""
    cfg = state['cfg']['observation']
    if elapsed < state['next_observation_s']:
        return
    state['next_observation_s'] = elapsed + float(cfg['period_s'])
    row = dict(t=round(elapsed, 2), ego_s_m=None if current_s is None else round(float(current_s), 1),
               active=len(context.highway_npcs),
               per_lane={n: sum(1 for i in context.highway_npcs if i['lane'] == n) for n in state['lanes']})
    exit_s = getattr(context, 'highway_exit_s', None)
    if current_s is not None and exit_s:
        row['progress'] = round(min(1., max(0., float(current_s) / float(exit_s))), 3)
    if state.get('route_state'):
        # Along Ego's own route (it curves through the ramp, a heading corridor misses the highway there).
        rels = [i['route_rel'] for i in context.highway_npcs if i.get('route_rel') is not None]
        gaps = [r for r in rels if 0 < r <= float(cfg['radius_m'])]
        row.update(ahead_route=len(gaps), nearest_ahead_route_m=round(min(gaps), 1) if gaps else None,
                   around_route=sum(1 for r in rels if abs(r) <= 120.), behind_route=sum(1 for r in rels if -150. <= r < 0), ahead_far_route=sum(1 for r in rels if 0 < r <= 300.),
                   ego_route_s_m=round(state.get('ego_route_s', 0.), 1))
    yaw = ego_state.get('yaw_deg', ego_state.get('yaw')) if isinstance(ego_state, dict) else None
    if ego is not None and yaw is not None and math.isfinite(float(yaw)):
        fx, fy = math.cos(math.radians(float(yaw))), math.sin(math.radians(float(yaw)))
        ahead, behind = [], []
        for item in context.highway_npcs:
            if 'x_now' not in item:
                continue
            dx, dy = item['x_now'] - ego[0], item['y_now'] - ego[1]
            along, side = dx * fx + dy * fy, -dx * fy + dy * fx
            if abs(side) <= float(cfg['corridor_half_width_m']) and abs(along) <= float(cfg['radius_m']):
                (ahead if along > 0 else behind).append(abs(along))
        row.update(ahead=len(ahead), behind=len(behind),
                   nearest_ahead_m=round(min(ahead), 1) if ahead else None,
                   nearest_behind_m=round(min(behind), 1) if behind else None)
    state['observations'] = (state['observations'] + [row])[-_OBSERVATION_LIMIT:]


def update(context, elapsed, ego_state, current_s):
    if not enabled(context) or not has_state(context) or context.highway_traffic_cleared \
            or not context.highway_spawn_points_active:
        return
    state = context.highway_competition
    if elapsed < state['next_update_s']:
        return
    state['next_update_s'] = elapsed + float(state['cfg']['update_period_s'])
    states = _states(context)
    ego = _ego_position(ego_state)
    if not states or ego is None:
        state['telemetry_gaps'] += 1            # never delete or spawn from an incomplete snapshot
        return
    state['ego_progress'] = None if current_s is None else float(current_s)
    _sample_vehicles(context, state, states, elapsed)
    _remove_stalled(context, state, ego, elapsed)
    _remove_finished(context, state, ego, elapsed)
    _resupply(context, state, states, ego, elapsed)
    _feed(context, state, states, ego, ego_state, elapsed, current_s)
    _observe(context, state, ego, ego_state, elapsed, current_s)
    if elapsed >= state['next_persist_s']:
        state['next_persist_s'] = elapsed + float(state['cfg']['persist_period_s'])
        _persist(context, 'running')


# ----------------------------------------------------------------------------- evidence

def _summary(context, state):
    lanes = {}
    for name, lane in state['lanes'].items():
        items = [i for i in context.highway_npcs if i['lane'] == name]
        lanes[name] = dict(target_speed_kmh=lane['speed_kmh'], target_count=lane['target_count'], active=len(items),
                           speeds_distinct_targets=sorted({i['target_speed_kmh'] for i in items}),
                           mean_actual_speed_kmh=(round(sum(i['speed_sum_kmh'] for i in items) /
                                                  max(1, sum(i['speed_samples'] for i in items)), 1)
                                                  if any(i['speed_samples'] for i in items) else None),
                           supplied=lane['spawned'], retire_at_route_end=lane['retire_at_route_end'],
                           end_link=lane['end_link'], supply_link=lane['supply']['route'][0],
                           supply_offset_m=lane['supply']['offset_m'])
    buckets = {}
    for row in state['observations']:
        count = row.get('ahead_route', row.get('ahead'))
        if 'progress' in row and count is not None:
            bucket = min(9, int(row['progress'] * 10))
            buckets.setdefault(bucket, []).append(count)
    return dict(lanes=lanes,
                ahead_by_progress_decile={str(k): dict(min=min(v), mean=round(sum(v) / len(v), 2), samples=len(v))
                                          for k, v in sorted(buckets.items())},
                removed_by_reason={r: sum(1 for x in state['removed'] if x['reason'] == r)
                                   for r in sorted({x['reason'] for x in state['removed']})})


def _persist(context, phase):
    state = context.highway_competition
    records = [dict({k: v for k, v in item.items() if k not in ('actor', 'speed_sum_kmh')},
                    mean_actual_speed_kmh=(round(item['speed_sum_kmh'] / item['speed_samples'], 1)
                                           if item['speed_samples'] else None)) for item in context.highway_npcs]
    output = Path(context.global_cfg['paths']['runtime_root']) / _REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    document = dict(policy=POLICY, phase=phase, seed=context.run_seed, cap=state['cfg']['cap'],
                    live_spawn_enabled=state['cfg'].get('live_spawn_enabled', True),
                    initial_count=state['initial_count'], active=len(records), supplied=state['supplied'], fed=state['fed'], fed_by_source=state['fed_by_source'], feed_blocked=state['feed_blocked'],
                    rear_reduction=dict(state['cfg']['rear_reduction'], started=state.get('rear_started'), removed=state.get('rear_removed', 0)),
                    prewarm=state['prewarm'], distinct_models=len({r['model'] for r in records}),
                    no_spawn_zones=[z['name'] for z in state['zones']],
                    summary=_summary(context, state), removed=state['removed'][-_EVENT_LIMIT:],
                    lane_violations=state['lane_violations'], shortages=state['shortages'],
                    shortage_log=state['shortage_log'], spawn_failures=state['spawn_failures'],
                    telemetry_gaps=state['telemetry_gaps'], vehicles=records, observations=state['observations'])
    output.write_text(json.dumps(document, indent=2, default=str), encoding='utf-8')


def finish(context):
    """Last evidence write before the highway boundary clears every owned actor."""
    if has_state(context):
        _persist(context, 'finished')
