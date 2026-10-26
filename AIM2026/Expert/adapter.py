"""Reuse the existing route, signal policy and Expert input adapter; no sensor sockets."""
import json, math, os
from pathlib import Path
from types import SimpleNamespace

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

UDP_STEERING_SIGNS = {'normal': -1.0, 'competition': 1.0}

def configure_udp_steering(config, sim_profile='normal'):
    # These profiles describe the two builds measured on this PC. Keep the
    # transport adaptation in this run's private config, outside Expert core.
    config['common']['realtime_steering_sign'] = UDP_STEERING_SIGNS[sim_profile]

def configure_highway_speed(config, explicit_cap_kmh=None):
    """Set only the highway route ceiling; all later merge/hazard limits remain."""
    from AIM2026.Expert.src.config.parameters import highway
    from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
    speed = validate_max_velocity_kmh(highway.HIGHWAY_CRUISE_SPEED_KPH)
    if explicit_cap_kmh is not None:
        speed = min(speed, validate_max_velocity_kmh(explicit_cap_kmh))
    for zone in config['planning']['velocity_profile'].get('speed_limit_zones', []):
        if zone.get('name') == 'high_speed_to_tollgate':
            zone['max_velocity'] = speed
    return speed

def prepare(args, output):
    from scenario_runner.main_runner import load_configuration
    from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
    from scenario_runner.utils.map_loader import MGeoMapLoader
    if getattr(args, 'scenario_context', None):
        document = json.loads(args.scenario_context.read_text())
        if document.get('schema') != 've_gps_scenario_context_v1':
            raise ValueError('GPS 시나리오 설정 형식이 다릅니다')
        global_cfg, config = document['global_config'], document['scenario_config']
    else:
        global_cfg, _zone, config = load_configuration()
    global_cfg['paths']['runtime_root'] = str(output)
    config.update(expert_speed_cap_kmh=60. if args.speed_cap_kmh is None else args.speed_cap_kmh,
                  route_debug_enabled=False)
    # Only export the existing map/route and private Expert config. No simulator
    # connection, scenario setup, reset, spawn, or vehicle GT query is performed.
    mission = config.get('mission_mode')
    if mission:
        # Bounded mission (e.g. the tunnel from its entrance): rebuild exactly the scenario's sliced route.
        from scenario_runner.scenarios.segment_modes import events
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario as FullLoopScenario
    context = FullLoopScenario(grpc_client=SimpleNamespace(make_transform=lambda *v: v),
        map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
        global_cfg=global_cfg, scenario_cfg=config)
    if mission:
        events.prepare_mission(context)
    context.route_definition = context.load_route_definition()
    context.prepare_full_loop_route()
    window = getattr(context, 'mission_route', None)
    slice_start = window.first_index if window is not None else 0   # full-route index of route_points[0]
    if getattr(args, 'scenario_context', None):
        if [list(p) for p in context.route_points] != document['route_points']:
            raise ValueError('시나리오와 GPS Expert의 경로가 다릅니다')
    context.setup_expert_controller()
    controller = context.expert_controller
    path = Path(controller.runtime_config_path)
    config = json.loads(path.read_text())
    highway_speed = configure_highway_speed(config, args.speed_cap_kmh)
    print('[Expert] 일반 고속도로 목표속도 상한: {:g}km/h (highway.py; 합류·위험 제한 별도)'.format(highway_speed), flush=True)
    config['common']['perception_source'] = 'ros'  # Original tracked-object planner.
    config['common']['sync_grpc_inputs']['enabled'] = False
    configure_udp_steering(config, args.sim_profile)
    config['map']['traffic_light_control'] = False
    settings = config['planning']['traffic_light_stop']
    settings['stop_statuses'] = sorted(set(settings.get('stop_statuses', [1])) | {0})
    settings['hold_on_stale'] = True
    if getattr(args, 'competition', False):
        settings['ignored_signal_ids'] = []
    if getattr(args, 'ignore_signals', None):
        # Evaluation-only: the run's private config copy skips these signals (planner and signal policy);
        # the Expert source config is not modified.
        settings['ignored_signal_ids'] = sorted(set(settings.get('ignored_signal_ids', [])) | set(args.ignore_signals))
    for key in ('jamming_scene_geometry', 'jamming_trial_contract', 'jamming_trial_diagnostics_path'):
        config['planning'].pop(key, None)
    write(path, config)
    os.environ.update(GT_BEV_CONFIG_PATH=str(path), GT_BEV_PATH_CSV=controller.runtime_path_csv,
                      GT_BEV_MAP_DIR=config['map']['data_dir'])
    from scenario_runner.adapters.ve_signal import build_signal_targets, FullRouteSignalPolicy
    document = build_signal_targets(context, config)
    ignored = set(settings.get('ignored_signal_ids', []))
    # The Expert's highway planner changes lanes inside +-6 m of the highway centreline, so the global path
    # is not the driven line there; only a much larger offset counts as leaving the route.
    zone = context.route_definition['zones']['highway']
    exit_index = next(int(c['path_index']) for c in context.route_definition['checkpoints']
                      if str(c['id']) == str(zone['ego_exit_checkpoint'])) - slice_start
    def checkpoint_index(identity):
        return next(int(c['path_index']) for c in context.route_definition['checkpoints'] if str(c['id']) == str(identity)) - slice_start
    tunnel = context.route_definition['zones'].get('tunnel')
    # GPS blackout zone: lane keeping is not scored and avoidance may cross the centre line (rules v1.2).
    tunnel_zone = ((checkpoint_index(tunnel['entry_checkpoint']), checkpoint_index(tunnel['exit_checkpoint']), 12.),
                   ) if tunnel else ()
    policy = FullRouteSignalPolicy(context.route_points,
                                  [r for r in document['targets'] if r['signal_id'] not in ignored],
                                  wide_zone=(int(zone['activation_path_index']) - slice_start, exit_index, 12.),
                                  wide_zones=tunnel_zone)
    write(output/'signal_targets.json', document)
    return context, controller, policy


def object_alignment_seconds(packet, control_pose):
    """Receipt-clock approximation from the object frame to control position time.

    Old publishers have no object-frame stamp: do not guess using their oldest
    camera/IMU receipt or inference send time. Input expiry remains enforced by
    Inputs.ready(); never extend it here or extrapolate beyond that 0.5 s budget.
    """
    if control_pose is None:
        return 0.
    observed = packet.get('object_received_ns')
    position = (control_pose.get('selected_samples') or {}).get('gps', {}).get('received_ns')
    if type(observed) is not int or type(position) is not int:
        return 0.
    return min(.5, max(0., (position - observed) / 1e9))


def static_zone_box(box):
    """Only static obstacles stand in the static-obstacle section; drop VE motion uncertainty there.

    The tracker's uncertain/moving labels come from box jitter, which resets the avoidance plan.
    Raw tracker states stay in inputs.jsonl; only the planner input is normalized.
    """
    from AIM2026.Expert.src.config.parameters.sections import STATIC_OBSTACLE_CERTAIN_X, STATIC_OBSTACLE_CERTAIN_Y
    x, y = box['center_world']
    if (box['motion_state'] == 'stationary' or not STATIC_OBSTACLE_CERTAIN_X[0] <= x <= STATIC_OBSTACLE_CERTAIN_X[1]
            or not STATIC_OBSTACLE_CERTAIN_Y[0] <= y <= STATIC_OBSTACLE_CERTAIN_Y[1]):
        return box
    return dict(box, motion_state='stationary', planning_speed_mps=0., planning_speed_source='static_zone')


def planner_inputs(packet, control_pose=None, position_sources=('GPS',), coordinates=None):
    from AIM2026.Expert.src.vehicle_state import VehicleState
    from AIM2026.Expert.src.obstacle.object_info import ObjectInfo
    if coordinates is not None:
        packet = coordinates.expert_packet(packet)
        if control_pose is not None:
            control_pose = coordinates.expert_packet(control_pose)
    pose_packet = packet if control_pose is None else control_pose
    pose = pose_packet['ego_pose']
    if pose_packet.get('ego_position_source') not in position_sources or pose_packet.get('ego_status_source') != 'UDP_status_speed_IMU_heading':
        raise ValueError('GPS/IMU 입력 이외의 자차 위치가 전달됐습니다')
    values = [pose[k] for k in ('x', 'y', 'yaw_grid_rad', 'speed_mps')]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('자차 위치·방향·속도가 유효하지 않습니다')
    vehicle = VehicleState(*values)
    alignment_s = object_alignment_seconds(packet, control_pose)
    from AIM2026.Expert.src.config.parameters.highway import UNKNOWN_OBJECT_SPEED_BOUND
    boxes = [static_zone_box(box) for box in packet['boxes']]
    objects = [ObjectInfo(*box['center_world'],
        box['planning_speed_mps'] if box['planning_speed_mps'] is not None else UNKNOWN_OBJECT_SPEED_BOUND, -1,
        name='ve_tracked_unknown:'+box['motion_state'],
        is_static=box['motion_state'] == 'stationary', size_x=box['length_m'], size_y=box['width_m'],
        heading_deg=box['planning_heading_deg'], unique_id=box['track_id']) for box in boxes]
    for obj, box in zip(objects, boxes):
        # Confidence is independent of the nominal model/observed speed.
        # Highway planning uses the supplied speed even before motion history
        # matures. Keep this confidence flag without inventing zero velocity.
        obj.motion_uncertain = (box['motion_state'] == 'uncertain'
                                and box.get('planning_speed_source') not in ('carry_over', 'overlap_supported'))
        obj.speed_unknown = box['planning_speed_mps'] is None
        # Raw detections/tracker history stay at measurement time. Construct a
        # fresh planning position each tick, shared verbatim with the RViz
        # snapshot. Never translate static scenery by ego displacement, or use
        # the unknown-speed bound as a measured velocity.
        obj.observation_x, obj.observation_y = obj.position.x, obj.position.y
        obj.position_projection_s = 0.
        if (not obj.is_static and not obj.speed_unknown
                and math.isfinite(obj.velocity) and math.isfinite(obj.heading_deg)):
            yaw = math.radians(obj.heading_deg)
            obj.position = obj.position.translate(
                obj.velocity * math.cos(yaw) * alignment_s,
                obj.velocity * math.sin(yaw) * alignment_s)
            obj.position_projection_s = alignment_s
    return vehicle, objects
