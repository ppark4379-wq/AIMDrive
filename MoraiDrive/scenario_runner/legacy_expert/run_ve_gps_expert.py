#!/usr/bin/env python3
"""GPS/IMU + VE -> existing Expert -> UDP; optional gRPC start placement only."""
import argparse
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO/'MoraiDrive'), str(REPO/'VLDrive'),
               str(REPO/'AIM2026/Expert/src'), str(REPO/'MoraiDrive/scenario_runner/legacy_expert')]
from gt_udp_protocol import pack_control, normalized_steering
from run_gt_udp_expert import running_controllers, write


UDP_STEERING_SIGNS = {'normal': -1.0, 'competition': 1.0}


def configure_udp_steering(config, sim_profile='normal'):
    # These profiles describe the two builds measured on this PC. Keep the
    # transport adaptation in this run's private config, outside Expert core.
    config['common']['realtime_steering_sign'] = UDP_STEERING_SIGNS[sim_profile]


def encode_planner_control(command, max_steer_rad):
    if command.rear_steer != 0:
        raise ValueError('현재 GPS UDP 실행은 후륜 조향을 지원하지 않습니다')
    return pack_control(command.accel, command.brake,
                        normalized_steering(command.front_steer, max_steer_rad))


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
    config.update(expert_speed_cap_kmh=args.speed_cap_kmh, route_debug_enabled=False)
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
    config['common']['perception_source'] = 'ros'  # Original tracked-object planner.
    config['common']['sync_grpc_inputs']['enabled'] = False
    configure_udp_steering(config, args.sim_profile)
    config['map']['traffic_light_control'] = False
    settings = config['planning']['traffic_light_stop']
    settings['stop_statuses'] = sorted(set(settings.get('stop_statuses', [1])) | {0})
    settings['hold_on_stale'] = True
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
    from inference.realtime_ve_objects import CHECKPOINTS
    checkpoint = CHECKPOINTS['P1']
    if not Path(checkpoint['path']).is_file() or not args.signal_head.is_file():
        raise FileNotFoundError('P1 가중치 또는 기존 좌회전 신호 head를 찾지 못했습니다')
    publisher = json.loads((REPO/'VLDrive/configs/vision_encoder_realtime_best_20261003.json').read_text())
    publisher.update(checkpoint=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
                     signal_head=str(args.signal_head.resolve()))
    publisher['runtime'].update(ego_source='gps_imu_udp', bind=args.bind_ip,
                                camera_receiver_mode='process', lidar_frame_policy='morai_cyclic_frame_timestamp',
                                control_pose_hz=50.)
    publisher['ports'].update(vehicle_status=args.status_port, gps=args.gps_port, imu=args.imu_port)
    publisher['runtime']['static_geometry_fusion'] = bool(args.static_geometry_fusion)
    publisher['runtime']['ignored_signals'] = list(args.ignore_signals)
    if args.gps_blackout_hold:
        hold = dict(enabled=True, max_hold_s=float(os.environ.get('VE_HOLD_MAX_S', 120.)), max_distance_m=250., latency_s=args.gps_latency_s)
        if args.simulate_gps_blackout:
            # Development only: discard real fixes inside the Expert Jamming section (tunnel) to imitate the blackout.
            section = json.loads((REPO/'AIM2026/Expert/src/config/jamming_section.json').read_text())
            hold['simulate_zone'] = dict(start_xy=section['start_enu'][:2], end_xy=section['end_enu'][:2], half_width_m=15.)
        publisher['runtime']['gps_blackout_hold'] = hold
    write(output/'publisher_config.json', publisher)
    return context, controller, policy, checkpoint


def planner_inputs(packet, control_pose=None, position_sources=('GPS',)):
    from AIM2026.Expert.src.vehicle_state import VehicleState
    from AIM2026.Expert.src.obstacle.object_info import ObjectInfo
    pose_packet = packet if control_pose is None else control_pose
    pose = pose_packet['ego_pose']
    if pose_packet.get('ego_position_source') not in position_sources or pose_packet.get('ego_status_source') != 'UDP_status_speed_IMU_heading':
        raise ValueError('GPS/IMU 입력 이외의 자차 위치가 전달됐습니다')
    values = [pose[k] for k in ('x', 'y', 'yaw_grid_rad', 'speed_mps')]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('자차 위치·방향·속도가 유효하지 않습니다')
    vehicle = VehicleState(*values)
    objects = [ObjectInfo(*box['center_world'], box['planning_speed_mps'], -1,
        name='ve_tracked_unknown:'+box['motion_state'],
        is_static=box['motion_state'] == 'stationary', size_x=box['length_m'], size_y=box['width_m'],
        heading_deg=box['planning_heading_deg'], unique_id=box['track_id']) for box in packet['boxes']]
    return vehicle, objects


def place_route_start(stub, actor_info, coordinates):
    """Setup writes only; no actor/signal/object GT is queried."""
    from morai.actor.actor_set_pb2 import EgoCruiseControl, SetTransformParam, SetVelocityParam
    from morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    x, y, z, yaw = coordinates
    transform = SetTransformParam(actor_info=actor_info)
    transform.transform.location.x, transform.transform.location.y, transform.transform.location.z = x, y, z
    transform.transform.rotation.z = yaw
    steps = [
        ('cruise_off', stub.SetVehicleEgoCruise, EgoCruiseControl(actor_info=actor_info, cruise_on=False)),
        ('velocity_zero', stub.SetVelocity, SetVelocityParam(actor_info=actor_info, velocity=0.)),
        ('route_start', stub.SetTransform, transform),
        ('settle_velocity', stub.SetVelocity, SetVelocityParam(actor_info=actor_info, velocity=0.))]
    for label, rpc, request in steps:
        response = rpc(request, timeout=1.)
        if response.status != STATUS_CODE_SUCCESS:
            raise RuntimeError('출발 위치 준비 실패 {}: {}'.format(label, response.description))


def reset_route_start(context, host):
    # Avoid SimulationWorld: its constructor/finalizer changes the scene.
    sys.path.insert(0, str(Path(context.global_cfg['paths']['grpc_src'])/'proto'))
    import grpc
    from morai.actor.actor_pb2_grpc import ActorStub
    from morai.common.object_info_pb2 import ObjectInfo
    from morai.common.object_identifier_pb2 import ObjectIdentifier
    target = os.environ.get('MORAI_GRPC_TARGET', '').strip() or '{}:{}'.format(
        os.environ.get('MORAI_GRPC_HOST', host) if os.environ.get('MORAI_GRPC_HOST', host) != 'auto' else host,
        os.environ.get('MORAI_GRPC_PORT', context.global_cfg['grpc'].get('port',7789)))
    actor = ObjectInfo(id=ObjectIdentifier(value='Ego'), object_type=1,
                       client_key=context.global_cfg['grpc'].get('client_key','kcity26_scenario_runner'))
    with grpc.insecure_channel(target) as channel:
        place_route_start(ActorStub(channel), actor, context.start_tf)
    time.sleep(.4)
    print('[출발] 공식 경로 시작 위치로 배치 완료. 주행 위치 확인은 GPS 사용.', flush=True)


def stop_child(process):
    if process is None:
        return
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', action='store_true', help='UDP 제어 송신. 생략하면 인지·계획만 검사')
    parser.add_argument('--from-start', action='store_true', help='출발 전 gRPC 설정 명령으로 공식 경로 시작 위치에 배치; 주행 입력은 GPS/VE 유지')
    parser.add_argument('--check', action='store_true', help='연결·GPU 실행 없이 경로·설정만 검사')
    parser.add_argument('--scenario-context', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--startup-timeout', type=float, default=90.)
    parser.add_argument('--speed-cap-kmh', type=float, default=60.)
    parser.add_argument('--sim-profile', choices=sorted(UDP_STEERING_SIGNS), default='normal',
                        help='현재 PC에서 확인한 UDP 조향 방향: normal=기본 MORAI, competition=대회판')
    parser.add_argument('--bind-ip', default='0.0.0.0')
    parser.add_argument('--sim-ip', default='auto')
    parser.add_argument('--status-port', type=int, default=8000)
    parser.add_argument('--gps-port', type=int, default=10506)
    parser.add_argument('--imu-port', type=int, default=10507)
    parser.add_argument('--control-port', type=int, default=7601)
    parser.add_argument('--control-source-port', type=int, default=7600)
    parser.add_argument('--signal-head', type=Path,
        default=REPO/'MoraiDrive/.runtime/signal_state_p1_20261014/run_003/signal_head_final.pt')
    parser.add_argument('--ignore-signals', type=lambda text: [x for x in text.split(',') if x], default=['LCS01'],
                        help='Comma-separated signal IDs the Expert ignores. Default LCS01: no signal stands there in the '
                             'competition (confirmed by the team), but the map has one and the VE head cannot read it. '
                             'Pass an empty string to ignore nothing.')
    parser.add_argument('--static-geometry-fusion', action='store_true',
                        help='stationary VE 박스를 world 좌표로 융합해 Expert에 전달 (기본: 프레임별 raw)')
    parser.add_argument('--moving-carry-s', type=float, default=0.,
                        help='움직임이 확인된 트랙의 근거가 잠깐 끊겨도 마지막 속도를 이 시간(초, 0~2) 유지 (기본 0=끔)')
    parser.add_argument('--stationary-hold-s', type=float, default=0.,
                        help='3초 이상 정지가 확인된 물체를 약한 근거 구간(박스 떨림, 트랙 재생성) 동안 정지로 유지 (초, 0~6; 기본 0=끔)')
    parser.add_argument('--slow-window-s', type=float, default=0.,
                        help='저속(0.8~3 m/s) 이동체용 긴 확인 창(초, 0.8 초과 2.4 이하; 기본 0=끔)')
    parser.add_argument('--gps-blackout-hold', action='store_true',
                        help='GPS 음영(blackout) 동안 마지막 fix에서 IMU 방향 + 상태 속도로 위치 유지 (기본: 정지)')
    parser.add_argument('--gps-latency-s', type=float, default=0.,
                        help='GPS 수신 전 지연(초). 음영 진입 시 마지막 fix를 속도x지연만큼 앞으로 보정 (기본 0, 실측 후 설정)')
    parser.add_argument('--simulate-gps-blackout', action='store_true',
                        help='개발용: Jamming 구간(터널) 안의 실제 GPS를 버려 음영을 흉내냄. --gps-blackout-hold 필요')
    args = parser.parse_args(argv)
    if args.simulate_gps_blackout and not args.gps_blackout_hold:
        parser.error('--simulate-gps-blackout requires --gps-blackout-hold')
    if not 0 <= args.gps_latency_s <= .5:
        parser.error('--gps-latency-s must be within 0..0.5')
    if args.from_start and not (args.drive or args.check):
        parser.error('--from-start는 --drive와 함께 사용하세요 (--check는 실제 배치 없음)')
    from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
    args.speed_cap_kmh = validate_max_velocity_kmh(args.speed_cap_kmh)
    if not math.isfinite(args.seconds) or args.seconds < 0 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error('seconds/startup-timeout 값이 유효하지 않습니다')
    ports = [args.status_port, args.gps_port, args.imu_port, args.control_port,
             args.control_source_port, 10501, 10502, 10503, 10504, 2368, 12050]
    if len(set(ports)) != len(ports) or not all(1 <= p <= 65535 for p in ports):
        parser.error('UDP 포트가 중복됐거나 유효하지 않습니다')
    output = args.output or REPO/'MoraiDrive/.runtime/ve_gps_expert'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    context, controller, policy, checkpoint = prepare(args, output)
    plan = dict(mode='drive' if args.drive else 'shadow', position='GPS', heading='IMU',
                speed='UDP_signed_status_kmh_div_3.6', obstacles='VE_P1_tracked', signals='existing_VE_heads',
                control='UDP_single_steer55', GT_inputs=False, grpc_used=args.from_start, ROS_used=False,
                grpc_scope='start_placement_writes_only' if args.from_start else None,
                from_start=args.from_start, start_transform=list(context.start_tf) if args.from_start else None,
                simulator_reset_or_spawn=args.from_start, sim_profile=args.sim_profile,
                scenario_context=str(args.scenario_context) if args.scenario_context else None,
                realtime_steering_sign=UDP_STEERING_SIGNS[args.sim_profile],
                UDP_positive_steering='right' if args.sim_profile == 'normal' else 'left',
                UDP_steering_units='normalized_desired_angle_div_max_angle',
                control_pose='latest_GPS_IMU_status_independent_of_VE',
                control_pose_hz=50., control_pose_max_age_s=.15,
                detection_transform_pose='unchanged_perception_snapshot',
                output=str(output))
    write(output/'plan.json', plan)
    if args.check:
        print(json.dumps(dict(status='CHECKED', **plan), ensure_ascii=False, indent=2))
        return 0
    conflicts = running_controllers()
    if conflicts:
        raise RuntimeError('기존 주행/VE 실행을 먼저 종료하세요: '+repr(conflicts))
    if args.sim_ip == 'auto':
        from scenario_runner.adapters.morai_grpc import _windows_host_gateway
        args.sim_ip = _windows_host_gateway()  # IP discovery only; no gRPC channel.
    from inference.ve_objects_udp import PORT, MAX_PACKET_BYTES, decode_packet
    from utils.common.ve_control_pose import SCHEMA as POSE_SCHEMA, MAX_AGE_S as POSE_MAX_AGE, decode_control_pose
    from inference.ve_object_tracking import VeObjectTracker
    from python_environment import build_analysis_environment
    from runtime_paths import ANALYSIS_PYTHON
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', PORT))
    receiver.setblocking(False)
    sender = process = None
    generation = str(uuid.uuid4())
    tracker = VeObjectTracker(static_geometry_fusion=args.static_geometry_fusion,
                              moving_carry_s=args.moving_carry_s, slow_window_s=args.slow_window_s,
                              stationary_hold_s=args.stationary_hold_s)
    position_sources = ('GPS', 'GPS_dead_reckoning') if args.gps_blackout_hold else ('GPS',)
    stopped = False
    old_handlers = {}
    def stop(*_):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    logs = {}
    started = time.monotonic()
    packet, last_sequence, last_tick, last_log = None, 0, started, 0.
    control_pose, pose_sequence, pose_count, pose_rejected = None, 0, 0, 0
    ready_since = missing_since = None
    count = commands = rejected = 0
    status, error, reason = 'STOPPED', None, 'waiting_for_VE_GPS'
    start_verified = not (args.from_start or args.scenario_context)
    try:
        for name in ('publisher', 'expert', 'inputs', 'control'):
            logs[name] = (output/(name+'.jsonl' if name in ('inputs', 'control') else name+'.log')).open('w', buffering=1)
        with contextlib.redirect_stdout(logs['expert']):
            from AIM2026.Expert.src.autonomous_driving import AutonomousDriving
            driving = AutonomousDriving()
            if os.environ.get('VE_CORRIDOR_TIMEOUT_S'):
                # Experiment only: the Jamming corridor search stops after 8 s of wall-clock time; allow longer.
                jam = getattr(driving.lattice_planning_process, 'recorded_jamming', None)
                if jam is not None:
                    jam.local.corridor_search_timeout = float(os.environ['VE_CORRIDOR_TIMEOUT_S'])
                    print('[실험] corridor_search_timeout=%ss' % os.environ['VE_CORRIDOR_TIMEOUT_S'], flush=True)
        if args.drive:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sender.bind((args.bind_ip, args.control_source_port))
        if args.from_start:
            for _ in range(5):
                sender.sendto(pack_control(0.,1.), (args.sim_ip,args.control_port))
                time.sleep(.03)
            reset_route_start(context, args.sim_ip)
        process = subprocess.Popen([str(ANALYSIS_PYTHON), '-u', '-B', str(REPO/'VLDrive/inference/ve_objects_udp.py'),
            '--config', str(output/'publisher_config.json'), '--generation', generation, '--seconds', '0',
            '--audit-dir', str(output)], env=build_analysis_environment(ANALYSIS_PYTHON),
            stdout=logs['publisher'], stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
        print('[VE+GPS] GPS 위치 · IMU 방향 · 상태 속도 · P1 장애물/기존 신호 head · 제어={}'.format(
            'UDP' if args.drive else '송신 없음'), flush=True)
        print('[조향] MORAI 프로필={} · 방향 부호={:+.0f} · 최대각 기준 정규화'.format(
            args.sim_profile, UDP_STEERING_SIGNS[args.sim_profile]), flush=True)
        print('[자세] GPS·IMU·상태 최신값 50Hz 별도 전달 → 제어 30Hz; VE 객체 좌표 변환 시점 유지', flush=True)
        print('[로그] '+str(output), flush=True)
        period = 1./driving.realtime_sampling_rate
        dump_paths = (output/'paths.jsonl').open('w', buffering=1) if os.environ.get('VE_DUMP_PATHS') == '1' else None
        last_path_dump = 0.
        while not stopped:
            now = time.monotonic()
            if process.poll() is not None:
                raise RuntimeError('VE 프로세스 종료: '+str(output/'publisher.log'))
            for _ in range(64):
                try:
                    data, _address = receiver.recvfrom(MAX_PACKET_BYTES+1)
                except BlockingIOError:
                    break
                is_pose = False
                try:
                    document = json.loads(data)
                    is_pose = document.get('schema') == POSE_SCHEMA
                    if is_pose:
                        candidate = decode_control_pose(document, generation, checkpoint['sha256'], time.monotonic_ns(),
                                                        position_sources)
                        if candidate['sequence'] > pose_sequence:
                            control_pose, pose_sequence = candidate, candidate['sequence']
                            pose_count += 1
                        continue
                    candidate = decode_packet(data, generation, checkpoint['sha256'], time.monotonic_ns())
                    if candidate['sequence'] <= last_sequence:
                        continue
                    last_sequence = candidate['sequence']
                    if candidate.get('ego_position_source') not in position_sources:
                        raise ValueError('GPS 자차 입력이 아닙니다')
                    packet = tracker.update(candidate)
                    policy.output(now)
                    if count == 0:
                        pose = packet['ego_pose']
                        policy.last_index = min(range(len(context.route_points)), key=lambda i:
                            (context.route_points[i][0]-pose['x'])**2+(context.route_points[i][1]-pose['y'])**2)
                    policy.accept(packet, context.route_points, time.monotonic())
                    count += 1
                    logs['inputs'].write(json.dumps(packet, allow_nan=False)+'\n')
                except (ValueError, KeyError, TypeError) as exc:
                    reason = str(exc)
                    if is_pose:
                        control_pose = None
                        pose_rejected += 1
                    else:
                        packet = None
                        policy.invalidate()
                        rejected += 1
            now = time.monotonic()
            perception_fresh = packet is not None and 0 <= now-packet['sensor_received_ns']/1e9 <= .5
            pose_fresh = control_pose is not None and 0 <= now-control_pose['sensor_received_ns']/1e9 <= POSE_MAX_AGE
            fresh = perception_fresh and pose_fresh
            code = policy.output(now)
            command = None
            if fresh:
                vehicle, objects = planner_inputs(packet, control_pose, position_sources)
                if not start_verified:
                    x, y, _z, _yaw = context.start_tf
                    if math.hypot(vehicle.position.x-x,vehicle.position.y-y) > 1.5:
                        raise RuntimeError('시작 위치 배치 후 GPS 좌표가 1.5m 이내로 확인되지 않았습니다')
                    start_verified = True
                    print('[PASS] 공식 시작 위치 GPS 확인 완료', flush=True)
                if policy.reason == 'off_route':
                    raise RuntimeError('GPS 위치가 기준 경로에서 6m 이상 벗어났습니다')
                traffic = [policy.target_id, code, 0, now]
                with contextlib.redirect_stdout(logs['expert']):
                    command, _path = driving.execute(vehicle, objects, traffic, sync_mode=False,
                        control_dt=max(.001, now-last_tick), ego_header_seq=pose_sequence,
                        ego_source_ns=control_pose['selected_samples']['vehicle_status'].get('source_stamp_ns'))
                if dump_paths is not None and now-last_path_dump >= .2:
                    # Development diagnostic (env VE_DUMP_PATHS=1): the path the planner handed to the controller.
                    last_path_dump = now
                    try:
                        points = [[float(getattr(p, 'x', p[0] if not hasattr(p, 'x') else 0.)),
                                   float(getattr(p, 'y', p[1] if not hasattr(p, 'y') else 0.))] for p in list(_path or [])[:80]]
                        dump_paths.write(json.dumps(dict(monotonic=now, ego=[vehicle.position.x, vehicle.position.y, vehicle.yaw],
                                                         speed=control_pose['ego_pose'].get('speed_mps'), path=points), allow_nan=False)+'\n')
                    except Exception as error:
                        dump_paths.write(json.dumps(dict(monotonic=now, error=repr(error)))+'\n')
                reason = 'ready'
                missing_since = None
                ready_since = now if ready_since is None else ready_since
            elif ready_since is not None:
                reason = 'GPS_IMU_or_VE_stale'
                missing_since = now if missing_since is None else missing_since
                if now-missing_since > 2:
                    raise RuntimeError('GPS/IMU/VE 입력이 2초 이상 유효하지 않습니다')
            if sender is not None:
                sender.sendto(encode_planner_control(command, driving.max_steering)
                              if command is not None else pack_control(0., 1.),
                              (args.sim_ip, args.control_port))
                commands += 1
            last_tick = now
            logs['control'].write(json.dumps(dict(monotonic=now, VE_sequence=last_sequence,
                control_pose_sequence=pose_sequence,
                control_pose_selected_samples=control_pose['selected_samples'] if pose_fresh else None,
                control_pose_age_ms=(now-control_pose['sensor_received_ns']/1e9)*1000 if pose_fresh else None,
                perception_age_ms=(now-packet['sensor_received_ns']/1e9)*1000 if perception_fresh else None,
                position=control_pose['ego_pose'] if fresh else None,
                perception_position=packet['ego_pose'] if perception_fresh else None,
                signal_id=policy.target_id, signal_code=code, signal_head=policy.head_source,
                vehicle_observed_control=control_pose.get('ego_control') if pose_fresh else None,
                planned_command=(dict(accel=float(command.accel), brake=float(command.brake),
                                      front_steer_rad=float(command.front_steer),
                                      UDP_steer_normalized=normalized_steering(command.front_steer,driving.max_steering)) if command is not None else
                                 dict(accel=0., brake=1., front_steer_rad=0.)),
                control_sent=sender is not None), allow_nan=False)+'\n')
            if now-last_log >= 2:
                row = dict(status='DRIVING' if args.drive and fresh else 'SHADOW' if fresh else 'WAITING',
                    elapsed_s=round(now-started, 1), VE_frames=count, rejected=rejected,
                    control_pose_frames=pose_count, control_pose_rejected=pose_rejected,
                    control_pose_age_ms=round((now-control_pose['sensor_received_ns']/1e9)*1000,1) if pose_fresh else None,
                    objects=len(packet['boxes']) if fresh else None,
                    GPS_pose=control_pose['ego_pose'] if fresh else None, signal_id=policy.target_id,
                    signal_code=code, signal_head=policy.head_source, reason=reason,
                    control_packets=commands, GT_inputs=False)
                write(output/'status.json', row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                last_log = now
            if ready_since is None and now-started > args.startup_timeout:
                raise RuntimeError('GPS·IMU·카메라·LiDAR·차량 상태 준비 시간 초과: '+reason)
            if args.seconds and ready_since is not None and now-ready_since >= args.seconds:
                status = 'COMPLETE'
                break
            time.sleep(max(.001, period-(time.monotonic()-now)))
    except Exception as exc:
        status, error = 'FAILED', str(exc)
        print('[FAIL] '+error, flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)       # the message alone (e.g. a bare KeyError) does not say where it came from
        sys.stdout.flush()
    finally:
        if sender is not None:
            try:
                for _ in range(10):
                    sender.sendto(pack_control(0., 1.), (args.sim_ip, args.control_port))
                    time.sleep(.03)
            except OSError as exc:
                status, error = 'FAILED', '종료 제동 송신 실패: '+str(exc)
            finally:
                sender.close()
        stop_child(process)
        receiver.close()
        for log in logs.values():
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        result = dict(status=status, error=error, VE_frames=count, control_packets=commands,
                      control_pose_frames=pose_count, control_pose_rejected=pose_rejected,
                      elapsed_s=time.monotonic()-started, plan=plan)
        write(output/'report.json', result)
        write(output/'status.json', result)
        print(json.dumps(dict(status=status, report=str(output/'report.json')), ensure_ascii=False), flush=True)
    return int(status == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
