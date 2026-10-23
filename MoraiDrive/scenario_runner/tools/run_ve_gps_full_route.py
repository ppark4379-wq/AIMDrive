#!/usr/bin/env python3
"""Existing full-loop scene lifecycle with the independent GPS/VE UDP driver."""
import argparse
import copy
import datetime
import json
import math
import os
import random
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'MoraiDrive'), str(REPO/'AIM2026/Expert/tools')]
from scenario_runner.main_runner import load_configuration, resolve_static_preset
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.scenarios.segment_modes.modes import MISSION_MODES
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.tools.ve_full_route_campaign import validate_aim2026_options
from run_ve_gps_expert import stop_child
from run_gt_udp_expert import running_controllers, write


TUNNEL_LAYOUT_LIBRARY = REPO/'MoraiDrive/scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/search/approved_candidates.json'


def pick_tunnel_layout(spec, seed):
    """One of the team's 121 curated tunnel layouts: an index, or 'random' (reproducible from the seed)."""
    rows = json.loads(TUNNEL_LAYOUT_LIBRARY.read_text())['candidates']
    if str(spec) == 'random':
        index = random.Random(seed).randrange(len(rows))
    else:
        index = int(spec)
        if not 0 <= index < len(rows):
            raise ValueError('--tunnel-layout must be 0..%d or random' % (len(rows)-1))
    row = rows[index]
    return dict(index=index, sha256=row['candidate_sha256'], requested_pattern=row.get('requested_pattern'),
                obstacles=[{k: o[k] for k in ('actor_kind', 'model', 'scale', 'yaw_deg', 'position', 'station_m',
                                              'lateral_m', 'planned_bbox')} for o in row['obstacles']])


def configure_scenario(config, args):
    result = copy.deepcopy(config)
    # Do not start the old ROS VE session or a second Expert process.
    for key in list(result):
        if key.startswith('ve_'):
            result.pop(key)
    result.update(clean_road_mode=False, drive_control_mode='aimdrive_expert',
                  expert_speed_cap_kmh=args.speed_cap_kmh, expert_ros_remaps=[],
                  expert_traffic_light_control=False, max_laps=1,
                  route_setup_attempts=1, random_seed=args.seed,
                  route_debug_enabled=False, ve_stop_before_tunnel=args.stop_before_tunnel,
                  static_verified_pair=not bool(args.static_preset))
    result.setdefault('sync_runtime', {})['enabled'] = False
    result.setdefault('highway_merge_static_vehicle', {})['enabled'] = False
    result['highway_merge_static_vehicle_explicitly_disabled'] = True
    result.setdefault('traffic_light_wait_override', {})['enabled'] = False
    result.setdefault('progress_watchdog', {}).update(enabled=True, timeout_sec=180.)
    if args.ego_trace:
        result['ego_trace_path'] = str(args.ego_trace.resolve())
    if args.static_preset:
        result['static_obstacle_preset'] = resolve_static_preset(args.static_preset)
    else:
        result.pop('static_obstacle_preset', None)
    if args.mission:
        # A bounded mission (e.g. the tunnel from its entrance): the scenario slices the official route itself.
        result['mission_mode'] = args.mission
    if args.tunnel_dynamic_speed is not None:
        result.setdefault('tunnel_dynamic', {})['speed_mps'] = args.tunnel_dynamic_speed
    if args.tunnel_layout is not None:
        result['gps_tunnel_layout'] = pick_tunnel_layout(args.tunnel_layout, args.seed if args.seed is not None else 0)
    if args.tunnel_candidate_id:
        # set_waypoint pedestrians jump to the goal in MORAI; the candidate path uses native straight movement.
        result.setdefault('tunnel_dynamic', {}).update(trial_candidate_id=args.tunnel_candidate_id, candidate_policy='trial')
    if args.tunnel_vehicle_behavior:
        result.setdefault('tunnel_vehicle_dynamic', {}).update(behavior=args.tunnel_vehicle_behavior, random_behavior=False)
    elif args.tunnel_vehicle_random:
        result.setdefault('tunnel_vehicle_dynamic', {})['random_behavior'] = True
    return result


class GpsController:
    """Lifecycle only; perception/control are owned by the tested GPS driver."""
    def __init__(self, base, context):
        self.base, self.context = base, context
        self.output = context.gps_output/'driver'
        self.process = self.log = None
        self.context_path = context.gps_output/'scenario_context.json'
        write(self.context_path, dict(schema='ve_gps_scenario_context_v1',
            global_config=context.global_cfg, scenario_config=context.cfg,
            route_points=[list(p) for p in context.route_points]))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def command(self, check=False):
        args = self.context.gps_args
        return [sys.executable, '-u', '-B', str(REPO/'AIM2026/Expert/tools/run_ve_gps_expert.py'),
                '--check' if check else '--drive', '--scenario-context', str(self.context_path),
                '--output', str(self.output), '--speed-cap-kmh', str(args.speed_cap_kmh),
                '--sim-profile', args.sim_profile, '--signal-head', str(args.signal_head),
                '--bind-ip', args.bind_ip, '--sim-ip', args.sim_ip,
                '--status-port', str(args.status_port), '--gps-port', str(args.gps_port),
                '--imu-port', str(args.imu_port), '--control-port', str(args.control_port),
                '--control-source-port', str(args.control_source_port)] + (
                    ['--aim2026-config', str(args.aim2026_config)] if getattr(args, 'aim2026_config', None) else []) + (
                    ['--static-geometry-fusion'] if getattr(args, 'static_geometry_fusion', False) else []) + (
                    ['--moving-carry-s', str(args.moving_carry_s)] if getattr(args, 'moving_carry_s', 0.) else []) + (
                    ['--slow-window-s', str(args.slow_window_s)] if getattr(args, 'slow_window_s', 0.) else []) + (
                    ['--stationary-hold-s', str(args.stationary_hold_s)] if getattr(args, 'stationary_hold_s', 0.) else []) + (
                    ['--ignore-signals', ','.join(args.ignore_signals)] if getattr(args, 'ignore_signals', None) else []) + (
                    ['--gps-blackout-hold', '--gps-latency-s', str(args.gps_latency_s)]
                    if getattr(args, 'gps_blackout_hold', False) else []) + (
                    ['--simulate-gps-blackout'] if getattr(args, 'simulate_gps_blackout', False) else [])

    def start_process(self):
        if self.is_running():
            return
        command = self.command()
        write(self.context.gps_output/'driver_command.json', command)
        self.log = (self.context.gps_output/'driver.console.log').open('w', buffering=1)
        self.process = subprocess.Popen(command, cwd=REPO, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        print('[GPS+VE 주행 로그] '+str(self.context.gps_output/'driver.console.log'), flush=True)
        deadline = time.monotonic()+100.
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError('GPS 실행기 종료: '+str(self.context.gps_output/'driver.console.log'))
            status = self.output/'status.json'
            if status.is_file() and json.loads(status.read_text()).get('status') == 'DRIVING':
                print('[PASS] GPS·IMU·UDP 속도 / VE 장애물·신호 / UDP 제어 준비 완료', flush=True)
                return
            time.sleep(.1)
        raise RuntimeError('GPS 실행기 준비 시간 초과: '+str(self.context.gps_output/'driver.console.log'))

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def stop_process(self):
        stop_child(self.process)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


class GpsDriverMixin:
    def require_async_ros_control_path(self):
        print('[GPS+VE] ROS 제어 연결 불필요; GPS 실행기의 센서·UDP 검사 사용', flush=True)

    def setup_expert_controller(self):
        super().setup_expert_controller()
        self.expert_controller = GpsController(self.expert_controller, self)


class GpsFullLoop(GpsDriverMixin, FullLoopScenario):
    pass


class LayoutMission(MissionScenario):
    """Tunnel mission that can spawn a team-library layout; the driver (GPS+VE or the GT Expert) is chosen by the subclass."""
    udp_stop = False

    def setup(self):
        super().setup()          # prepare_mission resets the cross-track limit for the empty-road mission
        if self.cfg.get('gps_tunnel_layout'):
            # Same limit the team's static-obstacle mission uses: bypassing an obstacle leaves the centreline.
            self.cfg['expert_max_cross_track_error_m'] = 5.0
            # Development knobs for "how long does the planner need": longer no-progress / total limits.
            if os.environ.get('VE_PROGRESS_WATCHDOG_S'):
                self.cfg['progress_watchdog']['timeout_sec'] = float(os.environ['VE_PROGRESS_WATCHDOG_S'])
            if os.environ.get('VE_RUN_TIMEOUT_S'):
                self.cfg['timeout_sec'] = float(os.environ['VE_RUN_TIMEOUT_S'])

    def _spawn_layout_item(self, layout, number, item):
        from proto.morai.common.type_pb2 import Vector3
        label = 'tunnel_layout_%d_%d' % (layout['index'], number)
        transform = self.grpc.make_transform(*(item['position'][axis] for axis in 'xyz'), item['yaw_deg'])
        if item['actor_kind'] == 'vehicle':
            actor = self.grpc.world.spawn_vehicle(transform=transform, model_name=item['model'], label=label,
                                                  request_id=label, velocity=0.0, pause=False, multi_ego=False)
        else:
            actor = self.grpc.world.spawn_obstacle(transform=transform, model_name=item['model'], label=label,
                                                   request_id=label, scale=Vector3(x=1., y=1., z=1.))
        if actor is None:
            raise RuntimeError('터널 배치 스폰 실패: '+label+' '+item['model'])
        self.scenario_actors.append(actor)
        self.layout_actors[number] = actor
        if item['actor_kind'] == 'obstacle':
            # MORAI advances physics between SpawnObstacle and SetPhysics, so some props (barrels, boxes, firewalls) fall
            # through the road before they exist in the inventory. The open-road static scenario freezes them with
            # physics off + the transform re-applied; do the same and prove the pose holds.
            from scenario_runner.scenarios.full_loop_parts.static_obstacle import lock_static_obstacle_transform
            state, diagnostics = lock_static_obstacle_transform(
                self, actor, str(actor.get_id()), transform, tuple(item['position'][axis] for axis in 'xyz'))
            self.layout_lock_diagnostics[number] = diagnostics
            if state is None:
                raise RuntimeError('터널 배치 물체를 도로 높이에 고정하지 못했습니다: %s %s %s' % (
                    label, item['model'], json.dumps(diagnostics[-1], ensure_ascii=False, default=str)[:300]))
        return actor

    def _layout_presence(self, layout):
        """Read the simulator inventory back: does each layout object stand where it was put (xy and height)?"""
        from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.safety import inventory
        rows = inventory(self.grpc.world._sim_adapter._actor_stub)
        report = []
        for item in layout['obstacles']:
            spot, z = (item['position']['x'], item['position']['y']), item['position']['z']
            near = [(math.dist((s.transform.location.x, s.transform.location.y), spot), s.transform.location.z, identity)
                    for identity, s in rows.items() if str(identity) != 'Ego']
            best = min(near, default=(None, None, None))
            report.append(dict(model=item['model'], expected=spot, nearest_actor_distance_m=best[0], actor_z=best[1], actor_id=best[2],
                               present=best[0] is not None and best[0] <= 2.0 and abs(best[1]-z) <= 1.0))
        return report

    def _ensure_layout(self, layout, reason):
        """Respawn whatever is missing or has fallen through the road; returns the final presence report."""
        time.sleep(1.5) if reason == 'setup' else None
        report = self._layout_presence(layout)
        for number, (item, row) in enumerate(zip(layout['obstacles'], report), 1):
            if row['present']:
                continue
            print('[TunnelLayout] %s: %s 없음/낙하(거리 %s, z %s) → 다시 스폰' % (reason, item['model'], row['nearest_actor_distance_m'], row['actor_z']), flush=True)
            self.layout_respawns.append(dict(reason=reason, model=item['model'], nearest=row['nearest_actor_distance_m'], z=row['actor_z']))
            old = self.layout_actors.get(number)
            if old is not None:
                try:
                    old.destroy()
                except Exception as error:
                    print('[TunnelLayout] 정리 경고: '+repr(error)[:100], flush=True)
                if old in self.scenario_actors:
                    self.scenario_actors.remove(old)
            self._spawn_layout_item(layout, number, item)
        if self.layout_respawns and any(not row['present'] for row in report):
            time.sleep(1.5)
            report = self._layout_presence(layout)
        return report

    def spawn_static_obstacle(self):
        layout = self.cfg.get('gps_tunnel_layout')
        if not layout:
            return super().spawn_static_obstacle()
        # Plain spawn of the library layout (same models/poses/yaw the team validated); the VE, not this runner, must see them.
        self.layout_actors, self.layout_respawns, self.layout_checks_done, self.layout_lock_diagnostics = {}, [], set(), {}
        for number, item in enumerate(layout['obstacles'], 1):
            self._spawn_layout_item(layout, number, item)
        report = self._ensure_layout(layout, 'setup')
        write(self.gps_output/'tunnel_layout.json', layout)
        for item, row in zip(layout['obstacles'], report):
            print('[TunnelLayout] %s %s station=%.1f lateral=%.2f 시뮬 확인=%s (가장 가까운 액터 %s m, z %s)' % (
                item['actor_kind'], item['model'], item['station_m'], item['lateral_m'], row['present'],
                None if row['nearest_actor_distance_m'] is None else round(row['nearest_actor_distance_m'], 2),
                None if row['actor_z'] is None else round(row['actor_z'], 1)), flush=True)
        write(self.gps_output/'tunnel_layout_presence.json', dict(setup=report, respawns=self.layout_respawns))

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s, *args, **kwargs):
        result = super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s, *args, **kwargs)
        layout = self.cfg.get('gps_tunnel_layout')
        if layout and getattr(self, 'layout_actors', None):
            # The tunnel road collider may load late: check again as the Ego closes in, and respawn what fell through.
            spot = (layout['obstacles'][0]['position']['x'], layout['obstacles'][0]['position']['y'])
            distance = math.dist((float(ego_state['x']), float(ego_state['y'])), spot)
            for limit in (65., 45.):
                if distance <= limit and limit not in self.layout_checks_done:
                    self.layout_checks_done.add(limit)
                    report = self._ensure_layout(layout, 'ego<%dm' % limit)
                    write(self.gps_output/'tunnel_layout_presence.json',
                          dict(setup=report, respawns=self.layout_respawns, last_check=limit))
        return result

    def run(self):
        # The tunnel mission verifies its final stop with an exclusive ROS /ctrl_cmd brake owner. This runner has
        # no ROS control path (the driver stops the car over UDP), so that check cannot succeed and is skipped.
        from scenario_runner.scenarios.segment_modes import tunnel_baseline
        original = tunnel_baseline.park

        def udp_stop(context, **_):
            print('[GPS+VE] ROS 정지 검증 생략: 드라이버가 UDP로 정지합니다', flush=True)

        if self.udp_stop:
            tunnel_baseline.park = udp_stop
        original_occupancy = tunnel_baseline._check_occupancy
        if self.cfg.get('gps_tunnel_layout'):
            # The baseline mission demands an empty road; here the library layout is the test fixture itself.
            def layout_occupancy(context, elapsed):
                try:
                    return original_occupancy(context, elapsed)
                except RuntimeError as error:
                    prefix = 'Tunnel baseline road is occupied: '
                    if not str(error).startswith(prefix):
                        raise
                    # The blockers are recorded before the raise: accept only those standing on a layout obstacle.
                    spots = [(o['position']['x'], o['position']['y']) for o in context.cfg['gps_tunnel_layout']['obstacles']]
                    foreign = [row for row in context.tunnel_baseline_report['last_blocking_actors']
                               if min(math.dist(row['world_bbox_center_xyz'][:2], spot) for spot in spots) > 4.]
                    if foreign:       # only the layout's own actors may stand on the road
                        raise
                    states, ego_id = tunnel_baseline._inventory(context)
                    state = next(item for item in states if str(item.actor_info.id.value) == ego_id)
                    location = state.transform.location
                    return [float(location.x), float(location.y), float(location.z)]
            tunnel_baseline._check_occupancy = layout_occupancy
        try:
            return super().run()
        finally:
            tunnel_baseline.park = original
            tunnel_baseline._check_occupancy = original_occupancy


class GpsMission(GpsDriverMixin, LayoutMission):
    udp_stop = True


class GtLayoutMission(LayoutMission):
    """The standard GT/ROS Expert drives; used only to give the passive VE recorder a pass along each library layout."""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='시뮬레이터 변경 없이 경로·입력 설정 검사')
    parser.add_argument('--stop-before-tunnel', action='store_true')
    parser.add_argument('--mission', choices=sorted(MISSION_MODES),
                        help='풀루프 대신 미션 구간만 주행 (예: full_loop_tunnel_normal = 터널 입구 약 40m 전부터)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--static-preset')
    parser.add_argument('--ego-trace', type=Path)
    parser.add_argument('--speed-cap-kmh', '--expert-speed-cap-kmh', type=float, default=60.)
    parser.add_argument('--sim-profile', choices=('normal', 'competition'), default='normal')
    parser.add_argument('--signal-head', '--ve-signal-head', type=Path,
        default=Path(os.environ.get('VE_SIGNAL_HEAD', str(REPO/'MoraiDrive/.runtime/signal_state_p1_20261014/run_003/signal_head_final.pt'))))
    parser.add_argument('--ignore-signals', type=lambda text: [x for x in text.split(',') if x], default=[],
                        help='Comma-separated signal IDs the Expert ignores this run (static-obstacle tests only)')
    parser.add_argument('--moving-carry-s', type=float, default=0., help='tracker: 움직임 확인 트랙의 속도 유지 시간(초, 0=끔)')
    parser.add_argument('--stationary-hold-s', type=float, default=0., help='tracker: 3초 이상 정지한 물체의 정지 유지 시간(초, 0=끔)')
    parser.add_argument('--slow-window-s', type=float, default=0., help='tracker: 저속 이동체 확인 창(초, 0=끔)')
    parser.add_argument('--static-geometry-fusion', action='store_true',
                        help='stationary VE 박스를 world 좌표로 융합해 Expert에 전달 (기본: 프레임별 raw)')
    parser.add_argument('--gt-expert', action='store_true',
                        help='GT/ROS Expert가 주행(VE는 별도 관찰기가 옆에서 기록만 함). --mission full_loop_tunnel_normal 필요')
    parser.add_argument('--tunnel-layout',
                        help='full_loop_tunnel_normal 미션에 팀원 터널 정적 라이브러리(121개)의 배치 하나를 스폰: 0~120 또는 random(--seed로 재현)')
    parser.add_argument('--tunnel-candidate-id',
                        help='full_loop_tunnel_dynamic 미션에서 터널 후보 경로(예: cross_left_to_right_01)를 native 직선 이동으로 사용')
    parser.add_argument('--tunnel-dynamic-speed', type=float,
                        help='full_loop_tunnel_dynamic 미션의 보행자 속도(m/s, 0 초과 5 이하; 기본 2.5)')
    vehicle_behaviors = ('vehicle_slow', 'vehicle_sudden_brake', 'vehicle_oncoming', 'vehicle_cut_in')
    parser.add_argument('--tunnel-vehicle-behavior', choices=vehicle_behaviors,
                        help='full_loop_tunnel_vehicle 미션의 터널 안 차량 행동 하나를 고정')
    parser.add_argument('--tunnel-vehicle-random', action='store_true',
                        help='full_loop_tunnel_vehicle 미션에서 --seed로 행동을 무작위 선택')
    parser.add_argument('--aim2026-config', type=Path,
                        help='Optional AIM2026 YAML sensor/DR backend for the existing common Expert.')
    parser.add_argument('--gps-blackout-hold', action='store_true',
                        help='GPS 음영 동안 IMU 방향 + 상태 속도로 위치 유지 (터널 통과 시험에는 --stop-before-tunnel 없이 사용)')
    parser.add_argument('--gps-latency-s', type=float, default=0.,
                        help='음영 진입 시 마지막 GPS fix 지연 보정(초), 기본 0')
    parser.add_argument('--simulate-gps-blackout', action='store_true',
                        help='개발용: 터널(Jamming 구간) 안의 실제 GPS를 버려 음영 흉내. --gps-blackout-hold 필요')
    parser.add_argument('--bind-ip', default='0.0.0.0')
    parser.add_argument('--sim-ip', default='auto')
    for name, port in (('status',8000), ('gps',10506), ('imu',10507), ('control',7601), ('control-source',7600)):
        parser.add_argument('--'+name+'-port', type=int, default=port)
    args = parser.parse_args(argv)
    validate_aim2026_options(args, parser)
    if args.simulate_gps_blackout and not args.gps_blackout_hold:
        parser.error('--simulate-gps-blackout requires --gps-blackout-hold')
    if not 0 <= args.gps_latency_s <= .5:
        parser.error('--gps-latency-s must be within 0..0.5')
    if args.gt_expert and args.mission != 'full_loop_tunnel_normal':
        parser.error('--gt-expert requires --mission full_loop_tunnel_normal')
    if args.tunnel_layout is not None and args.mission != 'full_loop_tunnel_normal':
        parser.error('--tunnel-layout requires --mission full_loop_tunnel_normal')
    if args.tunnel_candidate_id and args.mission != 'full_loop_tunnel_dynamic':
        parser.error('--tunnel-candidate-id requires --mission full_loop_tunnel_dynamic')
    if args.tunnel_dynamic_speed is not None and (args.mission != 'full_loop_tunnel_dynamic' or not 0. < args.tunnel_dynamic_speed <= 5.):
        parser.error('--tunnel-dynamic-speed requires --mission full_loop_tunnel_dynamic and 0 < speed <= 5 m/s')
    if (args.tunnel_vehicle_behavior or args.tunnel_vehicle_random) and args.mission != 'full_loop_tunnel_vehicle':
        parser.error('--tunnel-vehicle-* requires --mission full_loop_tunnel_vehicle')
    if args.tunnel_vehicle_behavior and args.tunnel_vehicle_random:
        parser.error('--tunnel-vehicle-behavior and --tunnel-vehicle-random are exclusive')
    if args.mission and (args.stop_before_tunnel or args.static_preset):
        parser.error('--mission cannot be combined with --stop-before-tunnel or --static-preset')
    from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
    args.speed_cap_kmh = validate_max_velocity_kmh(args.speed_cap_kmh)
    args.signal_head = args.signal_head.expanduser().resolve()
    if not args.aim2026_config and not args.signal_head.is_file():
        parser.error('신호 head 없음: '+str(args.signal_head))
    ports = [args.status_port,args.gps_port,args.imu_port,args.control_port,args.control_source_port,
             10501,10502,10503,10504,2368,12050]
    if len(set(ports)) != len(ports) or not all(1 <= p <= 65535 for p in ports):
        parser.error('UDP 포트가 중복됐거나 유효하지 않습니다')
    if not args.check and not args.gt_expert:       # GT mode runs beside the passive VE recorder on purpose
        conflicts = running_controllers()
        if conflicts:
            raise RuntimeError('기존 주행/VE 실행을 먼저 종료하세요: '+repr(conflicts))
    global_cfg, _zone, base = load_configuration()
    output = REPO/'MoraiDrive/scenario_runner/.runtime/ve_gps_full_route'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    global_cfg['paths']['runtime_root'] = str(output)
    config = configure_scenario(base, args)
    config['profile_state_path'] = str(output/'profile_state.json')
    config.setdefault('ego_trace_path', str(output/'scenario_ego_trace.csv'))
    plan = dict(driving_inputs=dict(position='GPS',heading='IMU',speed='UDP_status',
                obstacles='VE_P1_tracked',signals='existing_VE_heads',control='UDP'),
                GT_driving_inputs=False, ROS_driving_inputs=False,
                grpc_scope='scene_setup_NPC_events_collision_and_completion_monitoring_only',
                speed_cap_kmh=args.speed_cap_kmh, stop_before_tunnel=args.stop_before_tunnel,
                gps_blackout_hold=args.gps_blackout_hold, gps_latency_s=args.gps_latency_s,
                simulate_gps_blackout=args.simulate_gps_blackout,
                aim2026_config=str(args.aim2026_config) if args.aim2026_config else None,
                mission=args.mission, tunnel_vehicle_behavior=args.tunnel_vehicle_behavior,
                tunnel_dynamic_speed=args.tunnel_dynamic_speed,
                tunnel_candidate_id=args.tunnel_candidate_id,
                tunnel_layout=config.get('gps_tunnel_layout'),
                static_verified_pair=config['static_verified_pair'], highway_merge_static_vehicle=False,
                output=str(output))
    write(output/'plan.json', plan)
    grpc = None
    if args.check:
        client = SimpleNamespace(make_transform=lambda *v:v)
    else:
        from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
        grpc = MoraiGrpcClient(global_cfg)
        grpc.connect()
        client = grpc
        if args.sim_ip == 'auto':
            args.sim_ip = grpc.host
    scenario_class = (GtLayoutMission if args.gt_expert else GpsMission) if args.mission else GpsFullLoop
    scenario = scenario_class(grpc_client=client, map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
                              global_cfg=global_cfg, scenario_cfg=config)
    scenario.gps_args, scenario.gps_output = args, output
    if args.check:
        if args.mission:
            from scenario_runner.scenarios.segment_modes import events
            events.prepare_mission(scenario)
        scenario.route_definition = scenario.load_route_definition()
        scenario.prepare_full_loop_route()
        scenario.setup_expert_controller()
        result = subprocess.run(scenario.expert_controller.command(check=True), cwd=REPO)
        if result.returncode:
            return result.returncode
        print(json.dumps(dict(status='CHECKED', **plan), ensure_ascii=False, indent=2))
        return 0
    def interrupt(*_):
        raise KeyboardInterrupt
    old_term = signal.signal(signal.SIGTERM, interrupt)
    status, error = 'COMPLETE', None
    try:
        print('[시험] GPS+VE 풀루트 1회 · 속도 상한 {}km/h · 정적 배치/이동 NPC 유지'.format(args.speed_cap_kmh), flush=True)
        print('[입력] 주행 GT 대체 없음. gRPC는 장면 구성·NPC 이벤트·충돌/완주 판정에만 사용.', flush=True)
        print('[로그] '+str(output), flush=True)
        scenario.run()
    except KeyboardInterrupt:
        status = 'STOPPED'
    except Exception as exc:
        status, error = 'FAILED', str(exc)
        print('[FAIL] '+error, flush=True)
    finally:
        try:
            grpc.stop()
        finally:
            signal.signal(signal.SIGTERM, old_term)
            write(output/'report.json', dict(status=status,error=error,plan=plan))
            print(json.dumps(dict(status=status,report=str(output/'report.json')),ensure_ascii=False), flush=True)
    return int(status == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
