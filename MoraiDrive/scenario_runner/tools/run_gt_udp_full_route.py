#!/usr/bin/env python3
"""Separate GT filming entry point: UDP Ego/objects/control, gRPC or UDP signals."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'MoraiDrive'), str(REPO / 'MoraiDrive/scenario_runner/legacy_expert')]
from scenario_runner.main_runner import load_configuration
from scenario_runner.adapters.expert_controller import ExpertController
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.scenarios.segment_modes import events as mission_events
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.utils.highway_density import add_highway_density_argument, apply_highway_density
from run_gt_udp_expert import running_controllers, write


def configure_scenario(config, args):
    result = copy.deepcopy(config)
    apply_highway_density(result, getattr(args, 'highway_density', None))
    result.setdefault('sync_runtime', {})['enabled'] = False
    result['clean_road_mode'] = False
    result['expert_speed_cap_kmh'] = args.speed_cap_kmh
    result['expert_ros_remaps'] = []
    result['static_verified_pair'] = True
    result.setdefault('highway_merge_static_vehicle', {})['enabled'] = False
    result['highway_merge_static_vehicle_explicitly_disabled'] = True
    # Natural GT signal phases, including the left-turn arrow. No forced green.
    result.setdefault('traffic_light_wait_override', {})['enabled'] = False
    # Retain stop-line binding for pedestrian placement; the existing guard
    # is a scenario event policy, not an Expert perception source.
    result['ve_stop_before_tunnel'] = args.stop_before_tunnel
    result['route_debug_enabled'] = False
    result['random_seed'] = args.seed
    if getattr(args, 'scenario', 'full_loop') == 'full_loop_highway':
        result['mission_mode'] = 'full_loop_highway'
        result['ve_stop_before_tunnel'] = False
    for key in list(result):
        if key.startswith('ve_') and key != 've_stop_before_tunnel':
            result.pop(key)
    return result


class UdpController:
    def __init__(self, base, args, output, before_start=None):
        self.base, self.args, self.output = base, args, output
        self.before_start = before_start
        self.process = None
        self.log = None
        doc = json.loads(Path(base.runtime_config_path).read_text())
        doc['common']['perception_source'] = 'ros'
        doc['common']['sync_grpc_inputs']['enabled'] = False
        doc['map']['traffic_light_control'] = False
        if args.signal_source == 'grpc':
            settings = doc['planning']['traffic_light_stop']
            settings['stop_statuses'] = sorted(set(settings.get('stop_statuses', [1])) | {0})
            settings['hold_on_stale'] = True
        for key in ('jamming_scene_geometry', 'jamming_trial_contract', 'jamming_trial_diagnostics_path'):
            doc['planning'].pop(key, None)
        write(Path(base.runtime_config_path), doc)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def start_process(self):
        if self.is_running():
            return
        if self.before_start is not None:
            self.before_start()
        self.output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-B', str(REPO / 'MoraiDrive/scenario_runner/legacy_expert/run_gt_udp_expert.py'),
                   '--config', self.base.runtime_config_path, '--path', self.base.runtime_path_csv,
                   '--output', str(self.output), *network_args(self.args)]
        self.log = (self.output / 'expert.log').open('w', buffering=1)
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, cwd=REPO)
        print('[GT UDP Expert] 로그: ' + str(self.output / 'expert.log'), flush=True)
        deadline = time.monotonic() + 90.
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('GT UDP Expert 시작 실패: ' + str(self.output / 'expert.log'))
            status_path = self.output / 'status.json'
            if status_path.is_file():
                state = json.loads(status_path.read_text())
                if state['status'] == 'DRIVING':
                    print('[PASS] GT 자차·장애물·제어 UDP · 신호={} 연결 완료'.format(self.args.signal_source), flush=True)
                    return
            time.sleep(.1)
        raise RuntimeError('GT UDP Expert 시작 시간 초과: ' + str(self.output / 'expert.log'))

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def stop_process(self):
        if self.process is not None:
            for sig, timeout in ((signal.SIGINT, 5.), (signal.SIGTERM, 3.), (signal.SIGKILL, 2.)):
                if not self.is_running():
                    break
                try:
                    os.killpg(self.process.pid, sig)
                    self.process.wait(timeout=timeout)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


class GtUdpFullLoop(FullLoopScenario):
    """Only this launcher swaps transport; ordinary VE runner stays intact."""
    def require_async_ros_control_path(self):
        print('[GT UDP] ROS 차량 연결 검사 생략; 별도 UDP 입력 검사 사용', flush=True)

    def setup_expert_controller(self):
        super().setup_expert_controller()
        self.expert_controller = UdpController(
            self.expert_controller, self.gt_args, self.gt_output / 'expert',
            before_start=self.check_startup_contact)

    def check_startup_contact(self):
        # Check after placing Ego, before the UDP backend can send acceleration.
        # An isolated 'Ego' is not proof of harmless stale/self contact.
        state = self.get_ego_motion_state()
        collisions = sorted({str(value).strip() for value in state.get('collision_objects', [])
                             if str(value).strip()})
        write(self.gt_output / 'startup_contact.json', dict(
            phase='before_UDP_Expert_start', raw_state=state,
            collision_objects=collisions, control_backend_started=False))
        if collisions:
            raise RuntimeError(
                '출발 전부터 MORAI 충돌 표시가 남아 있어 UDP Expert를 시작하지 않았습니다: '
                + repr(collisions) + '. 자차 Reset 후 Play 상태에서 다시 확인하세요. '
                + '진단: ' + str(self.gt_output / 'startup_contact.json'))


class GtUdpMission(GtUdpFullLoop, MissionScenario):
    """Use existing segment placement/events with the same UDP Expert."""


def scenario_class(args):
    return GtUdpMission if getattr(args, 'scenario', 'full_loop') == 'full_loop_highway' else GtUdpFullLoop


def run_scenarios(grpc, global_config, config, args, output):
    """Each completed run owns a fresh scenario, controller and runtime folder."""
    sequence = 1
    limit = args.max_runs if args.max_runs is not None else (None if args.repeat else 1)
    while True:
        run_output = output / ('run_{:04d}'.format(sequence))
        run_output.mkdir(parents=True, exist_ok=True)
        run_global = copy.deepcopy(global_config)
        run_global['paths']['runtime_root'] = str(run_output)
        scenario = scenario_class(args)(grpc_client=grpc,
            map_loader=MGeoMapLoader(run_global['paths']['mgeo_root']),
            global_cfg=run_global, scenario_cfg=copy.deepcopy(config))
        scenario.gt_args, scenario.gt_output = args, run_output
        print('[GT UDP] scenario={} sequence={}'.format(args.scenario, sequence), flush=True)
        scenario.run()
        if limit is not None and sequence >= limit:
            return
        sequence += 1
        print('[GT UDP] 구간 시작점에서 다음 회차를 시작합니다.', flush=True)
        time.sleep(2.)


def network_args(args):
    return ['--bind-ip', args.bind_ip, '--sim-ip', args.sim_ip, '--ego-port', str(args.ego_port),
            '--object-port', str(args.object_port), '--signal-port', str(args.signal_port),
            '--control-port', str(args.control_port), '--control-source-port', str(args.control_source_port),
            '--control-format', args.control_format,
            '--signal-source', args.signal_source, '--grpc-target', args.grpc_target]


def prepare_controller(global_config, config, args, output):
    context = scenario_class(args)(
        grpc_client=SimpleNamespace(make_transform=lambda *values: values),
        map_loader=MGeoMapLoader(global_config['paths']['mgeo_root']),
        global_cfg=global_config, scenario_cfg=copy.deepcopy(config))
    if isinstance(context, GtUdpMission):
        mission_events.prepare_mission(context)
    context.route_definition = context.load_route_definition()
    context.prepare_full_loop_route()
    context.gt_args, context.gt_output = args, output
    context.setup_expert_controller()
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_highway_density_argument(parser)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--probe', action='store_true', help='소켓으로 수신만 검사; 주행·시나리오 변경 없음')
    parser.add_argument('--scenario', choices=('full_loop', 'full_loop_highway'), default='full_loop')
    parser.add_argument('--repeat', action='store_true', help='완주 후 구간 시작점에서 반복; Ctrl+C로 종료')
    parser.add_argument('--max-runs', type=int, help='실행 횟수 제한 (기본 1회, --repeat이면 무제한)')
    parser.add_argument('--stop-before-tunnel', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--speed-cap-kmh', type=float, default=60.)
    parser.add_argument('--bind-ip', default='0.0.0.0')
    parser.add_argument('--sim-ip', default='auto')
    parser.add_argument('--ego-port', type=int, default=8000)
    parser.add_argument('--object-port', type=int, default=8100)
    parser.add_argument('--signal-port', type=int, default=7700)
    parser.add_argument('--signal-source', choices=('udp', 'grpc', 'none'), default=None,
                        help='대회판 촬영 기본값: 신호만 gRPC GT 조회. 자차·장애물·제어는 UDP')
    parser.add_argument('--grpc-target', default='auto')
    parser.add_argument('--control-port', type=int, default=7601)
    parser.add_argument('--control-source-port', type=int, default=7600)
    parser.add_argument('--control-format', choices=('single_steer55', 'front_rear59'), default='single_steer55')
    args = parser.parse_args(argv)
    if args.highway_density is not None and args.scenario != 'full_loop_highway':
        parser.error('--highway-density requires --scenario full_loop_highway')
    from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
    args.speed_cap_kmh = validate_max_velocity_kmh(args.speed_cap_kmh)
    if args.signal_source is None:
        args.signal_source = 'none' if args.scenario == 'full_loop_highway' else 'grpc'
    if args.signal_source == 'none' and args.scenario != 'full_loop_highway':
        parser.error('--signal-source none is only supported for the highway segment')
    if args.max_runs is not None and args.max_runs < 1:
        parser.error('--max-runs must be at least 1')
    if args.scenario == 'full_loop_highway' and args.stop_before_tunnel:
        parser.error('고주로 전용 실행에서는 --stop-before-tunnel을 빼세요')
    if args.probe and args.signal_source != 'grpc':
        from run_gt_udp_expert import main as probe
        return probe(['--probe', *network_args(args)])
    global_config, _zone, base_config = load_configuration()
    if args.grpc_target == 'auto':
        explicit = os.environ.get('MORAI_GRPC_TARGET', '').strip()
        host = os.environ.get('MORAI_GRPC_HOST', str(global_config['grpc'].get('host', 'auto')))
        port = os.environ.get('MORAI_GRPC_PORT', str(global_config['grpc'].get('port', 7789)))
        args.grpc_target = explicit or (host+':'+port if host != 'auto' else 'auto')
    output = REPO / 'MoraiDrive/scenario_runner/.runtime/gt_udp_full_route' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    global_config['paths']['runtime_root'] = str(output)
    config = configure_scenario(base_config, args)
    # Isolate the profile history, keeping the ordinary VE run's history intact.
    config['profile_state_path'] = str(output / 'profile_state.json')
    plan = dict(scenario=args.scenario, repeat=args.repeat, max_runs=args.max_runs,
                transport='UDP_and_gRPC_signal_GT' if args.signal_source == 'grpc' else 'UDP',
                signal_source=args.signal_source, Expert='existing_AutonomousDriving', VE_used=False,
                control_format=args.control_format,
                sensor_capture=False, gt_fallback=False, stop_before_tunnel=args.stop_before_tunnel,
                udp=dict(ego=args.ego_port, objects=args.object_port,
                         signal=args.signal_port if args.signal_source == 'udp' else None,
                         control_destination=args.control_port, control_source=args.control_source_port),
                simulation_setup='existing_scenario_runner_gRPC', output=str(output))
    write(output / 'plan.json', plan)
    if args.probe:
        context = prepare_controller(global_config, config, args, output)
        from run_gt_udp_expert import main as probe
        return probe(['--probe', '--config', context.expert_controller.runtime_config_path,
                      '--path', context.expert_controller.runtime_path_csv, *network_args(args)])
    if args.check:
        context = prepare_controller(global_config, config, args, output)
        result = subprocess.run([sys.executable, '-B', str(REPO / 'MoraiDrive/scenario_runner/legacy_expert/run_gt_udp_expert.py'),
            '--check', '--config', context.expert_controller.runtime_config_path,
            '--path', context.expert_controller.runtime_path_csv, *network_args(args)], cwd=REPO)
        if result.returncode:
            return result.returncode
        print(json.dumps(dict(status='CHECKED', **plan), ensure_ascii=False, indent=2))
        return 0
    conflicts = running_controllers()
    if conflicts:
        raise RuntimeError('기존 주행 실행을 먼저 종료하세요: ' + json.dumps(conflicts, ensure_ascii=False))
    # Verify all three input streams before changing the world or spawning NPCs.
    from run_gt_udp_expert import main as probe
    probe_arguments = ['--probe', '--startup-timeout', '30', *network_args(args)]
    if args.signal_source == 'grpc':
        context = prepare_controller(global_config, config, args, output)
        probe_arguments += ['--config', context.expert_controller.runtime_config_path,
                            '--path', context.expert_controller.runtime_path_csv]
    if probe(probe_arguments):
        return 2
    from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
    grpc = MoraiGrpcClient(global_config)
    grpc.connect()
    if args.sim_ip == 'auto':
        args.sim_ip = grpc.host
    args.grpc_target = '{}:{}'.format(grpc.host, grpc.port)
    status, error = 'COMPLETE', None
    try:
        print('[GT UDP] 자차/장애물/제어=UDP · 신호={} GT'.format(args.signal_source), flush=True)
        print('[로그] ' + str(output), flush=True)
        run_scenarios(grpc, global_config, config, args, output)
    except KeyboardInterrupt:
        status = 'STOPPED'
    except Exception as exc:
        status, error = 'FAILED', str(exc)
        print('[FAIL] ' + error, flush=True)
    finally:
        grpc.stop()
        write(output / 'report.json', dict(status=status, error=error, plan=plan))
    return int(status == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
