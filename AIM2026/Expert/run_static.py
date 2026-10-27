"""One static-obstacle mission with integrated Expert and failure diagnosis."""

import argparse
import copy
from datetime import datetime
import errno
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'MoraiDrive')]

from AIM2026.Expert import run_highway as shared
from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.segment_modes import events
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.utils.map_loader import MGeoMapLoader

MISSION = 'full_loop_static'
# --avoid: one obstacle on the lane split after the static section, so the Expert has to change lane.
AVOID_MISSION = 'full_loop_static_avoid'
AVOID_LINKS = ('A2256W000202', 'A2256W000009')      # ego lane, and the right lane that splits off it
AVOID_OBSTACLE_Y = -83.
FOOTPRINTS = ROOT / 'MoraiDrive/scenario_runner/config/static_obstacle_footprints_v1.json'
GEOMETRY_REGISTRY = (ROOT / 'MoraiDrive/data_collection/internal_sensor_record/evidence/'
                     'object_geometry_registry_default3d_v2/registry.json')
# The only route signal on the static section (driver log: signal target).
STATIC_ROUTE_SIGNALS = ('C1256W000091',)


def obstacle_extents(models):
    """{model: (width, length)} in metres at native scale; models without evidence are left out.

    Width is across the model's yaw direction. LiDAR-verified footprints win over the geometry registry.
    """
    result, registry = {}, None
    catalog = json.loads(FOOTPRINTS.read_text(encoding='utf-8')).get('models', {})
    for name in models:
        if name in catalog:
            result[name] = (float(catalog[name]['width_m']), float(catalog[name]['length_m']))
            continue
        if registry is None:
            try:
                registry = json.loads(GEOMETRY_REGISTRY.read_text(encoding='utf-8')).get('models', {})
            except (OSError, ValueError):
                registry = {}
        size = registry.get('obstacle|{}|1,1,1'.format(name), {}).get('geometry_3d', {}).get('size_lwh_m')
        if size:
            result[name] = (float(size[1]), float(size[0]))
    return result


def avoidance_preset(map_loader, models, map_name, seed=None, model=None, obstacle_y=AVOID_OBSTACLE_Y):
    """One obstacle on the ego lane where the right lane has opened enough to pass it.

    A model is usable when its narrowest side, turned across the road if that helps, leaves the Expert's
    collision margin between the obstacle and a car centred in the right lane. Models are picked by
    ``seed`` (consecutive seeds of a --runs batch cycle through all of them) or at random.
    """
    from AIM2026.Expert.src.config.parameters import lattice
    lane, side = (map_loader.get_link_points(link) for link in AVOID_LINKS)
    index = min(range(1, len(lane) - 1), key=lambda i: abs(lane[i][1] - obstacle_y))
    x, y, z = (float(value) for value in lane[index][:3])
    gap = min(math.hypot(point[0] - x, point[1] - y) for point in side)
    heading = math.degrees(math.atan2(lane[index + 1][1] - lane[index - 1][1],
                                      lane[index + 1][0] - lane[index - 1][0]))
    limit = 2. * (gap - lattice.VEHICLE_WIDTH / 2. - lattice.COLLISION_MARGIN)
    extents = obstacle_extents(models)
    usable, excluded = [], {}
    for name in models:
        if name not in extents:
            excluded[name] = 'no size evidence'
        elif min(extents[name]) > limit:
            excluded[name] = 'narrowest side {:.2f} m > {:.2f} m'.format(min(extents[name]), limit)
        else:
            usable.append(name)
    if not usable:
        raise ValueError('No static obstacle model fits the lane gap {:.2f} m'.format(gap))
    if model is not None:
        if model not in usable:
            raise ValueError('{} cannot be passed here: {}'.format(model, excluded.get(model, 'unknown model')))
    elif seed is not None:
        model = usable[seed % len(usable)]
    else:
        model = random.choice(usable)
    width, length = extents[model]
    yaw = heading + 180. + (90. if width > length else 0.)       # long side along the road when it is wider than the lane gap allows
    yaw = (yaw + 180.) % 360. - 180.
    return {
        'schema': 'morai_static_obstacle_preset_v1', 'name': 'static_avoid_' + model,
        'map_name': map_name, 'scenario': AVOID_MISSION,
        'obstacles': [{'model': model, 'position': {'x': x, 'y': y, 'z': z}, 'yaw_deg': yaw,
                       'scale': [1., 1., 1.]}],
        'generation': {'lane_gap_m': gap, 'max_obstacle_width_m': limit, 'usable_models': usable,
                       'excluded_models': excluded},
    }


def latest_control_status(path):
    """A Windows writer can briefly give WSL ENODATA while the file exists."""
    try:
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            rows = stream.read().splitlines()
    except OSError as error:
        if error.errno in (errno.ENODATA, errno.EAGAIN):
            return None
        raise
    for raw in reversed(rows):
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            continue
    return None


def silent_sensor_note(status_path):
    """Name the UDP streams that never delivered a packet (wrong or unloaded MORAI sensor set)."""
    try:
        streams = json.loads(status_path.read_text(encoding='utf-8')).get('streams') or {}
    except (OSError, ValueError):
        return ''
    silent = sorted(name for name, row in streams.items() if not row.get('packets'))
    return ' (no packets from: {})'.format(', '.join(silent)) if silent else ''


def start_rviz(root):
    """Start this checkout's RViz viewer; it follows this checkout's newest driver folder."""
    environment = os.environ.copy()
    # The viewer uses system ROS Python; keep the runner's capture/Conda paths out of it.
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE'):
        environment.pop(name, None)
    print('[Static] RViz: bash AIM2026/Visualization/run_rviz.sh --runs-root {} '
          '(a viewer already running from another checkout must be stopped first)'.format(
              root / 'AIM2026/outputs/expert'), flush=True)
    return subprocess.Popen(
        ['bash', str(root / 'AIM2026/Visualization/run_rviz.sh'),
         '--runs-root', str(root / 'AIM2026/outputs/expert')],
        cwd=root, env=environment, start_new_session=True)


def stop_rviz(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.)
    except (OSError, subprocess.TimeoutExpired):
        pass


def static_expert_environment(environment):
    """Keep Conda packages ahead of system ROS capture dependencies."""
    child = environment.copy()
    child.pop('PYTHONPATH', None)
    # The static corridor planner solves small QPs every control cycle. With
    # multi-threaded BLAS next to the inference process it took ~280 ms (up to
    # 800 ms in run static-20261026-214507-855814) instead of ~45 ms, which
    # starved the pose/detection packets and kept the Expert braking.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        child[name] = '1'
    return child


class StaticExpertController(shared.ExpertController):
    def command(self, check=False):
        command = super().command(check=check)
        ignored = getattr(self.scenario.args, 'ignore_signals', None)
        if ignored:
            command.extend(['--ignore-signals', ','.join(ignored)])
        return command

    def child_environment(self):
        return static_expert_environment(super().child_environment())

    def start_process(self):
        if self.is_running():
            return
        self.scenario.check_startup_contact()
        command = self.command()
        (self.scenario.output / 'driver_command.json').write_text(json.dumps(command), encoding='utf-8')
        self.log = (self.scenario.output / 'driver.console.log').open('w', encoding='utf-8', buffering=1)
        child_options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                         if os.name == 'nt' else {'start_new_session': True})
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log,
                                        env=self.child_environment(),
                                        stderr=subprocess.STDOUT, **child_options)
        print('[Static] Expert log: ' + str(self.scenario.output / 'driver.console.log'), flush=True)
        deadline = time.monotonic() + 120.
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError('AIM2026 Expert stopped; inspect driver.console.log')
            row = latest_control_status(self.scenario.driver_output / 'control.jsonl')
            if row and row.get('status') == 'RUNNING' and row.get('control_sent'):
                print('[Static] AIM2026 UDP + perception + Expert ready', flush=True)
                return
            time.sleep(.1)
        raise RuntimeError('AIM2026 sensor/inference startup timed out{}; inspect driver.console.log'.format(
            silent_sensor_note(self.scenario.driver_output / 'inference_status.json')))


class StaticScenario(shared.HighwayScenario):
    """Reuse the safe lifecycle while owning static validation and startup."""

    def prepare_full_loop_route(self):
        if self.args.validation_events and not self.args.check:
            start_validation(self)
        return MissionScenario.prepare_full_loop_route(self)

    def setup_expert_controller(self):
        MissionScenario.setup_expert_controller(self)
        self.expert_controller = StaticExpertController(self.expert_controller, self)


def start_validation(scenario):
    if getattr(scenario, '_runner_validation_log', None) is not None:
        return
    from scenario_runner.validation.run_log import ValidationRunLog

    validation_root = ROOT / 'MoraiDrive/scenario_runner/.runtime'
    log = ValidationRunLog(validation_root / 'validation_runs', getattr(scenario.args, 'mission', MISSION))
    scenario._runner_validation_log = log
    scenario._runner_scenario_name = getattr(scenario.args, 'mission', MISSION)
    scenario._validation_controller = 'aim2026_static'
    options = ['--seed', str(scenario.run_seed), '--validation-events']
    if scenario.args.static_obstacle_count is not None:
        options.extend(['--static-obstacle-count', str(scenario.args.static_obstacle_count)])
    log.enable_incidents(validation_root / 'incidents', scenario, run_options=options)
    log.incident_recorder.connect_pipeline(scenario.driver_output / 'pipeline.jsonl')


def main(argv=None):
    from AIM2026.Expert.run_repeat import run_many, split_repeat_options
    repeat, argv = split_repeat_options(argv)
    if repeat.runs > 1:
        if '--check' in argv:
            print('[Batch] --check validates once; --runs is ignored', flush=True)
        else:
            return run_many(main, argv, repeat, label='static')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without connecting to MORAI')
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config_local.yaml')
    parser.add_argument('--expert-python', default=os.environ.get('AIM_STATIC_EXPERT_PYTHON')
                        or shared.expert_python())
    parser.add_argument('--speed-cap-kmh', type=validate_max_velocity_kmh, default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--static-obstacle-count', type=int, choices=(1, 2))
    parser.add_argument('--avoid', action='store_true',
                        help='Lane-split avoidance test: the static section is extended and one obstacle '
                             'is placed on the split (use run_static_avoid.sh)')
    parser.add_argument('--model', help='With --avoid: obstacle model (default: by seed, else random)')
    parser.add_argument('--obstacle-y', type=float, default=AVOID_OBSTACLE_Y,
                        help='With --avoid: obstacle map y [m] on the ego lane (default %(default)s)')
    parser.add_argument('--validation-events', action='store_true', default=True,
                        help='Enabled by default: record observations and diagnose Incidents')
    parser.add_argument('--rviz', action='store_true',
                        help='Also start the RViz viewer for this run (WSL with ROS Noetic and a display)')
    parser.add_argument('--no-signal', action='store_true',
                        help='Drive through the route traffic light: the Expert ignores its signal')
    parser.add_argument('--ignore-signals', type=lambda text: [x for x in text.split(',') if x],
                        default=None, help='Comma-separated map signal IDs the Expert ignores')
    parser.add_argument('--strict-collisions', action='store_true')
    args = parser.parse_args(argv)
    args.mission = AVOID_MISSION if args.avoid else MISSION
    if not args.avoid and (args.model or args.obstacle_y != AVOID_OBSTACLE_Y):
        parser.error('--model and --obstacle-y need --avoid')
    if args.avoid and args.static_obstacle_count is not None:
        parser.error('--avoid always places one obstacle; drop --static-obstacle-count')
    if args.no_signal:
        # Passing the option replaces the Expert's development default (LCS01).
        args.ignore_signals = sorted({'LCS01', *STATIC_ROUTE_SIGNALS, *(args.ignore_signals or [])})
    args.config = args.config.resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))

    if not args.check:
        probe = subprocess.run([args.expert_python, '-c', 'import torch'],
                               env=static_expert_environment(os.environ),
                               capture_output=True, text=True, timeout=30)
        if probe.returncode:
            parser.error('Expert Python cannot import torch: {}. Activate vldrive-ve or pass '
                         '--expert-python /path/to/vldrive-ve/bin/python3. {}'.format(
                             args.expert_python, probe.stderr.strip()[-500:]))
        print('[Static] Expert Python: {}'.format(args.expert_python), flush=True)

    stamp = 'static-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'AIM2026/outputs/scenarios' / stamp
    output.mkdir(parents=True)
    # Preserve the selected build's packet layout. Without local_status_bytes,
    # VehicleStatusReceiver uses the competition 181-byte format (verified on
    # this MORAI build); local 229/245-byte builds must opt in via --config.
    sensor_config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    # Run static-20261026-212827-521332: with 'right' the Expert's +15deg left
    # command went out as negative wire steer, yaw fell 60 -> -13deg, steer
    # saturated and the pose left the route corridor. A positive wire value
    # turns left on this 229-byte build.
    sensor_config.setdefault('Control', {}).setdefault('positive_steer', 'left')
    static_config = output / 'sensor_config.yaml'
    static_config.write_text(yaml.safe_dump(sensor_config, sort_keys=False), encoding='utf-8')
    args.config = static_config

    global_cfg, _, base = load_configuration()
    config = copy.deepcopy(base)
    config.setdefault('highway_merge_static_vehicle', {})['enabled'] = False
    for key in list(config):
        if key.startswith('ve_'):
            config.pop(key)
    config.update(mission_mode=args.mission, drive_control_mode='aimdrive_expert',
                  expert_speed_cap_kmh=60. if args.speed_cap_kmh is None else args.speed_cap_kmh,
                  expert_ros_remaps=[], expert_traffic_light_control=False,
                  route_debug_enabled=False, max_laps=1, auto_repeat=False,
                  route_setup_attempts=1)
    config.setdefault('sync_runtime', {})['enabled'] = False
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.static_obstacle_count is not None:
        config['static_obstacle_count'] = args.static_obstacle_count
    map_loader = MGeoMapLoader(global_cfg['paths']['mgeo_root'])
    if args.avoid:
        try:
            preset = avoidance_preset(map_loader, config['static_obstacle_models'],
                                      global_cfg['morai']['map_name'], seed=args.seed,
                                      model=args.model, obstacle_y=args.obstacle_y)
        except ValueError as error:
            parser.error(str(error))
        config['static_obstacle_preset'] = str(output / 'static_avoid_preset.yaml')
        (output / 'static_avoid_preset.yaml').write_text(yaml.safe_dump(preset, sort_keys=False), encoding='utf-8')
        item, info = preset['obstacles'][0], preset['generation']
        print('[Static] Avoid test: {} at ({:.2f}, {:.2f}) yaw={:.1f} lane gap={:.2f} m; {} of {} models usable, '
              'excluded: {}'.format(item['model'], item['position']['x'], item['position']['y'], item['yaw_deg'],
                                    info['lane_gap_m'], len(info['usable_models']), len(config['static_obstacle_models']),
                                    info['excluded_models'] or 'none'), flush=True)

    global_cfg['paths']['runtime_root'] = str(output)
    global_cfg['paths']['expert_root'] = str(ROOT / 'AIM2026/Expert')
    client = SimpleNamespace(make_transform=lambda *values: values)
    if not args.check:
        from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
        if os.name == 'nt' and global_cfg['grpc'].get('host') == 'auto':
            global_cfg['grpc']['host'] = '127.0.0.1'
        client = MoraiGrpcClient(global_cfg)
    scenario = StaticScenario(grpc_client=client,
        map_loader=map_loader, global_cfg=global_cfg, scenario_cfg=config)
    scenario.args, scenario.output = args, output
    scenario.driver_output = ROOT / 'AIM2026/outputs/expert' / stamp
    if args.check:
        events.prepare_mission(scenario)
        scenario.route_definition = scenario.load_route_definition()
        scenario.prepare_full_loop_route()
        if args.avoid:      # the same binding the real run does when it spawns: obstacle lies on the mission route
            from scenario_runner.scenarios.full_loop_parts.static_obstacle import load_static_obstacle_preset
            bound = load_static_obstacle_preset(config['static_obstacle_preset'], global_cfg['morai']['map_name'],
                                                scenario.route_points, scenario.mission_route.first_index)
            print('[Static] Avoid test preset binds to route index {} ({:.3f} m off route)'.format(
                bound['placements'][0]['route_path_index'], bound['placements'][0]['route_distance_m']), flush=True)
        scenario.setup_expert_controller()
        return subprocess.run(scenario.expert_controller.command(check=True), cwd=ROOT,
                              env=scenario.expert_controller.child_environment()).returncode

    def interrupt(*_):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    status, error = 'run_returned', None
    rviz = None
    try:
        client.connect()
        if args.rviz:
            rviz = start_rviz(ROOT)
        print('[Static] One run: {} -> AIM2026/main.py -> Expert'.format(args.mission), flush=True)
        scenario.run()
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[Static] stopped', flush=True)
    except BaseException as exc:
        status, error = 'exception', exc
        raise
    finally:
        stop_rviz(rviz)
        try:
            try:
                scenario.stop_expert_controller()
            finally:
                client.stop()
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            # Building the HTML report parses the whole pipeline log (about
            # 10 s); a second Ctrl+C here used to abort it before report.html.
            print('[Validation] building report; wait ~10 s (Ctrl+C is ignored until it is written)',
                  flush=True)
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                shared._finish_validation(scenario, output, status, error)
            except Exception as finish_error:
                print('[Validation] finalization failed: ' + str(finish_error), flush=True)
            finally:
                signal.signal(signal.SIGINT, old_sigint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
