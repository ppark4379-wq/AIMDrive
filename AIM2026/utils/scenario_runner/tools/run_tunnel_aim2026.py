#!/usr/bin/env python3
"""Tunnel-segment scenarios using the same integrated UDP Expert as AIM2026/main.py, with the virtual GPS.

Same flow as Expert/run_highway.py and run_roundabout.py (scene lifecycle here, driving in
AIM2026/main.py: vision encoder, Expert and the shared GPS/IMU-speed Localization), selecting the
tunnel mission windows. The Expert package is only imported; nothing in it is edited.

The simulator never blacks out GPS, so the virtual GPS (dead reckoning) only engages in the tunnel
while the Localization test_blackout zone is enabled. That is done in a per-run copy of the
configuration; Localization/config.yaml and the base config are never edited.
"""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import AIM_ROOT, RUNNER_ROOT, activate_project_paths
activate_project_paths()

import argparse
import copy
import dataclasses
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import yaml

from AIM2026.Expert import run_highway as harness
from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.full_loop_parts import competition_static
from scenario_runner.scenarios.segment_modes import events
from scenario_runner.scenarios.segment_modes.modes import get_mission_mode
from scenario_runner.utils.map_loader import MGeoMapLoader

# The four tunnel scenes of the full course (legacy sudden brake, static, pedestrian/animal, vehicle)
# plus the empty tunnel. The Expert rebuilds the route window from the mission name, so brake and static
# keep the normal mission (its window starts 40 m before the tunnel link, the run-up the sudden-brake NPC
# needs) and only switch the tunnel zone events on in this launcher's scene.
MISSIONS = {
    'normal': 'full_loop_tunnel_normal',
    'brake': 'full_loop_tunnel_normal',
    'static': 'full_loop_tunnel_normal',
    'pedestrian': 'full_loop_tunnel_dynamic',
    'vehicle': 'full_loop_tunnel_vehicle',
}
# --variant all: one scene per run, in the order brake, static, pedestrian, vehicle, then again.
ROTATION = ('brake', 'static', 'pedestrian', 'vehicle')
VEHICLE_BEHAVIORS = ('vehicle_slow', 'vehicle_sudden_brake', 'vehicle_oncoming', 'vehicle_cut_in')
RUN_GAP_S = 3.
INTERRUPTED = 130  # Ctrl+C / SIGTERM: cleanup is done, but the whole campaign must stop
ROOT = harness.ROOT
# Everything this launcher writes stays inside the runner (git-ignored .runtime).
RUNTIME = RUNNER_ROOT / '.runtime/tunnel_aim2026'


def _load(path):
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict):
        raise ValueError('Expected a YAML mapping: ' + str(path))
    return document


def virtual_gps_config(base_config, folder):
    """Copy of the runtime config whose Localization masks real GPS inside the tunnel zone.

    The Localization path is relative to AIM2026 so a Windows interpreter resolves it as well.
    """
    runtime = _load(base_config)
    localization = runtime.setdefault('Localization', {})
    source = Path(localization.get('config', 'Localization/config.yaml'))
    settings = _load(source if source.is_absolute() else AIM_ROOT / source)
    zone = settings.get('test_blackout')
    if not isinstance(zone, dict) or not all(key in zone for key in ('start_xy', 'end_xy')):
        raise ValueError('Localization test_blackout needs start_xy and end_xy: ' + str(source))
    settings['test_blackout'] = dict(zone, enabled=True)
    folder.mkdir(parents=True)
    (folder / 'localization.yaml').write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding='utf-8')
    localization.update(enabled=True, config=(folder / 'localization.yaml').relative_to(AIM_ROOT).as_posix())
    destination = folder / 'runtime.yaml'
    destination.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return destination


class TunnelScenario(harness.HighwayScenario):
    """The highway/roundabout harness scene with one tunnel scene selected by ``variant``."""
    variant = 'normal'
    fixed_pedestrian = False
    vehicle_behavior = None
    static_report_dir = None

    def prepare_full_loop_route(self):
        self._configure_variant()
        result = super().prepare_full_loop_route()
        self._validation_controller = 'aim2026_tunnel'
        if self.variant == 'static':
            self.competition_static_report_path = (
                Path(self.static_report_dir or self.output)
                / (getattr(self, 'run_instance_id', 'check') + '_static.json'))
            competition_static.prepare(self)
        return result

    def _configure_variant(self):
        cfg = self.cfg
        if self.variant in ('brake', 'static'):
            # Normal window with the tunnel zone on. The dynamic mission's name keeps speed-zone filtering and
            # leaves the empty-road baseline recorder (normal mission only) off; only the chosen event is armed.
            dynamic = get_mission_mode('full_loop_tunnel_dynamic')
            self.mission_mode = dataclasses.replace(self.mission_mode, name=dynamic.name, zones=dynamic.zones)
            cfg.setdefault('tunnel_dynamic', {})['enabled'] = False
            cfg.setdefault('tunnel_vehicle_dynamic', {})['enabled'] = False
            cfg.setdefault('tunnel_sudden_brake', {})['enabled'] = self.variant == 'brake'
            if self.variant == 'static':
                cfg['expert_max_cross_track_error_m'] = 5.0  # as the static mission: avoidance leaves the line
        elif self.variant == 'pedestrian':
            pedestrians = cfg.setdefault('tunnel_dynamic', {})
            if not self.fixed_pedestrian:
                # Same explicit random-mix (pedestrians and animals) as the full-course pedestrian scene.
                pedestrians.pop('trial_candidate_id', None)
                pedestrians['candidate_policy'] = 'trial'
                pedestrians.setdefault('random_mix', {})['enabled'] = True
        elif self.variant == 'vehicle':
            vehicles = cfg.setdefault('tunnel_vehicle_dynamic', {})
            if self.vehicle_behavior:
                vehicles.update(behavior=self.vehicle_behavior, random_behavior=False)
            else:
                vehicles['random_behavior'] = True

    def configure_drive_with_retries(self):
        if self.variant == 'static':
            # Environment only, verified before the Expert starts.
            competition_static.arm(self)
        return super().configure_drive_with_retries()

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        if self.variant == 'static':
            competition_static.update(self, elapsed, ego_state, current_s)

    def cleanup(self):
        try:
            return super().cleanup()
        finally:
            if self.variant == 'static':
                competition_static.cleanup(self)


def run_one(argv, variant, options=None):
    """One tunnel run: the highway/roundabout harness flow with one tunnel scene."""
    if variant not in MISSIONS:
        raise ValueError('Unsupported tunnel scene: ' + str(variant))
    mission = MISSIONS[variant]
    options = options or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without connecting to MORAI')
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config_local.yaml')
    parser.add_argument('--expert-python', default=harness.expert_python())
    parser.add_argument('--speed-cap-kmh', type=validate_max_velocity_kmh, default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--validation-events', action='store_true')
    parser.add_argument('--strict-collisions', action='store_true')
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    global_cfg, _, base = load_configuration()
    config = copy.deepcopy(base)
    for key in list(config):
        if key.startswith('ve_'):
            config.pop(key)
    config.update(mission_mode=mission, drive_control_mode='aimdrive_expert',
                  expert_speed_cap_kmh=60. if args.speed_cap_kmh is None else args.speed_cap_kmh,
                  expert_ros_remaps=[], expert_traffic_light_control=False, route_debug_enabled=False,
                  max_laps=1, auto_repeat=False, route_setup_attempts=1)
    config.setdefault('sync_runtime', {})['enabled'] = False
    if args.seed is not None:
        config['random_seed'] = args.seed
    stamp = 'tunnel_' + variant + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = RUNTIME / 'scenarios' / stamp
    output.mkdir(parents=True)
    global_cfg['paths']['runtime_root'] = str(output)
    global_cfg['paths']['expert_root'] = str(ROOT / 'AIM2026/Expert')
    client = SimpleNamespace(make_transform=lambda *v: v)
    if not args.check:
        from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
        if os.name == 'nt' and global_cfg['grpc'].get('host') == 'auto':
            global_cfg['grpc']['host'] = '127.0.0.1'
        client = MoraiGrpcClient(global_cfg)
    scenario = TunnelScenario(grpc_client=client, map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
                              global_cfg=global_cfg, scenario_cfg=config)
    args.mission = mission
    scenario.variant = variant
    scenario.fixed_pedestrian = bool(options.get('fixed_pedestrian'))
    scenario.vehicle_behavior = options.get('vehicle_behavior')
    scenario.static_report_dir = output
    scenario.args, scenario.output = args, output
    scenario.driver_output = RUNTIME / 'expert' / stamp
    if args.check:
        # setup() assigns these in a real run; the check path never runs it.
        scenario.run_seed = args.seed if args.seed is not None else 0
        scenario.run_instance_id = 'check'
        events.prepare_mission(scenario)
        scenario.route_definition = scenario.load_route_definition()
        scenario.prepare_full_loop_route()
        scenario.setup_expert_controller()
        return subprocess.run(scenario.expert_controller.command(check=True), cwd=ROOT,
                              env=scenario.expert_controller.child_environment()).returncode

    def interrupt(*_):
        raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGTERM, interrupt)
    status, error = 'run_returned', None
    try:
        client.connect()
        print('[Scenario] One run: tunnel {} ({}) -> AIM2026/main.py -> Expert'.format(variant, mission), flush=True)
        scenario.run()
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[Tunnel] stopped', flush=True)
    except BaseException as exc:
        status, error = 'exception', exc
        raise
    finally:
        try:
            try:
                scenario.stop_expert_controller()
            finally:
                client.stop()
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            try:
                harness._finish_validation(scenario, output, status, error)
            except Exception as finish_error:
                print('[Validation] finalization failed: ' + str(finish_error), flush=True)
    return INTERRUPTED if status == 'interrupted' else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Other options go to the single-run flow '
                                     '(--check, --speed-cap-kmh, --validation-events, --strict-collisions, ...).')
    parser.add_argument('--variant', choices=[*MISSIONS, 'all'], default='normal',
                        help='normal: empty tunnel; brake: sudden-brake NPC; static: approved obstacle layout; '
                             'pedestrian: pedestrians and animals; vehicle: NPC vehicle behaviours; '
                             'all: brake, static, pedestrian, vehicle in turn, one scene per run')
    parser.add_argument('--fixed-pedestrian', action='store_true',
                        help='pedestrian: the single default pedestrian instead of the random pedestrian/animal mix')
    parser.add_argument('--vehicle-behavior', choices=VEHICLE_BEHAVIORS,
                        help='vehicle: fix one behaviour (default: random)')
    parser.add_argument('--runs', type=int, default=None,
                        help='Repeat the tunnel segment this many times, each in a fresh scene '
                             '(default 1; 4 with --variant all)')
    parser.add_argument('--no-virtual-gps', dest='virtual_gps', action='store_false',
                        help='Keep real GPS inside the tunnel; the dead-reckoning provider stays idle')
    parser.add_argument('--config', type=Path, default=AIM_ROOT / 'config_local.yaml')
    parser.add_argument('--seed', type=int, help='First run seed; later runs use seed+1, seed+2, ...')
    args, remaining = parser.parse_known_args(argv)
    if args.runs is not None and args.runs < 1:
        parser.error('--runs must be at least 1')
    if '--check' in remaining and args.runs is not None:
        parser.error('--check validates one run per scene; omit --runs')
    rotating = args.variant == 'all'
    runs = args.runs if args.runs is not None else (len(ROTATION) if rotating else 1)
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    options = dict(fixed_pedestrian=args.fixed_pedestrian, vehicle_behavior=args.vehicle_behavior)
    if args.fixed_pedestrian and args.variant not in ('pedestrian', 'all'):
        parser.error('--fixed-pedestrian is for --variant pedestrian or all')
    if args.vehicle_behavior and args.variant not in ('vehicle', 'all'):
        parser.error('--vehicle-behavior is for --variant vehicle or all')
    config = args.config
    if args.virtual_gps:
        config = virtual_gps_config(args.config, RUNTIME / 'config' / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        print('[Tunnel] virtual GPS: Localization test_blackout zone enabled in ' + str(config), flush=True)
    else:
        print('[Tunnel] virtual GPS masking off; config ' + str(config), flush=True)

    def arguments(index):
        values = [*remaining, '--config', str(config)]
        if args.seed is not None:
            values += ['--seed', str(args.seed + index)]
        return values

    def variant_of(index):
        return ROTATION[index % len(ROTATION)] if rotating else args.variant

    if '--check' in remaining:
        # One validation per scene; with all that is every scene of the rotation.
        checks = [run_one(arguments(0), variant, options) for variant in (ROTATION if rotating else (args.variant,))]
        return 0 if not any(checks) else 1
    if runs == 1:
        return run_one(arguments(0), variant_of(0), options)
    results = []
    for index in range(runs):
        variant = variant_of(index)
        print('[Tunnel] run {}/{}: {}'.format(index + 1, runs, variant), flush=True)
        try:
            code = run_one(arguments(index), variant, options)
            if code == INTERRUPTED:
                results.append((variant, 'interrupted'))
                break
            results.append((variant, 'completed' if code == 0 else 'exit code {}'.format(code)))
        except KeyboardInterrupt:
            results.append((variant, 'interrupted'))
            break
        except Exception as error:
            results.append((variant, 'failed: {}: {}'.format(type(error).__name__, error)))
            print('[Tunnel] run {} failed; continuing with a fresh scene'.format(index + 1), flush=True)
        if index + 1 < runs:
            time.sleep(RUN_GAP_S)
    for index, (variant, result) in enumerate(results, 1):
        print('[Tunnel] run {} ({}): {}'.format(index, variant, result), flush=True)
    done = {name: sum(1 for variant, result in results if variant == name and result == 'completed')
            for name in dict.fromkeys(variant for variant, _ in results)}
    print('[Tunnel] completed per scene: ' + ', '.join('{} {}'.format(n, c) for n, c in done.items()), flush=True)
    if results and results[-1][1] == 'interrupted':
        return INTERRUPTED
    return 0 if all(result == 'completed' for _, result in results) and len(results) == runs else 1

if __name__ == '__main__':
    raise SystemExit(main())
