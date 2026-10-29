#!/usr/bin/env python3
"""Full-course competition runs with the integrated UDP Expert of AIM2026/main.py, with the virtual GPS.

``full_loop_competition`` (whole course, tunnel scene rotating legacy -> static -> pedestrian -> vehicle, one per
run) driven the way run_tunnel_aim2026.py drives the tunnel: scene lifecycle here, driving in AIM2026/main.py (vision
encoder, Expert and the shared GPS/IMU-speed Localization), so it also runs on the competition MORAI, which only
speaks UDP. ``--competition-morai`` keeps the static tunnel scene to obstacle models that MORAI ships.

The simulator never blacks out GPS, so the virtual GPS (dead reckoning) engages only in the tunnel zone of the
Localization test_blackout setting, in a per-run copy of the configuration; no shared configuration is edited.
"""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import RUNNER_ROOT, activate_project_paths
activate_project_paths()

import argparse
import copy
from datetime import datetime
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import types
from types import SimpleNamespace
import zipfile

from AIM2026.Expert import run_highway as harness
from AIM2026.Expert import wsl_network
from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.full_loop_competition import FullLoopCompetitionScenario, TUNNEL_SCENES
from scenario_runner.tools import run_tunnel_aim2026 as tunnel
from scenario_runner.utils.map_loader import MGeoMapLoader

ROOT = harness.ROOT
# Everything this launcher writes stays inside the runner (git-ignored .runtime).
RUNTIME = RUNNER_ROOT / '.runtime/competition_aim2026'
MISSION = 'full_loop_competition'   # the name the harness's validation log and report use
# --tunnel-scene names -> the rotation's own scene names (the sudden-brake NPC scene is called "legacy" there).
SCENE_NAMES = {'brake': 'legacy', 'static': 'static', 'pedestrian': 'pedestrian', 'vehicle': 'vehicle'}
RUNS = len(TUNNEL_SCENES)   # one pass over the tunnel scene rotation
# A whole course lasts several times the tunnel window, and signals/roundabouts hold the Ego for a while.
ROUND_TIMEOUT_S = 900.
STALL_S = 90.
INTERRUPTED = tunnel.INTERRUPTED
RUN_GAP_S = tunnel.RUN_GAP_S
RoundTimeout = tunnel.RoundTimeout


def apply_highway_options(config, options):
    """Highway-section choices of one run.

    Like the highway-only launcher (Expert/run_highway.py), the course leaves out the one stopped car on the merge ramp
    by default (moving traffic is unchanged); ``merge_static_vehicle`` keeps the course's own setting.
    """
    if not options.get('merge_static_vehicle'):
        config.setdefault('highway_merge_static_vehicle', {})['enabled'] = False


def sequence_for_scene(previous, scene):
    """The next rotation number after ``previous`` that falls on ``scene``.

    The saved rotation record must stay consistent (scene = TUNNEL_SCENES[(sequence - 1) % 4]), so a chosen scene
    skips ahead to its own number instead of writing an arbitrary one; later unforced runs continue from there.
    """
    return previous + 1 + (TUNNEL_SCENES.index(scene) - previous) % len(TUNNEL_SCENES)


def graft(overlay, base, name):
    """``overlay``'s methods on top of ``base`` instead of its own parent class.

    The harness's HighwayScenario sits on MissionScenario, whose route window, zone and model handling follow a
    segment mission mode; the competition course has none. Its UDP Expert, collision evidence and startup checks do
    not depend on that, so they are rebuilt over FullLoopCompetitionScenario (the harness itself stays untouched).
    A method's zero-argument ``super()`` is bound to the class that defines it, hence the new ``__class__`` cell.
    """
    attributes = {key: value for key, value in vars(overlay).items()
                  if key not in ('__dict__', '__weakref__', '__module__', '__qualname__', '__doc__')}
    grafted = type(name, (base,), attributes)
    for key, value in attributes.items():
        if isinstance(value, types.FunctionType) and '__class__' in value.__code__.co_freevars:
            closure = list(value.__closure__)
            closure[value.__code__.co_freevars.index('__class__')] = types.CellType(grafted)
            function = types.FunctionType(value.__code__, value.__globals__, value.__name__, value.__defaults__,
                                          tuple(closure))
            function.__kwdefaults__ = value.__kwdefaults__
            function.__doc__, function.__qualname__ = value.__doc__, name + '.' + value.__name__
            setattr(grafted, key, function)
    return grafted


class CompetitionScenario(graft(harness.HighwayScenario, FullLoopCompetitionScenario, 'UdpCompetitionScenario')):
    """The competition course with the harness's UDP Expert, collision evidence and startup-contact checks."""
    stall_s = STALL_S
    forced_tunnel_scene = None
    watch_stall = tunnel.TunnelScenario.watch_stall

    def _select_scene(self):
        if self.forced_tunnel_scene is None:
            return super()._select_scene()
        previous = self.read_profile_state().get('competition', {}).get('sequence', 0)
        self.competition_sequence = sequence_for_scene(previous, self.forced_tunnel_scene)
        self.competition_tunnel_scene = self.forced_tunnel_scene
        self._configure_scene()

    def prepare_full_loop_route(self):
        result = super().prepare_full_loop_route()
        self._validation_controller = 'aim2026_competition'
        return result

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        self.watch_stall(elapsed, current_s)


def run_one(argv, options=None, record=None, limits=None, scene=None):
    """One course run: the next tunnel scene of the rotation (``scene`` forces one, for --check)."""
    options = options or {}
    record = {} if record is None else record
    limits = limits or {}
    round_timeout_s = float(limits.get('round_timeout_s', ROUND_TIMEOUT_S))
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
    config.update(drive_control_mode='aimdrive_expert',
                  expert_speed_cap_kmh=60. if args.speed_cap_kmh is None else args.speed_cap_kmh,
                  expert_ros_remaps=[], expert_traffic_light_control=False, route_debug_enabled=False,
                  max_laps=1, auto_repeat=False, route_setup_attempts=1)
    config.setdefault('sync_runtime', {})['enabled'] = False
    apply_highway_options(config, options)
    if args.seed is not None:
        config['random_seed'] = args.seed
    stamp = 'competition-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = RUNTIME / 'scenarios' / stamp
    output.mkdir(parents=True)
    global_cfg['paths']['runtime_root'] = str(output)
    global_cfg['paths']['expert_root'] = str(ROOT / 'AIM2026/Expert')
    client = SimpleNamespace(make_transform=lambda *v: v)
    if not args.check:
        from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
        import os
        if os.name == 'nt' and global_cfg['grpc'].get('host') == 'auto':
            global_cfg['grpc']['host'] = '127.0.0.1'
        client = MoraiGrpcClient(global_cfg)
    scenario = CompetitionScenario(grpc_client=client, map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
                                   global_cfg=global_cfg, scenario_cfg=config)
    args.mission = MISSION
    scenario.competition_static_installed_models = options.get('installed_models')
    scenario.forced_tunnel_scene = options.get('tunnel_scene')
    scenario.stall_s = float(limits.get('stall_s', STALL_S))
    scenario.args, scenario.output = args, output
    scenario.driver_output = RUNTIME / 'expert' / stamp
    record.update(output=str(output), driver_output=str(scenario.driver_output))
    if args.check:
        # setup() assigns these in a real run; the check path never runs it, and never advances the rotation.
        scenario.run_seed = args.seed if args.seed is not None else 0
        scenario.run_instance_id = 'check'
        scenario.competition_sequence = TUNNEL_SCENES.index(scene) + 1
        scenario.competition_tunnel_scene = scene
        scenario._configure_scene()
        scenario.route_definition = scenario.load_route_definition()
        scenario.prepare_full_loop_route()
        scenario.setup_expert_controller()
        return subprocess.run(scenario.expert_controller.command(check=True), cwd=ROOT,
                              env=scenario.expert_controller.child_environment()).returncode

    def interrupt(*_):
        raise KeyboardInterrupt
    def round_expired(*_):
        raise RoundTimeout(round_timeout_s)
    old_handler = signal.signal(signal.SIGTERM, interrupt)
    old_alarm = None
    if round_timeout_s > 0 and hasattr(signal, 'setitimer'):
        old_alarm = signal.signal(signal.SIGALRM, round_expired)
        signal.setitimer(signal.ITIMER_REAL, round_timeout_s)
    status, error = 'run_returned', None
    try:
        client.connect()
        print('[Scenario] One run: full_loop_competition -> AIM2026/main.py -> Expert', flush=True)
        scenario.run()
    except RoundTimeout as exc:
        status, error = 'exception', exc
        print('[Competition] run stopped: more than {:g} s'.format(round_timeout_s), flush=True)
        raise
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[Competition] stopped', flush=True)
    except BaseException as exc:
        status, error = 'exception', exc
        raise
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        record.update(run_seed=getattr(scenario, 'run_seed', None),
                      run_instance_id=getattr(scenario, 'run_instance_id', None),
                      tunnel_scene=getattr(scenario, 'competition_tunnel_scene', None))
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


def ensure_static_library(parser):
    """The static tunnel scene needs the curated layout library; unpack it from the shared archive when missing."""
    library = tunnel.static_library()
    if not library.is_file():
        archive = tunnel.static_archive()
        if archive is not None:
            print('[Competition] unpacking the static layout library from {} (first run only)'.format(archive),
                  flush=True)
            try:
                tunnel.install_static_library(archive, library.parents[3])
            except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
                print('[Competition] could not unpack {}: {}'.format(archive, error), flush=True)
    if not library.is_file():
        parser.error('the static layout library is not installed ({}); the course rotates through the static tunnel '
                     'scene, so copy the tunnel_static_trials folder into {} or put tunnel_static_trials.tar.gz '
                     'next to the repository'.format(library, library.parents[3]))
    fixed = tunnel.restore_lf_evidence()
    if fixed:
        print('[Competition] restored LF line ends in {} evidence JSON file(s) (Windows checkout)'.format(len(fixed)),
              flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument('--runs', type=int, default=None,
                        help='Course runs in a row, each in a fresh scene (default {}: one pass over the tunnel '
                             'scenes {}, fewer with --skip, or 1 with --tunnel-scene; without --skip/--tunnel-scene the rotation '
                             'continues where the last run stopped)'.format(RUNS, ' -> '.join(TUNNEL_SCENES)))
    parser.add_argument('--round-timeout-s', type=float, default=ROUND_TIMEOUT_S,
                        help='Stop a run (counted as failed, the next one starts) after this many wall-clock seconds, '
                             'scene setup included (default {:g}; 0 = no limit)'.format(ROUND_TIMEOUT_S))
    parser.add_argument('--stall-s', type=float, default=STALL_S,
                        help='Fail a run when the Ego makes less than {:g} m of progress along the route for this many '
                             'simulated seconds (default {:g}; 0 = off)'.format(tunnel.STALL_MIN_PROGRESS_M, STALL_S))
    parser.add_argument('--results', type=Path,
                        help='JSON-lines file with one row per run '
                             '(default .runtime/competition_aim2026/results/competition-<time>.jsonl)')
    parser.add_argument('--no-virtual-gps', dest='virtual_gps', action='store_false',
                        help='Keep real GPS inside the tunnel; the dead-reckoning provider stays idle')
    parser.add_argument('--merge-static-vehicle', action='store_true',
                        help='Highway section: also place the one stopped car on the merge ramp (left out by default, '
                             'as in the highway-only launcher; moving traffic is the same either way)')
    parser.add_argument('--tunnel-scene', choices=sorted(SCENE_NAMES),
                        help='Run this tunnel scene every time instead of the rotation {} (the whole course is run '
                             'either way); brake is the sudden-brake NPC scene'.format(' -> '.join(TUNNEL_SCENES)))
    parser.add_argument('--skip', nargs='+', choices=sorted(SCENE_NAMES), default=[], metavar='SCENE',
                        help='Leave these tunnel scenes out of the rotation ({}); the remaining ones follow each '
                             'other in the usual order, starting with the first, e.g. --skip static'.format(
                                 ', '.join(sorted(SCENE_NAMES))))
    parser.add_argument('--competition-morai', action='store_true',
                        help='The simulator is the competition MORAI: static tunnel layouts that use a model it does '
                             'not ship are left out (model list: config/competition_morai_models.txt), and the competition build\'s '
                             'vehicle interface (181 B status, single_steer55, positive_steer left) is used')
    parser.add_argument('--morai-assets', type=Path, metavar='FOLDER',
                        help='Like --competition-morai, but read the models from this MORAI launcher folder '
                             '(or its Bundle folder)')
    parser.add_argument('--config', type=Path, default=tunnel.AIM_ROOT / 'config_local.yaml')
    parser.add_argument('--seed', type=int, help='First run seed; later runs use seed+1, seed+2, ...')
    args, remaining = parser.parse_known_args(argv)
    # In WSL without --config: the UDP settings written for this PC (addresses read from the running WSL, see
    # AIM2026/Expert/wsl_network.py). Elsewhere, or with an explicit --config, nothing changes.
    network = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    network.add_argument('--expert-python', default=None)
    wsl_network.add_arguments(network)
    network_args, remaining = network.parse_known_args(remaining)
    if network_args.expert_python is not None:
        remaining = [*remaining, '--expert-python', network_args.expert_python]
    checking = '--check' in remaining
    if args.runs is not None and args.runs < 1:
        parser.error('--runs must be at least 1')
    if checking and args.runs is not None:
        parser.error('--check validates every tunnel scene once; omit --runs')
    if args.round_timeout_s < 0 or args.stall_s < 0:
        parser.error('--round-timeout-s and --stall-s must not be negative')
    if args.competition_morai and args.morai_assets is not None:
        parser.error('use either --competition-morai or --morai-assets')
    if args.skip and args.tunnel_scene:
        parser.error('--skip is for the rotation; leave out --tunnel-scene')
    skipped = {SCENE_NAMES[name] for name in args.skip}
    scenes = ([SCENE_NAMES[args.tunnel_scene]] if args.tunnel_scene
              else [scene for scene in TUNNEL_SCENES if scene not in skipped])
    if not scenes:
        parser.error('--skip leaves no tunnel scene to run')
    chosen = args.tunnel_scene is not None or bool(args.skip)      # a scene is named for every run
    if 'static' in scenes:
        ensure_static_library(parser)
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    network_args.config = args.config
    network_args.expert_python = network_args.expert_python or harness.expert_python()
    if wsl_network.configure_launcher(network_args, ROOT, parser):
        return 0
    generated = Path(network_args.config) != args.config
    args.config = Path(network_args.config)
    options = {}
    if args.merge_static_vehicle:
        options['merge_static_vehicle'] = True
    if (args.competition_morai or args.morai_assets is not None) and 'static' in scenes:
        source = args.morai_assets or tunnel.COMPETITION_MODELS
        models = (tunnel.installed_models(args.morai_assets) if args.morai_assets is not None
                  else tunnel.competition_models())
        if not models:
            parser.error('No obstacle/vehicle/pedestrian assets found in ' + str(source))
        options['installed_models'] = models
        print('[Competition] MORAI models: {} from {}'.format(len(models), source), flush=True)
    config = args.config
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if args.competition_morai:
        config = tunnel.competition_config(config, RUNTIME / 'config' / (stamp + '-competition'))
        if generated:
            size = wsl_network.keep_status_layout(args.config, config)
            if size is not None:
                print('[Competition] Ego status layout kept at {} B (the size MORAI sends on this PC; --status-bytes)'.format(size),
                      flush=True)
        print('[Competition] competition MORAI: Ego cmd_layout=single_steer55, {} status, positive_steer=left '
              'set in '.format('{} B'.format(size) if generated and size is not None else '181 B') + str(config), flush=True)
    if args.virtual_gps:
        config = tunnel.virtual_gps_config(config, RUNTIME / 'config' / stamp)
        print('[Competition] virtual GPS: Localization test_blackout zone enabled in ' + str(config), flush=True)
    else:
        print('[Competition] virtual GPS masking off; config ' + str(config), flush=True)

    def arguments(index):
        values = [*remaining, '--config', str(config)]
        if args.seed is not None:
            values += ['--seed', str(args.seed + index)]
        return values

    limits = dict(round_timeout_s=args.round_timeout_s, stall_s=args.stall_s)

    def options_for(index):
        return dict(options, tunnel_scene=scenes[index % len(scenes)]) if chosen else options

    if checking:
        checks = [run_one(arguments(0), dict(options, tunnel_scene=scene) if chosen else options, scene=scene)
                  for scene in (scenes if chosen else TUNNEL_SCENES)]
        return 0 if not any(checks) else 1
    runs = args.runs if args.runs is not None else (len(scenes) if chosen else RUNS)
    results_path = args.results or RUNTIME / 'results' / datetime.now().strftime('competition-%Y%m%d-%H%M%S.jsonl')
    print('[Competition] results file: ' + str(results_path), flush=True)
    results, rows = [], []
    for index in range(runs):
        print('[Competition] run {}/{}'.format(index + 1, runs), flush=True)
        record, started = {}, datetime.now()
        outcome, reason = 'completed', None
        try:
            code = run_one(arguments(index), options_for(index), record=record, limits=limits)
            if code == INTERRUPTED:
                outcome = 'interrupted'
            elif code != 0:
                outcome, reason = 'failed', 'exit code {}'.format(code)
        except RoundTimeout as error:
            outcome, reason = tunnel.classify_failure(error), 'more than {:g} s'.format(args.round_timeout_s)
            print('[Competition] run {} failed ({}); continuing with a fresh scene'.format(index + 1, outcome), flush=True)
        except KeyboardInterrupt:
            outcome = 'interrupted'
        except Exception as error:
            outcome, reason = tunnel.classify_failure(error), '{}: {}'.format(type(error).__name__, error)
            print('[Competition] run {} failed ({}); continuing with a fresh scene'.format(index + 1, outcome), flush=True)
        ended = datetime.now()
        report = Path(record.get('output', '')) / 'failure_report' / 'report.html' if record.get('output') else None
        row = dict(run=index + 1, tunnel_scene=record.get('tunnel_scene'), result=outcome, reason=reason,
                   seed=record.get('run_seed'), requested_seed=None if args.seed is None else args.seed + index,
                   run_instance_id=record.get('run_instance_id'),
                   started_at=started.isoformat(timespec='seconds'), ended_at=ended.isoformat(timespec='seconds'),
                   duration_s=round((ended - started).total_seconds(), 1),
                   output=record.get('output'), driver_output=record.get('driver_output'),
                   report=str(report) if report is not None and report.is_file() else None)
        rows.append(row)
        tunnel.append_result(results_path, row)
        results.append((row['tunnel_scene'], outcome if outcome in ('completed', 'interrupted')
                        else 'failed ({}): {}'.format(outcome, reason)))
        if outcome == 'interrupted':
            break
        if index + 1 < runs:
            time.sleep(RUN_GAP_S)
    for index, (scene, result) in enumerate(results, 1):
        print('[Competition] run {} (tunnel {}): {}'.format(index, scene, result), flush=True)
    failed = [row for row in rows if row['result'] not in ('completed', 'interrupted')]
    if failed:
        print('[Competition] failed runs (seed, reason, folder):', flush=True)
        for row in failed:
            print('  run {run} tunnel {tunnel_scene}: seed={seed} {result} -> {output}'.format(**row), flush=True)
    print('[Competition] results file: ' + str(results_path), flush=True)
    if results and results[-1][1] == 'interrupted':
        return INTERRUPTED
    return 0 if all(result == 'completed' for _, result in results) and len(results) == runs else 1


if __name__ == '__main__':
    raise SystemExit(main())
