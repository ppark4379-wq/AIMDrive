"""Isolated full-route trials. GT is used by the runner for scoring, not perception."""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import REPO_ROOT, project_path, activate_project_paths
activate_project_paths()

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import signal
import subprocess
import time

REPO = REPO_ROOT
ROOT = project_path('MoraiDrive/scenario_runner/.runtime/ve_full_route_campaign')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def classify(code, text, forced=None):
    if forced:
        return forced
    if 'collision during scenario verification' in text:
        return 'collision'
    if 'FATAL_STALL' in text or 'progress watchdog abort' in text:
        return 'stalled_180s'
    if code == 0 and 'GOAL REACHED' in text:
        return 'success'
    if 'VE/자차 입력이 2초 이상 끊겼습니다' in text:
        return 'perception_input_stale'
    # The normal startup banner contains "timeout=disabled".
    if any(token in text.lower() for token in (
            'timeouterror:', 'full-loop timeout:', 'crossing timeout:',
            '센서 준비 시간 초과', '토픽 연결 시간 초과')):
        return 'timeout'
    return 'runtime_error' if 'AIMDrive expert running.' in text else 'startup_error'


class MovementWatch:
    def __init__(self, seconds):
        self.seconds = seconds
        self.anchor = None
        self.moved_at = None

    def update(self, x, y, now):
        if not all(math.isfinite(v) for v in (x, y)):
            return
        if self.anchor is None or math.hypot(x-self.anchor[0], y-self.anchor[1]) >= 2.0:
            self.anchor = (x, y)
            self.moved_at = now

    def stalled(self, now):
        return self.moved_at is not None and now-self.moved_at >= self.seconds


# Expert console states of the static-obstacle planner that mean "no usable path yet":
# blocked (remaining path collides / no route) or still planning. Following a path
# while stopped for a traffic signal is deliberately not one of them.
# The GPS/VE runner prints the Expert log into its own driver directory, announced in the trial console.
DRIVER_LOG = re.compile(r'\[GPS\+VE 주행 로그\]\s*(\S+)/driver\.console\.log')
STATIC_TROUBLE = re.compile(
    r'\[PLAN\] reason=(static_recorded_blocked|static_recorded_planning|no_complete_route|'
    r'planning_error|no_global_return)|\[STATIC_FAILURE\]')


SIGNAL_LIMIT = re.compile(r"\[STATIC\] \{.*'speed_limits': \{[^}]*'signal': ([-0-9.e]+)")


class StaticStuckWatch:
    """Ego has not moved for ``seconds`` while the static planner reported trouble within that time.

    Time spent held by a traffic signal (the Expert's static-zone signal speed limit is 0) is a
    signal wait, not a static failure: it restarts the stationary clock.
    """
    def __init__(self, seconds):
        self.movement = MovementWatch(seconds)
        self.seconds = seconds
        self.trouble_at = None

    def update(self, x, y, now):
        self.movement.update(x, y, now)

    def console(self, line, now):
        if STATIC_TROUBLE.search(line):
            self.trouble_at = now
        signal = SIGNAL_LIMIT.search(line)
        if signal and float(signal.group(1)) <= 1e-3 and self.movement.moved_at is not None:
            self.movement.moved_at = now

    def stuck(self, now):
        return (self.trouble_at is not None and now-self.trouble_at <= self.seconds
                and self.movement.stalled(now))


def descendants():
    """The subreaper adopts orphaned grandchildren, including new-session VE workers."""
    records = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            records[int(path.parent.name)] = (int(fields[1]), fields[0])
        except (OSError, ValueError, IndexError):
            pass
    owned = {os.getpid()}
    while True:
        added = {pid for pid, (parent, _) in records.items() if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    return {pid for pid in owned if pid != os.getpid() and records[pid][1] != 'Z'}


def cleanup(child):
    # First let the runner brake, remove its NPCs, and unregister its ROS nodes.
    if child.poll() is None:
        try:
            os.kill(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in descendants():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+5
        while descendants() and time.monotonic() < deadline:
            time.sleep(.1)
    child.wait(timeout=2)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break
    if descendants():
        raise RuntimeError('Campaign-owned child processes remain; next trial is blocked')


def trial_seeds(runs, seed=None):
    """New entropy by default; a logged root seed reproduces the unique trial list."""
    seed = secrets.randbits(64) if seed is None else seed
    return seed, random.Random(seed).sample(range(1, 2**31), runs)


def validate_aim2026_options(args, parser):
    """Resolve only the explicitly selected backend and reject conflicting DR switches."""
    selected = getattr(args, 'aim2026_config', None)
    if selected is None:
        return
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = project_path(path)
    path = path.resolve()
    if not path.is_file():
        parser.error('--aim2026-config file not found: '+str(path))
    if args.simulate_gps_blackout:
        parser.error('--aim2026-config uses explicit Localization test_blackout settings; omit --simulate-gps-blackout')
    if args.gps_latency_s != 0.:
        parser.error('--aim2026-config aligned localization requires --gps-latency-s 0')
    if getattr(args, 'gt_expert', False):
        parser.error('--aim2026-config uses the common GPS Expert and cannot be combined with --gt-expert')
    import yaml
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ValueError('expected a YAML mapping')
        localization = config.get('Localization')
        if localization is None:
            localization = {}
        if not isinstance(localization, dict):
            raise ValueError('Localization must be a YAML mapping')
        enabled = localization.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('Localization.enabled must be boolean')
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error('--aim2026-config: '+str(exc))
    if enabled and not args.gps_blackout_hold:
        parser.error('enabled AIM2026 Localization requires --gps-blackout-hold')
    args.aim2026_config = path


def trial_command(args, seed, trace):
    command = ['bash', str(project_path('MoraiDrive/scenario_runner/tools/run_ve_full_route.sh')),
               '--seed', str(seed), '--ego-trace', str(trace)]
    if getattr(args, 'aim2026_config', None):
        command += ['--aim2026-config', str(args.aim2026_config)]
    if args.stop_before_tunnel:
        command.append('--stop-before-tunnel')
    if args.static_geometry_fusion:
        command.append('--static-geometry-fusion')
    if args.moving_carry_s:
        command += ['--moving-carry-s', str(args.moving_carry_s)]
    if args.slow_window_s:
        command += ['--slow-window-s', str(args.slow_window_s)]
    if args.stationary_hold_s:
        command += ['--stationary-hold-s', str(args.stationary_hold_s)]
    if args.ignore_signals:
        command += ['--ignore-signals', args.ignore_signals]
    if args.speed_cap_kmh != 60.:
        command += ['--speed-cap-kmh', str(args.speed_cap_kmh)]
    if args.mission:
        command += ['--mission', args.mission]
    if args.sensor_only:
        command += ['--sensor-only', '--gps-hold-method', args.gps_hold_method,
                    '--gps-anchor-window-s', str(args.gps_anchor_window_s)]
    if args.tunnel_vehicle_behavior:
        command += ['--tunnel-vehicle-behavior', args.tunnel_vehicle_behavior]
    if args.tunnel_dynamic_speed is not None:
        command += ['--tunnel-dynamic-speed', str(args.tunnel_dynamic_speed)]
    if args.tunnel_candidate_id:
        command += ['--tunnel-candidate-id', args.tunnel_candidate_id]
    if args.tunnel_layout is not None:
        command += ['--tunnel-layout', str(args.tunnel_layout)]
    if args.gps_blackout_hold:
        command += ['--gps-blackout-hold', '--gps-latency-s', str(args.gps_latency_s)]
    if args.simulate_gps_blackout:
        command.append('--simulate-gps-blackout')
    return command


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--stall-seconds', type=int, default=180)
    parser.add_argument('--max-trial-seconds', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional root seed to reproduce a campaign; default is fresh random entropy.')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--stop-before-tunnel', dest='stop_before_tunnel', action='store_true',
                       help='Finish successfully at highway exit checkpoint 13 (default).')
    scope.add_argument('--include-tunnel', dest='stop_before_tunnel', action='store_false',
                       help='Explicitly test the complete loop including the tunnel.')
    parser.set_defaults(stop_before_tunnel=True)
    parser.add_argument('--speed-cap-kmh', type=float, default=60.,
                        help='Expert target-speed cap for every trial (default 60). Above 60 the campaign leaves the '
                             'speed range VE was trained on; use it to measure, not to compare with 60 km/h runs.')
    parser.add_argument('--ignore-signals', default='',
                        help='Comma-separated signal IDs ignored by the Expert in every trial (static-obstacle tests; recorded in the report)')
    parser.add_argument('--static-stuck-seconds', type=float, default=30.,
                        help='End the trial as static_stuck when the ego is stopped this long while the static '
                             'planner is blocked/planning (traffic-signal stops do not count); 0 disables.')
    parser.add_argument('--moving-carry-s', type=float, default=0.)
    parser.add_argument('--slow-window-s', type=float, default=0.)
    parser.add_argument('--stationary-hold-s', type=float, default=0.)
    parser.add_argument('--static-geometry-fusion', action='store_true',
                        help='Publish fused footprints for stationary VE boxes (A/B: reuse --seed for paired layouts).')
    parser.add_argument('--mission', default='',
                        help='Run a bounded mission instead of the full loop, e.g. full_loop_tunnel_normal (starts ~40 m before the tunnel).')
    parser.add_argument('--tunnel-candidate-id', default=None)
    parser.add_argument('--tunnel-layout', default=None,
                        help='Spawn one of the team\'s tunnel static layouts (0..120, or random from each trial seed).')
    parser.add_argument('--tunnel-dynamic-speed', type=float, default=None,
                        help='With --mission full_loop_tunnel_dynamic: pedestrian speed in m/s (0 < v <= 5, default 2.5).')
    parser.add_argument('--tunnel-vehicle-behavior', default='',
                        help='With --mission full_loop_tunnel_vehicle: fix the tunnel vehicle behavior (vehicle_slow, vehicle_sudden_brake, vehicle_oncoming, vehicle_cut_in).')
    parser.add_argument('--aim2026-config', type=Path,
                        help='Optional AIM2026 YAML backend for the existing common Expert; defaults remain unchanged.')
    parser.add_argument('--gps-blackout-hold', action='store_true',
                        help='Hold the position through a GPS blackout with IMU heading + status speed (needed to drive the tunnel).')
    parser.add_argument('--sensor-only', action='store_true',
                        help='Empty tunnel evaluation with GPS/IMU/status only; no VE models or camera/LiDAR inputs.')
    parser.add_argument('--gps-latency-s', type=float, default=0.,
                        help='GPS receipt latency compensated at the blackout anchor (default 0; measure live first).')
    parser.add_argument('--gps-hold-method', choices=('legacy', 'aligned'), default='aligned')
    parser.add_argument('--gps-anchor-window-s', type=float, default=1.)
    parser.add_argument('--simulate-gps-blackout', action='store_true',
                        help='Development: mask GPS inside the mapped tunnel entrance/exit for comparison.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sensor_only:
        parser.error('--sensor-only is not part of this port: it needs VLDrive sensor-audit modules that main does not have')
    validate_aim2026_options(args, parser)
    if args.mission:
        args.stop_before_tunnel = False
    if args.sensor_only and (args.mission != 'full_loop_tunnel_normal' or args.tunnel_layout is not None):
        parser.error('--sensor-only requires --mission full_loop_tunnel_normal without --tunnel-layout')
    if args.sensor_only and args.aim2026_config:
        parser.error('--sensor-only uses the sensor audit driver; omit --aim2026-config')
    if args.simulate_gps_blackout and not args.gps_blackout_hold:
        parser.error('--simulate-gps-blackout requires --gps-blackout-hold')
    if args.gps_blackout_hold and args.stop_before_tunnel:
        parser.error('--gps-blackout-hold only matters inside the tunnel; add --include-tunnel')
    if not 0 <= args.gps_latency_s <= .5:
        parser.error('--gps-latency-s must be within 0..0.5')
    if min(args.runs, args.stall_seconds, args.max_trial_seconds, args.speed_cap_kmh) <= 0 or args.static_stuck_seconds < 0:
        parser.error('counts and timeouts must be positive')
    return args


def main():
    args = parse_arguments()
    campaign_seed, seeds = trial_seeds(args.runs, args.seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Linux PR_SET_CHILD_SUBREAPER: cleanup owns only descendants of this campaign.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'status.json').exists():
        raise RuntimeError('Output already contains a campaign; choose a new directory')
    report = dict(status='RUNNING', pid=os.getpid(), requested=args.runs,
                  started_unix=time.time(), stall_seconds=args.stall_seconds,
                  movement_threshold_m=2, max_trial_seconds=args.max_trial_seconds,
                  perception='none_sensor_only' if args.sensor_only else 'P1 objects; native straight signals; C052 three-class left signal',
                  sensor_only=args.sensor_only, model_weights_required=not args.sensor_only,
                  speed_cap_kmh=args.speed_cap_kmh,
                  unsupported_signals=[] if args.sensor_only else ['LCS01', 'other left intersections'],
                  static_layout_policy='empty_road' if args.sensor_only else 'historically_gt_verified_exact_pair',
                  highway_merge_static_vehicle=False,
                  static_geometry_fusion=args.static_geometry_fusion,
                  moving_carry_s=args.moving_carry_s, slow_window_s=args.slow_window_s, stationary_hold_s=args.stationary_hold_s,
                  gps_blackout_hold=args.gps_blackout_hold, gps_latency_s=args.gps_latency_s,
                  gps_hold_method=args.gps_hold_method, gps_anchor_window_s=args.gps_anchor_window_s,
                  simulate_gps_blackout=args.simulate_gps_blackout,
                  aim2026_config=str(args.aim2026_config) if args.aim2026_config else None,
                  static_stuck_seconds=args.static_stuck_seconds,
                  ignored_signals=[x for x in args.ignore_signals.split(',') if x],
                  campaign_seed=campaign_seed, trial_seeds=seeds,
                  seed_policy='random_without_replacement_from_logged_root_seed',
                  between_runs_delay_seconds=0,
                  route_scope=('mission:' + args.mission if args.mission else
                               'start_to_highway_exit_before_tunnel' if args.stop_before_tunnel else 'full_loop'),
                  trials=[], success=0, completed=0)
    paths = ['VLDrive/inference/ve_object_tracking.py',
             'VLDrive/inference/ve_objects_udp.py',
             'VLDrive/inference/realtime_ve_objects.py',
             'MoraiDrive/scenario_runner/utils/perception.py',
             'MoraiDrive/scenario_runner/utils/environment.py',
             'MoraiDrive/scenario_runner/tools/project_env.sh',
             'MoraiDrive/scenario_runner/_compat.py',
             'MoraiDrive/scenario_runner/adapters/ve_signal.py',
             'MoraiDrive/scenario_runner/adapters/ve/signal.py',
             'MoraiDrive/scenario_runner/tools/run_ve_full_route.sh',
             'MoraiDrive/scenario_runner/scenarios/full_loop_parts/before_tunnel.py',
             'MoraiDrive/scenario_runner/scenarios/full_loop_parts/route_profile.py',
             'MoraiDrive/scenario_runner/scenarios/full_loop_parts/static_obstacle.py',
             'MoraiDrive/scenario_runner/tools/ve_full_route_campaign.py']
    if args.gps_blackout_hold:
        paths += ['AIM2026/Localization/gps_blackout_hold.py',
                  'MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.py',
                  'MoraiDrive/scenario_runner/adapters/udp/gps_ve_driver.py',
                  'MoraiDrive/scenario_runner/adapters/udp/gt_driver.py',
                  'MoraiDrive/scenario_runner/adapters/udp/protocol.py',
                  'MoraiDrive/scenario_runner/tools/analyze_tunnel_hold.py']
    report['source_sha256'] = {p: hashlib.sha256((project_path(p)).read_bytes()).hexdigest() for p in paths}

    def publish():
        report['updated_unix'] = time.time()
        report['completed'] = len(report['trials'])
        report['success'] = sum(t['result'] == 'success' for t in report['trials'])
        report['success_rate_completed'] = (report['success']/report['completed']
                                            if report['completed'] else None)
        save(output/'status.json', report)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    publish()
    print('[캠페인] root_seed={} · 각 회차 무작위 시드 · 종점={} · 정리 후 즉시 다음 회차'.format(
        campaign_seed, args.mission or ('터널 앞 체크포인트 13' if args.stop_before_tunnel else '전체 풀루프')), flush=True)
    try:
        for number in range(1, args.runs+1):
            seed = seeds[number-1]
            trial = output / ('trial_%02d' % number)
            trial.mkdir()
            trace = trial/'ego.csv'
            console = trial/'console.log'
            command = trial_command(args, seed, trace)
            save(trial/'command.json', command)
            report['active_trial'] = number
            report['active_seed'] = seed
            report['active_console'] = str(console)
            publish()
            print('[시작] {}/{} seed={}'.format(number, args.runs, seed), flush=True)
            started = time.monotonic()
            watcher = MovementWatch(args.stall_seconds)
            static_watch = StaticStuckWatch(args.static_stuck_seconds) if args.static_stuck_seconds else None
            offset = console_offset = expert_offset = 0
            expert_log = None
            forced = None
            with console.open('w', buffering=1) as stream:
                child = subprocess.Popen(command, cwd=str(REPO), stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while child.poll() is None:
                        now = time.monotonic()
                        if trace.exists():
                            with trace.open() as source:
                                source.seek(offset)
                                for line in source:
                                    try:
                                        fields = line.split(',')
                                        watcher.update(float(fields[1]), float(fields[2]), now)
                                        if static_watch:
                                            static_watch.update(float(fields[1]), float(fields[2]), now)
                                    except (ValueError, IndexError):
                                        pass
                                offset = source.tell()
                        if static_watch:
                            for path, position in ((console, 'console'), (expert_log, 'expert')):
                                if path is None or not path.exists():
                                    continue
                                with path.open('rb') as source:
                                    source.seek(console_offset if position == 'console' else expert_offset)
                                    chunk = source.read()
                                complete = chunk[:chunk.rfind(b'\n')+1]
                                if position == 'console':
                                    console_offset += len(complete)
                                else:
                                    expert_offset += len(complete)
                                for line in complete.decode(errors='replace').splitlines():
                                    static_watch.console(line, now)
                                    found = DRIVER_LOG.search(line) if position == 'console' else None
                                    if found and expert_log is None:
                                        expert_log = Path(found.group(1))/'driver'/'expert.log'
                        if static_watch and static_watch.stuck(now):
                            forced = 'static_stuck'
                        elif watcher.stalled(now):
                            forced = 'stalled_180s'
                        elif watcher.anchor is None and now-started > 180:
                            forced = 'startup_timeout'
                        elif now-started > args.max_trial_seconds:
                            forced = 'wall_timeout'
                        report['active_elapsed_s'] = round(now-started, 1)
                        report['stationary_s'] = (round(now-watcher.moved_at, 1)
                                                   if watcher.moved_at is not None else None)
                        publish()
                        if forced:
                            break
                        time.sleep(1)
                finally:
                    cleanup(child)
            text = console.read_text(errors='replace')
            result = classify(child.returncode, text, forced)
            evidence = [line for line in text.splitlines() if any(token in line for token in
                        ('FATAL_', 'GOAL REACHED', 'Error:', '[VE 신호]', '[KCityFullLoop] lap='))]
            entry = dict(trial=number, seed=seed, result=result,
                         returncode=child.returncode, elapsed_s=round(time.monotonic()-started, 1),
                         console=str(console), ego_trace=str(trace), tail_evidence=evidence[-12:])
            save(trial/'result.json', entry)
            report['trials'].append(entry)
            publish()
            print('[결과] {}/{} {} · 성공 {}/{}'.format(number, args.runs, result,
                  report['success'], report['completed']), flush=True)
        report['status'] = 'COMPLETE'
    except KeyboardInterrupt:
        report['status'] = 'STOPPED'
    except Exception as error:
        report['status'] = 'BLOCKED'
        report['error'] = repr(error)
        raise
    finally:
        report['finished_unix'] = time.time()
        report['active_trial'] = None
        report['active_seed'] = None
        publish()
        save(output/'report.json', report)


if __name__ == '__main__':
    main()
