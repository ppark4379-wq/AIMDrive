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
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import time
from types import SimpleNamespace

import yaml
import zipfile

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
# Reduced from 40 m for the tunnel-only brake scene (see TunnelScenario._configure_variant).
BRAKE_TEST_MIN_GAP_M = 20.
# Route-departure limit for every tunnel scene: the same 10 m the pedestrian and vehicle scenes already use
# (the normal mission's 1.5 m, and the static mission's 5 m, are too tight for scenes that steer around a stopped obstacle).
CROSS_TRACK_LIMIT_M = 10.
# Tunnel pedestrians: reaction time assumed when choosing how far ahead of the Ego a pedestrian appears. The scenario's
# own value is 1.0 s (inherited from dynamic_obstacle); this launcher sets 2.0 s in tunnel_dynamic.interaction_trigger
# only, so other zones keep 1.0 s. Override with --pedestrian-reaction-s (1 restores the scenario value).
PEDESTRIAN_REACTION_S = 2.0
VEHICLE_BEHAVIORS = ('vehicle_slow', 'vehicle_sudden_brake', 'vehicle_oncoming', 'vehicle_cut_in')
RUN_GAP_S = 3.
# Per-run limits for unattended repeats. The pedestrian and vehicle missions run without the scenario's own timeout
# and progress watchdog, so a stopped Ego would hold up every later run. 0 turns a limit off.
ROUND_TIMEOUT_S = 300.   # wall-clock time of one run, scene setup included
STALL_S = 60.            # simulated seconds without STALL_MIN_PROGRESS_M of progress along the route
STALL_MIN_PROGRESS_M = 2.
INTERRUPTED = 130  # Ctrl+C / SIGTERM: cleanup is done, but the whole campaign must stop
ROOT = harness.ROOT
# Everything this launcher writes stays inside the runner (git-ignored .runtime).
RUNTIME = RUNNER_ROOT / '.runtime/tunnel_aim2026'


class RoundTimeout(KeyboardInterrupt):
    """One run exceeded its wall-clock limit. A KeyboardInterrupt so the scene's interrupt cleanup runs as for Ctrl+C,
    but reported as a failed run (the campaign goes on with the next one)."""


def classify_failure(error):
    """Coarse reason of a failed run from the scene's own failure messages."""
    if isinstance(error, RoundTimeout):
        return 'round_timeout'
    text = str(error).lower()
    for key, reason in (('collision', 'collision'), ('left the official route', 'off_route'),
                        ('stall abort', 'stalled'), ('progress watchdog', 'stalled'), ('timeout', 'timeout')):
        if key in text:
            return reason
    return 'error'


def append_result(path, row):
    """One JSON line per run, written as soon as the run ends so a later hang or crash keeps the earlier runs."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError as error:
        print('[Tunnel] could not write {}: {}'.format(path, error), flush=True)


def _load(path):
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict):
        raise ValueError('Expected a YAML mapping: ' + str(path))
    return document


def clear_actors_with(stub, filter_type, info_type, attempts=3, pause_s=.5):
    """Destroy every actor but the Ego; returns the (id, type) pairs removed. Raises if some stay after ``attempts``."""
    removed = []
    for _ in range(attempts):
        states = stub.GetAllActorsState(
            filter_type(client_key="", vehicle=True, pedestrian=True, obstacle=True), timeout=5).states
        left = [state.actor_info for state in states if str(state.actor_info.id.value).lower() != 'ego']
        if not left:
            return removed
        for actor in left:
            info = info_type()
            info.id.value = actor.id.value
            info.object_type = actor.object_type
            stub.DestroyActor(info)
            removed.append((str(actor.id.value), int(actor.object_type)))
        time.sleep(pause_s)
    states = stub.GetAllActorsState(
        filter_type(client_key="", vehicle=True, pedestrian=True, obstacle=True), timeout=5).states
    stuck = [str(state.actor_info.id.value) for state in states if str(state.actor_info.id.value).lower() != 'ego']
    if stuck:
        raise RuntimeError('Leftover actors could not be removed: ' + ', '.join(stuck))
    return removed


def clear_leftover_actors():
    """Between runs: remove whatever a failed run left in MORAI (every actor except the Ego).

    A failed run can leave a half-placed obstacle behind. The next run then stops at its first check ("Foreign actors
    occupy the tunnel preview corridor") and so does every run after it. Only for a simulator used for these runs.
    """
    from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
    global_cfg, _, _ = load_configuration()
    client = MoraiGrpcClient(global_cfg)
    client.connect()
    from proto.morai.actor.actor_get_pb2 import GetAllActorsFilter
    from proto.morai.common.object_info_pb2 import ObjectInfo
    try:
        removed = clear_actors_with(client.client._sim_adapter._actor_stub, GetAllActorsFilter, ObjectInfo)
    finally:
        try:
            client.client.disconnect()
        except Exception:
            pass
    if removed:
        print('[Tunnel] removed {} actor(s) left in MORAI: {}'.format(
            len(removed), ', '.join('{}(type {})'.format(*item) for item in removed)), flush=True)
    return removed


SENSOR_UDP_PORTS = (9090, 2368)     # camera/GPS/IMU and LiDAR packets MORAI sends to the Expert
_PORT_HOLDERS = ("$ports = @(%s); $rows = @(); foreach ($e in Get-NetUDPEndpoint | "
                 "Where-Object { $ports -contains $_.LocalPort }) { "
                 "$p = Get-CimInstance Win32_Process -Filter \"ProcessId=$($e.OwningProcess)\"; "
                 "$alive = $false; if ($p.CommandLine -match 'parent_pid=(\\d+)') { "
                 "$alive = [bool](Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue) }; "
                 "$rows += [pscustomobject]@{pid=[int]$e.OwningProcess; port=[int]$e.LocalPort; "
                 "cmd=[string]$p.CommandLine; parent_alive=$alive} }; "
                 "ConvertTo-Json -InputObject @($rows) -Compress")


def stale_sensor_receivers(holders):
    """Split port holders into (orphans, others).

    An orphan is a multiprocessing worker whose parent Expert is gone: it only keeps the port bound and
    swallows the packets the next run waits for. Anything else (e.g. a live Expert in another terminal)
    is never touched.
    """
    orphans, others = [], []
    for holder in holders:
        command = holder.get('cmd') or ''
        spawned = 'multiprocessing.spawn' in command and re.search(r'parent_pid=\d+', command)
        (orphans if spawned and not holder.get('parent_alive') else others).append(holder)
    return orphans, others


def clear_stale_sensor_receivers(ports=SENSOR_UDP_PORTS):
    """Stop Windows Python workers a killed run left on the sensor UDP ports; refuse if something else holds them."""
    shell = shutil.which('powershell.exe') or shutil.which('powershell')
    if shell is None:
        return
    script = _PORT_HOLDERS % ','.join(str(port) for port in ports)
    # errors='replace': a non-ASCII Windows user name in a command line is not valid UTF-8 for WSL.
    result = subprocess.run([shell, '-NoProfile', '-Command', script], capture_output=True, text=True,
                            errors='replace', timeout=60)
    try:
        holders = json.loads(result.stdout.strip() or '[]')
    except ValueError:
        print('[Tunnel] Could not read the sensor port owners; continuing', flush=True)
        return
    orphans, others = stale_sensor_receivers(holders)
    if orphans:
        pids = sorted({holder['pid'] for holder in orphans})
        subprocess.run([shell, '-NoProfile', '-Command', 'Stop-Process -Force -Id ' + ','.join(map(str, pids))],
                       capture_output=True, text=True, errors='replace', timeout=60)
        print('[Tunnel] Stopped leftover sensor receivers of an earlier run: PID {}'.format(
            ', '.join(map(str, pids))), flush=True)
    if others:
        raise SystemExit('Sensor UDP ports are in use by another process (a run still active?): {}. '
                         'Stop it first; this launcher only removes leftovers of a killed run.'.format(
                             ', '.join('PID {} port {}'.format(h['pid'], h['port']) for h in others)))


def competition_config(base_config, folder):
    """Copy of the runtime config with the competition MORAI build's vehicle interface.

    The competition build (25.S4) sends the 181 B Competition Vehicle Status, takes the 55 B front-steer-only command
    and steers positive = left. A config written for another build (245 B status, 59 B command with rear steer, the
    opposite steer sign) leaves the Expert without any vehicle status. Only these three values are set; addresses,
    ports and everything else stay the PC's own.
    """
    runtime = _load(base_config)
    ego = runtime.setdefault('Ego', {})
    ego['cmd_layout'] = 'single_steer55'
    ego.pop('local_status_bytes', None)
    runtime.setdefault('Control', {})['positive_steer'] = 'left'
    folder.mkdir(parents=True)
    destination = folder / 'runtime.yaml'
    destination.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return destination


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
    pedestrian_reaction_s = PEDESTRIAN_REACTION_S
    pedestrian_margin_m = None
    vehicle_behavior = None
    static_report_dir = None
    stall_s = STALL_S

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
        cfg['expert_max_cross_track_error_m'] = CROSS_TRACK_LIMIT_M
        if self.variant == 'normal':
            # Same empty tunnel window and settings, but not the ROS-based baseline recorder (it holds the
            # brake through /ctrl_cmd at the end); this launcher drives and stops over UDP.
            dynamic = get_mission_mode('full_loop_tunnel_dynamic')
            self.mission_mode = dataclasses.replace(self.mission_mode, name=dynamic.name)
        elif self.variant in ('brake', 'static'):
            # Normal window with the tunnel zone on. The dynamic mission's name keeps speed-zone filtering and
            # leaves the empty-road baseline recorder (normal mission only) off; only the chosen event is armed.
            dynamic = get_mission_mode('full_loop_tunnel_dynamic')
            self.mission_mode = dataclasses.replace(self.mission_mode, name=dynamic.name, zones=dynamic.zones)
            cfg.setdefault('tunnel_dynamic', {})['enabled'] = False
            cfg.setdefault('tunnel_vehicle_dynamic', {})['enabled'] = False
            cfg.setdefault('tunnel_sudden_brake', {})['enabled'] = self.variant == 'brake'
            if self.variant == 'brake':
                # Test condition: this window starts 22.1 m before the entrance, so the NPC (placed 5 m inside)
                # is only 27.1 m ahead of the Ego. The full course keeps the original 40 m minimum.
                cfg['tunnel_sudden_brake']['minimum_spawn_gap_m'] = BRAKE_TEST_MIN_GAP_M
                # The sudden-brake NPC is spawned through the highway vehicle helper, which reads the
                # fixed model that the highway NPC preparation (off in this scene) normally sets.
                self.highway_fixed_model = cfg.get('tunnel_sudden_brake', {}).get(
                    'model', cfg.get('highway_npc', {}).get('model', 'DEFAULT_SEDAN'))
        elif self.variant == 'pedestrian':
            pedestrians = cfg.setdefault('tunnel_dynamic', {})
            if not self.fixed_pedestrian:
                # Same explicit random-mix (pedestrians and animals) as the full-course pedestrian scene.
                pedestrians.pop('trial_candidate_id', None)
                pedestrians['candidate_policy'] = 'trial'
                pedestrians.setdefault('random_mix', {})['enabled'] = True
            # Tunnel-only stopping-distance assumptions behind the spawn distance (tunnel_dynamic.interaction_trigger).
            # The margin stays at the scenario's own value unless given.
            trigger = pedestrians.setdefault('interaction_trigger', {})
            if self.pedestrian_reaction_s is not None:
                trigger['reaction_time_sec'] = self.pedestrian_reaction_s
            if self.pedestrian_margin_m is not None:
                trigger['stopping_safety_margin_m'] = self.pedestrian_margin_m
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

    def watch_stall(self, elapsed, current_s):
        """Fail the run when the Ego has made no progress along the route for ``stall_s`` simulated seconds.

        The base loop's progress watchdog is off in the pedestrian and vehicle missions; this one is the launcher's own."""
        if not self.stall_s:
            return
        anchor = getattr(self, '_stall_anchor', None)
        if anchor is None or current_s - anchor[0] >= STALL_MIN_PROGRESS_M:
            self._stall_anchor = (current_s, elapsed)
        elif elapsed - anchor[1] >= self.stall_s:
            self.stop_drive_control()
            raise RuntimeError('EGO stall abort: less than {:g} m of progress in {:g} s (s={:.1f} m)'.format(
                STALL_MIN_PROGRESS_M, self.stall_s, current_s))

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        self.watch_stall(elapsed, current_s)
        if self.variant == 'static':
            competition_static.update(self, elapsed, ego_state, current_s)

    def cleanup(self):
        try:
            return super().cleanup()
        finally:
            if self.variant == 'static':
                competition_static.cleanup(self)


def run_one(argv, variant, options=None, record=None, limits=None):
    """One tunnel run: the highway/roundabout harness flow with one tunnel scene.

    ``record`` (a dict) receives the run's seed, instance id and output folders; ``limits`` may hold
    ``round_timeout_s`` (wall clock, 0 = off) and ``stall_s`` (0 = off).
    """
    if variant not in MISSIONS:
        raise ValueError('Unsupported tunnel scene: ' + str(variant))
    mission = MISSIONS[variant]
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
    scenario.pedestrian_reaction_s = options.get('pedestrian_reaction_s')
    scenario.pedestrian_margin_m = options.get('pedestrian_margin_m')
    scenario.static_report_dir = output
    scenario.competition_static_installed_models = options.get('installed_models')
    scenario.stall_s = float(limits.get('stall_s', STALL_S))
    scenario.args, scenario.output = args, output
    scenario.driver_output = RUNTIME / 'expert' / stamp
    record.update(output=str(output), driver_output=str(scenario.driver_output))
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
        print('[Scenario] One run: tunnel {} ({}) -> AIM2026/main.py -> Expert'.format(variant, mission), flush=True)
        scenario.run()
    except RoundTimeout as exc:
        status, error = 'exception', exc
        print('[Tunnel] run stopped: more than {:g} s'.format(round_timeout_s), flush=True)
        raise
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[Tunnel] stopped', flush=True)
    except BaseException as exc:
        status, error = 'exception', exc
        raise
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        record.update(run_seed=getattr(scenario, 'run_seed', None),
                      run_instance_id=getattr(scenario, 'run_instance_id', None))
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


STATIC_ARCHIVE_NAMES = ('tunnel_static_trials.tar.gz', 'tunnel_static_trials.zip')


def static_archive(repo_root=None):
    """A tunnel_static_trials archive next to the repository or in the user's folder, Desktop or Downloads."""
    repo_root = Path(repo_root or ROOT)
    folders = [repo_root.parent]
    parts = repo_root.parts
    if len(parts) >= 5 and parts[1] == 'mnt' and parts[3] == 'Users':  # WSL view of C:\Users\<name>
        home = Path(*parts[:5])
        folders += [home, home / 'Desktop', home / 'Downloads']
    for folder in dict.fromkeys(folders):
        for name in STATIC_ARCHIVE_NAMES:
            if (folder / name).is_file():
                return folder / name
    return None


def install_static_library(archive, runtime_dir):
    """Unpack a tunnel_static_trials archive into the runner's .runtime folder (the folder is created if missing)."""
    archive, runtime_dir = Path(archive), Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    root = runtime_dir.resolve()
    if archive.name.endswith('.zip'):
        opened, names = zipfile.ZipFile(archive), lambda a: a.namelist()
    else:
        opened, names = tarfile.open(archive), lambda a: a.getnames()
    with opened as package:
        members = names(package)
        # An archive made from inside the folder has no top-level tunnel_static_trials entry.
        target = root if any(name.replace('\\', '/').split('/')[0] == 'tunnel_static_trials' for name in members) \
            else root / 'tunnel_static_trials'
        target.mkdir(parents=True, exist_ok=True)
        for name in members:
            destination = (target / name).resolve()
            if destination != target.resolve() and target.resolve() not in destination.parents:
                raise ValueError('Unsafe path in the archive: ' + name)
        package.extractall(target)


def static_library():
    """The approved static-layout library behind the static scene. It lives under .runtime, which git ignores."""
    from scenario_runner.scenarios.segment_modes.tunnel_static_replay import DEFAULT_LIBRARY
    return Path(DEFAULT_LIBRARY)


MORAI_ASSET_SUFFIXES = ('.obstacle', '.vehicle', '.pedestrian')
COMPETITION_MODELS = RUNNER_ROOT / 'config' / 'competition_morai_models.txt'


def competition_models(path=COMPETITION_MODELS):
    """Normalized asset names listed in the repository's copy of the competition MORAI's model list."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return frozenset(competition_static.model_key(line) for line in lines if line.strip() and not line.startswith('#'))


def installed_models(folder):
    """Normalized names of the obstacle/vehicle/pedestrian assets a MORAI install ships.

    ``folder`` is the MORAI launcher folder, its ``<name>_Data/Bundle`` folder or one bundle version inside it.
    """
    folder = Path(folder).expanduser()
    roots = [folder, *folder.glob('*'), *folder.glob('Bundle/*'), *folder.glob('*_Data/Bundle/*')]
    names = set()
    for root in roots:
        if root.is_dir():
            names.update(competition_static.model_key(path.stem) for path in root.iterdir()
                         if path.suffix in MORAI_ASSET_SUFFIXES)
    return frozenset(names)


def restore_lf_evidence(roots=(RUNNER_ROOT / 'data' / 'geometry', RUNNER_ROOT / 'config')):
    """Put CRLF line ends back to LF in the static scene's evidence and configuration JSON files.

    The static scene hashes its evidence files byte for byte. A Windows checkout (git autocrlf) turns their LF line
    ends into CRLF, which fails the hash check ("Static 3D evidence hash mismatch"). Only the line ends change. The map
    data (data/R_KR_PR_K-city_2025/*.json) is stored with CRLF in git and is left alone.
    """
    fixed = []
    for root in roots:
        for path in sorted(Path(root).rglob('*.json')):
            try:
                data = path.read_bytes()
                if b'\r\n' in data:
                    path.write_bytes(data.replace(b'\r\n', b'\n'))
                    fixed.append(path)
            except OSError as error:
                print('[Tunnel] could not restore LF line ends in {}: {}'.format(path, error), flush=True)
    return fixed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Other options go to the single-run flow '
                                     '(--check, --speed-cap-kmh, --validation-events, --strict-collisions, ...).')
    parser.add_argument('--variant', choices=[*MISSIONS, 'all'], default='normal',
                        help='normal: empty tunnel; brake: sudden-brake NPC; static: approved obstacle layout; '
                             'pedestrian: pedestrians and animals; vehicle: NPC vehicle behaviours; '
                             'all: brake, static, pedestrian, vehicle in turn, one scene per run (see --skip)')
    parser.add_argument('--fixed-pedestrian', action='store_true',
                        help='pedestrian: the single default pedestrian instead of the random pedestrian/animal mix')
    parser.add_argument('--pedestrian-reaction-s', type=float, default=None,
                        help='pedestrian: reaction time assumed when choosing how far ahead of the Ego a pedestrian '
                             'appears (default {:g}; the scenario\'s own value is 1.0). Larger = spawned farther '
                             'away'.format(PEDESTRIAN_REACTION_S))
    parser.add_argument('--pedestrian-margin-m', type=float, default=None,
                        help='pedestrian: extra stopping margin in metres (scenario default 3.0), e.g. 5')
    parser.add_argument('--vehicle-behavior', choices=VEHICLE_BEHAVIORS,
                        help='vehicle: fix one behaviour (default: random)')
    parser.add_argument('--skip', nargs='+', choices=ROTATION, default=[], metavar='SCENE',
                        help='with --variant all: leave these scenes out of the rotation ({}), e.g. --skip brake'.format(
                            ', '.join(ROTATION)))
    parser.add_argument('--runs', type=int, default=None,
                        help='Repeat the tunnel segment this many times, each in a fresh scene '
                             '(default 1; 4 with --variant all)')
    parser.add_argument('--round-timeout-s', type=float, default=ROUND_TIMEOUT_S,
                        help='Stop a run (counted as failed, the next one starts) after this many wall-clock seconds, '
                             'scene setup included (default {:g}; 0 = no limit)'.format(ROUND_TIMEOUT_S))
    parser.add_argument('--stall-s', type=float, default=STALL_S,
                        help='Fail a run when the Ego makes less than {:g} m of progress along the route for this many '
                             'simulated seconds (default {:g}; 0 = off)'.format(STALL_MIN_PROGRESS_M, STALL_S))
    parser.add_argument('--results', type=Path,
                        help='JSON-lines file with one row per run when repeating '
                             '(default .runtime/tunnel_aim2026/results/tunnel-<time>.jsonl)')
    parser.add_argument('--clear-leftovers', action='store_true',
                        help='Before every run, remove all actors but the Ego from MORAI, so a failed run cannot block '
                             'the following ones (only for a simulator used for these runs)')
    parser.add_argument('--no-virtual-gps', dest='virtual_gps', action='store_false',
                        help='Keep real GPS inside the tunnel; the dead-reckoning provider stays idle')
    parser.add_argument('--config', type=Path, default=AIM_ROOT / 'config_local.yaml')
    parser.add_argument('--seed', type=int, help='First run seed; later runs use seed+1, seed+2, ...')
    parser.add_argument('--competition-morai', action='store_true',
                        help='static: the simulator is the competition MORAI; layouts that use a model it does not '
                             'ship are left out (model list: config/competition_morai_models.txt), and the competition build\'s '
                             'vehicle interface (181 B status, single_steer55, positive_steer left) is used')
    parser.add_argument('--morai-assets', type=Path, metavar='FOLDER',
                        help='static: like --competition-morai, but read the models from this MORAI launcher folder '
                             '(or its Bundle folder)')
    args, remaining = parser.parse_known_args(argv)
    if args.runs is not None and args.runs < 1:
        parser.error('--runs must be at least 1')
    if args.round_timeout_s < 0 or args.stall_s < 0:
        parser.error('--round-timeout-s and --stall-s must not be negative')
    if '--check' in remaining and args.runs is not None:
        parser.error('--check validates one run per scene; omit --runs')
    rotating = args.variant == 'all'
    if args.skip and not rotating:
        parser.error('--skip is for --variant all')
    rotation = tuple(name for name in ROTATION if name not in args.skip)
    if not rotation:
        parser.error('--skip leaves no scene to run')
    if 'static' in (rotation if rotating else (args.variant,)):
        library = static_library()
        if not library.is_file():
            archive = static_archive()
            if archive is not None:
                print('[Tunnel] unpacking the static layout library from {} (first run only)'.format(archive), flush=True)
                try:
                    install_static_library(archive, library.parents[3])
                except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
                    print('[Tunnel] could not unpack {}: {}'.format(archive, error), flush=True)
        if not library.is_file():
            hint = ('the static layout library is not installed ({}); copy the tunnel_static_trials folder into '
                    '{} to run the static scene'.format(library, library.parents[3]))
            if not rotating:
                parser.error(hint)
            print('[Tunnel] ' + hint + '. The static scene is left out of this run.', flush=True)
            rotation = tuple(name for name in rotation if name != 'static')
            if not rotation:
                parser.error('--skip leaves no scene to run')
        else:
            fixed = restore_lf_evidence()
            if fixed:
                print('[Tunnel] restored LF line ends in {} evidence JSON file(s) (Windows checkout)'.format(len(fixed)),
                      flush=True)
    runs = args.runs if args.runs is not None else (len(rotation) if rotating else 1)
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    for name, value in (('--pedestrian-reaction-s', args.pedestrian_reaction_s),
                        ('--pedestrian-margin-m', args.pedestrian_margin_m)):
        if value is None:
            continue
        if args.variant not in ('pedestrian', 'all'):
            parser.error(name + ' is for --variant pedestrian or all')
        if not 0. <= value <= 10.:
            parser.error(name + ' must be within 0..10')
    reaction = PEDESTRIAN_REACTION_S if args.pedestrian_reaction_s is None else args.pedestrian_reaction_s
    options = dict(fixed_pedestrian=args.fixed_pedestrian, vehicle_behavior=args.vehicle_behavior,
                   pedestrian_reaction_s=reaction, pedestrian_margin_m=args.pedestrian_margin_m)
    if args.morai_assets is not None or args.competition_morai:
        if args.morai_assets is not None and args.competition_morai:
            parser.error('use either --competition-morai or --morai-assets')
        has_static = 'static' in (rotation if rotating else (args.variant,))
        if args.morai_assets is not None and not has_static:
            parser.error('--morai-assets is for the static scene (--variant static, or all without --skip static)')
        if has_static:
            source = args.morai_assets or COMPETITION_MODELS
            models = installed_models(args.morai_assets) if args.morai_assets is not None else competition_models()
            if not models:
                parser.error('No obstacle/vehicle/pedestrian assets found in ' + str(source))
            options['installed_models'] = models
            print('[Tunnel] MORAI models: {} from {}'.format(len(models), source), flush=True)
    if args.fixed_pedestrian and args.variant not in ('pedestrian', 'all'):
        parser.error('--fixed-pedestrian is for --variant pedestrian or all')
    if args.vehicle_behavior and args.variant not in ('vehicle', 'all'):
        parser.error('--vehicle-behavior is for --variant vehicle or all')
    config = args.config
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if args.competition_morai:
        config = competition_config(config, RUNTIME / 'config' / (stamp + '-competition'))
        print('[Tunnel] competition MORAI: Ego cmd_layout=single_steer55, 181 B status, positive_steer=left '
              'set in ' + str(config), flush=True)
    if args.virtual_gps:
        config = virtual_gps_config(config, RUNTIME / 'config' / stamp)
        print('[Tunnel] virtual GPS: Localization test_blackout zone enabled in ' + str(config), flush=True)
    else:
        print('[Tunnel] virtual GPS masking off; config ' + str(config), flush=True)

    def arguments(index):
        values = [*remaining, '--config', str(config)]
        if args.seed is not None:
            values += ['--seed', str(args.seed + index)]
        return values

    limits = dict(round_timeout_s=args.round_timeout_s, stall_s=args.stall_s)

    def variant_of(index):
        return rotation[index % len(rotation)] if rotating else args.variant

    if '--check' in remaining:
        # One validation per scene; with all that is every scene of the rotation.
        checks = [run_one(arguments(0), variant, options) for variant in (rotation if rotating else (args.variant,))]
        return 0 if not any(checks) else 1
    def prepare_run():
        clear_stale_sensor_receivers()      # a killed run's workers keep the sensor ports and fail every later run
        if args.clear_leftovers:
            try:
                clear_leftover_actors()
            except Exception as error:
                print('[Tunnel] could not clear leftover actors: {}: {}'.format(type(error).__name__, error), flush=True)

    if runs == 1:
        try:
            prepare_run()
            return run_one(arguments(0), variant_of(0), options, limits=limits)
        except RoundTimeout:
            return 1
    results = []
    rows = []
    results_path = args.results or RUNTIME / 'results' / datetime.now().strftime('tunnel-%Y%m%d-%H%M%S.jsonl')
    print('[Tunnel] results file: ' + str(results_path), flush=True)
    for index in range(runs):
        variant = variant_of(index)
        print('[Tunnel] run {}/{}: {}'.format(index + 1, runs, variant), flush=True)
        record, started = {}, datetime.now()
        outcome, reason = 'completed', None
        try:
            prepare_run()
            code = run_one(arguments(index), variant, options, record=record, limits=limits)
            if code == INTERRUPTED:
                outcome = 'interrupted'
            elif code != 0:
                outcome, reason = 'failed', 'exit code {}'.format(code)
        except RoundTimeout as error:
            outcome, reason = classify_failure(error), 'more than {:g} s'.format(args.round_timeout_s)
            print('[Tunnel] run {} failed ({}); continuing with a fresh scene'.format(index + 1, outcome), flush=True)
        except KeyboardInterrupt:
            outcome = 'interrupted'
        except Exception as error:
            outcome, reason = classify_failure(error), '{}: {}'.format(type(error).__name__, error)
            print('[Tunnel] run {} failed ({}); continuing with a fresh scene'.format(index + 1, outcome), flush=True)
        ended = datetime.now()
        report = Path(record.get('output', '')) / 'failure_report' / 'report.html' if record.get('output') else None
        row = dict(run=index + 1, scene=variant, mission=MISSIONS[variant], result=outcome, reason=reason,
                   seed=record.get('run_seed'), requested_seed=None if args.seed is None else args.seed + index,
                   run_instance_id=record.get('run_instance_id'),
                   started_at=started.isoformat(timespec='seconds'), ended_at=ended.isoformat(timespec='seconds'),
                   duration_s=round((ended - started).total_seconds(), 1),
                   output=record.get('output'), driver_output=record.get('driver_output'),
                   report=str(report) if report is not None and report.is_file() else None)
        rows.append(row)
        append_result(results_path, row)
        results.append((variant, outcome if outcome in ('completed', 'interrupted')
                        else 'failed ({}): {}'.format(outcome, reason)))
        if outcome == 'interrupted':
            break
        if index + 1 < runs:
            time.sleep(RUN_GAP_S)
    for index, (variant, result) in enumerate(results, 1):
        print('[Tunnel] run {} ({}): {}'.format(index, variant, result), flush=True)
    failed = [row for row in rows if row['result'] not in ('completed', 'interrupted')]
    if failed:
        print('[Tunnel] failed runs (seed, reason, folder):', flush=True)
        for row in failed:
            print('  run {run} {scene}: seed={seed} {result} -> {output}'.format(**row), flush=True)
    print('[Tunnel] results file: ' + str(results_path), flush=True)
    done = {name: sum(1 for variant, result in results if variant == name and result == 'completed')
            for name in dict.fromkeys(variant for variant, _ in results)}
    print('[Tunnel] completed per scene: ' + ', '.join('{} {}'.format(n, c) for n, c in done.items()), flush=True)
    if results and results[-1][1] == 'interrupted':
        return INTERRUPTED
    return 0 if all(result == 'completed' for _, result in results) and len(results) == runs else 1

if __name__ == '__main__':
    raise SystemExit(main())
