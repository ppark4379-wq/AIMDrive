"""Bounded scenario/expert -> D raw -> timing audit. Never publishes training data.

Simulation Time is an explicit diagnostic mode, with separately recorded
scenario and expert clock contracts. No expert algorithm is modified here.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT_ROOT)]
os.environ['MORAI_GRPC_SRC'] = str(
    PROJECT_ROOT / 'data_collection/third_party/grpc_inha_univ/src')
from async_capture.journal import write_json, digest
from async_capture.sim_mode import require_free_running_mode
from process_guard import process_lock
from python_environment import build_analysis_environment, probe_analysis_environment
from scenario_collection import (configure_collection_scenario,
    tunnel_collection_evidence, collection_tunnel_phase)
from ssd_storage import SSD_ROOT, require_ssd
from scenario_runner.main_runner import (
    PACKAGE_ROOT as SCENARIO_PACKAGE_ROOT,
    SUPPORTED_SCENARIOS,
    load_configuration,
    resolve_static_preset,
)
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.scenarios.segment_modes.modes import MISSION_MODES, PRIMARY_SCENARIOS
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.tools.verify_separation import verify_package
from async_capture.scenario_simulation import (SimulationObservationClock, RosClockGuard,
    cleanup_unreachable_expert_publishers, publish_final_brake,
    start_simulation_time_world, wait_for_control_release, wait_for_native_capture_armed)
from async_capture.scenario_catalog import COLLECTION_SCENARIOS, TUNNEL_STATIC_SCENARIO
from async_capture.tunnel_normal_collection import finalize_normal_collection
from async_capture.traffic_light_collection import (collection_evidence as traffic_light_collection_evidence,
    finalize_traffic_light_collection, PERMISSION_POLICY)
from async_capture.rosbag_recording import ScenarioRosbag
from async_capture.route_signal_contract import build_route_signal_contract
from native_capture.ground_truth import default_grpc_target


PREPARED_NATIVE_ROLES = {
    'CAMERA_1', 'CAMERA_2', 'CAMERA_3', 'CAMERA_4',
    'LIDAR_5', 'GPS_6', 'IMU_7',
}


def require_prepared_output_root(output):
    """Accept only the empty native root prepared by the capture owner."""
    output = Path(output)
    if not output.is_dir() or output.is_symlink():
        raise ValueError('Prepared output root must be a real directory')
    expected = PREPARED_NATIVE_ROLES | {'SESSION.json', 'supervisor.log'}
    observed = {path.name for path in output.iterdir()}
    if observed != expected:
        raise ValueError(
            'Prepared output root has unexpected entries: '
            +repr(sorted(observed ^ expected)))
    session_path = output/'SESSION.json'
    log_path = output/'supervisor.log'
    if (not session_path.is_file() or session_path.is_symlink()
            or not log_path.is_file() or log_path.is_symlink()):
        raise ValueError('Prepared output ownership files are invalid')
    try:
        session = json.loads(session_path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError('Prepared output SESSION.json is invalid') from exc
    plan = session.get('plan') if isinstance(session, dict) else None
    if (not isinstance(session, dict)
            or session.get('applied') is not True or not isinstance(plan, dict)
            or plan.get('session') != output.name):
        raise ValueError('Prepared output SESSION.json ownership mismatch')
    for role in PREPARED_NATIVE_ROLES:
        directory = output/role
        if (not directory.is_dir() or directory.is_symlink()
                or any(directory.iterdir())):
            raise ValueError(
                'Prepared native role must be an empty real directory: '+role)


def software_inventory(expert_root):
    files = [
        Path(__file__).resolve(),
        ROOT / 'scenario_collection.py',
        ROOT / 'async_capture/traffic_light_collection.py',
        ROOT / 'tools/collect_async_raw_proven.py',
        ROOT / 'async_capture/rosbag_recording.py',
        ROOT / 'tools/exec_owned_rosbag.py',
        ROOT / 'async_capture/bundle_screen.py',
        ROOT / 'async_capture/temporal_groups.py',
    ]
    files += sorted((PROJECT_ROOT / 'scenario_runner').rglob('*.py'))
    files += sorted((PROJECT_ROOT / 'scenario_runner/config').rglob('*.yaml'))
    files += sorted(
        path for path in Path(expert_root).rglob('*')
        if path.is_file() and path.suffix.lower() in {
            '.py', '.json', '.yaml', '.yml',
        }
    )
    inventory = {}
    for source in files:
        source = source.resolve()
        if ROOT == source or ROOT in source.parents:
            name = source.relative_to(ROOT).as_posix()
        else:
            name = 'project/' + source.relative_to(PROJECT_ROOT).as_posix()
        inventory[name] = digest(source)
    return inventory


def _git_value(*arguments):
    process = subprocess.run(
        ['git', '-C', str(PROJECT_ROOT), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def expert_provenance(mode, expert_root):
    expert_root = Path(expert_root).resolve()
    files = sorted(
        path for path in expert_root.rglob('*')
        if path.is_file() and path.suffix.lower() in {
            '.py', '.json', '.yaml', '.yml',
        }
    )
    hashes = {
        path.relative_to(expert_root).as_posix(): digest(path)
        for path in files
    }
    tree_sha256 = hashlib.sha256(
        json.dumps(
            hashes, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    base_config = expert_root / 'src/config/config.json'
    status = _git_value('status', '--porcelain', '--', '../AIM2026/Expert')
    return dict(
        mode=mode,
        root=str(expert_root),
        git_head=_git_value('rev-parse', 'HEAD'),
        worktree_dirty=bool(status),
        source_tree_sha256=tree_sha256,
        source_file_count=len(hashes),
        base_config_sha256=digest(base_config),
    )


def require_expert_root(mode):
    if mode != 'current':
        raise ValueError('Only current is supported; the proven runtime was removed')
    root = (PROJECT_ROOT.parent / 'AIM2026' / 'Expert').resolve()
    marker = root / 'src/config/config.json'
    if not marker.is_file():
        raise RuntimeError(
            'Selected external Expert is incomplete: ' + str(marker)
        )
    return root


class SimulationTimeClient(MoraiGrpcClient):
    def start_world(self, ego_transform):
        start_simulation_time_world(self, ego_transform)


class DiagnosticWindowComplete(Exception):
    pass


class InventoryObservedActor:
    """Diagnostic read adapter; all actor control stays on the original handle.

    This MORAI build raised in GetActorState for a pedestrian whose fresh
    GetAllActorsState entry was present. Never substitute a nearby actor or a
    cached state. This adapter changes no recorded sensor/teacher payload.
    """
    def __init__(self, actor, read_states, event, *, attempts=6, retry_seconds=0.05):
        self.actor, self.read_states, self.event = actor, read_states, event
        self.attempts = max(1, int(attempts))
        self.retry_seconds = max(0.0, float(retry_seconds))

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def get_actor_state(self):
        identity = self.actor.get_object_info()
        started = time.monotonic_ns()
        matches = []
        observations = []
        for attempt in range(1, self.attempts + 1):
            states = list(self.read_states())
            matches = [s for s in states
                       if str(s.actor_info.id.value) == str(identity.id.value)
                       and s.actor_info.object_type == identity.object_type]
            observations.append(dict(attempt=attempt, inventory_size=len(states),
                                     exact_matches=len(matches)))
            if len(matches) == 1:
                break
            if attempt < self.attempts:
                time.sleep(self.retry_seconds)
        received = time.monotonic_ns()
        if len(matches) != 1:
            raise RuntimeError('Expected one fresh inventory actor %s after %d reads; found %d; observations=%s' %
                               (identity.id.value, self.attempts, len(matches), observations))
        state = matches[0]
        self.event('scenario_actor_inventory_observation', actor_id=identity.id.value,
                   object_type=identity.object_type, query_start_monotonic_ns=started,
                   query_end_monotonic_ns=received,
                   query_attempts=len(observations), observations=observations,
                   position=dict(x=state.transform.location.x, y=state.transform.location.y,
                                 z=state.transform.location.z),
                   source='GetAllActorsState', training_visible=False)
        return state


def control_topology(state, topic):
    code, message, (publishers, subscribers, services) = state
    if code != 1:
        raise RuntimeError(message)
    existing = dict(publishers).get(topic, [])
    listeners = dict(subscribers).get(topic, [])
    if existing or not listeners:
        raise RuntimeError('Control must have no other publisher and a simulator subscriber: ' +
                           str(dict(topic=topic, publishers=existing, subscribers=listeners)))
    return dict(topic=topic, publishers=existing, subscribers=listeners)


def raw_stop_record(outcome):
    # Raw byte completion and expert success are separate axes. Collision data
    # is useful diagnostic evidence, never an approved teacher episode.
    return dict(status='completed', purpose='diagnostic_raw_only',
                episode_outcome=outcome, training_visible=False,
                receive_monotonic_ns=time.monotonic_ns())


def summarize_episode(episode):
    episode = Path(episode)
    timing = json.loads((episode/'quality/timing_report.json').read_text())
    raw = json.loads((episode/'raw/raw_manifest.json').read_text())
    index = json.loads((episode/'quality/INDEX_COMPLETE.json').read_text())
    events = [json.loads(line) for line in (episode/'events.jsonl').read_text().splitlines()]
    progress = [e for e in events if e['event'] == 'progress']
    return dict(raw_records=index['records'], raw_bytes=index['bytes_verified'],
        recorder_statistics=raw['statistics'], duration_seconds=timing['active_duration_s'],
        source_clock_diagnostics=timing['streams'], clock_findings=timing['clock_findings'],
        moving_pose_holds=timing['moving_pose_holds'], associations=timing['associations'],
        route_progress_m=max((e['route_s'] for e in progress), default=0),
        max_cross_track_error_m=max((e['cross_track_error_m'] for e in progress), default=None),
        collision_events=[e for e in events if e['event']=='collision'],
        outcome=json.loads((episode/'outcome.json').read_text()),
        physical_simultaneity_verified=False, training_visible=False)


def audit_episode(episode, python, policy_study=None, time_mode='realtime', require_ros_clock=False,
                  clock_trace=None, clock_trace_owner=None, scope='full'):
    episode = Path(episode)
    analysis_env = build_analysis_environment(
        python, os.environ,
        OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
        PYTHONFAULTHANDLER='1',
    )
    analysis_environment = probe_analysis_environment(python, analysis_env)
    steps = [('quality', 'audit_local_alignment.py', ['--raw-run', str(episode/'raw'),
              '--output', str(episode/'quality')])]
    if require_ros_clock:
        steps.append(('clock','audit_ros_clock_alignment.py',['--quality-dir',str(episode/'quality'),
                      '--output',str(episode/'clock_review')]))
        if clock_trace is not None:
            args=['--quality-dir',str(episode/'quality'),'--clock-trace',str(clock_trace),
                  '--output',str(episode/'clock_provenance')]
            if clock_trace_owner is not None:args+=['--trace-owner',str(clock_trace_owner)]
            steps.append(('clock_provenance','audit_clock_provenance.py',args))
    if scope=='timing':
        pass
    elif time_mode=='simulation_free_run':
        steps.append(('groups','build_temporal_groups.py',['--raw-run',str(episode/'raw'),
                      '--output',str(episode/'temporal_groups')]))
        steps.append(('motion','compare_fixed_sensor_motion.py',['--quality-dir',str(episode/'quality'),
                      '--pairing-dir',str(episode/'temporal_groups'),'--pairing-kind','internal10',
                      '--lidar-estimator','multistart','--output',str(episode/'motion_review')]))
        steps.append(('witness','audit_lidar_rotation_witnesses.py',['--quality-dir',str(episode/'quality'),
                      '--geometry-dir',str(episode/'motion_review'),'--clock','source_timestamp_ns',
                      '--output',str(episode/'motion_witnesses')]))
        steps.append(('phase','audit_internal_motion_phase.py',['--quality-dir',str(episode/'quality'),
                      '--motion-dir',str(episode/'motion_review'),'--dense-groups',str(episode/'temporal_groups'),
                      '--output',str(episode/'phase_review')]))
        steps.append(('exposure_content','audit_exposure_observability.py',['--episode',str(episode),
                      '--edges-only','--edge-stride','1','--output',str(episode/'exposure_review')]))
        steps.append(('bundle_screen','screen_temporal_bundles.py',['--episode',str(episode),
                      '--output',str(episode/'bundle_screen')]))
    else:
        steps.append(('delivery', 'audit_realtime_delivery.py', ['--quality-dir', str(episode/'quality'),
                      '--output', str(episode/'delivery_review')]))
    if policy_study is not None:
        steps.append(('pairing', 'study_realtime_motion.py', ['--quality-dir', str(episode/'quality'),
            '--policy-study', str(policy_study), '--output', str(episode/'pairing_review')]))
    results = []
    for stage, tool, args in steps:
        with (episode/(stage+'_audit.log')).open('x') as log:
            proc = subprocess.run(
                [python, '-X', 'faulthandler', str(ROOT/'tools'/tool)] + args,
                cwd=ROOT, env=analysis_env,
                stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        results.append(dict(stage=stage, exit_code=proc.returncode))
        if proc.returncode:
            return dict(status='failed', stages=results,
                        analysis_environment=analysis_environment)
    write_json(episode/'comparison_summary.json', summarize_episode(episode))
    return dict(status='diagnostic_complete', stages=results,
                analysis_environment=analysis_environment)


# A stalled or off-route expert must end the capture instead of recording until
# the outer wall bound (static36 lap 35 stood still for ~985 simulation seconds
# and produced 10,013 bundles). The scenario clock is simulation time in
# simulation_free_run; the longest observed red-light wait was 30 s.
COLLECTION_PROGRESS_WATCHDOG = dict(enabled=True, timeout_sec=180.0,
                                    startup_grace_sec=30.0, minimum_progress_m=2.0)
COLLECTION_OFF_ROUTE_TIMEOUT_SEC = 10.0


class RawScenario(FullLoopScenario):
    def __init__(self, *args, output, contract_path, seconds, collision_policy='abort', time_mode='realtime', expert_clock='host_wall_ros_rate', full_lap=False, record_rosbag=False, sensor_record_coordination=None, **kwargs):
        if collision_policy not in ('abort', 'record'):
            raise ValueError('Unknown collision policy')
        super().__init__(*args, **kwargs)
        # 신호 구간의 45초 watchdog·3초 이탈 계약은 Runner 설정을 그대로 유지한다.
        if self.cfg.get('mission_mode') != 'full_loop_traffic_light':
            self.cfg['progress_watchdog'] = dict(COLLECTION_PROGRESS_WATCHDOG)
            self.cfg['off_route_timeout_sec'] = COLLECTION_OFF_ROUTE_TIMEOUT_SEC
        self.output = Path(output).resolve()
        self.output.mkdir(exist_ok=False)
        self.contract_path = Path(contract_path).resolve()
        self.seconds = seconds
        self.full_lap = full_lap
        self.record_rosbag = record_rosbag
        self.sensor_record_coordination=(Path(sensor_record_coordination).resolve()
                                         if sensor_record_coordination else None)
        self.rosbag = None
        self.time_mode = time_mode
        self.expert_clock = expert_clock
        self.drive_start_wall = None
        self.collector = None
        self.capture_log = None
        self.cleaned = False
        self.event_count = 0
        self.last_progress = float('-inf')
        self.last_mode_check = float('-inf')
        self.collision_policy = collision_policy
        self.contact_ids = ()
        self.collision_events = 0
        self.collision_samples = 0
        self.first_collision_receive_ns = None
        self.outcome = dict(status='failed', reason='setup_incomplete', training_visible=False)

    def event(self, event, **fields):
        row = dict(event=event, sequence=self.event_count, wall_ns=time.time_ns(),
                   receive_monotonic_ns=time.monotonic_ns(), **fields)
        self.event_count += 1
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        if event != 'progress':
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def setup_expert_controller(self):
        if self.collector is not None:
            raise RuntimeError('No reset/retry inside an existing raw episode')
        super().setup_expert_controller()
        runtime_path = Path(self.expert_controller.runtime_config_path)
        runtime = json.loads(runtime_path.read_text())
        maximum = float(runtime['planning']['velocity_profile']['max_velocity'])
        if self.cfg.get('mission_mode') == 'full_loop_traffic_light':
            # Runner가 검증한 실제 설정 바이트를 그대로 보존한다.
            if maximum != 30. or any(float(zone['max_velocity']) > maximum for zone in
                                    runtime['planning']['velocity_profile'].get('speed_limit_zones', [])):
                raise ValueError('좌회전 신호 Expert 속도 계약이 수집 전에 변경됐습니다')
        else:
            for zone in runtime['planning']['velocity_profile'].get('speed_limit_zones', []):
                zone['max_velocity'] = min(float(zone['max_velocity']), maximum)
            runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
        expert = dict(self.global_cfg.get('expert_provenance', {}))
        setup = dict(scenario_name=self.scenario_name,
                     expert_mode=expert.get('mode'),
                     expert_root=self.expert_controller.expert_root,
                     expert=expert,
                     seed=self.run_seed, profile=self.profile, time_hour=self.scenario_hour,
                     global_config=self.global_cfg, scenario_config=self.cfg,
                     route_length_m=self.route_length_m, scenario_clock=self.clock.domain,
                     global_path_sha256=digest(Path(self.cfg['global_path'])),
                     expert_inputs={str(Path(p).relative_to(self.output)): digest(Path(p)) for p in
                                    (self.expert_controller.runtime_config_path,
                                     self.expert_controller.runtime_path_csv)},
                     collision_policy=self.collision_policy, time_mode=self.time_mode,
                     expert_clock=self.expert_clock, duration_clock='host_monotonic',
                     static_obstacle_preset=(
                         dict(
                             path=str(self.cfg['static_obstacle_preset']),
                             sha256=digest(Path(self.cfg['static_obstacle_preset'])),
                         )
                         if self.cfg.get('static_obstacle_preset') else None
                     ),
                     tunnel_mission=tunnel_collection_evidence(self),
                     training_visible=False, object_geometry_verified=False)
        if self.cfg.get('mission_mode') == 'full_loop_traffic_light':
            setup['traffic_light_mission'] = traffic_light_collection_evidence(self)
        if self.cfg.get('_tunnel_static_collection'):
            trial = getattr(self,'tunnel_static_trial_context',None)
            if trial is None:
                raise RuntimeError('Tunnel collection requires its measured trial before raw capture')
            setup['tunnel_static_trial'] = dict(
                directory=str(trial.output_dir),
                drive_report_path=str(trial.output_dir/'drive_trial.json'),
                preview_report_path=str(trial.output_dir/'preview.json'),
                planner_diagnostics_path=str(trial.output_dir/'planner_diagnostics.json'),
                selected_candidate_sha256=self.cfg.get('_tunnel_static_candidate_sha256'),
                runtime_source_sha256=dict(trial.runtime_sources),
                road_boundary_policy_sha256=trial.boundary.policy_sha256,
                finalized=False,
            )
        write_json(self.output / 'episode.json', setup)
        self.capture_log = (self.output / 'capture.log').open('x')
        command = ['/usr/bin/python3', '-u', str(ROOT / 'tools/collect_async_raw_proven.py'),
                   '--output', str(self.output / 'raw'),
                   '--seconds', str(self.seconds + (120 if self.cfg.get('_tunnel_static_collection') else 60)),
                   '--max-gib', '15', '--sensor-contract', str(self.contract_path),
                   '--ready-file', str(self.output / 'ready.json'),
                   '--stop-file', str(self.output / 'stop.json'),
                   '--episode-metadata', str(self.output / 'episode.json')]
        command.extend(['--time-mode',self.time_mode])
        if self.record_rosbag:
            command.extend(['--sensor-ready-timeout','600'])
        self.collector = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                         stdout=self.capture_log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.event('waiting_for_raw_sensors', pid=self.collector.pid)
        deadline = time.monotonic() + (620 if self.record_rosbag else 50)
        while not (self.output / 'ready.json').exists():
            if self.collector.poll() is not None:
                raise RuntimeError('Raw recorder exited before sensor readiness')
            if time.monotonic() > deadline:
                raise TimeoutError('Raw sensor readiness timeout')
            time.sleep(.1)
        if hasattr(self, 'clock_guard'):
            self.event('ros_clock_ready_before_expert', **self.clock_guard.arm())
        self.event('raw_ready_before_expert')
        if self.sensor_record_coordination is not None:
            self.event('waiting_for_native_recording_before_expert')
            armed = wait_for_native_capture_armed(self.sensor_record_coordination,self.collector,timeout=65.)
            self.event('native_recording_ready_before_expert',**armed)
        if self.record_rosbag:
            self.rosbag = ScenarioRosbag(self.output/'rosbag', self.event)
            self.rosbag.start(timeout=600)

    def _spawn_active_dynamic_obstacle(self, *args, **kwargs):
        super()._spawn_active_dynamic_obstacle(*args, **kwargs)
        actor = self.dynamic_obstacle
        if actor is not None and self.time_mode == 'simulation_free_run':
            observed = InventoryObservedActor(actor, self.grpc.get_all_pedestrian_states, self.event)
            self.scenario_actors[self.scenario_actors.index(actor)] = observed
            self.dynamic_obstacle = observed
            self.event('scenario_actor_observation_source', actor_id=str(actor.get_id()),
                       source='GetAllActorsState', reason='diagnostic_server_GetActorState_failure')

    def observe_collision(self, elapsed, ego_state, current_s):
        ids = tuple(sorted(set(str(x) for x in ego_state.get('collision_objects', []))))
        if ids:
            self.collision_samples += 1
            if self.first_collision_receive_ns is None:
                self.first_collision_receive_ns = time.monotonic_ns()
        # Keep raw CollisionData/Ego unchanged; compact only supervisor events.
        if ids != self.contact_ids:
            if ids:
                self.collision_events += 1
                self.event('collision', ego=ego_state, elapsed_seconds=elapsed, route_s=current_s,
                           policy=self.collision_policy, previous_contact_ids=list(self.contact_ids),
                           continued=self.collision_policy == 'record')
            elif self.contact_ids:
                self.event('collision_cleared', elapsed_seconds=elapsed, route_s=current_s,
                           previous_contact_ids=list(self.contact_ids),
                           meaning='Supervisor observed empty collision IDs; not a physical contact certificate')
            self.contact_ids = ids
        if ids and self.collision_policy == 'abort':
            raise RuntimeError('Expert collision: ' + str(ego_state['collision_objects']))

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        if self.collector is None or self.collector.poll() is not None:
            raise RuntimeError('Raw recorder stopped during expert drive')
        if self.rosbag is not None:
            self.rosbag.check()
        self.observe_collision(elapsed, ego_state, current_s)
        now = time.monotonic()
        if now - self.last_mode_check >= 5:
            self.mode_check()
            if hasattr(self, 'clock_guard'): self.clock_guard.check()
            require_ssd(self.output)
            self.last_mode_check = now
        # Preserve ordinary mission events; a bounded window is not a full lap.
        super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        if now - self.last_progress >= .5:
            _, cte = self.project_on_route_near_progress(ego_state['x'], ego_state['y'], current_s)
            self.event('progress', elapsed_seconds=elapsed, ego=ego_state, route_s=current_s,
                       remaining_s=remaining_s, cross_track_error_m=cte,
                       dynamic_phase=self.dynamic_obstacle_phase, tunnel_phase=collection_tunnel_phase(self),
                       highway_active=self.highway_spawn_points_active,
                       roundabout_phase=getattr(self, 'nearby_roundabout_state', 'unmanaged'),
                       traffic_wait_light=(self.traffic_wait_state['stop_key'][0]
                                           if self.traffic_wait_state else None),
                       traffic_override_attempts=self.traffic_override_attempts,
                       traffic_override_verified=self.traffic_override_verified,
                       actor_ids=[str(a.get_id()) for a in self.scenario_actors])
            self.last_progress = now
        if (time.monotonic()-self.drive_start_wall if self.drive_start_wall is not None else elapsed) >= self.seconds:
            raise DiagnosticWindowComplete()

    def run(self):
        self.event('setup_started')
        try:
            self.setup()
            self.drive_start_wall = time.monotonic()
            self.event('expert_started', pid=self.expert_controller.process.pid)
            self.run_timeline()
            self.outcome = dict(status=('diagnostic_lap_completed_with_collisions'
                                       if self.collision_events else 'mission_completed'),
                                reason='runner_goal_reached', full_mission_completed=True,
                                training_visible=False)
        except DiagnosticWindowComplete:
            self.outcome = dict(status=('diagnostic_lap_incomplete' if self.full_lap else 'diagnostic_window_completed'), reason='bounded_window_elapsed',
                                full_mission_completed=False, training_visible=False)
        except BaseException as error:
            self.outcome = dict(status='failed', reason=type(error).__name__ + ': ' + str(error),
                                training_visible=False)
        finally:
            self.cleanup()
        return self.outcome

    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True
        self.outcome.update(collision_policy=self.collision_policy,
            collision_summary=dict(contact_events=self.collision_events,
                nonempty_contact_observations=self.collision_samples,
                first_collision_receive_monotonic_ns=self.first_collision_receive_ns,
                last_observed_contact_ids=list(self.contact_ids)),
            teacher_trajectory_approved=False, training_visible=False)
        errors = []
        warnings = []
        for name, action in [('stop_expert', self.stop_expert_controller),
                             ('brake_ego', self.grpc.stop_ego_control)]:
            try:
                action()
            except Exception as error:
                errors.append(name + ': ' + str(error))
        if hasattr(self,'final_brake'):
            try:self.event('final_brake_observed',**self.final_brake())
            except Exception as error:
                finding='final_brake: '+str(error)
                if self.outcome.get('full_mission_completed') is True:
                    warnings.append(finding)
                else:
                    errors.append(finding)
        # When Sensor Record ends early at the route goal, keep the raw ROS/GT
        # recorder alive until the UI owner confirms Record OFF.  This makes
        # the native tail observable instead of producing unmatched files.
        coordination=self.sensor_record_coordination
        if coordination is not None and (coordination/'recording_armed.json').is_file():
            try:
                write_json(coordination/'stop.request',dict(reason='scenario_capture_window_finished',
                    request_monotonic_ns=time.monotonic_ns(),training_visible=False))
                deadline=time.monotonic()+20
                while not (coordination/'recording_stopped.json').is_file():
                    if time.monotonic()>deadline:raise TimeoutError('Sensor Record OFF acknowledgement timeout')
                    time.sleep(.05)
                stopped=json.loads((coordination/'recording_stopped.json').read_text(encoding='utf-8-sig'))
                if stopped.get('error'):
                    raise RuntimeError('Sensor Record owner reported stop failure: '+str(stopped['error']))
                self.event('sensor_record_stop_acknowledged',coordination=str(coordination),
                           owner_stopwatch_ticks=stopped.get('stopwatch_ticks'))
            except Exception as error:errors.append('sensor_record_coordination: '+str(error))
        if self.rosbag is not None:
            try:
                self.outcome['rosbag'] = self.rosbag.stop()
            except Exception as error:
                errors.append('rosbag_finalize: '+str(error))
        # Finalize the raw journal before deleting any owned scene actors.
        if self.collector is not None:
            try:
                if self.collector.poll() is None:
                    write_json(self.output / 'stop.json', raw_stop_record(self.outcome))
                    try:
                        self.collector.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.collector.pid, signal.SIGINT)
                        self.collector.wait(timeout=10)
                if self.collector.returncode != 0:
                    errors.append('raw_recorder_exit_' + str(self.collector.returncode))
            except Exception as error:
                errors.append('raw_finalize: ' + str(error))
                if self.collector.poll() is None:
                    os.killpg(self.collector.pid, signal.SIGKILL)
                    self.collector.wait(timeout=5)
            finally:
                self.capture_log.close()
        try:
            super().cleanup()
        except Exception as error:
            errors.append('scenario_cleanup: ' + str(error))
        if warnings:
            self.outcome['cleanup_warnings']=warnings
        if errors:
            self.outcome.update(status='failed', cleanup_errors=errors)
        tunnel_evidence = tunnel_collection_evidence(self)
        if tunnel_evidence is not None:
            self.outcome['tunnel_mission'] = tunnel_evidence
        if self.cfg.get('mission_mode') == 'full_loop_traffic_light':
            self.outcome['traffic_light_mission'] = traffic_light_collection_evidence(self)
        self.event('episode_finished', **self.outcome)
        write_json(self.output / 'outcome.json', self.outcome)


class RawMissionScenario(RawScenario, MissionScenario):
    """Collection lifecycle layered on the main segment scenario."""

    def run(self):
        if self.cfg.get('mission_mode') == 'full_loop_traffic_light':
            # cleanup에서 파일 기록이 실패해도 변경한 신호의 복원 시도는 보장한다.
            result, primary_error, restoration_errors = None, None, []
            try:
                result = super().run()
            except BaseException as error:
                primary_error = error
            finally:
                restoration_errors = self._restore_traffic_signal(result, primary_error)
            if primary_error is not None:
                if restoration_errors:
                    raise RuntimeError('수집 오류: ' + str(primary_error) + '; 신호 복원/기록 오류: '
                                       + '; '.join(restoration_errors)) from primary_error
                raise primary_error
            return result
        result = super().run()
        # RawScenario owns run/cleanup in this MRO, so MissionScenario.run's
        # final parking block is not reached. Preserve that baseline check
        # explicitly before sealing the empty-road collection evidence.
        if self.cfg.get('mission_mode') == 'full_loop_tunnel_normal':
            from scenario_runner.scenarios.segment_modes import tunnel_baseline
            try:
                tunnel_baseline.park(self)
            except Exception as error:
                tunnel_baseline.finish(self, 'failed', str(error) or type(error).__name__)
                detail = str(error) or type(error).__name__
                result.update(status='failed', reason='tunnel_normal_parking_failed: ' + detail,
                              tunnel_normal_parking_error=detail)
                self.event('tunnel_normal_parking_failed', error=detail)
            else:
                self.event('tunnel_normal_parking_verified')
        return result

    def _restore_traffic_signal(self, result, primary_error):
        from scenario_runner.scenarios.segment_modes import traffic_light
        errors = []
        report = getattr(self, 'traffic_light_report', {})
        required = report.get('original') is not None
        try:
            traffic_light.restore(self)
            if required and report.get('restore', {}).get('verified') is not True:
                raise RuntimeError('좌회전 신호의 원래 상태 복원이 실제 readback으로 확인되지 않았습니다')
            if not required and isinstance(result, dict) and result.get('status') == 'mission_completed':
                raise RuntimeError('완료된 신호 수집에 원래 신호 상태가 없습니다')
        except BaseException as error:
            errors.append(str(error) or type(error).__name__)
        if primary_error is not None or errors:
            reason = '; '.join(([str(primary_error)] if primary_error is not None else []) + errors)
            try:
                traffic_light.finish(self, 'failed', reason)
            except BaseException as error:
                errors.append('복원 보고서 기록: ' + (str(error) or type(error).__name__))
        if required or errors:
            try:
                self.event('traffic_light_restore_failed' if errors else 'traffic_light_restore_verified',
                           **({'error': '; '.join(errors)} if errors else {}))
            except BaseException as error:
                errors.append('복원 이벤트 기록: ' + (str(error) or type(error).__name__))
        if errors and isinstance(result, dict):
            detail = '; '.join(errors)
            result.update(status='failed', reason='traffic_light_restore_failed: ' + detail,
                          traffic_light_restore_error=detail)
        return errors


def collection_scenario_class(name):
    """Select only a main Scenario Runner class; keep collection as an adapter."""
    base = SUPPORTED_SCENARIOS.get(name)
    if base is FullLoopScenario:
        return RawScenario
    if base is MissionScenario:
        return RawMissionScenario
    raise ValueError("Unsupported collection scenario: " + str(name))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--scenario', choices=COLLECTION_SCENARIOS, default='full_loop',
                   help='Main Scenario Runner mode to execute (default: full_loop)')
    p.add_argument('--static-preset',
                   help='Static editor preset name or YAML path; full_loop_static only')
    p.add_argument(
        '--scenario-weather', choices=('SUNNY', 'FOGGY'),
        help='Force the run weather from a sealed collection plan',
    )
    p.add_argument(
        '--scenario-hour', type=int, choices=(11, 13, 15),
        help='Force the run hour from a sealed collection plan',
    )
    p.add_argument(
        '--expert-runtime', choices=('current',), default='current',
        help='Execute AIM2026/Expert; the frozen proven mode has been removed',
    )
    p.add_argument('--nearby-actors', action='store_true', help='Compatibility option; actor lifecycle is owned by the main scenario runner')
    p.add_argument('--rosbag', action='store_true', help='Record original ROS topics including ReplayInfo before expert launch')
    p.add_argument('--audit-scope', choices=('full','timing'), default='full',
                   help='timing: raw integrity and clock checks; defer expensive image geometry studies')
    p.add_argument('--time-mode',choices=('realtime','simulation_free_run'),default='realtime')
    p.add_argument('--expert-wall-clock',action='store_true',
                   help='Explicit diagnostic wall-time expert scheduling; restore use_sim_time after run')
    p.add_argument('--output', type=Path)
    p.add_argument(
        '--prepared-output-root', action='store_true',
        help=('Use the empty native Sensor Record root prepared by the outer '
              'capture owner; rejected unless its ownership seal is exact'),
    )
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--seconds', type=float, default=90, help='Maximum driving wall seconds; stop earlier at route completion')
    p.add_argument('--full-lap', action='store_true', help='Require selected-scenario route completion; permit a bounded wall budget up to 3540 seconds')
    p.add_argument('--max-speed-kmh', type=float, default=None,
                   help='일반 생략값은 20km/h, 좌회전 신호는 30km/h 고정, 터널 정적은 원래 Expert 속도')
    p.add_argument('--seed', type=int, default=260910)
    p.add_argument('--profile-state',type=Path,
                   help='Optional D-local state shared by separate one-lap processes')
    p.add_argument('--collision-policy', choices=('abort','record'), default='abort',
                   help='record: retain collision and post-collision raw data without collision-triggered abort; diagnostic only')
    p.add_argument('--grpc-target')
    p.add_argument('--control-topic', default='/ctrl_cmd')
    p.add_argument('--audit-python', default=sys.executable)
    p.add_argument('--clock-trace',type=Path,
                   help='Require one observed upstream clock and audit its unchanged sequence after capture')
    p.add_argument('--clock-trace-owner',type=Path,
                   help='Contemporaneous owner evidence for legacy diagnostic trace headers')
    p.add_argument('--policy-study', type=Path,
                   help='Optional frozen policy study; compare after each capture without fitting offsets')
    p.add_argument('--sensor-record-coordination',type=Path,
                   help='Optional D-local stop handshake with the external Sensor Record UI owner')
    p.add_argument('--raw-camera-transport',choices=('required','unrecorded','bridge_dropped','absent'),
                   default='required',
                   help='required: write ROS cameras; unrecorded: omit raw writes; bridge_dropped: discard at bridge ingress; absent: diagnostic only')
    a = p.parse_args()
    if a.scenario == 'full_loop_traffic_light':
        if a.max_speed_kmh not in (None, 30.):
            p.error('좌회전 신호 수집은 --max-speed-kmh 30 고정 계약을 사용합니다')
        a.max_speed_kmh = 30.
    if a.max_speed_kmh is None and a.scenario != TUNNEL_STATIC_SCENARIO:
        a.max_speed_kmh = 20.
    expert_root = require_expert_root(a.expert_runtime)
    os.environ['MORAI_EXPERT_ROOT'] = str(expert_root)
    os.environ['MORAI_EXPERT_MODE'] = a.expert_runtime
    print('[Expert] mode=' + a.expert_runtime, flush=True)
    print('[Expert] root=' + str(expert_root), flush=True)
    if a.static_preset:
        if a.scenario != 'full_loop_static':
            p.error('--static-preset requires --scenario full_loop_static')
        try:
            a.static_preset = resolve_static_preset(a.static_preset)
        except ValueError as error:
            p.error(str(error))
    if (a.scenario_weather is None) != (a.scenario_hour is None):
        p.error('--scenario-weather and --scenario-hour must be provided together')
    if a.time_mode=='simulation_free_run' and a.policy_study:
        p.error('Simulation diagnostic cannot use a RealTime policy study')
    ros_clock_expert = a.time_mode=='simulation_free_run' and not a.expert_wall_clock
    if a.clock_trace is not None:
        if not ros_clock_expert:p.error('Clock provenance requires actual ROS simulation clock')
        a.clock_trace=a.clock_trace.resolve()
        require_ssd(a.clock_trace,reserve_bytes=0)
        if not a.clock_trace.is_file():p.error('Running bridge clock trace missing')
    if a.clock_trace_owner is not None:
        if a.clock_trace is None or not a.clock_trace_owner.is_file():p.error('Invalid clock trace owner evidence')
        a.clock_trace_owner=a.clock_trace_owner.resolve()
    if a.sensor_record_coordination is not None:
        a.sensor_record_coordination=a.sensor_record_coordination.resolve()
        require_ssd(a.sensor_record_coordination,reserve_bytes=0)
    if a.profile_state is not None:
        a.profile_state=a.profile_state.expanduser().resolve()
        require_ssd(a.profile_state.parent,reserve_bytes=0)
        if a.profile_state.is_dir():p.error('--profile-state must name a file')
    expert_clock = 'ros_clock' if ros_clock_expert else 'host_wall_ros_rate'
    max_seconds = 3540 if a.full_lap else 900
    if (not 1 <= a.episodes <= 5 or not 20 <= a.seconds <= max_seconds
            or (a.max_speed_kmh is not None and not 5 <= a.max_speed_kmh <= 60)):
        p.error('Require bounded episodes=1..5, seconds=20..%d, speed=5..60' % max_seconds)
    global_cfg, _, cfg = load_configuration()
    if not a.grpc_target:
        if a.check:
            a.grpc_target = '{}:{}'.format(
                global_cfg['grpc']['host'], global_cfg['grpc']['port'])
        else:
            a.grpc_target = default_grpc_target()
    # The main adapter supports environment discovery; the collection command
    # remains authoritative about the exact live target. Offline --check uses
    # only the checked-in configuration and must not probe a PC network.
    os.environ['MORAI_GRPC_TARGET'] = a.grpc_target
    resolved_expert_root = Path(global_cfg['paths']['expert_root']).resolve()
    if resolved_expert_root != expert_root:
        raise RuntimeError(
            'Scenario Runner resolved a different Expert root: '
            f'expected={expert_root}, actual={resolved_expert_root}'
        )
    global_cfg['grpc']['host'], port = a.grpc_target.rsplit(':', 1)
    global_cfg['grpc']['port'] = int(port)
    cfg.update(random_seed=a.seed,
               expert_max_velocity_kmh=a.max_speed_kmh,
               expert_python_executable='/usr/bin/python3',
               expert_ros_remaps=['/ctrl_cmd:=' + a.control_topic])
    if a.scenario_weather is not None:
        cfg['forced_weather'] = a.scenario_weather
        cfg['forced_time_hour'] = a.scenario_hour
    if a.scenario in MISSION_MODES:
        cfg['mission_mode'] = a.scenario
    configure_collection_scenario(cfg, a.scenario)
    if a.static_preset:
        cfg['static_obstacle_preset'] = a.static_preset
    scenario_type = collection_scenario_class(a.scenario)
    provenance = expert_provenance(a.expert_runtime, expert_root)
    global_cfg['expert_provenance'] = provenance
    # Scenario policy is loaded only from the main scenario_runner. This
    # collector owns capture, collision recording and the outer wall bound.
    verify_package(SCENARIO_PACKAGE_ROOT, global_cfg, cfg, a.scenario)
    if a.check:
        print(
            'Offline scenario configuration check passed; no simulator calls; '
            f'expert_mode={a.expert_runtime}; expert_root={expert_root}'
        )
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt('supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    from native_capture.ground_truth import GrpcGroundTruth
    observer = GrpcGroundTruth(a.grpc_target, timeout=2)
    def mode_check():
        return require_free_running_mode(observer.call(observer.simulation.GetSynchronousMode, observer.Empty()), a.time_mode)
    output = a.output or SSD_ROOT / 'runs' / datetime.datetime.now().strftime('scenario_review_%Y%m%d_%H%M%S')
    output = output.resolve()
    restore_parameter = None
    clock_guard = None
    try:
        with process_lock('campaign'):
            # Check collector ownership before any world/vehicle mutation.
            with process_lock('capture'):
                mode = mode_check()
                master = xmlrpc.client.ServerProxy(os.environ.get('ROS_MASTER_URI', 'http://127.0.0.1:11311'))
                removed = cleanup_unreachable_expert_publishers(a.control_topic)
                if removed:
                    print('[Control] removed unreachable expert registrations: ' +
                          ', '.join(removed), flush=True)
                topology_state = wait_for_control_release(
                    lambda: master.getSystemState('/scenario_review_preflight'),
                    a.control_topic, timeout=10.0)
                topology = control_topology(topology_state, a.control_topic)
                existed=master.hasParam('/scenario_review_preflight','/use_sim_time')[2]
                original=master.getParam('/scenario_review_preflight','/use_sim_time')[2] if existed else None
                if ros_clock_expert and original is not True:
                    raise ValueError('Simulation expert requires use_sim_time=true and actual /clock')
                if original and not a.expert_wall_clock and not ros_clock_expert:
                    raise ValueError('RealTime expert requires use_sim_time=false')
                storage = require_ssd(output, check_volume=True, reserve_bytes=int(float(os.environ.get('MORAI_STORAGE_RESERVE_GIB', '70')) * 1024**3))
                if a.policy_study is not None:
                    a.policy_study = a.policy_study.resolve()
                    require_ssd(a.policy_study, reserve_bytes=0)
                    if not (a.policy_study/'report.json').is_file():
                        raise ValueError('Frozen policy study report missing')
                if a.prepared_output_root:
                    if a.output is None:
                        p.error('--prepared-output-root requires --output')
                    require_prepared_output_root(output)
                else:
                    output.mkdir(parents=True, exist_ok=False)
                if ros_clock_expert:
                    os.environ.update(ROS_LOG_DIR=str(output/'ros_logs'), ROS_HOME=str(output/'ros_home'))
                    clock_guard=RosClockGuard()
                    write_json(output/'clock_preflight.json', clock_guard.require_progress())
                if original and a.expert_wall_clock:
                    restore_parameter=(master,original)
                    if master.setParam('/scenario_review_preflight','/use_sim_time',False)[0]!=1:
                        raise RuntimeError('Cannot configure explicit wall-clock expert')
            os.environ.update(ROS_LOG_DIR=str(output / 'ros_logs'), ROS_HOME=str(output / 'ros_home'),
                              MORAI_REVIEW_OWNER_PID=str(os.getpid()))
            plan = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                mode=mode, topology=topology, storage=storage, training_visible=False, nas_used=False,
                original_use_sim_time=original, expert_clock=expert_clock,
                note='Same seed/profile inputs per episode; explicit speed cap and wall budget; only runner goal confirms a full lap',
                software=software_inventory(expert_root),
                expert=provenance)
            write_json(output / 'plan.json', plan)
            contract = json.loads((ROOT/'configs/encoder_mission_candidate_v1.json').read_text())
            contract['capture']['actor_ground_truth']['target'] = a.grpc_target
            contract['capture']['expert_control_topic'] = a.control_topic
            contract['capture']['record_expert_paths'] = True
            contract['capture']['require_ros_clock'] = ros_clock_expert
            contract['capture']['require_single_clock_source'] = a.clock_trace is not None
            contract['capture']['record_ros_cameras'] = a.raw_camera_transport=='required'
            contract['capture']['ros_camera_transport_policy'] = a.raw_camera_transport
            contract['capture']['traffic_light_ground_truth'] = build_route_signal_contract(
                global_cfg['paths']['mgeo_root'],cfg['route_definition_path'],
                cfg.get('route_definition_key','full_loop'))
            if a.scenario == 'full_loop_traffic_light':
                # 기존 색 분류 정답은 유지하고 행동 허용 근거만 별도로 기록한다.
                contract['traffic_light_permission_target'] = copy.deepcopy(PERMISSION_POLICY)
            contract_path = output / 'sensor_contract.json'
            write_json(contract_path, contract)
            results = []
            for index in range(a.episodes):
                mode_check()
                removed = cleanup_unreachable_expert_publishers(a.control_topic)
                if removed:
                    print('[Control] removed unreachable expert registrations: ' +
                          ', '.join(removed), flush=True)
                control_topology(wait_for_control_release(
                    lambda: master.getSystemState('/scenario_review_preflight'),
                    a.control_topic, timeout=10.0), a.control_topic)
                episode = output / ('episode_%03d' % index)
                gc, sc = copy.deepcopy(global_cfg), copy.deepcopy(cfg)
                gc['grpc']['client_key'] = output.name + '_' + str(index)
                gc['paths']['runtime_root'] = str(episode / 'runtime')
                sc['profile_state_path'] = str(a.profile_state or (episode / 'profile_state.json'))
                sc['route_debug_dir'] = str(episode / 'route_debug')
                client = (SimulationTimeClient if a.time_mode=='simulation_free_run' else MoraiGrpcClient)(gc)
                scenario = None
                try:
                    client.connect()
                    loader = MGeoMapLoader(gc['paths']['mgeo_root'])
                    options = dict(output=episode, contract_path=contract_path, seconds=a.seconds,
                        collision_policy=a.collision_policy,time_mode=a.time_mode,expert_clock=expert_clock,full_lap=a.full_lap,
                        record_rosbag=a.rosbag,sensor_record_coordination=a.sensor_record_coordination,
                        clock=SimulationObservationClock(observer) if a.time_mode=='simulation_free_run' else None)
                    def configure_capture(scenario_instance):
                        if clock_guard is not None:
                            clock_guard.armed=False
                            clock_guard.reset_seen=False
                            scenario_instance.clock_guard=clock_guard
                        scenario_instance.mode_check = mode_check
                        scenario_instance.final_brake = lambda: publish_final_brake(a.control_topic,observer)
                    if a.scenario == TUNNEL_STATIC_SCENARIO:
                        from scenario_runner.scenarios.full_loop_parts.tunnel_static import run_collection_episode
                        result = run_collection_episode(client,loader,gc,sc,
                            scenario_factory=scenario_type,scenario_kwargs=options,
                            configure_scenario=configure_capture,
                            output_dir=output/'tunnel_static_trials'/episode.name,
                            seed=a.seed,history_path=(a.profile_state.parent if a.profile_state else output)
                            /'tunnel_static_selection_history.json')
                    else:
                        scenario = scenario_type(client,loader,gc,sc,**options)
                        configure_capture(scenario)
                        result = scenario.run()
                        # RawMissionScenario includes the normal baseline's
                        # final parking check before writing collection proof.
                        finalize_normal_collection(scenario, result)
                        finalize_traffic_light_collection(scenario, result)
                finally:
                    client.stop()
                raw = episode / 'raw'
                audit = dict(status='not_run_raw_unsealed')
                seal = raw / 'RAW_COMPLETE.json'
                if seal.exists() and json.loads(seal.read_text()).get('status') == 'completed':
                    audit = audit_episode(episode, a.audit_python, a.policy_study,a.time_mode,
                        require_ros_clock=ros_clock_expert,clock_trace=a.clock_trace,
                        clock_trace_owner=a.clock_trace_owner,scope=a.audit_scope)
                if a.rosbag and result.get('rosbag',{}).get('status')=='recorded':
                    with (episode/'rosbag_audit.log').open('x') as log:
                        bag_audit=subprocess.run([sys.executable,str(ROOT/'tools/audit_scenario_rosbag.py'),
                            '--directory',str(episode/'rosbag')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    audit['rosbag_audit_exit_code']=bag_audit.returncode
                    if bag_audit.returncode:
                        audit['status']='failed'
                results.append(dict(episode=str(episode), outcome=result, audit=audit))
                write_json(output / ('episode_%03d_result.json' % index), results[-1])
                if result['status'] in ('failed', 'diagnostic_lap_incomplete') or audit['status'] != 'diagnostic_complete':
                    break
            write_json(output / 'REVIEW_COMPLETE.json', dict(status='bounded_review_finished', episodes=results,
                requested_episodes=a.episodes, training_visible=False, physical_simultaneity_verified=False))
            if any(r['outcome']['status'] in ('failed', 'diagnostic_lap_incomplete') or r['audit']['status']!='diagnostic_complete' for r in results):
                raise SystemExit(2)
    finally:
        if restore_parameter is not None:
            master,original=restore_parameter
            master.setParam('/scenario_review_preflight','/use_sim_time',original)
        if clock_guard is not None: clock_guard.close()
        observer.close()


if __name__ == '__main__':
    main()
