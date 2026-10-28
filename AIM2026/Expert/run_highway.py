#!/usr/bin/env python3
"""Temporary scenario harness: existing highway scene -> AIM2026/main.py.

Only scene lifecycle and the scenario route live here. Driving always uses
the same main entry, configuration, inference and Expert as ordinary operation.
"""
import argparse
import copy
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'AIM2026/utils')]

from scenario_runner.project_paths import activate_project_paths
activate_project_paths()
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.segment_modes import events
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.adapters.expert_controller import validate_max_velocity_kmh
from AIM2026.main import runtime_python
from AIM2026.Expert import wsl_network
from scenario_runner.core.contact_verification import SpawnContactVerifier

# Validation runs, incidents and the batch scratch of the runner (git-ignored), relative to the repository root.
RUNNER_RUNTIME = 'AIM2026/utils/scenario_runner/.runtime'


def expert_python():
    return str(runtime_python())


class StartupContactBaseline:
    """Journal pre-drive self/anonymous records and verified removed NPCs.

    This is an explicit monitoring exception, not proof of collision-free driving.
    Once absent, the label loses its exemption for the rest of this run.
    """
    def __init__(self, state, live_actor_ids=None):
        initial = {str(value).strip() for value in state.get('collision_objects', [])
                   if str(value).strip()}
        self.initial = initial.copy()
        self.initial_live_actor_ids = (sorted(str(v) for v in live_actor_ids)
                                       if live_actor_ids is not None else None)
        self.seen = initial.copy()
        self.previous = initial.copy()
        numeric = {value for value in initial if value.isdecimal()}
        absent = (numeric - set(live_actor_ids)) if live_actor_ids is not None else set()
        allowed = {'Ego', '-100'} | absent
        self.ignored = initial if 'Ego' in initial and initial <= allowed else set()
        self.removed_actor_ids = self.ignored & absent
        self.active = bool(self.ignored)

    def apply(self, state, live_actor_ids=None):
        state = dict(state)
        raw = list(dict.fromkeys(str(value).strip() for value in state.get('collision_objects', [])
                                 if str(value).strip()))
        previously_ignored = self.ignored.copy()
        # Each exemption can only disappear, never grow after the baseline.
        # A cleared label that later reappears is a NEW contact, including -100.
        self.ignored.intersection_update(raw)
        # Reused actor IDs must not hide a new contact. If the live actor query
        # fails, numeric exemptions are revoked instead of assuming an empty scene.
        if self.removed_actor_ids & self.ignored:
            self.ignored.difference_update(self.removed_actor_ids if live_actor_ids is None
                                           else self.removed_actor_ids & set(live_actor_ids))
        self.active = bool(self.ignored)
        state['raw_collision_objects'] = raw
        state['startup_contact_baseline_active'] = self.active
        state['startup_contact_ignored'] = sorted(self.ignored)
        state['collision_objects'] = [item for item in raw if item not in self.ignored]
        state['contact_classification'] = {
            item: ('startup_residual' if item in self.ignored else
                   'startup_exemption_revoked' if item in previously_ignored else
                   'new_report' if item not in self.seen else
                   'reappeared_report' if item not in self.previous else
                   'preexisting_unverified' if item in self.initial else 'continuing_report')
            for item in raw
        }
        self.seen.update(raw)
        self.previous = set(raw)
        return state


class ExpertController:
    """Own one UDP Expert; the scene runner never supplies driving observations."""
    def __init__(self, base, scenario):
        self.base = base
        self.scenario = scenario
        self.process = self.log = None
        self.stop_file = scenario.output / 'stop_expert'
        self.context_path = scenario.output / 'scenario_context.json'
        self.windows_child = os.name != 'nt' and scenario.args.expert_python.lower().endswith('.exe')
        self._paths = {}
        document = dict(schema='ve_gps_scenario_context_v1', global_config=scenario.global_cfg,
                        scenario_config=scenario.cfg, route_points=[list(p) for p in scenario.route_points])
        self.context_path.write_text(json.dumps(self.portable(document), ensure_ascii=False), encoding='utf-8')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def child_path(self, path):
        path = str(Path(path).resolve())
        if self.windows_child and path not in self._paths:
            parts = path.split('/')
            if len(parts) > 3 and parts[1] == 'mnt' and len(parts[2]) == 1 and parts[2].isalpha():
                # Mounted drive paths need no subprocess/fork of the live gRPC runner.
                self._paths[path] = parts[2].upper() + ':\\' + '\\'.join(parts[3:])
            else:
                self._paths[path] = subprocess.check_output(['wslpath', '-w', path], text=True).strip()
        return self._paths.get(path, path)

    def portable(self, value):
        if isinstance(value, dict):
            return {key: self.portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.portable(item) for item in value]
        if self.windows_child and isinstance(value, str) and value.startswith(str(ROOT) + '/'):
            return self.child_path(value)
        return value

    def command(self, check=False):
        args = self.scenario.args
        command = [args.expert_python, '-X', 'utf8', '-u', '-B', self.child_path(ROOT / 'AIM2026/main.py'),
                '--check' if check else '--drive', '--visualize',
                '--config', self.child_path(args.config),
                '--scenario-context', self.child_path(self.context_path),
                '--output', self.child_path(self.scenario.driver_output),
                '--stop-file', self.child_path(self.stop_file)]
        if args.speed_cap_kmh is not None:
            command.extend(['--speed-cap-kmh', str(args.speed_cap_kmh)])
        if getattr(args, 've_object_speed', False):
            command.append('--ve-object-speed')
        if getattr(args, 'validation_events', False) and not check:
            command.extend(['--run-id', str(self.scenario.run_instance_id)])
        return command

    def child_environment(self):
        environment = os.environ.copy()
        # The unified entry must use the same interpreter selected by this
        # harness, including an explicit --expert-python override.
        selected = self.scenario.args.expert_python
        environment['AIM_EXPERT_PYTHON'] = (self.child_path(selected)
                                           if self.windows_child else selected)
        return environment

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
        print('[Highway] Expert log: ' + str(self.scenario.output / 'driver.console.log'), flush=True)
        deadline = time.monotonic() + 120.
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError('AIM2026 Expert stopped; inspect driver.console.log')
            control_log = self.scenario.driver_output / 'control.jsonl'
            if control_log.is_file():
                try:
                    with control_log.open('rb') as stream:
                        stream.seek(max(0, control_log.stat().st_size - 65536))
                        rows = stream.read().splitlines()
                except OSError as error:
                    # A Windows writer can briefly give WSL ENODATA while the file exists; look again.
                    if error.errno not in (errno.ENODATA, errno.EAGAIN, errno.EINTR, errno.ENOENT):
                        raise
                    rows = []
                for raw in reversed(rows):
                    try:
                        row = json.loads(raw)
                    except (ValueError, UnicodeError):
                        continue
                    if row.get('status') == 'RUNNING' and row.get('control_sent'):
                        self.scenario._contact_drive_active = True
                        print('[Highway] AIM2026 UDP + perception + Expert ready', flush=True)
                        return
                    break
            time.sleep(.1)
        raise RuntimeError('AIM2026 sensor/inference startup timed out; inspect driver.console.log')

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def stop_process(self):
        # A shared stop file also works across WSL -> Windows. The Expert's
        # existing finally block brakes, closes sockets and stops inference.
        if self.is_running():
            self.stop_file.touch()
            try:
                self.process.wait(timeout=20.)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('Expert did not acknowledge stop; inspect driver.console.log') from exc
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


class HighwayScenario(MissionScenario):
    def record_live_spawn_contact_evidence(self, actor, requested_xy, started):
        """Own-NPC birth evidence; never called by the perception/control pipeline."""
        verifier = getattr(self, '_spawn_contact_verifier', None)
        ego = getattr(self, '_contact_latest_ego', None)
        if verifier is None or not getattr(self, '_contact_drive_active', False) or ego is None:
            return
        event = dict(actor_id=str(actor.get_id()), request_started=started,
                     ego=ego, requested_xy=requested_xy)
        try:
            before = time.monotonic()
            state = actor.get_actor_state()
            duration = time.monotonic() - before
            if state is None or str(state.actor_info.id.value) != event['actor_id']:
                raise ValueError('Spawn lookup did not return the requested actor')
            actual = (float(state.transform.location.x), float(state.transform.location.y))
            event.update(actual_xy=actual, query_s=duration)
            if duration <= verifier.MAX_QUERY_S:
                verifier.record_spawn(event['actor_id'], started, ego['time'], ego['xy'], requested_xy, actual)
            event['registered'] = event['actor_id'] in verifier.births
        except Exception as exc:
            event['error'] = str(exc)  # missing evidence never grants an exception
        with (self.output / 'spawn_contact_evidence.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')

    def _verify_runtime_contacts(self, state):
        verifier = getattr(self, '_spawn_contact_verifier', None)
        if verifier is None or not getattr(self, '_contact_drive_active', False):
            return state
        reports = state.get('collision_objects', [])
        ego_xy, targets, error = None, {}, None
        before = time.monotonic()
        if any(identity in verifier.births or identity in verifier.continuing for identity in reports):
            try:
                snapshot = self.grpc.world.get_all_actors_state(vehicle=True, pedestrian=True, obstacle=True)
                if snapshot is None or not hasattr(snapshot, 'states'):
                    raise ValueError('No actor snapshot')
                for actor in snapshot.states:
                    identity = str(actor.actor_info.id.value)
                    if identity == 'Ego' or identity in reports:
                        xy = (float(actor.transform.location.x), float(actor.transform.location.y))
                        if identity == 'Ego':
                            ego_xy = xy
                        else:
                            targets[identity] = xy
            except Exception as exc:
                ego_xy, targets, error = None, {}, str(exc)
        now = time.monotonic()
        decisions = verifier.assess(reports, now, now-before, ego_xy, targets)
        if not decisions:
            return state
        deferred = [identity for identity, decision in decisions.items()
                    if decision['action'] == 'continue_with_review']
        evidence = dict(observed_at=datetime.now().astimezone().isoformat(), monotonic=now,
                        ego_xy=ego_xy, targets=targets, query_s=now-before,
                        query_error=error, reports=reports, decisions=decisions)
        # Persist the raw report AND its proof before filtering the stop list.
        # This is not a clean pass; final reports/batches retain REVIEW_REQUIRED.
        with (self.output / 'runtime_contact_checks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(evidence) + '\n')
        state = dict(state, collision_objects=[v for v in reports if v not in deferred],
                     runtime_contact_deferred=deferred, runtime_contact_checks=decisions)
        signature = tuple((identity, d['action'], d['reason']) for identity, d in decisions.items())
        if signature != getattr(self, '_last_runtime_contact_signature', None):
            for identity, decision in decisions.items():
                print('[Contact verification] {}: {} ({})'.format(
                    identity, decision['action'], decision['reason']), flush=True)
            self._last_runtime_contact_signature = signature
        return state

    def prepare_full_loop_route(self):
        # FullLoopScenario.setup assigns run_seed and run_instance_id immediately
        # before this hook. Bind validation to those actual per-run values.
        if self.args.validation_events and not self.args.check:
            _start_validation(self)
        return super().prepare_full_loop_route()

    def live_actor_ids(self):
        """Scenario verification only; never feed actor states to the Expert."""
        world = getattr(getattr(self, 'grpc', None), 'world', None)
        if world is None:
            return None
        try:
            response = world.get_all_actors_state(vehicle=True, pedestrian=True, obstacle=True)
            if response is None or not hasattr(response, 'states'):
                return None
            return {str(item.actor_info.id.value) for item in response.states}
        except Exception as exc:
            print('[Highway] Cannot verify retired collision actors: {}'.format(exc), flush=True)
            return None

    def get_ego_motion_state(self):
        sampled = time.monotonic()
        state = super().get_ego_motion_state()
        if SpawnContactVerifier.point((state.get('x'), state.get('y'))) is not None:
            self._contact_latest_ego = dict(time=sampled, xy=(state['x'], state['y']))
        baseline = getattr(self, '_startup_contact_baseline', None)
        if baseline is None:
            return state
        actor_ids = (self.live_actor_ids() if baseline.removed_actor_ids & baseline.ignored else None)
        state = baseline.apply(state, actor_ids)
        if baseline.removed_actor_ids:
            state['startup_contact_live_actor_ids'] = sorted(actor_ids) if actor_ids is not None else None
        state = self._verify_runtime_contacts(state)
        signature = (tuple(sorted(state['raw_collision_objects'])),
                     tuple(state['startup_contact_ignored']), tuple(sorted(state['collision_objects'])))
        if signature != getattr(self, '_last_contact_signature', None):
            now = time.monotonic()
            event = dict(monotonic=now, observed_at=datetime.now().astimezone().isoformat(),
                         since_baseline_s=now - getattr(self, '_contact_started_monotonic', now),
                         source='MORAI_gRPC_vehicle_state.collision_objects', **state)
            first_reports = getattr(self, '_contact_first_reports', {})
            for item in state['raw_collision_objects']:
                first_reports.setdefault(item, event.copy())
            self._contact_first_reports = first_reports
            with (self.output / 'contact_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event) + '\n')
            self._last_contact_signature = signature
        return state

    def record_collision_stop(self, state, elapsed, phase='driving'):
        """Called before launch or AFTER stopping control, before actor cleanup.

        Actor snapshots are verification evidence only, never Expert input.
        Observation time is not a simulator-provided collision timestamp.
        """
        baseline = getattr(self, '_startup_contact_baseline', None)
        report = dict(
            observed_at=datetime.now().astimezone().isoformat(), scenario_elapsed_s=elapsed,
            phase=phase, control_backend_started=phase != 'startup',
            decision='stop_on_unexempted_MORAI_report', ego_state=state,
            startup_raw=sorted(baseline.initial) if baseline else [],
            startup_live_actor_ids=baseline.initial_live_actor_ids if baseline else None,
            first_observations=getattr(self, '_contact_first_reports', {}),
            collision_time_known=False, target_snapshots={},
            note='Positions below are sampled after stop request, not at physical impact. '
                 'Any runtime distant-spawn exception is separately recorded in runtime_contact_checks.jsonl.')
        destination = self.output / 'collision_stop.json'
        # Persist the triggering observation even if the subsequent actor query fails.
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        try:
            from google.protobuf.json_format import MessageToDict
            response = self.grpc.world.get_all_actors_state(vehicle=True, pedestrian=True, obstacle=True)
            if response is None or not hasattr(response, 'states'):
                raise RuntimeError('Actor query returned no states')
            requested = set(state.get('collision_objects', []))
            for actor in response.states:
                actor_id = str(actor.actor_info.id.value)
                if actor_id in requested:
                    report['target_snapshots'][actor_id] = MessageToDict(
                        actor, preserving_proto_field_name=True)
            report['targets_absent_at_query'] = sorted(requested - set(report['target_snapshots']))
            report['actor_snapshot_at'] = datetime.now().astimezone().isoformat()
        except Exception as exc:
            report['actor_query_error'] = str(exc)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[Highway] Collision evidence: {} | ids={} | classification={} | ego=({}, {})'.format(
            destination, state.get('collision_objects'), state.get('contact_classification'),
            state.get('x'), state.get('y')), flush=True)

    def configure_drive_from_start(self):
        # Preserve evidence before the existing runner teleports/reset-motion.
        # This separates a pre-existing simulator contact from placement contact.
        state = self.get_ego_motion_state()
        self._contact_started_monotonic = time.monotonic()
        self._contact_first_reports = {}
        self._last_contact_signature = None
        self._contact_drive_active = False
        self._spawn_contact_verifier = None if self.args.strict_collisions else SpawnContactVerifier()
        (self.output / 'before_placement.json').write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        if not self.args.strict_collisions:
            actor_ids = (self.live_actor_ids() if any(str(v).isdecimal()
                         for v in state.get('collision_objects', [])) else None)
            self._startup_contact_baseline = StartupContactBaseline(state, actor_ids)
            if self._startup_contact_baseline.active:
                print('[Highway] INTERACTIVE: pre-existing contact baseline {} is recorded; '
                      'new contacts and cleared labels reappearing still stop the run. '
                      'This is not a collision-free guarantee.'.format(
                          sorted(self._startup_contact_baseline.ignored)), flush=True)
        else:
            # Strict mode records the same evidence but grants no exemptions.
            self._startup_contact_baseline = StartupContactBaseline(state)
            self._startup_contact_baseline.ignored.clear()
            self._startup_contact_baseline.active = False
        return super().configure_drive_from_start()

    def check_startup_contact(self):
        state = self.get_ego_motion_state()
        collisions = sorted({str(value).strip() for value in state.get('collision_objects', [])
                             if str(value).strip()})
        destination = self.output / 'startup_contact.json'
        destination.write_text(json.dumps(dict(ego_state=state, collision_objects=collisions,
            control_backend_started=False), ensure_ascii=False, indent=2), encoding='utf-8')
        if collisions:
            try:
                self.record_collision_stop(state, 0., phase='startup')
            except Exception as exc:
                print('[Highway] Startup collision evidence failed: {}'.format(exc), flush=True)
            before_path = self.output / 'before_placement.json'
            before = json.loads(before_path.read_text(encoding='utf-8')) if before_path.exists() else {}
            initial = {str(v).strip() for v in before.get('collision_objects', [])}
            added = sorted(set(collisions) - initial)
            phase = ('new reports after placement: ' + repr(added) if added else
                     'already present before placement; not verified as retired actors')
            raise RuntimeError('MORAI startup collision ' + repr(collisions) + ': ' + phase
                + '. UDP Expert was NOT started. Reset/reload Ego in MORAI and verify contact clears. '
                + 'Evidence: ' + str(destination))

    def require_async_ros_control_path(self):
        print('[Highway] Driving: AIM2026 UDP. Scene setup/NPC monitoring: gRPC.', flush=True)

    def setup_expert_controller(self):
        super().setup_expert_controller()
        self.expert_controller = ExpertController(self.expert_controller, self)


def _start_validation(scenario):
    if getattr(scenario, '_runner_validation_log', None) is not None:
        return
    from scenario_runner.validation.run_log import ValidationRunLog

    mission = scenario.args.mission
    validation_root = ROOT / RUNNER_RUNTIME
    log = ValidationRunLog(validation_root / 'validation_runs', mission)
    scenario._runner_validation_log = log
    scenario._runner_scenario_name = mission
    scenario._validation_controller = ('aim2026_highway' if mission == 'full_loop_highway'
                                       else 'aim2026_roundabout')
    options = ['--seed', str(scenario.run_seed), '--validation-events']
    if getattr(scenario.args, 'no_live_spawn', False):
        options.append('--no-live-spawn')
    if getattr(scenario.args, 've_object_speed', False):
        options.append('--ve-object-speed')
    if getattr(scenario.args, 'highway_density', None):
        options += ['--highway-density', scenario.args.highway_density]
    if getattr(scenario.args, 'rear_npcs', 'normal') != 'normal':
        options += ['--rear-npcs', scenario.args.rear_npcs]
    log.enable_incidents(validation_root / 'incidents', scenario, run_options=options)
    log.incident_recorder.connect_pipeline(scenario.driver_output / 'pipeline.jsonl')


def _finish_validation(scenario, output, status, error):
    from scenario_runner.validation.finish import finish_validation_run

    validation_log = getattr(scenario, '_runner_validation_log', None)
    verifier = getattr(scenario, '_spawn_contact_verifier', None)
    review_ids = sorted(verifier.review_ids) if verifier else []
    validation_status = 'review_required' if status == 'run_returned' and review_ids else status
    finish_validation_run(validation_log, scenario, validation_status, error)
    report = dict(status={'run_returned': 'COMPLETE', 'interrupted': 'STOPPED',
                          'exception': 'FAILED'}[status],
                  error=str(error) if error else None, run_id=scenario.run_instance_id,
                  plan=dict(mission=scenario.args.mission, seed=scenario.run_seed,
                            driver_output=str(scenario.driver_output)))
    if review_ids:
        report['collision_review'] = dict(required=True, actor_ids=review_ids,
                                         evidence=str(output / 'runtime_contact_checks.jsonl'))
        if status == 'run_returned':
            report['status'] = 'REVIEW_REQUIRED'
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    if (scenario.driver_output / 'pipeline.jsonl').is_file():
        try:
            (scenario.driver_output / 'report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            from scenario_runner.validation.failure_report import build_report
            html, _ = build_report(scenario.driver_output, output / 'failure_report')
            print('[Validation] HTML report=' + str(html), flush=True)
        except Exception as report_error:
            print('[Validation] HTML report unavailable: ' + str(report_error), flush=True)


def main(argv=None, *, mission='full_loop_highway'):
    if mission not in ('full_loop_highway', 'full_loop_roundabout',
                       'full_loop_roundabout_nonuniform', 'full_loop_roundabout_speed_random',
                       'full_loop_roundabout_brake'):
        raise ValueError('Unsupported verification mission: ' + mission)
    from AIM2026.Expert.run_repeat import run_many, split_repeat_options
    repeat, argv = split_repeat_options(argv)
    if repeat.runs > 1:
        if '--check' in argv:
            print('[Batch] --check validates once; --runs is ignored', flush=True)
        else:
            return run_many(lambda one: main(one, mission=mission), argv, repeat,
                            label=mission.replace('full_loop_', '', 1))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-live-spawn', action='store_true',
                        help='Competition traffic diagnostic: keep initial NPCs but do not create replacements during driving')
    parser.add_argument('--ve-object-speed', action='store_true',
                        help='Experiment: forward raw VE object speed option to Expert')
    parser.add_argument('--check', action='store_true', help='Validate without connecting to MORAI')
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config_local.yaml')
    parser.add_argument('--expert-python', default=expert_python())
    parser.add_argument('--speed-cap-kmh', type=validate_max_velocity_kmh, default=None,
                        help='Optional additional ceiling; tune highway.py for the highway speed')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--validation-events', action='store_true', default=True,
                        help='On by default: record Ego observations, incidents and the VE/Planning/Control journal, '
                             'and write the failure report (use --no-validation-events to switch it off)')
    parser.add_argument('--no-validation-events', dest='validation_events', action='store_false',
                        help='Do not record validation events or write a failure report')
    parser.add_argument('--strict-collisions', action='store_true',
                        help='Stop even for verified persistent pre-start contact records')
    parser.add_argument('--rear-npcs', choices=('normal', 'few'), default='normal',
                        help='With --highway-density competition: few = fewer NPCs coming from behind once Ego has '
                             'entered the highway lane (thinned out from the merge entry on)')
    parser.add_argument('--highway-density', choices=('dense', 'competition'), default=None,
                        help='Highway NPC traffic (run_highway.sh only): competition = independent lane-speed traffic')
    wsl_network.add_arguments(parser)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    if args.highway_density is not None and mission != 'full_loop_highway':
        parser.error('--highway-density is only for run_highway.sh')
    if args.rear_npcs != 'normal' and args.highway_density != 'competition':
        parser.error('--rear-npcs needs --highway-density competition')
    if wsl_network.configure_launcher(args, ROOT, parser):
        return 0
    global_cfg, _, base = load_configuration()
    # Keep moving traffic; this merge verification run has no stopped vehicle.
    config = copy.deepcopy(base)
    config.setdefault('highway_merge_static_vehicle', {})['enabled'] = False
    for key in list(config):
        if key.startswith('ve_'):
            config.pop(key)
    config.update(mission_mode=mission, drive_control_mode='aimdrive_expert',
                  expert_speed_cap_kmh=60. if args.speed_cap_kmh is None else args.speed_cap_kmh,
                  expert_ros_remaps=[],
                  expert_traffic_light_control=False, route_debug_enabled=False,
                  max_laps=1, auto_repeat=False, route_setup_attempts=1)
    config.setdefault('sync_runtime', {})['enabled'] = False
    from scenario_runner.utils.highway_density import apply_highway_density
    apply_highway_density(config, args.highway_density)
    if args.no_live_spawn:
        managed = config.get('highway_npc', {}).get('managed_traffic', {})
        if mission != 'full_loop_highway' or managed.get('policy') != 'competition':
            parser.error('--no-live-spawn needs highway competition traffic')
        managed['competition']['live_spawn_enabled'] = False
    if args.rear_npcs == 'few':
        config['highway_npc']['managed_traffic']['competition']['rear_reduction']['enabled'] = True
    if args.seed is not None:
        config['random_seed'] = args.seed
    stamp = mission.replace('full_loop_', '', 1) + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'AIM2026/outputs/scenarios' / stamp
    output.mkdir(parents=True)
    global_cfg['paths']['runtime_root'] = str(output)
    global_cfg['paths']['expert_root'] = str(ROOT / 'AIM2026/Expert')
    client = SimpleNamespace(make_transform=lambda *v: v)
    if not args.check:
        from scenario_runner.adapters.morai_grpc import MoraiGrpcClient
        if os.name == 'nt' and global_cfg['grpc'].get('host') == 'auto':
            global_cfg['grpc']['host'] = '127.0.0.1'
        client = MoraiGrpcClient(global_cfg)
    scenario = HighwayScenario(grpc_client=client,
        map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']), global_cfg=global_cfg, scenario_cfg=config)
    args.mission = mission
    scenario.args, scenario.output = args, output
    scenario.driver_output = ROOT / 'AIM2026/outputs/expert' / stamp
    if args.check:
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
        print('[Scenario] One run: {} -> AIM2026/main.py -> Expert'.format(mission), flush=True)
        scenario.run()
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[Highway] stopped', flush=True)
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
                _finish_validation(scenario, output, status, error)
            except Exception as finish_error:
                print('[Validation] finalization failed: ' + str(finish_error), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
