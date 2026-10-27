"""Dynamic scenarios using the unmodified AIM2026 Expert entry point."""
import argparse
import copy
import errno
import json
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'MoraiDrive')]

from AIM2026.Expert.run_highway import ExpertController, HighwayScenario, expert_python
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.segment_modes import events
from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
from scenario_runner.utils.map_loader import MGeoMapLoader


def control_ready(path):
    """Poll complete journal rows; transient DrvFS reads do not imply readiness."""
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            data = stream.read()
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.EAGAIN, errno.EINTR, 61):
            return False
        raise
    # The writer may still be appending the last JSON row.
    rows = data.split(b'\n')[:-1]
    for raw in reversed(rows):
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict):
            return row.get('status') == 'RUNNING' and bool(row.get('control_sent'))
    return False


class DynamicController(ExpertController):
    def child_environment(self):
        environment = super().child_environment()
        # Do not inject system-Python capture packages into the conda Expert.
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        return environment

    def start_process(self):
        if self.is_running():
            return
        self.scenario.check_startup_contact()
        command = self.command()
        (self.scenario.output / 'driver_command.json').write_text(json.dumps(command), encoding='utf-8')
        self.log = (self.scenario.output / 'driver.console.log').open('w', encoding='utf-8', buffering=1)
        options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == 'nt' else {'start_new_session': True})
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log,
            stderr=subprocess.STDOUT, env=self.child_environment(), **options)
        print('[Dynamic] Expert log: ' + str(self.scenario.output / 'driver.console.log'), flush=True)
        deadline = time.monotonic() + 120.
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError('AIM2026 Expert stopped; inspect driver.console.log')
            if control_ready(self.scenario.driver_output / 'control.jsonl'):
                print('[Dynamic] AIM2026 UDP + perception + Expert ready', flush=True)
                return
            time.sleep(.1)
        raise RuntimeError('AIM2026 sensor/inference startup timed out; inspect driver.console.log')


class DynamicScenario(HighwayScenario):
    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        try:
            return super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        finally:
            if getattr(self.args, 'pedestrian_trace', False):
                from scenario_runner.tools.pedestrian_diagnostics import record_snapshot
                record_snapshot(self, current_s)

    def prepare_full_loop_route(self):
        if self.args.validation_events and not self.args.check:
            from scenario_runner.tools.dynamic_validation import start_validation
            start_validation(self)
        return super().prepare_full_loop_route()

    def event(self, name, **payload):
        # Keep evidence per run: the shared runtime profile is overwritten
        # when --repeat starts the next scenario.
        try:
            with (self.output / 'dynamic_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time_unix=time.time(), event=name,
                                             payload=payload), ensure_ascii=False) + '\n')
        except OSError as error:
            print(f'[Dynamic] Event recording failed: {error}', flush=True)
        if name == 'dynamic_obstacle_failed':
            recorder = getattr(getattr(self, '_runner_validation_log', None), 'incident_recorder', None)
            if recorder is not None:
                try:
                    recorder.capture_dynamic_failure(self.run_instance_id, payload.get('reason', name))
                except Exception as error:
                    print(f'[Dynamic] Incident recording failed: {error}', flush=True)

    def setup_expert_controller(self):
        MissionScenario.setup_expert_controller(self)
        self.expert_controller = DynamicController(self.expert_controller, self)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config_local.yaml')
    parser.add_argument('--expert-python', default=expert_python())
    parser.add_argument('--strict-collisions', action='store_true')
    parser.add_argument('--seed', type=int, help='Repeat random scenario selection with this seed')
    parser.add_argument('--validation-events', action='store_true',
                        help='Record seed, Ego/NPC observations and VE/Planning/Control diagnosis')
    parser.add_argument('--repeat', action='store_true',
                        help='Repeat the fixed dynamic segment until Ctrl+C; stop on errors')
    parser.add_argument('--once', action='store_false', dest='repeat', help='Run one scenario')
    parser.add_argument('--random-locations', type=int, choices=(2, 3, 4),
                        help='Choose exactly this many distinct locations among 1,2,3,4,5')
    parser.add_argument('--roads', type=int, nargs='+', choices=(1, 4, 5),
                        help='Ordinary-road locations to combine with --opposing-crosswalk')
    parser.add_argument('--dynamic-obstacle-count', type=int, choices=(1, 2), default=1,
                        help='1: fixed Man1 crossing; 2: exactly two random sequential crossings per run')
    parser.add_argument('--dynamic-location-mix', action='store_true',
                        help='Randomly combine 2-5 distinct crossing locations with Ego-aware yield timing')
    parser.add_argument('--only-location', type=int, choices=(5,),
                        help='Spawn one pedestrian only at location 5 to check the complete crossing')
    parser.add_argument('--opposing-crosswalk', type=int, choices=(2, 3),
                        help='Two pedestrians from opposite sides of crosswalk 2 or 3 in the same tick')
    parser.add_argument('--pedestrian-models', nargs='+', help='Cycle one fixed model per run (opposing mode only)')
    parser.add_argument('--exclude-pedestrian-models', nargs='+', default=[], help='Exclude models in this diagnostic run only')
    parser.add_argument('--pedestrian-trace', action='store_true', help='Record full pedestrian inventory including Z at up to 10 Hz')
    args = parser.parse_args(argv)
    if args.random_locations and (args.opposing_crosswalk or args.only_location or args.dynamic_obstacle_count != 1):
        parser.error('--random-locations cannot be combined with other scenario selectors')
    if args.roads and (not args.opposing_crosswalk or len(args.roads) != len(set(args.roads))):
        parser.error('--roads requires --opposing-crosswalk and distinct road numbers')
    if (args.pedestrian_models or args.exclude_pedestrian_models) and not args.opposing_crosswalk:
        parser.error('Pedestrian diagnostic options require --opposing-crosswalk')
    if args.pedestrian_models:
        args.pedestrian_models = [m for m in args.pedestrian_models if m not in args.exclude_pedestrian_models]
        if not args.pedestrian_models:
            parser.error('All requested models were excluded')
    if args.opposing_crosswalk and (args.only_location or args.dynamic_location_mix):
        parser.error('--opposing-crosswalk is a separate scenario; omit --only-location and --dynamic-location-mix')
    args.config = args.config.resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    args.speed_cap_kmh = None
    args.mission = 'full_loop_dynamic'
    run_number = 1
    try:
        while True:
            print('[Dynamic] Run {}'.format(run_number), flush=True)
            args.pedestrian_model = (args.pedestrian_models[(run_number - 1) % len(args.pedestrian_models)]
                                     if args.pedestrian_models else None)
            result = run_once(args)
            if result != 0 or args.check or not args.repeat:
                return result
            run_number += 1
    except KeyboardInterrupt:
        print('[Dynamic] repeat stopped', flush=True)
        return 130


def build_dynamic_config(base, count, location_mix=False, only_location=None, opposing_crosswalk=None,
                         random_locations=None, roads=None):
    if random_locations is not None:
        if random_locations not in (2, 3, 4) or opposing_crosswalk or only_location or count != 1:
            raise ValueError('Invalid random location combination')
        location_mix = True
    if roads and (not opposing_crosswalk or len(set(roads)) != len(roads) or not set(roads) <= {1, 4, 5}):
        raise ValueError('Road combinations require an opposing crosswalk and distinct roads 1,4,5')
    if opposing_crosswalk is not None:
        if opposing_crosswalk not in (2, 3) or only_location is not None:
            raise ValueError('Opposing crossing requires crosswalk 2 or 3')
        location_mix = True
    if only_location is not None:
        if only_location != 5 or count != 1:
            raise ValueError('Location-only verification requires location 5 and one pedestrian')
        location_mix = True
    config = copy.deepcopy(base)
    for key in list(config):
        if key.startswith('ve_'):
            config.pop(key)
    config.update(mission_mode='full_loop_dynamic', drive_control_mode='aimdrive_expert',
                  expert_ros_remaps=[], expert_traffic_light_control=False,
                  route_debug_enabled=False, max_laps=1, auto_repeat=False,
                  route_setup_attempts=1, random_seed=42,
                  dynamic_obstacle_review=[dict(id='legacy_left_to_right', model='Man1', speed_mps=2.5)])
    config.setdefault('sync_runtime', {})['enabled'] = False
    if count == 2:
        config.pop('dynamic_obstacle_review', None)
        config['random_seed'] = None
        config.setdefault('dynamic_obstacle', {}).update(
            max_count=2, required_count=2, single_event_probability=0.0)
    if location_mix:
        config.pop('dynamic_obstacle_review', None)
        config['random_seed'] = None
        dynamic = config.setdefault('dynamic_obstacle', {})
        dynamic.pop('required_count', None)
        dynamic.update(
            location_mix=True, max_count=5, minimum_event_spacing_m=0.0,
            minimum_trigger_distance_m=5.0, minimum_ego_speed_mps=1.0,
            interaction_lead_sec=0.5, maximum_eta_delta_sec=1.0,
            retire_after_conflict_m=0.5, event_recovery_distance_m=0.5,
            early_following_spawn=True, following_spawn_interval_sec=2.0,
            state_recovery_grace_wall_sec=2.0, state_retry_interval_wall_sec=0.2,
            continue_on_confirmed_actor_loss=True,
            require_ego_response=True, assessment_policy='strict')
        if count == 2:
            dynamic.update(required_count=2, max_count=2)
        if only_location is not None:
            dynamic.update(only_location=only_location, required_count=1, max_count=1)
        if opposing_crosswalk is not None:
            dynamic.update(opposing_crosswalk=opposing_crosswalk, required_count=2, max_count=2)
            if roads:
                dynamic.update(opposing_roads=list(roads), required_count=2+len(roads), max_count=2+len(roads))
        if random_locations is not None:
            dynamic.update(required_count=random_locations, max_count=random_locations)
        from scenario_runner.tools.pedestrian_compatibility import apply_compatibility
        apply_compatibility(config)
    return config


def run_once(args):
    global_cfg, _, base = load_configuration()
    config = build_dynamic_config(base, args.dynamic_obstacle_count, args.dynamic_location_mix,
                                  args.only_location, args.opposing_crosswalk,
                                  getattr(args, 'random_locations', None), getattr(args, 'roads', None))
    from scenario_runner.tools.pedestrian_diagnostics import configure_models
    configure_models(config, getattr(args, 'pedestrian_model', None),
                     getattr(args, 'exclude_pedestrian_models', []))
    compatible = config.get('dynamic_obstacle', {}).get('compatible_models')
    if compatible is not None:
        blocked = sorted(set(config.get('dynamic_obstacle_models', [])) - set(compatible))
        print('[Dynamic] Live-tested pedestrian models: {}; excluded: {}'.format(
            len(compatible), ', '.join(blocked)), flush=True)
    if args.seed is not None:
        config['random_seed'] = args.seed
    stamp = 'dynamic-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
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
    scenario = DynamicScenario(grpc_client=client,
        map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
        global_cfg=global_cfg, scenario_cfg=config)
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
    status, run_error = 'run_returned', None
    try:
        client.connect()
        description = (f'Opposite-side pair at crosswalk {args.opposing_crosswalk}' if args.opposing_crosswalk else
                       'Location 5 only: one complete pedestrian crossing' if args.only_location else
                       'Random distinct-location yielding sequence' if args.dynamic_location_mix else
                       'Exactly two random sequential crossings' if args.dynamic_obstacle_count == 2
                       else 'Fixed Man1 crossing, 2.5 m/s')
        print('[Dynamic] ' + description + '; AIM2026 Expert + VE', flush=True)
        scenario.run()
    except KeyboardInterrupt as error:
        status, run_error = 'interrupted', error
        print('[Dynamic] stopped', flush=True)
        return 130
    except Exception as error:
        status, run_error = 'exception', error
        try:
            (output / 'runner_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        except OSError as error:
            print(f'[Dynamic] Error recording failed: {error}', flush=True)
        raise
    finally:
        try:
            scenario.stop_expert_controller()
        finally:
            try:
                client.stop()
            finally:
                signal.signal(signal.SIGTERM, old_handler)
                if args.validation_events:
                    from scenario_runner.tools.dynamic_validation import finish_validation
                    try:
                        finish_validation(scenario, status, run_error)
                    except Exception as error:
                        print(f'[Dynamic] Validation finalization failed: {error}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
