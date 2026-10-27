"""Offline contract checks for the integrated highway / roundabout validation path (no MORAI)."""

import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'AIM2026/utils')]

from scenario_runner.validation.failure_report import build_report
from scenario_runner.validation.failure_analysis import _derive_failure
from scenario_runner.validation.pipeline_log import PipelineWriter, read_window
from scenario_runner.validation.run_log import ValidationRunLog

RUNTIME = 'AIM2026/utils/scenario_runner/.runtime'
MISSIONS = (
    ('full_loop_highway', 'aim2026_highway', 'bash AIM2026/Expert/run_highway.sh --seed 124 --validation-events'),
    ('full_loop_roundabout', 'aim2026_roundabout',
     'bash AIM2026/Expert/run_roundabout.sh --variant normal --seed 124 --validation-events'),
    ('full_loop_roundabout_nonuniform', 'aim2026_roundabout',
     'bash AIM2026/Expert/run_roundabout.sh --variant nonuniform --seed 124 --validation-events'),
    ('full_loop_roundabout_speed_random', 'aim2026_roundabout',
     'bash AIM2026/Expert/run_roundabout.sh --variant speed_random --seed 124 --validation-events'),
    ('full_loop_roundabout_brake', 'aim2026_roundabout',
     'bash AIM2026/Expert/run_roundabout.sh --variant brake --seed 124 --validation-events'),
)


class HighwayValidationTests(unittest.TestCase):
    def test_validation_starts_after_scenario_assigns_run_identity_for_every_mission(self):
        from AIM2026.Expert import run_highway
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario

        for mission, controller, _ in MISSIONS:
            with self.subTest(mission=mission), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scenario = object.__new__(run_highway.HighwayScenario)
                scenario.args = SimpleNamespace(validation_events=True, check=False, mission=mission)
                scenario.run_seed = 124
                scenario.run_instance_id = 'run-124'
                scenario.driver_output = root / 'driver'
                with patch.object(run_highway, 'ROOT', root), \
                     patch.object(MissionScenario, 'prepare_full_loop_route'):
                    scenario.prepare_full_loop_route()
                self.assertEqual(scenario._validation_controller, controller)
                self.assertEqual(scenario._runner_scenario_name, mission)
                recorder = scenario._runner_validation_log.incident_recorder
                self.assertEqual(recorder.run_options, ['--seed', '124', '--validation-events'])
                self.assertEqual(recorder.pipeline_path, root / 'driver/pipeline.jsonl')

    def test_validation_is_not_started_without_the_option_or_in_check_mode(self):
        from AIM2026.Expert import run_highway
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario

        for options in (dict(validation_events=False, check=False), dict(validation_events=True, check=True)):
            with self.subTest(**options):
                scenario = object.__new__(run_highway.HighwayScenario)
                scenario.args = SimpleNamespace(mission='full_loop_highway', **options)
                with patch.object(MissionScenario, 'prepare_full_loop_route'):
                    scenario.prepare_full_loop_route()
                self.assertIsNone(getattr(scenario, '_runner_validation_log', None))

    def test_watchdog_creates_stall_incident_without_changing_failure(self):
        from scenario_runner.validation.finish import finish_validation_run
        for mission, controller, _ in MISSIONS:
            with self.subTest(mission=mission), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scenario = SimpleNamespace(run_instance_id='run-124', run_seed=124,
                    cfg={'mission_mode': mission}, global_cfg={'paths': {'mgeo_root': 'map'}},
                    _runner_scenario_name=mission, _validation_controller=controller)
                log = ValidationRunLog(root / 'validation_runs', mission)
                log.enable_incidents(root / 'incidents', scenario,
                                     run_options=['--seed', '124', '--validation-events'])
                log.observe('run-124', 180., {'x': 1., 'y': 2., 'z': 0.,
                    'speed_mps': 0., 'collision_objects': []})
                error = RuntimeError('EGO progress watchdog abort: reason=no_route_progress stalled=180.0s')
                finish_validation_run(log, scenario, 'exception', error)
                bundles = list((root / 'incidents').glob('incident_*_stall'))
                self.assertEqual(len(bundles), 1)
                manifest = json.loads((bundles[0] / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['failure']['event_type'], 'scenario_stall')
                self.assertEqual(manifest['failure']['reason'], str(error))
                self.assertTrue((bundles[0] / 'diagnosis.json').is_file())

    def test_full_loop_timeout_creates_timeout_incident_for_roundabout(self):
        from scenario_runner.validation.finish import finish_validation_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = SimpleNamespace(run_instance_id='run-124', run_seed=124,
                cfg={'mission_mode': 'full_loop_roundabout'}, global_cfg={'paths': {'mgeo_root': 'map'}},
                _runner_scenario_name='full_loop_roundabout', _validation_controller='aim2026_roundabout')
            log = ValidationRunLog(root / 'validation_runs', 'full_loop_roundabout')
            log.enable_incidents(root / 'incidents', scenario)
            log.observe('run-124', 600., {'x': 1., 'y': 2., 'z': 0., 'speed_mps': 3., 'collision_objects': []})
            finish_validation_run(log, scenario, 'exception', RuntimeError('K-City full-loop timeout: 600.0s'))
            bundles = list((root / 'incidents').glob('incident_*_timeout'))
            self.assertEqual(len(bundles), 1)
            manifest = json.loads((bundles[0] / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['failure']['event_type'], 'scenario_timeout')

    def test_optional_planner_debug_keeps_valid_fields_when_limit_is_nonfinite(self):
        from AIM2026.Expert.runtime import validation_json_value
        result = validation_json_value({'static_avoidance': {
            'reason': 'yield', 'speed_limits': {'front': float('inf'), 'route': 3.0}}})
        self.assertEqual(result['static_avoidance']['reason'], 'yield')
        self.assertIsNone(result['static_avoidance']['speed_limits']['front'])
        self.assertEqual(result['static_avoidance']['speed_limits']['route'], 3.0)

    def test_launcher_records_collision_before_existing_failure_propagates(self):
        from AIM2026.Expert import run_highway
        for mission, controller, rerun in MISSIONS:
            with self.subTest(mission=mission), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / 'config.yaml'
                config.write_text('placeholder', encoding='utf-8')
                scenario = SimpleNamespace(cfg={'mission_mode': mission},
                    global_cfg={'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}},
                    stop_expert_controller=Mock())

                def failed_run():
                    scenario.run_instance_id, scenario.run_seed = 'run-124', 124
                    run_highway._start_validation(scenario)
                    scenario._runner_validation_log.observe('run-124', 12.5,
                        {'x': 1., 'y': 2., 'z': 0., 'speed_mps': 2.,
                         'collision_objects': ['actor-7']})
                    raise RuntimeError('AIMDrive expert collision during scenario verification')

                scenario.run = failed_run
                client = Mock()
                with patch.object(run_highway, 'ROOT', root), \
                     patch.object(run_highway, 'load_configuration', return_value=(
                         {'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}}, {}, {})), \
                     patch.object(run_highway, 'HighwayScenario', return_value=scenario), \
                     patch.object(run_highway, 'MGeoMapLoader'), \
                     patch('scenario_runner.adapters.morai_grpc.MoraiGrpcClient', return_value=client):
                    with self.assertRaisesRegex(RuntimeError, 'collision'):
                        run_highway.main(['--config', str(config), '--seed', '124', '--validation-events'],
                                         mission=mission)
                summary = root / RUNTIME / 'validation_runs/run-124/summary.json'
                self.assertEqual(json.loads(summary.read_text(encoding='utf-8'))['event_count'], 1)
                incidents = list((root / RUNTIME / 'incidents').glob('incident_*'))
                self.assertEqual(len(incidents), 1)
                manifest = json.loads((incidents[0] / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['failure']['actor_ids'], ['actor-7'])
                self.assertEqual(manifest['scenario_name'], mission)
                self.assertEqual(manifest['controller'], controller)
                self.assertEqual(json.loads((incidents[0] / 'rerun.json').read_text(encoding='utf-8'))[
                    'command'], rerun)
                client.stop.assert_called_once()

    def test_no_validation_events_option_writes_no_validation_files(self):
        from AIM2026.Expert import run_highway
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('placeholder', encoding='utf-8')
            scenario = SimpleNamespace(cfg={}, global_cfg={'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}},
                                       stop_expert_controller=Mock(), run=Mock())
            with patch.object(run_highway, 'ROOT', root), \
                 patch.object(run_highway, 'load_configuration', return_value=(
                     {'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}}, {}, {})), \
                 patch.object(run_highway, 'HighwayScenario', return_value=scenario), \
                 patch.object(run_highway, 'MGeoMapLoader'), \
                 patch('scenario_runner.adapters.morai_grpc.MoraiGrpcClient', return_value=Mock()):
                self.assertEqual(run_highway.main(['--config', str(config), '--no-validation-events']), 0)
            self.assertFalse((root / RUNTIME).exists())

    def test_direct_runtime_pipeline_can_build_html_without_morai(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PipelineWriter(root / 'pipeline.jsonl', 'run-124')
            writer.write('perception', cycle_id=0, ve_valid=True, ve_sequence=1,
                         objects=[], controller='aim2026_integrated_udp')
            writer.write('planning', cycle_id=0, ve_sequence=1,
                         planning_input_source='aim2026_ve_udp', input_objects=[],
                         planned_trajectory=[[0., 0.], [1., 0.]],
                         target_velocity_mps=4., controller='aim2026_integrated_udp')
            writer.write('control', cycle_id=0, planning_cycle_id=0,
                         final_wire_command={'accel': .2, 'brake': 0., 'steer_normalized': 0.},
                         sent=True, observed_ego={'x': 0., 'y': 0., 'speed_mps': 0.},
                         controller='aim2026_integrated_udp')
            writer.close()
            pipeline_rows = [json.loads(line) for line in (root / 'pipeline.jsonl').read_text(
                encoding='utf-8').splitlines()]
            stamp = pipeline_rows[-1]['timestamp_ns']
            (root / 'inputs.jsonl').write_text(json.dumps(dict(
                sequence=1, sensor_received_ns=stamp - 10_000_000,
                signal={'code': 1, 'confidence': .95},
                native_signal={'code': 1, 'confidence': .99})) + '\n', encoding='utf-8')
            (root / 'control.jsonl').write_text(json.dumps(dict(
                monotonic=stamp / 1e9, VE_sequence=1,
                signal_code=1, signal_id='signal-1', signal_head='native_binary',
                signal_gate={'gate_output_code': 1}, status='RUNNING', control_sent=True)) + '\n',
                encoding='utf-8')
            (root / 'report.json').write_text(json.dumps({'status': 'COMPLETE', 'error': None}),
                                              encoding='utf-8')
            html, payload = build_report(root)
            self.assertTrue(html.is_file())
            self.assertEqual(payload['run_status'], 'COMPLETE')
            self.assertEqual(payload['run_id'], 'run-124')
            self.assertIsNone(payload['stage_error'])
            self.assertEqual(payload['stage']['meta']['cycles'], 1)

    def test_cross_host_incident_window_requires_verified_shared_file_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pipeline.jsonl'
            now = time.time_ns()
            row = dict(schema='aimdrive_pipeline_v1', run_id='run-124',
                       sequence=0, stage='control', timestamp_ns=12345,
                       clock_domain='host_monotonic_ns',
                       clock_host='windows' if platform.system().lower() != 'windows' else 'linux',
                       wall_time_ns=now, cycle_id=0, sent=True)
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            rows, issue = read_window(path, 'run-124', 99999, 30., until_wall_ns=now)
            self.assertEqual(len(rows), 1)
            self.assertIn('cross-host wall window', issue)
            row['wall_time_ns'] = now - 10_000_000_000
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            rows, issue = read_window(path, 'run-124', 99999, 30., until_wall_ns=now)
            self.assertEqual(rows, [])
            self.assertIn('not verified', issue)

    def test_cross_host_failure_uses_verified_pipeline_anchor_not_runner_monotonic(self):
        manifest = {'failure': {'event_type': 'scenario_stall', 'reason': 'watchdog',
            'captured_monotonic_sec': 100., 'pipeline_anchor_ns': 999_000_000_000,
            'pipeline_window_basis': 'cross-host wall window; monotonic clocks not aligned'}}
        result = _derive_failure('incident', Path('.'), manifest, None, None, None,
                                 [{'ns': 999_000_000_000}])
        self.assertEqual(result['type'], 'stall')
        self.assertEqual(result['ns'], 999_000_000_000)
        self.assertIn('cross-host', result['basis'])


if __name__ == '__main__':
    unittest.main()
