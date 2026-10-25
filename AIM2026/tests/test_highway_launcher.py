"""Scenario-to-UDP ownership, command and graceful-stop regressions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from AIM2026.Expert.run_highway import ExpertController, HighwayScenario, StartupContactBaseline, ROOT
from AIM2026 import main as integrated_main


class HighwayLauncherTest(unittest.TestCase):
    def test_launcher_disables_only_stopped_vehicle_in_private_scene_config(self):
        from AIM2026.Expert import run_highway
        base = {'highway_merge_static_vehicle': {'enabled': True, 'route_link': '411'},
                'highway_spawnpoints': [1, 13, 17]}
        config_path = self.output / 'ports.yaml'
        config_path.write_text('Ego: {cmd_port: 9093}', encoding='utf-8')
        with patch.object(run_highway, 'ROOT', self.output), \
             patch.object(run_highway, 'load_configuration', return_value=(
                 {'paths': {'mgeo_root': 'map'}}, {}, base)), \
             patch.object(run_highway, 'HighwayScenario') as scenario, \
             patch.object(run_highway, 'MGeoMapLoader'), \
             patch.object(run_highway.events, 'prepare_mission'), \
             patch.object(run_highway.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(run_highway.main(['--check', '--config', str(config_path)]), 0)
            selected = scenario.call_args.kwargs['scenario_cfg']
        self.assertFalse(selected['highway_merge_static_vehicle']['enabled'])
        self.assertEqual(selected['highway_spawnpoints'], [1, 13, 17])
        self.assertTrue(base['highway_merge_static_vehicle']['enabled'])
        self.assertEqual(config_path.read_text(encoding='utf-8'), 'Ego: {cmd_port: 9093}')
        from scenario_runner.scenarios.full_loop_parts.highway import spawn_highway_merge_static_vehicle
        scene = SimpleNamespace(cfg=selected, highway_spawn_points_active=True, grpc=Mock())
        self.assertIsNone(spawn_highway_merge_static_vehicle(scene))
        scene.grpc.spawn_vehicle.assert_not_called()

    def test_integrated_entry_prefers_lmdrive_then_installed_vldrive(self):
        repo = self.output / 'owner' / 'repo'
        repo.mkdir(parents=True)
        vldrive = repo.parent / 'anaconda3/envs/vldrive/python.exe'
        vldrive.parent.mkdir(parents=True)
        vldrive.touch()
        with patch.dict('os.environ', {}, clear=True), patch.object(integrated_main, 'ROOT', repo):
            self.assertEqual(integrated_main.runtime_python(), vldrive)
            lmdrive = repo.parent / 'anaconda3/envs/lmdrive/python.exe'
            lmdrive.parent.mkdir(parents=True)
            lmdrive.touch()
            self.assertEqual(integrated_main.runtime_python(), lmdrive)

    def test_integrated_entry_forwards_scene_arguments_and_check_never_drives(self):
        interpreter = self.output / 'python.exe'
        interpreter.touch()
        scene = ['--scenario-context', 'scenario.json', '--stop-file', 'stop_expert']
        for options, drives in ((scene, True), (['--check', *scene], False)):
            with patch.object(integrated_main, 'runtime_python', return_value=interpreter), \
                 patch.object(integrated_main.os, 'chdir'), patch.object(integrated_main.os, 'execv') as execute:
                integrated_main.main(options)
            command = execute.call_args.args[1]
            self.assertIn('AIM2026/Expert/main.py', command)
            self.assertEqual('--drive' in command, drives)
            self.assertEqual(command[-len(options):], options)

    def test_persistent_self_baseline_is_recorded_but_other_contacts_still_stop(self):
        baseline = StartupContactBaseline({'collision_objects': ['Ego']})
        result = baseline.apply({'collision_objects': ['Ego']})
        self.assertEqual(result['collision_objects'], [])
        self.assertEqual(result['raw_collision_objects'], ['Ego'])
        result = baseline.apply({'collision_objects': ['Ego', 'vehicle_7']})
        self.assertEqual(result['collision_objects'], ['vehicle_7'])

    def test_self_contact_after_clear_is_not_exempt(self):
        baseline = StartupContactBaseline({'collision_objects': ['Ego']})
        baseline.apply({'collision_objects': []})
        self.assertEqual(baseline.apply({'collision_objects': ['Ego']})['collision_objects'], ['Ego'])

    def test_only_preexisting_anonymous_contact_is_baselined(self):
        baseline = StartupContactBaseline({'collision_objects': ['Ego', '-100']})
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '-100']})['collision_objects'], [])
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '-100', '7']})['collision_objects'], ['7'])
        baseline.apply({'collision_objects': ['Ego']})
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '-100']})['collision_objects'], ['-100'])
        # It was not present before placement: never add it to the exception.
        baseline = StartupContactBaseline({'collision_objects': ['Ego']})
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '-100']})['collision_objects'], ['-100'])

    def test_mixed_real_object_and_anonymous_contact_are_not_baselined(self):
        raw = ['Ego', '-100', 'barrier']
        baseline = StartupContactBaseline({'collision_objects': raw})
        self.assertEqual(baseline.apply({'collision_objects': raw})['collision_objects'], raw)

    def test_removed_numeric_actor_must_be_absent_before_and_after_placement(self):
        raw = {'collision_objects': ['Ego', '27']}
        for ids in (None, {'Ego', '27'}):
            baseline = StartupContactBaseline(raw, ids)
            self.assertEqual(baseline.apply(raw, ids)['collision_objects'], raw['collision_objects'])
        baseline = StartupContactBaseline(raw, {'Ego'})
        result = baseline.apply(raw, {'Ego'})
        self.assertEqual(result['collision_objects'], [])
        self.assertEqual(result['raw_collision_objects'], ['Ego', '27'])
        self.assertEqual(baseline.apply({'collision_objects':['Ego','27','28']}, {'Ego','28'})['collision_objects'], ['28'])

    def test_reused_actor_or_failed_query_revokes_retired_actor_exception(self):
        raw = {'collision_objects': ['Ego', '27']}
        for ids in (None, {'Ego', '27'}):
            baseline = StartupContactBaseline(raw, {'Ego'})
            self.assertEqual(baseline.apply(raw, ids)['collision_objects'], ['27'])
        baseline = StartupContactBaseline(raw, {'Ego'})
        baseline.apply({'collision_objects':['Ego']}, {'Ego'})
        self.assertEqual(baseline.apply(raw, {'Ego'})['collision_objects'], ['27'])

    def test_retired_actor_check_is_used_by_real_scenario_wrapper(self):
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
        scenario = object.__new__(HighwayScenario)
        scenario.output = self.output
        scenario.args = SimpleNamespace(strict_collisions=False)
        scenario.grpc = SimpleNamespace(world=Mock())
        actor = SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value='Ego')))
        scenario.grpc.world.get_all_actors_state.return_value = SimpleNamespace(states=[actor])
        raw = {'collision_objects':['Ego','27']}
        with patch.object(MissionScenario,'get_ego_motion_state',return_value=raw), \
             patch.object(MissionScenario,'configure_drive_from_start',return_value=True):
            self.assertTrue(scenario.configure_drive_from_start())
            scenario.check_startup_contact()
            scenario.grpc.world.get_all_actors_state.return_value = None
            with self.assertRaisesRegex(RuntimeError, '27'):
                scenario.check_startup_contact()

    def test_clean_or_mixed_start_does_not_create_exemption(self):
        for contacts in ([], ['vehicle_7'], ['Ego', 'vehicle_7']):
            baseline = StartupContactBaseline({'collision_objects': contacts})
            self.assertEqual(baseline.apply({'collision_objects': ['Ego']})['collision_objects'], ['Ego'])

    def test_scene_handoff_and_timeline_share_baseline_policy(self):
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
        scenario = object.__new__(HighwayScenario)
        scenario.output = self.output
        scenario.args = SimpleNamespace(strict_collisions=False)
        raw = {'collision_objects': ['Ego']}
        with patch.object(MissionScenario, 'get_ego_motion_state', return_value=raw), \
             patch.object(MissionScenario, 'configure_drive_from_start', return_value=True):
            self.assertTrue(scenario.configure_drive_from_start())
            scenario.check_startup_contact()
            self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
            raw['collision_objects'] = ['Ego', 'barrier']
            self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], ['barrier'])
            with self.assertRaisesRegex(RuntimeError, 'barrier'):
                scenario.check_startup_contact()

    def test_strict_mode_preserves_startup_rejection(self):
        from scenario_runner.scenarios.segment_modes.scenario import MissionScenario
        scenario = object.__new__(HighwayScenario)
        scenario.output = self.output
        scenario.args = SimpleNamespace(strict_collisions=True)
        with patch.object(MissionScenario, 'get_ego_motion_state', return_value={'collision_objects': ['Ego']}), \
             patch.object(MissionScenario, 'configure_drive_from_start', return_value=True):
            scenario.configure_drive_from_start()
            with self.assertRaisesRegex(RuntimeError, 'Ego'):
                scenario.check_startup_contact()

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.scenario = SimpleNamespace(
            output=self.output, driver_output=self.output / 'driver',
            global_cfg={'paths': {'expert_root': str(ROOT / 'AIM2026/Expert')}},
            cfg={'mission_mode': 'full_loop_highway', 'highway_merge_static_vehicle': {'enabled': True}},
            route_points=[(1., 2., 3.), (2., 3., 4.)],
            check_startup_contact=Mock(),
            args=SimpleNamespace(expert_python='python3', config=ROOT / 'AIM2026/config_local.yaml',
                                 speed_cap_kmh=60.))
        self.controller = ExpertController(SimpleNamespace(runtime_path_csv='base.csv'), self.scenario)

    def test_current_entry_yaml_and_scene_route_are_forwarded(self):
        command = self.controller.command()
        self.assertIn(str(ROOT / 'AIM2026/main.py'), command)
        self.assertNotIn(str(ROOT / 'AIM2026/Expert/main.py'), command)
        self.assertIn('--drive', command)
        self.assertIn('--visualize', command)
        self.assertEqual(command[command.index('--config') + 1], str(self.scenario.args.config))
        self.assertNotIn('--status-port', command)
        self.assertNotIn('--transport', command)
        self.assertIn('--check', self.controller.command(check=True))
        self.assertNotIn('--drive', self.controller.command(check=True))
        document = json.loads(self.controller.context_path.read_text(encoding='utf-8'))
        self.assertEqual(document['route_points'], [[1., 2., 3.], [2., 3., 4.]])
        self.assertTrue(document['scenario_config']['highway_merge_static_vehicle']['enabled'])
        self.assertEqual(self.controller.runtime_path_csv, 'base.csv')

    def test_stop_request_precedes_wait_without_killing_other_processes(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = lambda **_: self.assertTrue(self.controller.stop_file.is_file())
        self.controller.process = process
        self.controller.stop_process()
        process.wait.assert_called_once_with(timeout=20.)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()
        self.assertIsNone(self.controller.process)

    def test_startup_contact_blocks_launch_including_ego_only(self):
        state = {'x': 1., 'y': 2., 'speed_mps': 0., 'collision_objects': ['Ego']}
        self.scenario.get_ego_motion_state = Mock(return_value=state)
        self.scenario.check_startup_contact = lambda: HighwayScenario.check_startup_contact(self.scenario)
        with patch('AIM2026.Expert.run_highway.subprocess.Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Ego'):
                self.controller.start_process()
            launch.assert_not_called()
        snapshot = json.loads((self.output / 'startup_contact.json').read_text(encoding='utf-8'))
        self.assertEqual(snapshot['collision_objects'], ['Ego'])
        self.assertFalse(snapshot['control_backend_started'])

    def test_clear_startup_contact_is_recorded_and_allowed(self):
        self.scenario.get_ego_motion_state = Mock(return_value={'collision_objects': []})
        HighwayScenario.check_startup_contact(self.scenario)
        snapshot = json.loads((self.output / 'startup_contact.json').read_text(encoding='utf-8'))
        self.assertEqual(snapshot['collision_objects'], [])

    def test_start_waits_for_current_udp_runtime_status(self):
        self.scenario.driver_output.mkdir()
        (self.scenario.driver_output / 'control.jsonl').write_text(
            json.dumps({'status': 'RUNNING', 'control_sent': True}) + '\n', encoding='utf-8')
        process = Mock()
        process.poll.return_value = None
        with patch('AIM2026.Expert.run_highway.subprocess.Popen', return_value=process) as launch:
            self.controller.start_process()
        self.assertEqual(launch.call_args.args[0], self.controller.command())
        self.controller.stop_process()

    def test_waiting_or_shadow_is_not_ready(self):
        self.scenario.driver_output.mkdir()
        path = self.scenario.driver_output / 'control.jsonl'
        process = Mock()
        process.poll.return_value = None
        for row in ({'status': 'WAITING', 'control_sent': True},
                    {'status': 'RUNNING', 'control_sent': False}):
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with patch('AIM2026.Expert.run_highway.subprocess.Popen', return_value=process), \
                 patch('AIM2026.Expert.run_highway.time.monotonic', side_effect=[0., 0., 121.]), \
                 patch('AIM2026.Expert.run_highway.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'startup timed out'):
                    self.controller.start_process()
            self.controller.stop_process()
