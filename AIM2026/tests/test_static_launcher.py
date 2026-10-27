"""Offline checks for the isolated static scenario launcher."""

import errno
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import yaml

from AIM2026.Expert import run_static


class StaticLauncherTests(unittest.TestCase):
    def test_expert_does_not_inherit_system_python_package_paths(self):
        parent = {'PYTHONPATH': '/tmp/system-capture-deps:/usr/lib/python3/dist-packages',
                  'CONDA_PREFIX': '/home/seoyeon/miniconda3/envs/vldrive-ve',
                  'AIM_EXPERT_PYTHON': '/home/seoyeon/miniconda3/envs/vldrive-ve/bin/python3'}
        child = run_static.static_expert_environment(parent)
        self.assertNotIn('PYTHONPATH', child)
        self.assertEqual(child['CONDA_PREFIX'], parent['CONDA_PREFIX'])
        self.assertEqual(child['AIM_EXPERT_PYTHON'], parent['AIM_EXPERT_PYTHON'])
        self.assertIn('PYTHONPATH', parent)
        self.assertEqual(child['OPENBLAS_NUM_THREADS'], '1')
        self.assertNotIn('OPENBLAS_NUM_THREADS', parent)

    def test_transient_control_log_read_does_not_claim_ready(self):
        source = MagicMock()
        source.is_file.return_value = True
        source.stat.return_value.st_size = 100
        source.open.return_value.__enter__.return_value.read.side_effect = OSError(
            errno.ENODATA, 'No data available')
        self.assertIsNone(run_static.latest_control_status(source))

    def test_static_setup_and_count_stay_out_of_highway_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('Ego: {local_status_bytes: 245}\n'
                              'Control: {positive_steer: right}', encoding='utf-8')
            scenario = Mock()
            scenario.expert_controller.command.return_value = ['python', '--check']
            scenario.expert_controller.child_environment.return_value = {}
            with patch.object(run_static, 'ROOT', root), \
                 patch.dict(run_static.os.environ,
                            {'AIM_STATIC_EXPERT_PYTHON': '/example/vldrive-ve/bin/python3'}), \
                 patch.object(run_static, 'load_configuration', return_value=(
                     {'paths': {'mgeo_root': 'map'}}, {}, {})), \
                 patch.object(run_static, 'StaticScenario', return_value=scenario) as constructor, \
                 patch.object(run_static, 'MGeoMapLoader'), \
                 patch.object(run_static.events, 'prepare_mission'), \
                 patch.object(run_static.subprocess, 'run',
                              return_value=SimpleNamespace(returncode=0)):
                result = run_static.main(['--check', '--config', str(config),
                                          '--static-obstacle-count', '2'])
            self.assertEqual(result, 0)
            chosen = constructor.call_args.kwargs['scenario_cfg']
            self.assertEqual(chosen['mission_mode'], 'full_loop_static')
            self.assertEqual(chosen['static_obstacle_count'], 2)
            self.assertTrue(scenario.args.validation_events)
            self.assertEqual(scenario.args.expert_python, '/example/vldrive-ve/bin/python3')
            self.assertEqual(yaml.safe_load(scenario.args.config.read_text())['Ego']['local_status_bytes'], 245)
            self.assertEqual(yaml.safe_load(scenario.args.config.read_text())['Control']['positive_steer'], 'right')

    def test_static_validation_preserves_count_in_rerun_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = SimpleNamespace(args=SimpleNamespace(static_obstacle_count=2),
                run_seed=124, run_instance_id='run-124', driver_output=root / 'driver',
                cfg={}, global_cfg={})
            with patch.object(run_static, 'ROOT', root):
                run_static.start_validation(scenario)
            self.assertEqual(scenario._validation_controller, 'aim2026_static')
            self.assertEqual(scenario._runner_validation_log.incident_recorder.run_options,
                             ['--seed', '124', '--validation-events',
                              '--static-obstacle-count', '2'])

    def test_competition_config_does_not_enable_local_status_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('Ego: {vehicle_status_port: 9099}\n', encoding='utf-8')
            scenario = Mock()
            scenario.expert_controller.command.return_value = ['python', '--check']
            scenario.expert_controller.child_environment.return_value = {}
            with patch.object(run_static, 'ROOT', root), \
                 patch.object(run_static, 'load_configuration', return_value=(
                     {'paths': {'mgeo_root': 'map'}}, {}, {})), \
                 patch.object(run_static, 'StaticScenario', return_value=scenario), \
                 patch.object(run_static, 'MGeoMapLoader'), \
                 patch.object(run_static.events, 'prepare_mission'), \
                 patch.object(run_static.subprocess, 'run',
                              return_value=SimpleNamespace(returncode=0)):
                self.assertEqual(run_static.main(['--check', '--config', str(config)]), 0)
            generated = yaml.safe_load(scenario.args.config.read_text(encoding='utf-8'))
            self.assertNotIn('local_status_bytes', generated['Ego'])

    def test_avoid_preset_keeps_every_model_that_can_be_passed(self):
        lane = [(0., -105. + 0.5 * i, 28.) for i in range(125)]
        side = [(min(3.7, 0.06 * i), -105. + 0.5 * i, 28.) for i in range(125)]
        loader = Mock()
        loader.get_link_points.side_effect = lambda link: lane if link == run_static.AVOID_LINKS[0] else side
        models = ['TrafficBarrel', 'Steel_Barricade', 'Rockslide', 'Barbed_Wirefence', 'No_Such_Model']
        preset = run_static.avoidance_preset(loader, models, 'map', seed=1)
        info = preset['generation']
        self.assertEqual(info['usable_models'], ['TrafficBarrel', 'Steel_Barricade', 'Barbed_Wirefence'])
        self.assertEqual(set(info['excluded_models']), {'Rockslide', 'No_Such_Model'})
        self.assertEqual(preset['obstacles'][0]['model'], 'Steel_Barricade')   # seed 1 -> second usable model
        self.assertAlmostEqual(preset['obstacles'][0]['position']['y'], -83., delta=0.25)
        # Wider than long (3.0 x 1.1 m): turned so its short side is across the road.
        self.assertLess(abs((preset['obstacles'][0]['yaw_deg'] + 180.) % 360. - 180.), 1.)   # road heading 90 + 270
        with self.assertRaises(ValueError):
            run_static.avoidance_preset(loader, models, 'map', model='Rockslide')

    def test_avoid_mission_extends_static_section_past_the_split(self):
        from scenario_runner.scenarios.segment_modes.modes import MISSION_MODES
        static, avoid = MISSION_MODES[run_static.MISSION], MISSION_MODES[run_static.AVOID_MISSION]
        self.assertEqual(avoid.start, static.start)
        self.assertGreater(avoid.end.index, static.end.index)
        self.assertTrue(avoid.zones.static)

    def test_only_orphaned_spawn_workers_are_removed_from_sensor_ports(self):
        worker = 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=11080, pipe_handle=2252)"'
        orphan = {'pid': 1, 'port': 9090, 'cmd': worker, 'parent_alive': False}
        live_child = {'pid': 2, 'port': 9090, 'cmd': worker, 'parent_alive': True}
        expert = {'pid': 3, 'port': 2368, 'cmd': 'python.exe -u AIM2026/main.py --drive', 'parent_alive': False}
        orphans, others = run_static.stale_sensor_receivers([orphan, live_child, expert])
        self.assertEqual([item['pid'] for item in orphans], [1])
        self.assertEqual([item['pid'] for item in others], [2, 3])      # a live run is never touched


if __name__ == '__main__':
    unittest.main()
