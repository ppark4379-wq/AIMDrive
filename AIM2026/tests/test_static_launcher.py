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


if __name__ == '__main__':
    unittest.main()
