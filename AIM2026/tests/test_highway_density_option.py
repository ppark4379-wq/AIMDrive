"""run_highway.sh --highway-density: the competition traffic reaches the scene config and the rerun command (no MORAI)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'AIM2026/utils')]

from AIM2026.Expert import run_highway


class HighwayDensityOptionTests(unittest.TestCase):
    def check(self, extra, mission='full_loop_highway'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'ports.yaml'
            config_path.write_text('Ego: {cmd_port: 9093}', encoding='utf-8')
            with patch.object(run_highway, 'ROOT', root), \
                 patch.object(run_highway, 'HighwayScenario') as scenario, \
                 patch.object(run_highway, 'MGeoMapLoader'), \
                 patch.object(run_highway.events, 'prepare_mission'), \
                 patch.object(run_highway.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = run_highway.main(['--check', '--config', str(config_path), *extra], mission=mission)
            self.last_args = scenario.return_value.args
            return code, scenario.call_args.kwargs['scenario_cfg']

    def test_competition_traffic_replaces_only_the_highway_npc_policy(self):
        _, default = self.check([])
        code, selected = self.check(['--highway-density', 'competition'])
        self.assertEqual(code, 0)
        managed = selected['highway_npc']['managed_traffic']
        self.assertEqual(managed['policy'], 'competition')
        self.assertIsNone(managed['speed_range_kmh'])
        self.assertNotIn('policy', default['highway_npc']['managed_traffic'])
        self.assertEqual({k: v for k, v in selected.items() if k != 'highway_npc'},
                         {k: v for k, v in default.items() if k != 'highway_npc'})

    def test_failure_diagnosis_is_on_by_default_and_can_be_switched_off(self):
        self.check([])
        self.assertTrue(self.last_args.validation_events)
        self.check(['--no-validation-events'])
        self.assertFalse(self.last_args.validation_events)
        self.check(['--validation-events'])
        self.assertTrue(self.last_args.validation_events)

    def test_rear_npcs_few_reaches_the_scene_config(self):
        _, normal = self.check(['--highway-density', 'competition'])
        self.assertFalse(normal['highway_npc']['managed_traffic']['competition']['rear_reduction']['enabled'])
        _, light = self.check(['--highway-density', 'competition', '--rear-npcs', 'few'])
        self.assertTrue(light['highway_npc']['managed_traffic']['competition']['rear_reduction']['enabled'])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.check(['--rear-npcs', 'few'])                               # needs the competition traffic

    def test_dense_choice_keeps_the_default_traffic(self):
        _, default = self.check([])
        _, dense = self.check(['--highway-density', 'dense'])
        self.assertEqual(dense['highway_npc'], default['highway_npc'])

    def test_the_option_is_for_the_highway_launcher_only(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.check(['--highway-density', 'competition'], mission='full_loop_roundabout')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.check(['--highway-density', 'unknown'])

    def test_the_incident_rerun_command_keeps_the_traffic_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('placeholder', encoding='utf-8')
            scenario = SimpleNamespace(cfg={'mission_mode': 'full_loop_highway'},
                global_cfg={'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}},
                stop_expert_controller=Mock())

            def failed_run():
                scenario.run_instance_id, scenario.run_seed = 'run-124', 124
                run_highway._start_validation(scenario)
                scenario._runner_validation_log.observe('run-124', 12.5,
                    {'x': 1., 'y': 2., 'z': 0., 'speed_mps': 2., 'collision_objects': ['actor-7']})
                raise RuntimeError('AIMDrive expert collision during scenario verification')
            scenario.run = failed_run
            base = ({'paths': {'mgeo_root': 'map'}, 'grpc': {'host': '127.0.0.1'}}, {}, {'highway_npc': {'managed_traffic': {}}})
            with patch.object(run_highway, 'ROOT', root), \
                 patch.object(run_highway, 'load_configuration', return_value=base), \
                 patch.object(run_highway, 'HighwayScenario', return_value=scenario), \
                 patch.object(run_highway, 'MGeoMapLoader'), \
                 patch('scenario_runner.utils.highway_density.apply_highway_density'), \
                 patch('scenario_runner.adapters.morai_grpc.MoraiGrpcClient', return_value=Mock()):
                with self.assertRaisesRegex(RuntimeError, 'collision'):
                    run_highway.main(['--config', str(config), '--seed', '124', '--validation-events',
                                      '--highway-density', 'competition'])
            incidents = list((root / run_highway.RUNNER_RUNTIME / 'incidents').glob('incident_*'))
            self.assertEqual(len(incidents), 1)
            command = json.loads((incidents[0] / 'rerun.json').read_text(encoding='utf-8'))['command']
            self.assertEqual(command, 'bash AIM2026/Expert/run_highway.sh --seed 124 --validation-events '
                                      '--highway-density competition')


if __name__ == '__main__':
    unittest.main()
