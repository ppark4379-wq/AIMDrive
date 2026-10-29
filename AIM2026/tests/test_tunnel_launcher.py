"""run_tunnel.sh: the runner's tunnel launcher with the WSL network setup, repeat runs and diagnosis (no MORAI)."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from AIM2026.Expert import run_repeat, run_tunnel

ROTATION = ('brake', 'static', 'pedestrian', 'vehicle')


def fake_tunnel():
    return SimpleNamespace(MISSIONS={'normal': 'n', 'brake': 'n', 'static': 'n', 'pedestrian': 'p', 'vehicle': 'v'},
                           ROTATION=ROTATION, RUNTIME=Path('runtime'), main=Mock(return_value=0))


class TunnelWrapperTests(unittest.TestCase):
    def call(self, argv, tunnel=None):
        tunnel = tunnel or fake_tunnel()
        with patch.object(run_tunnel, 'load_tunnel_launcher', return_value=tunnel), \
             patch.object(run_tunnel, 'wsl_network', Mock(configure_launcher=Mock(return_value=False))):
            code = run_tunnel.main(argv)
        return code, tunnel

    def test_the_real_launcher_loads_and_has_the_scenes_this_wrapper_expects(self):
        tunnel = run_tunnel.load_tunnel_launcher()
        self.assertEqual(tuple(tunnel.ROTATION), ROTATION)
        self.assertEqual(set(tunnel.MISSIONS), {'normal', *ROTATION})
        self.assertTrue(callable(tunnel.main))

    def test_a_single_run_goes_straight_to_the_tunnel_launcher_with_its_other_options(self):
        code, tunnel = self.call(['--variant', 'brake', '--seed', '7', '--validation-events', '--no-virtual-gps'])
        self.assertEqual(code, 0)
        (argv,), _ = tunnel.main.call_args
        self.assertEqual(argv[argv.index('--variant') + 1], 'brake')
        self.assertEqual(argv[argv.index('--seed') + 1], '7')
        self.assertIn('--validation-events', argv)
        self.assertIn('--no-virtual-gps', argv)
        self.assertEqual(tunnel.main.call_count, 1)

    def test_repeat_runs_are_batched_with_one_scene_label_and_the_tunnel_scenario_folder(self):
        tunnel = fake_tunnel()
        with patch.object(run_repeat, 'run_many', return_value=0) as many:
            self.call(['--variant', 'pedestrian', '--runs', '5', '--seed', '100', '--run-timeout-s', '600'], tunnel)
        options = many.call_args.args[2]
        self.assertEqual((options.runs, options.seed, options.run_timeout_s), (5, 100, 600))
        self.assertEqual(many.call_args.kwargs['label'], 'tunnel_pedestrian')
        self.assertEqual(many.call_args.kwargs['scenarios_dir'], Path('runtime') / 'scenarios')
        forwarded = many.call_args.args[1]
        self.assertNotIn('--runs', forwarded)
        self.assertNotIn('--variant', forwarded)                # the variant is chosen per run

    def test_each_repeat_run_calls_the_tunnel_launcher_once_with_a_single_scene_and_its_seed(self):
        tunnel = fake_tunnel()

        def many(once, argv, options, **kwargs):
            for index in range(options.runs):
                once(list(argv) + ['--seed', str(options.seed + index)])
            return 0
        with patch.object(run_repeat, 'run_many', side_effect=many):
            self.call(['--variant', 'vehicle', '--runs', '3', '--seed', '10'], tunnel)
        calls = [call.args[0] for call in tunnel.main.call_args_list]
        self.assertEqual(len(calls), 3)
        for index, argv in enumerate(calls):
            self.assertEqual(argv[argv.index('--variant') + 1], 'vehicle')
            self.assertEqual(argv[argv.index('--seed') + 1], str(10 + index))
            self.assertNotIn('--runs', argv)

    def test_variant_all_rotates_the_scenes_and_defaults_to_one_pass(self):
        tunnel = fake_tunnel()
        captured = {}

        def many(once, argv, options, **kwargs):
            captured['runs'], captured['label'] = options.runs, kwargs['label']
            for _ in range(options.runs + 1):
                once(list(argv))                                # one extra: the rotation wraps around
            return 0
        with patch.object(run_repeat, 'run_many', side_effect=many):
            self.call(['--variant', 'all'], tunnel)
        self.assertEqual((captured['runs'], captured['label']), (4, 'tunnel_all'))
        scenes = [call.args[0][call.args[0].index('--variant') + 1] for call in tunnel.main.call_args_list]
        self.assertEqual(scenes, [*ROTATION, ROTATION[0]])

    def test_an_explicit_runs_overrides_the_default_pass_of_variant_all(self):
        with patch.object(run_repeat, 'run_many', return_value=0) as many:
            self.call(['--variant', 'all', '--runs', '8'])
        self.assertEqual(many.call_args.args[2].runs, 8)

    def test_failure_diagnosis_is_on_by_default_for_the_tunnel_too(self):
        _, tunnel = self.call(['--variant', 'brake'])
        self.assertIn('--validation-events', tunnel.main.call_args.args[0])
        _, tunnel = self.call(['--variant', 'brake', '--no-validation-events'])
        argv = tunnel.main.call_args.args[0]
        self.assertNotIn('--validation-events', argv)
        self.assertNotIn('--no-validation-events', argv)               # the runner's own parser does not know it
        _, tunnel = self.call(['--variant', 'brake', '--check'])
        self.assertNotIn('--validation-events', tunnel.main.call_args.args[0])

    def test_check_validates_once_and_ignores_runs(self):
        code, tunnel = self.call(['--variant', 'brake', '--check', '--runs', '4', '--seed', '3'])
        self.assertEqual(code, 0)
        self.assertEqual(tunnel.main.call_count, 1)
        (argv,), _ = tunnel.main.call_args
        self.assertIn('--check', argv)
        self.assertNotIn('--runs', argv)

    def test_network_options_are_not_forwarded_to_the_tunnel_launcher(self):
        tunnel = fake_tunnel()
        wsl = Mock()

        def configure(args, root, parser):
            args.config = Path(args.config).parent / 'wsl.yaml'
            return False

        def add_arguments(parser):
            parser.add_argument('--morai-build', default='competition')
            parser.add_argument('--status-bytes', type=int, default=None)
            parser.add_argument('--prepare-network', action='store_true')
        wsl.configure_launcher.side_effect = configure
        wsl.add_arguments.side_effect = add_arguments
        with patch.object(run_tunnel, 'load_tunnel_launcher', return_value=tunnel), \
             patch.object(run_tunnel, 'wsl_network', wsl):
            run_tunnel.main(['--status-bytes', '229', '--morai-build', 'local', '--seed', '1'])
        (argv,), _ = tunnel.main.call_args
        for name in ('--status-bytes', '--morai-build', '--prepare-network'):
            self.assertNotIn(name, argv)
        self.assertEqual(Path(argv[argv.index('--config') + 1]).name, 'wsl.yaml')

    def test_prepare_network_stops_before_any_scene(self):
        tunnel = fake_tunnel()
        wsl = Mock()
        wsl.configure_launcher.return_value = True
        with patch.object(run_tunnel, 'load_tunnel_launcher', return_value=tunnel), \
             patch.object(run_tunnel, 'wsl_network', wsl):
            self.assertEqual(run_tunnel.main(['--prepare-network']), 0)
        tunnel.main.assert_not_called()

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.call(['--variant', 'nope'])


class RerunCommandTests(unittest.TestCase):
    """The incident bundle names the launcher to rerun, and the Expert launchers' timeout/stall aborts are captured."""
    def rerun_command(self, controller, **scene):
        import json
        import tempfile
        from scenario_runner.validation.finish import finish_validation_run
        from scenario_runner.validation.run_log import ValidationRunLog

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = SimpleNamespace(run_instance_id='run-124', run_seed=124, cfg={}, global_cfg={'paths': {}},
                                       _runner_scenario_name='scene', _validation_controller=controller, **scene)
            log = ValidationRunLog(root / 'validation_runs', 'scene')
            log.enable_incidents(root / 'incidents', scenario, run_options=['--seed', '124', '--validation-events'])
            log.observe('run-124', 180., {'x': 1., 'y': 2., 'z': 0., 'speed_mps': 0., 'collision_objects': []})
            error = RuntimeError('EGO progress watchdog abort: reason=no_route_progress stalled=180.0s')
            finish_validation_run(log, scenario, 'exception', error)
            bundles = list((root / 'incidents').glob('incident_*_stall'))
            self.assertEqual(len(bundles), 1)
            return json.loads((bundles[0] / 'rerun.json').read_text(encoding='utf-8'))['command']

    def test_tunnel_incident_reruns_its_own_scene(self):
        self.assertEqual(self.rerun_command('aim2026_tunnel', variant='pedestrian'),
                         'bash AIM2026/Expert/run_tunnel.sh --variant pedestrian --seed 124 --validation-events')



if __name__ == '__main__':
    unittest.main()
