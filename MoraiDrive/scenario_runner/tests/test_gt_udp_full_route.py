"""GT filming must reject pre-existing contacts before starting UDP control."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..tools.run_gt_udp_full_route import GtUdpFullLoop, UdpController
from ..tools import run_gt_udp_full_route as runner
from ..scenarios.segment_modes.modes import ROUNDABOUT_END, HIGHWAY_END


class GtUdpStartupTests(unittest.TestCase):
    def test_highway_probe_needs_no_traffic_light_input(self):
        with patch('run_gt_udp_expert.main', return_value=0) as probe:
            self.assertEqual(runner.main(['--scenario', 'full_loop_highway', '--probe']), 0)
        arguments = probe.call_args.args[0]
        self.assertEqual(arguments[arguments.index('--signal-source') + 1], 'none')

    def test_full_route_cannot_disable_traffic_light_input(self):
        with self.assertRaises(SystemExit):
            runner.main(['--signal-source', 'none', '--check'])

    def test_highway_prepares_segment_spawn_and_udp_expert(self):
        args = SimpleNamespace(scenario='full_loop_highway', speed_cap_kmh=60.,
                               stop_before_tunnel=False, seed=None, signal_source='grpc')
        global_config, _, base = runner.load_configuration()
        with tempfile.TemporaryDirectory() as directory:
            global_config['paths']['runtime_root'] = directory
            config = runner.configure_scenario(base, args)
            self.assertEqual(config['expert_speed_cap_kmh'], 60.)
            context = runner.prepare_controller(global_config, config, args, Path(directory))
            effective = json.loads(Path(context.expert_controller.runtime_config_path).read_text())
            self.assertLessEqual(effective['planning']['velocity_profile']['max_velocity'], 60.)
            self.assertEqual(context.start_tf, (*ROUNDABOUT_END.xyz, ROUNDABOUT_END.yaw_deg))
            self.assertEqual(context.mission_route.first_index, ROUNDABOUT_END.index)
            self.assertEqual(context.mission_route.last_index, HIGHWAY_END.index)
            self.assertTrue(context.mission_mode.zones.highway)
            self.assertFalse(context.mission_mode.zones.roundabout)
            self.assertFalse(context.mission_mode.zones.tunnel)
            self.assertIsInstance(context.expert_controller, UdpController)
            self.assertFalse(context.cfg['ve_stop_before_tunnel'])

    def test_repeated_runs_use_fresh_scenarios_and_configs(self):
        args = SimpleNamespace(scenario='full_loop_highway', repeat=True, max_runs=3)
        factory = Mock()
        instances = [Mock() for _ in range(3)]
        factory.side_effect = instances
        config = {'random_seed': None, 'nested': {'value': 1}}
        global_config = {'paths': {'mgeo_root': 'map', 'runtime_root': 'old'}}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, 'scenario_class', return_value=factory
        ), patch.object(runner, 'MGeoMapLoader'), patch.object(runner.time, 'sleep'):
            runner.run_scenarios(Mock(), global_config, config, args, Path(directory))
        self.assertEqual(factory.call_count, 3)
        for instance in instances:
            instance.run.assert_called_once_with()
        paths = [call.kwargs['global_cfg']['paths']['runtime_root'] for call in factory.call_args_list]
        self.assertEqual(len(set(paths)), 3)
        factory.call_args_list[0].kwargs['scenario_cfg']['nested']['value'] = 2
        self.assertEqual(factory.call_args_list[1].kwargs['scenario_cfg']['nested']['value'], 1)
        self.assertEqual(config['nested']['value'], 1)

    def test_repeat_stops_on_interrupt_or_failure(self):
        for error in (KeyboardInterrupt(), RuntimeError('failed')):
            factory = Mock(return_value=Mock(run=Mock(side_effect=error)))
            args = SimpleNamespace(scenario='full_loop_highway', repeat=True, max_runs=None)
            with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, 'scenario_class', return_value=factory
            ), patch.object(runner, 'MGeoMapLoader'):
                with self.assertRaises(type(error)):
                    runner.run_scenarios(Mock(), {'paths': {'mgeo_root': 'map'}}, {}, args, Path(directory))
            self.assertEqual(factory.call_count, 1)

    def test_clean_start_records_raw_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = dict(x=1., y=2., collision_objects=[])
            context = SimpleNamespace(gt_output=Path(directory),
                                      get_ego_motion_state=lambda: state)
            GtUdpFullLoop.check_startup_contact(context)
            record = json.loads((context.gt_output / 'startup_contact.json').read_text())
            self.assertEqual(record['raw_state'], state)
            self.assertFalse(record['control_backend_started'])

    def test_self_and_foreign_contacts_are_not_silently_filtered(self):
        for contacts in (['Ego'], ['vehicle_2'], ['Ego', 'vehicle_2']):
            with self.subTest(contacts=contacts), tempfile.TemporaryDirectory() as directory:
                context = SimpleNamespace(gt_output=Path(directory),
                    get_ego_motion_state=lambda: dict(collision_objects=contacts))
                with self.assertRaisesRegex(RuntimeError, '출발 전부터'):
                    GtUdpFullLoop.check_startup_contact(context)
                self.assertEqual(json.loads((context.gt_output / 'startup_contact.json').read_text())
                                 ['collision_objects'], sorted(contacts))

    def test_contact_failure_does_not_launch_backend(self):
        context = SimpleNamespace(is_running=lambda: False,
            before_start=Mock(side_effect=RuntimeError('contact')))
        # No process, log, or output directory access may occur after rejection.
        with self.assertRaisesRegex(RuntimeError, 'contact'):
            UdpController.start_process(context)


if __name__ == '__main__':
    unittest.main()
