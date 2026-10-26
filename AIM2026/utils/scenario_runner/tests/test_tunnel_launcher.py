"""Tunnel-segment launcher: same integrated Expert flow as the highway/roundabout launchers, virtual GPS on."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from ..tools import run_tunnel_aim2026 as run_tunnel


class TunnelLauncherTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'Localization').mkdir()
        self.localization = self.root / 'Localization/config.yaml'
        self.localization.write_text(yaml.safe_dump(dict(
            anchor_window_s=1., test_blackout=dict(enabled=False, start_xy=[1., 2.], end_xy=[3., 4.],
                                                    half_width_m=15., margin_m=0.))), encoding='utf-8')
        self.runtime = self.root / 'utils/scenario_runner/.runtime/tunnel_aim2026'
        self.base = self.root / 'config_local.yaml'
        self.base.write_text(yaml.safe_dump(dict(
            Ego=dict(cmd_port=9093), Localization=dict(enabled=True, config='Localization/config.yaml',
                                                       record_dir=None))), encoding='utf-8')

    def generated(self):
        with patch.object(run_tunnel, 'AIM_ROOT', self.root):
            return run_tunnel.virtual_gps_config(self.base, self.root / 'outputs/tunnel_config/one')

    def test_virtual_gps_enables_only_the_private_copy_of_the_tunnel_zone(self):
        before = (self.localization.read_bytes(), self.base.read_bytes())
        runtime = yaml.safe_load(self.generated().read_text(encoding='utf-8'))
        zone = yaml.safe_load((self.root / 'outputs/tunnel_config/one/localization.yaml').read_text())['test_blackout']
        self.assertTrue(zone['enabled'])
        self.assertEqual((zone['start_xy'], zone['end_xy'], zone['half_width_m']), ([1., 2.], [3., 4.], 15.))
        self.assertEqual(runtime['Ego'], {'cmd_port': 9093})
        self.assertTrue(runtime['Localization']['enabled'])
        # Relative to AIM2026, so a Windows interpreter reads the same file.
        self.assertEqual(runtime['Localization']['config'], 'outputs/tunnel_config/one/localization.yaml')
        self.assertTrue((self.root / runtime['Localization']['config']).is_file())
        self.assertEqual((self.localization.read_bytes(), self.base.read_bytes()), before)

    def test_zone_without_boundaries_is_rejected(self):
        self.localization.write_text(yaml.safe_dump(dict(test_blackout=dict(enabled=False))), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.generated()

    def run_main(self, argv, results=None):
        calls = []

        def fake(arguments, variant, options=None):
            calls.append((arguments, variant, options))
            outcome = results[len(calls) - 1] if results else 0
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                patch.object(run_tunnel, 'run_one', fake), patch.object(run_tunnel.time, 'sleep'):
            code = run_tunnel.main(['--config', str(self.base), *argv])
        return code, calls

    def test_single_run_uses_the_generated_config_and_selected_mission(self):
        code, calls = self.run_main(['--variant', 'pedestrian', '--speed-cap-kmh', '40'])
        self.assertEqual(code, 0)
        (arguments, variant, options), = calls
        self.assertEqual(variant, 'pedestrian')
        self.assertEqual(options, dict(fixed_pedestrian=False, vehicle_behavior=None))
        self.assertEqual(arguments[:2], ['--speed-cap-kmh', '40'])
        config = Path(arguments[arguments.index('--config') + 1])
        self.assertEqual(config.name, 'runtime.yaml')
        self.assertTrue(yaml.safe_load(config.read_text())['Localization']['enabled'])

    def test_virtual_gps_can_be_left_off(self):
        code, calls = self.run_main(['--no-virtual-gps'])
        arguments = calls[0][0]
        self.assertEqual(Path(arguments[arguments.index('--config') + 1]), self.base.resolve())
        self.assertFalse(self.runtime.exists())

    def test_generated_files_stay_inside_the_runner(self):
        code, calls = self.run_main([])
        config = Path(calls[0][0][calls[0][0].index('--config') + 1])
        self.assertTrue(config.is_relative_to(self.root / 'utils/scenario_runner/.runtime')
                        if hasattr(config, 'is_relative_to') else
                        str(config).startswith(str(self.root / 'utils/scenario_runner/.runtime')))
        self.assertEqual([p.name for p in self.root.iterdir() if p.is_dir()].count('outputs'), 0)

    def test_repeats_use_fresh_seeds_and_survive_a_failed_run(self):
        code, calls = self.run_main(['--runs', '3', '--seed', '10'],
                                    results=[0, RuntimeError('collision'), 0])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 3)
        seeds = [call[0][call[0].index('--seed') + 1] for call in calls]
        self.assertEqual(seeds, ['10', '11', '12'])

    def test_all_completed_runs_exit_zero_and_interrupt_stops_repeating(self):
        self.assertEqual(self.run_main(['--runs', '2'])[0], 0)
        code, calls = self.run_main(['--runs', '3'], results=[0, KeyboardInterrupt()])
        self.assertEqual((code, len(calls)), (run_tunnel.INTERRUPTED, 2))

    def test_all_rotates_brake_static_pedestrian_vehicle_one_scene_per_run(self):
        code, calls = self.run_main(['--variant', 'all', '--runs', '6', '--seed', '5'])
        self.assertEqual(code, 0)
        self.assertEqual([call[1] for call in calls],
                         ['brake', 'static', 'pedestrian', 'vehicle', 'brake', 'static'])
        seeds = [call[0][call[0].index('--seed') + 1] for call in calls]
        self.assertEqual(seeds, ['5', '6', '7', '8', '9', '10'])
        # One shared virtual-GPS configuration for the whole campaign.
        self.assertEqual(len({call[0][call[0].index('--config') + 1] for call in calls}), 1)

    def test_all_defaults_to_one_full_cycle_and_a_failed_scene_does_not_stop_the_rotation(self):
        code, calls = self.run_main(['--variant', 'all'], results=[0, RuntimeError('library'), 0, 0])
        self.assertEqual((code, [call[1] for call in calls]), (1, ['brake', 'static', 'pedestrian', 'vehicle']))

    def test_all_check_validates_every_scene_once(self):
        code, calls = self.run_main(['--variant', 'all', '--check'])
        self.assertEqual((code, [call[1] for call in calls]), (0, ['brake', 'static', 'pedestrian', 'vehicle']))
        self.assertTrue(all('--check' in call[0] for call in calls))
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), self.assertRaises(SystemExit):
            run_tunnel.main(['--config', str(self.base), '--variant', 'all', '--check', '--runs', '4'])

    def test_scene_options_are_accepted_with_all(self):
        code, calls = self.run_main(['--variant', 'all', '--runs', '4', '--fixed-pedestrian',
                                     '--vehicle-behavior', 'vehicle_slow'])
        self.assertEqual(calls[0][2], dict(fixed_pedestrian=True, vehicle_behavior='vehicle_slow'))

    def test_interrupted_scene_returns_the_interrupt_code_not_success(self):
        from unittest.mock import Mock
        config = self.root / 'ports.yaml'
        config.write_text('Ego: {cmd_port: 9093}', encoding='utf-8')
        scene = Mock()
        scene.run.side_effect = KeyboardInterrupt
        with patch.object(run_tunnel, 'ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                patch.object(run_tunnel, 'load_configuration',
                             return_value=({'paths': {'mgeo_root': 'm'}, 'grpc': {'host': 'h'}}, {}, {})), \
                patch.object(run_tunnel, 'TunnelScenario', return_value=scene), \
                patch.object(run_tunnel, 'MGeoMapLoader'), patch.object(run_tunnel.harness, '_finish_validation'), \
                patch('scenario_runner.adapters.morai_grpc.MoraiGrpcClient'):
            code = run_tunnel.run_one(['--config', str(config)], 'normal')
        self.assertEqual(code, run_tunnel.INTERRUPTED)
        scene.stop_expert_controller.assert_called_once_with()

    def test_an_interrupt_ends_the_whole_campaign_and_is_never_reported_completed(self):
        code, calls = self.run_main(['--variant', 'all', '--runs', '8'], results=[0, run_tunnel.INTERRUPTED])
        self.assertEqual((code, [call[1] for call in calls]), (run_tunnel.INTERRUPTED, ['brake', 'static']))
        code, calls = self.run_main(['--variant', 'brake'], results=[run_tunnel.INTERRUPTED])
        self.assertEqual((code, len(calls)), (run_tunnel.INTERRUPTED, 1))

    def test_check_validates_a_single_run(self):
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), self.assertRaises(SystemExit):
            run_tunnel.main(['--config', str(self.base), '--check', '--runs', '2'])

    def test_each_variant_builds_a_tunnel_scene_for_the_integrated_expert(self):
        config = self.root / 'ports.yaml'
        config.write_text('Ego: {cmd_port: 9093}', encoding='utf-8')
        base = {'highway_merge_static_vehicle': {'enabled': True}, 've_checkpoint': 'old'}
        for variant, mission in run_tunnel.MISSIONS.items():
            if variant == 'static':
                continue  # arms the approved library; covered below without touching it
            with patch.object(run_tunnel, 'ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                    patch.object(run_tunnel, 'load_configuration',
                                 return_value=({'paths': {'mgeo_root': 'm'}}, {}, base)), \
                    patch.object(run_tunnel, 'TunnelScenario') as scenario, \
                    patch.object(run_tunnel, 'MGeoMapLoader'), patch.object(run_tunnel.events, 'prepare_mission'), \
                    patch.object(run_tunnel.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                self.assertEqual(run_tunnel.run_one(['--check', '--config', str(config)], variant), 0)
                selected = scenario.call_args.kwargs['scenario_cfg']
            self.assertEqual(selected['mission_mode'], mission)
            self.assertEqual(selected['drive_control_mode'], 'aimdrive_expert')
            self.assertNotIn('ve_checkpoint', selected)
            self.assertFalse(selected['sync_runtime']['enabled'])
        self.assertIn('ve_checkpoint', base)
        with self.assertRaises(ValueError):
            run_tunnel.run_one(['--check'], 'highway')

    def scene(self, variant, mission='full_loop_tunnel_normal', **options):
        from ..scenarios.segment_modes.modes import get_mission_mode
        scene = object.__new__(run_tunnel.TunnelScenario)
        scene.mission_mode = get_mission_mode(mission)
        scene.cfg = {'tunnel_dynamic': {'enabled': True, 'trial_candidate_id': 'x'},
                     'tunnel_sudden_brake': {'enabled': True}, 'tunnel_vehicle_dynamic': {'enabled': False}}
        scene.variant = variant
        for name, value in options.items():
            setattr(scene, name, value)
        scene._configure_variant()
        return scene

    def test_brake_arms_only_the_sudden_brake_npc_in_the_normal_window(self):
        from ..scenarios.segment_modes.modes import get_mission_mode
        scene = self.scene('brake')
        normal = get_mission_mode('full_loop_tunnel_normal')
        self.assertEqual((scene.mission_mode.start, scene.mission_mode.end), (normal.start, normal.end))
        # Tunnel zone on (the event is gated by it); baseline recorder (normal mission name) stays off.
        self.assertTrue(scene.mission_mode.zones.tunnel)
        self.assertEqual(scene.mission_mode.name, 'full_loop_tunnel_dynamic')
        self.assertEqual((scene.cfg['tunnel_sudden_brake']['enabled'], scene.cfg['tunnel_dynamic']['enabled'],
                          scene.cfg['tunnel_vehicle_dynamic']['enabled']), (True, False, False))

    def test_static_arms_no_npc_event_and_allows_avoidance_cross_track(self):
        scene = self.scene('static')
        self.assertFalse(scene.cfg['tunnel_sudden_brake']['enabled'])
        self.assertFalse(scene.cfg['tunnel_dynamic']['enabled'])
        self.assertEqual(scene.cfg['expert_max_cross_track_error_m'], 5.0)

    def test_pedestrian_uses_the_random_pedestrian_animal_mix_unless_fixed(self):
        mixed = self.scene('pedestrian', 'full_loop_tunnel_dynamic').cfg['tunnel_dynamic']
        self.assertTrue(mixed['random_mix']['enabled'])
        self.assertEqual(mixed['candidate_policy'], 'trial')
        self.assertNotIn('trial_candidate_id', mixed)
        fixed = self.scene('pedestrian', 'full_loop_tunnel_dynamic', fixed_pedestrian=True).cfg['tunnel_dynamic']
        self.assertNotIn('random_mix', fixed)
        self.assertEqual(fixed['trial_candidate_id'], 'x')

    def test_vehicle_behaviour_is_random_unless_chosen(self):
        random = self.scene('vehicle', 'full_loop_tunnel_vehicle').cfg['tunnel_vehicle_dynamic']
        self.assertTrue(random['random_behavior'])
        chosen = self.scene('vehicle', 'full_loop_tunnel_vehicle',
                            vehicle_behavior='vehicle_oncoming').cfg['tunnel_vehicle_dynamic']
        self.assertEqual((chosen['behavior'], chosen['random_behavior']), ('vehicle_oncoming', False))

    def test_variant_specific_options_are_rejected_elsewhere(self):
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime):
            for argv in (['--fixed-pedestrian'], ['--variant', 'pedestrian', '--vehicle-behavior', 'vehicle_slow']):
                with self.assertRaises(SystemExit):
                    run_tunnel.main(['--config', str(self.base), *argv])

    def test_validation_label_names_the_tunnel_controller(self):
        scene = object.__new__(run_tunnel.TunnelScenario)
        scene.cfg, scene.variant = {}, 'normal'
        scene._validation_controller = 'aim2026_roundabout'
        with patch.object(run_tunnel.harness.HighwayScenario, 'prepare_full_loop_route', return_value='route'):
            self.assertEqual(run_tunnel.TunnelScenario.prepare_full_loop_route(scene), 'route')
        self.assertEqual(scene._validation_controller, 'aim2026_tunnel')


if __name__ == '__main__':
    unittest.main()
