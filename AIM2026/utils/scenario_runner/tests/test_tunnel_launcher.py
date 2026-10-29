"""Tunnel-segment launcher: same integrated Expert flow as the highway/roundabout launchers, virtual GPS on."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from ..tools import run_tunnel_aim2026 as run_tunnel


def setUpModule():
    # The real cleanup asks Windows PowerShell for the UDP port owners (seconds per run); it has its own tests.
    patcher = patch.object(run_tunnel, 'clear_stale_sensor_receivers')
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


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
        self.limits = []
        self.library = self.root / 'approved_candidates.json'
        self.library.write_text('{}', encoding='utf-8')
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

    def run_main(self, argv, results=None, library=None, archive=None):
        calls = []

        def fake(arguments, variant, options=None, record=None, limits=None):
            calls.append((arguments, variant, options))
            self.limits.append(limits)
            if record is not None:
                record.update(run_seed=1000 + len(calls), run_instance_id='id{}'.format(len(calls)),
                              output=str(self.root / 'run{}'.format(len(calls))))
            outcome = results[len(calls) - 1] if results else 0
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        library = self.library if library is None else library
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                patch.object(run_tunnel, 'run_one', fake), patch.object(run_tunnel.time, 'sleep'), \
                patch.object(run_tunnel, 'static_library', return_value=library), \
                patch.object(run_tunnel, 'static_archive', return_value=archive), \
                patch.object(run_tunnel, 'restore_lf_evidence', return_value=[]):
            code = run_tunnel.main(['--config', str(self.base), *argv])
        return code, calls

    def test_single_run_uses_the_generated_config_and_selected_mission(self):
        code, calls = self.run_main(['--variant', 'pedestrian', '--speed-cap-kmh', '40'])
        self.assertEqual(code, 0)
        (arguments, variant, options), = calls
        self.assertEqual(variant, 'pedestrian')
        self.assertEqual(options, dict(fixed_pedestrian=False, vehicle_behavior=None,
                                       pedestrian_reaction_s=2.0, pedestrian_margin_m=None))
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
        self.assertEqual(calls[0][2], dict(fixed_pedestrian=True, vehicle_behavior='vehicle_slow',
                                           pedestrian_reaction_s=2.0, pedestrian_margin_m=None))

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

    def result_rows(self, argv, results):
        code, calls = self.run_main(argv, results=results)
        files = sorted((self.runtime / 'results').glob('tunnel-*.jsonl'))
        rows = [json.loads(line) for line in files[-1].read_text(encoding='utf-8').splitlines()] if files else []
        return code, calls, rows

    def test_every_repeated_run_is_written_to_the_results_file_with_seed_and_reason(self):
        code, calls, rows = self.result_rows(
            ['--variant', 'all', '--runs', '4', '--seed', '50'],
            [0, RuntimeError('AIMDrive expert collision during scenario verification: [x]'),
             RuntimeError('AIMDrive expert left the official route: cte=11.00m'), 0])
        self.assertEqual(code, 1)
        self.assertEqual([r['run'] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r['scene'] for r in rows], ['brake', 'static', 'pedestrian', 'vehicle'])
        self.assertEqual([r['result'] for r in rows], ['completed', 'collision', 'off_route', 'completed'])
        self.assertIn('collision', rows[1]['reason'])
        self.assertEqual([r['requested_seed'] for r in rows], [50, 51, 52, 53])
        self.assertEqual([r['seed'] for r in rows], [1001, 1002, 1003, 1004])
        self.assertEqual(rows[1]['output'], str(self.root / 'run2'))
        self.assertIsNone(rows[0]['reason'])

    def test_results_file_path_can_be_chosen(self):
        target = self.root / 'mine.jsonl'
        self.run_main(['--runs', '2', '--results', str(target)])
        self.assertEqual(len(target.read_text(encoding='utf-8').splitlines()), 2)

    def test_a_run_over_its_time_limit_is_a_failed_run_and_the_campaign_goes_on(self):
        code, calls, rows = self.result_rows(['--runs', '3'], [0, run_tunnel.RoundTimeout(300.), 0])
        self.assertEqual((code, len(calls)), (1, 3))
        self.assertEqual([r['result'] for r in rows], ['completed', 'round_timeout', 'completed'])

    def test_an_interrupt_is_recorded_and_ends_the_campaign(self):
        code, calls, rows = self.result_rows(['--runs', '5'], [0, KeyboardInterrupt()])
        self.assertEqual((code, len(calls)), (run_tunnel.INTERRUPTED, 2))
        self.assertEqual([r['result'] for r in rows], ['completed', 'interrupted'])

    def test_failures_are_classified_from_the_scene_messages(self):
        for text, reason in (('AIMDrive expert collision during scenario verification: x', 'collision'),
                             ('AIMDrive expert left the official route: cte=3m', 'off_route'),
                             ('EGO progress watchdog abort: no progress', 'stalled'),
                             ('EGO stall abort: less than 2 m', 'stalled'),
                             ('K-City full-loop timeout: dist=3m', 'timeout'),
                             ('something else', 'error')):
            self.assertEqual(run_tunnel.classify_failure(RuntimeError(text)), reason)
        self.assertEqual(run_tunnel.classify_failure(run_tunnel.RoundTimeout(1.)), 'round_timeout')

    def test_limits_reach_every_run_and_can_be_turned_off(self):
        self.run_main(['--runs', '2'])
        self.assertEqual(self.limits, [dict(round_timeout_s=300., stall_s=60.)] * 2)
        self.limits.clear()
        self.run_main(['--runs', '1', '--round-timeout-s', '0', '--stall-s', '0'])
        self.assertEqual(self.limits, [dict(round_timeout_s=0., stall_s=0.)])
        with self.assertRaises(SystemExit):
            self.run_main(['--round-timeout-s', '-1'])

    def test_a_single_run_over_its_time_limit_exits_nonzero(self):
        code, _ = self.run_main(['--variant', 'brake'], results=[run_tunnel.RoundTimeout(1.)])
        self.assertEqual(code, 1)

    def test_the_stall_watch_fails_a_run_without_progress_but_not_a_slow_one(self):
        from unittest.mock import Mock
        scene = self.scene('pedestrian', 'full_loop_tunnel_dynamic')
        scene.stop_drive_control = Mock()
        for elapsed in range(0, 60, 5):
            scene.watch_stall(float(elapsed), 10.0 + (0.5 if elapsed > 30 else 0.))   # 0.5 m is not progress
        with self.assertRaisesRegex(RuntimeError, 'EGO stall abort'):
            scene.watch_stall(60., 10.5)
        scene.stop_drive_control.assert_called_once_with()
        moving = self.scene('pedestrian', 'full_loop_tunnel_dynamic')
        moving.stop_drive_control = Mock()
        for elapsed in range(0, 600, 5):
            moving.watch_stall(float(elapsed), elapsed * 0.5)    # 2.5 m per 5 s, never stalled
        moving.stop_drive_control.assert_not_called()
        off = self.scene('pedestrian', 'full_loop_tunnel_dynamic', stall_s=0.)
        off.watch_stall(0., 1.)
        off.watch_stall(1000., 1.)

    def test_the_wall_clock_limit_stops_a_hung_run_and_still_cleans_up(self):
        from unittest.mock import Mock
        import time as real_time
        config = self.root / 'ports.yaml'
        config.write_text('Ego: {cmd_port: 9093}', encoding='utf-8')
        scene = Mock()
        scene.run.side_effect = lambda: real_time.sleep(5)
        record = {}
        with patch.object(run_tunnel, 'ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                patch.object(run_tunnel, 'load_configuration',
                             return_value=({'paths': {'mgeo_root': 'm'}, 'grpc': {'host': 'h'}}, {}, {})), \
                patch.object(run_tunnel, 'TunnelScenario', return_value=scene), \
                patch.object(run_tunnel, 'MGeoMapLoader'), patch.object(run_tunnel.harness, '_finish_validation') as finish, \
                patch('scenario_runner.adapters.morai_grpc.MoraiGrpcClient'):
            started = real_time.monotonic()
            with self.assertRaises(run_tunnel.RoundTimeout):
                run_tunnel.run_one(['--config', str(config)], 'normal', record=record,
                                   limits=dict(round_timeout_s=0.2, stall_s=0.))
        self.assertLess(real_time.monotonic() - started, 3)
        scene.stop_expert_controller.assert_called_once_with()
        self.assertEqual(finish.call_args[0][2], 'exception')
        self.assertIsInstance(finish.call_args[0][3], run_tunnel.RoundTimeout)
        self.assertIn('output', record)
        import signal as real_signal
        self.assertEqual(real_signal.getitimer(real_signal.ITIMER_REAL)[0], 0.)

    def test_skip_leaves_scenes_out_of_the_rotation_and_the_default_cycle(self):
        code, calls = self.run_main(['--variant', 'all', '--runs', '6', '--skip', 'brake'])
        self.assertEqual([call[1] for call in calls], ['static', 'pedestrian', 'vehicle'] * 2)
        code, calls = self.run_main(['--variant', 'all', '--skip', 'brake'])
        self.assertEqual([call[1] for call in calls], ['static', 'pedestrian', 'vehicle'])
        code, calls = self.run_main(['--variant', 'all', '--runs', '4', '--skip', 'static', 'vehicle'])
        self.assertEqual([call[1] for call in calls], ['brake', 'pedestrian', 'brake', 'pedestrian'])

    def test_a_missing_static_library_leaves_static_out_of_the_rotation_instead_of_failing(self):
        missing = self.root / 'no/approved_candidates.json'
        code, calls = self.run_main(['--variant', 'all', '--runs', '6'], library=missing)
        self.assertEqual(code, 0)
        self.assertEqual([call[1] for call in calls], ['brake', 'pedestrian', 'vehicle', 'brake', 'pedestrian', 'vehicle'])
        # The default cycle shrinks with the rotation, and an explicit --skip static changes nothing.
        self.assertEqual([call[1] for call in self.run_main(['--variant', 'all'], library=missing)[1]],
                         ['brake', 'pedestrian', 'vehicle'])
        self.assertEqual([call[1] for call in self.run_main(['--variant', 'all', '--skip', 'static'], library=missing)[1]],
                         ['brake', 'pedestrian', 'vehicle'])

    def test_the_static_scene_alone_without_its_library_stops_with_an_explanation(self):
        missing = self.root / 'no/runtime/a/b/approved_candidates.json'
        with patch('sys.stderr') as stderr, self.assertRaises(SystemExit) as raised:
            self.run_main(['--variant', 'static'], library=missing)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('static layout library is not installed', ''.join(str(c.args[0]) for c in stderr.write.call_args_list))

    def test_scenes_other_than_static_never_need_the_library(self):
        missing = self.root / 'no/approved_candidates.json'
        for variant in ('brake', 'pedestrian', 'vehicle', 'normal'):
            code, calls = self.run_main(['--variant', variant], library=missing)
            self.assertEqual((code, [call[1] for call in calls]), (0, [variant]))

    def morai_folder(self, *names):
        bundle = self.root / 'morai/Sim_Data/Bundle/v.S4.test'
        bundle.mkdir(parents=True)
        for name in names:
            (bundle / name).write_text('', encoding='utf-8')
        return self.root / 'morai'

    def test_installed_models_are_read_from_the_morai_bundle_and_normalized(self):
        folder = self.morai_folder('rubber-cone.obstacle', 'Steel_Barricade.obstacle', 'steel_barricade.obstacle.manifest',
                                   'stinger_2020.vehicle', 'woman1.pedestrian', 'sensor.scene')
        models = run_tunnel.installed_models(folder)
        self.assertEqual(models, frozenset({'rubbercone', 'steelbarricade', 'stinger2020', 'woman1'}))
        self.assertEqual(run_tunnel.installed_models(folder / 'Sim_Data/Bundle'), models)
        self.assertEqual(run_tunnel.installed_models(self.root / 'nowhere'), frozenset())

    def test_layouts_with_a_model_the_morai_lacks_are_left_out(self):
        layout = [dict(model='Steel_Barricade'), dict(model='WoodBox')]
        installed = frozenset({'steelbarricade', 'woodbox', 'tree01'})
        self.assertTrue(run_tunnel.competition_static.layout_models_available(layout, installed))
        self.assertFalse(run_tunnel.competition_static.layout_models_available(
            layout + [dict(model='Rubber_Cone')], installed))
        self.assertFalse(run_tunnel.competition_static.layout_models_available(
            [dict(model='2020_Kia_Stinger')], frozenset({'stinger2020'})))      # only an exact name match counts
        self.assertTrue(run_tunnel.competition_static.layout_models_available(layout + [dict(model='Rubber_Cone')], None))

    def test_morai_assets_hand_the_installed_models_to_the_static_scene_only_when_asked(self):
        folder = self.morai_folder('woodbox.obstacle', 'cargobox.obstacle')
        code, calls = self.run_main(['--variant', 'static', '--morai-assets', str(folder)])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][2]['installed_models'], frozenset({'woodbox', 'cargobox'}))
        code, calls = self.run_main(['--variant', 'static'])
        self.assertNotIn('installed_models', calls[0][2])

    def test_the_competition_morai_model_list_filters_the_static_scene_without_any_folder(self):
        models = run_tunnel.competition_models()
        self.assertTrue({'bollard04', 'concretebarricade01', 'lane_separation_bar'.replace('_', '')} <= models)
        self.assertFalse({'rubbercone', 'colorcone', 'barbedwirefence'} & models)   # not shipped
        self.assertFalse({'woodbox', 'cargobox', 'pefirewallwhite', 'pefirewallorange', 'pedrum', 'redbarrel', 'yellowbarrel',
                           'steelbarricade'} & models)   # shipped, but fall through the tunnel floor
        code, calls = self.run_main(['--variant', 'static', '--competition-morai'])
        self.assertEqual((code, calls[0][2]['installed_models']), (0, models))
        with patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
            self.run_main(['--variant', 'static', '--competition-morai', '--morai-assets', str(self.root)])
        self.assertEqual(raised.exception.code, 2)

    def test_competition_morai_without_the_static_scene_needs_no_model_filter(self):
        code, calls = self.run_main(['--variant', 'brake', '--competition-morai'])
        self.assertEqual((code, 'installed_models' in calls[0][2]), (0, False))

    def test_competition_morai_sets_the_competition_vehicle_interface_on_a_copy_of_the_config(self):
        self.base.write_text(yaml.safe_dump(dict(
            Ego=dict(ip='10.0.0.5', cmd_port=9093, cmd_layout='front_rear59', local_status_bytes=245),
            Control=dict(positive_steer='right', max_speed_mps=None),
            Localization=dict(enabled=True, config='Localization/config.yaml'))), encoding='utf-8')
        before = self.base.read_text(encoding='utf-8')
        code, calls = self.run_main(['--variant', 'pedestrian', '--competition-morai'])
        used = Path(calls[0][0][calls[0][0].index('--config') + 1])
        document = yaml.safe_load(used.read_text(encoding='utf-8'))
        self.assertEqual(document['Ego'], dict(ip='10.0.0.5', cmd_port=9093, cmd_layout='single_steer55'))
        self.assertEqual(document['Control'], dict(positive_steer='left', max_speed_mps=None))
        self.assertEqual(self.base.read_text(encoding='utf-8'), before)           # the PC's own file is never edited

    def test_without_competition_morai_the_config_values_are_left_alone(self):
        self.base.write_text(yaml.safe_dump(dict(
            Ego=dict(cmd_layout='front_rear59', local_status_bytes=245), Control=dict(positive_steer='right'),
            Localization=dict(enabled=True, config='Localization/config.yaml'))), encoding='utf-8')
        code, calls = self.run_main(['--variant', 'pedestrian'])
        document = yaml.safe_load(Path(calls[0][0][calls[0][0].index('--config') + 1]).read_text(encoding='utf-8'))
        self.assertEqual((document['Ego']['cmd_layout'], document['Ego']['local_status_bytes'],
                          document['Control']['positive_steer']), ('front_rear59', 245, 'right'))

    def test_morai_assets_need_the_static_scene_and_a_folder_with_assets(self):
        folder = self.morai_folder('woodbox.obstacle')
        for argv in (['--variant', 'pedestrian', '--morai-assets', str(folder)],
                     ['--variant', 'all', '--skip', 'static', '--morai-assets', str(folder)],
                     ['--variant', 'static', '--morai-assets', str(self.root / 'empty')]):
            with patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
                self.run_main(argv)
            self.assertEqual(raised.exception.code, 2)

    @staticmethod
    def fake_world(actors, sticky=()):
        """A MORAI stand-in: DestroyActor removes an actor unless its id is in ``sticky``."""
        world = dict(actors)
        calls = []

        class Stub:
            def GetAllActorsState(self, query, timeout=None):
                return SimpleNamespace(states=[
                    SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=key), object_type=kind))
                    for key, kind in world.items()])

            def DestroyActor(self, info):
                calls.append((info.id.value, info.object_type))
                if info.id.value not in sticky:
                    world.pop(info.id.value, None)

        class Info:
            def __init__(self):
                self.id = SimpleNamespace(value='')
                self.object_type = 0
        return Stub(), Info, world, calls

    def test_leftover_actors_are_all_removed_except_the_ego(self):
        stub, info, world, calls = self.fake_world({'Ego': 1, '1': 3, '2': 2, '3': 3})
        removed = run_tunnel.clear_actors_with(stub, lambda **kw: None, info, pause_s=0)
        self.assertEqual(sorted(removed), [('1', 3), ('2', 2), ('3', 3)])
        self.assertEqual(world, {'Ego': 1})
        self.assertEqual(run_tunnel.clear_actors_with(stub, lambda **kw: None, info, pause_s=0), [])   # nothing to do

    def test_an_actor_that_will_not_go_is_reported_instead_of_ignored(self):
        stub, info, world, calls = self.fake_world({'Ego': 1, '7': 3}, sticky={'7'})
        with self.assertRaisesRegex(RuntimeError, 'could not be removed: 7'):
            run_tunnel.clear_actors_with(stub, lambda **kw: None, info, attempts=2, pause_s=0)

    def test_clear_leftovers_runs_before_every_run_only_when_asked(self):
        with patch.object(run_tunnel, 'clear_leftover_actors') as clear:
            self.run_main(['--variant', 'pedestrian', '--runs', '3'])
            self.assertEqual(clear.call_count, 0)
            self.run_main(['--variant', 'pedestrian', '--runs', '3', '--clear-leftovers'])
            self.assertEqual(clear.call_count, 3)
            clear.reset_mock()
            self.run_main(['--variant', 'pedestrian', '--clear-leftovers'])          # a single run too
            self.assertEqual(clear.call_count, 1)

    def test_a_failing_clean_up_does_not_stop_the_run(self):
        with patch.object(run_tunnel, 'clear_leftover_actors', side_effect=RuntimeError('boom')):
            code, calls = self.run_main(['--variant', 'pedestrian', '--runs', '2', '--clear-leftovers'])
        self.assertEqual((code, len(calls)), (0, 2))

    def test_a_present_library_keeps_static_and_restores_the_evidence_line_ends(self):
        calls = []
        with patch.object(run_tunnel, 'restore_lf_evidence', side_effect=lambda: calls.append(1) or [Path('a.json')]) as _:
            # run_main patches restore_lf_evidence itself, so drive main() directly.
            def fake(arguments, variant, options=None, record=None, limits=None):
                return 0
            with patch.object(run_tunnel, 'AIM_ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime), \
                    patch.object(run_tunnel, 'run_one', fake), patch.object(run_tunnel.time, 'sleep'), \
                    patch.object(run_tunnel, 'static_library', return_value=self.library):
                run_tunnel.main(['--config', str(self.base), '--variant', 'all', '--runs', '4'])
        self.assertEqual(calls, [1])

    def make_archive(self, kind, top_level=True, extra=None):
        import tarfile, zipfile
        source = self.root / 'pack_src'
        (source / 'mass_search_20261008/search').mkdir(parents=True, exist_ok=True)
        (source / 'mass_search_20261008/search/approved_candidates.json').write_text('{}', encoding='utf-8')
        prefix = 'tunnel_static_trials/' if top_level else ''
        archive = self.root / ('tunnel_static_trials.zip' if kind == 'zip' else 'tunnel_static_trials.tar.gz')
        entries = [(source / 'mass_search_20261008/search/approved_candidates.json',
                    prefix + 'mass_search_20261008/search/approved_candidates.json')]
        if extra:
            entries.append((source / 'mass_search_20261008/search/approved_candidates.json', extra))
        if kind == 'zip':
            with zipfile.ZipFile(archive, 'w') as package:
                for path, name in entries:
                    package.write(path, name)
        else:
            with tarfile.open(archive, 'w:gz') as package:
                for path, name in entries:
                    package.add(path, name)
        return archive

    def test_the_static_library_is_unpacked_from_a_tar_or_zip_with_or_without_the_top_folder(self):
        for kind in ('tar', 'zip'):
            for top_level in (True, False):
                runtime = self.root / 'unpack_{}_{}/.runtime'.format(kind, top_level)
                run_tunnel.install_static_library(self.make_archive(kind, top_level), runtime)
                self.assertTrue((runtime / 'tunnel_static_trials/mass_search_20261008/search/approved_candidates.json')
                                .is_file(), (kind, top_level))

    def test_an_archive_that_would_write_outside_the_runtime_folder_is_refused(self):
        for kind in ('tar', 'zip'):
            runtime = self.root / 'unsafe_{}/.runtime'.format(kind)
            with self.assertRaises(ValueError):
                run_tunnel.install_static_library(
                    self.make_archive(kind, extra='tunnel_static_trials/../../escaped.json'), runtime)
            self.assertFalse((runtime.parent / 'escaped.json').exists())

    def test_the_archive_is_found_beside_the_repository_and_in_the_wsl_user_folder(self):
        beside = self.root / 'beside'
        (beside / 'AIMDrive').mkdir(parents=True)
        self.assertIsNone(run_tunnel.static_archive(beside / 'AIMDrive'))
        (beside / 'tunnel_static_trials.zip').write_bytes(b'')
        self.assertEqual(run_tunnel.static_archive(beside / 'AIMDrive'), beside / 'tunnel_static_trials.zip')
        # C:\Users\<name>\Desktop\AIMDrive seen from WSL: the user's folder and Desktop are searched too.
        with patch.object(Path, 'is_file', lambda path: str(path) == '/mnt/c/Users/kim/tunnel_static_trials.tar.gz'):
            self.assertEqual(str(run_tunnel.static_archive(Path('/mnt/c/Users/kim/Desktop/AIMDrive'))),
                             '/mnt/c/Users/kim/tunnel_static_trials.tar.gz')

    def test_a_missing_library_is_unpacked_from_the_archive_and_static_stays_in_the_rotation(self):
        library = self.root / 'rt/a/b/c/approved_candidates.json'
        installed = []

        def install(archive, runtime):
            installed.append((archive, runtime))
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_text('{}', encoding='utf-8')
        with patch.object(run_tunnel, 'install_static_library', install):
            code, calls = self.run_main(['--variant', 'all', '--runs', '4'], library=library,
                                        archive=self.root / 'x.tar.gz')
        self.assertEqual([call[1] for call in calls], ['brake', 'static', 'pedestrian', 'vehicle'])
        self.assertEqual(installed, [(self.root / 'x.tar.gz', library.parents[3])])

    def test_an_unreadable_archive_leaves_static_out_instead_of_stopping_the_run(self):
        bad = self.root / 'broken.tar.gz'
        bad.write_bytes(b'not an archive')
        code, calls = self.run_main(['--variant', 'all', '--runs', '3'], library=self.root / 'none/a/b/c/x.json', archive=bad)
        self.assertEqual((code, [call[1] for call in calls]), (0, ['brake', 'pedestrian', 'vehicle']))

    def test_restore_lf_evidence_leaves_the_crlf_map_data_alone(self):
        roots = [Path(root) for root in run_tunnel.restore_lf_evidence.__defaults__[0]]
        self.assertEqual([root.relative_to(run_tunnel.RUNNER_ROOT).as_posix() for root in roots],
                         ['data/geometry', 'config'])
        # git stores these two map files with CRLF; rewriting them would show them as modified everywhere.
        map_data = run_tunnel.RUNNER_ROOT / 'data/R_KR_PR_K-city_2025'
        self.assertFalse(any(map_data in path.parents for root in roots for path in root.rglob('*.json')))

    def test_restore_lf_evidence_rewrites_only_the_files_with_crlf(self):
        crlf, lf = self.root / 'data/a/crlf.json', self.root / 'config/lf.json'
        for path, content in ((crlf, b'{\r\n  "a": 1\r\n}\r\n'), (lf, b'{\n  "a": 1\n}\n')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        other = self.root / 'data/keep.txt'
        other.write_bytes(b'x\r\ny\r\n')
        stamp = lf.stat().st_mtime_ns
        fixed = run_tunnel.restore_lf_evidence((self.root / 'data', self.root / 'config'))
        self.assertEqual(fixed, [crlf])
        self.assertEqual(crlf.read_bytes(), b'{\n  "a": 1\n}\n')
        self.assertEqual(other.read_bytes(), b'x\r\ny\r\n')
        self.assertEqual(lf.stat().st_mtime_ns, stamp)
        self.assertEqual(run_tunnel.restore_lf_evidence((self.root / 'data', self.root / 'config')), [])

    def test_skip_is_only_for_all_and_cannot_remove_every_scene(self):
        with self.assertRaises(SystemExit):
            self.run_main(['--variant', 'brake', '--skip', 'static'])
        with self.assertRaises(SystemExit):
            self.run_main(['--variant', 'all', '--skip', 'brake', 'static', 'pedestrian', 'vehicle'])

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

    def test_normal_keeps_the_empty_window_and_zones_but_not_the_ros_baseline_recorder(self):
        from ..scenarios.segment_modes import tunnel_baseline
        from ..scenarios.segment_modes.modes import get_mission_mode
        scene = self.scene('normal')
        normal = get_mission_mode('full_loop_tunnel_normal')
        self.assertEqual((scene.mission_mode.start, scene.mission_mode.end, scene.mission_mode.zones),
                         (normal.start, normal.end, normal.zones))
        self.assertFalse(tunnel_baseline._enabled(scene))

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

    def test_brake_lowers_only_this_scene_spawn_gap_below_the_window_gap(self):
        scene = self.scene('brake')
        gap = scene.cfg['tunnel_sudden_brake']['minimum_spawn_gap_m']
        self.assertEqual(gap, run_tunnel.BRAKE_TEST_MIN_GAP_M)
        self.assertLess(gap, 27.1)           # the planned gap of the 22.1 m run-up window
        other = self.scene('static')
        self.assertNotIn('minimum_spawn_gap_m', other.cfg['tunnel_sudden_brake'])

    def test_brake_provides_the_fixed_npc_model_the_vehicle_helper_reads(self):
        scene = self.scene('brake')
        self.assertEqual(scene.highway_fixed_model, 'DEFAULT_SEDAN')
        scene = object.__new__(run_tunnel.TunnelScenario)
        from ..scenarios.segment_modes.modes import get_mission_mode
        scene.mission_mode, scene.variant = get_mission_mode('full_loop_tunnel_normal'), 'brake'
        scene.cfg = {'tunnel_sudden_brake': {'model': 'Other'}, 'highway_npc': {'model': 'Highway'}}
        scene._configure_variant()
        self.assertEqual(scene.highway_fixed_model, 'Other')
        scene.cfg = {'highway_npc': {'model': 'Highway'}}
        scene._configure_variant()
        self.assertEqual(scene.highway_fixed_model, 'Highway')

    def test_static_arms_no_npc_event_and_allows_avoidance_cross_track(self):
        scene = self.scene('static')
        self.assertFalse(scene.cfg['tunnel_sudden_brake']['enabled'])
        self.assertFalse(scene.cfg['tunnel_dynamic']['enabled'])
        self.assertEqual(scene.cfg['expert_max_cross_track_error_m'], 10.0)

    def test_every_tunnel_scene_uses_the_same_route_departure_limit(self):
        for variant, mission in (('normal', 'full_loop_tunnel_normal'), ('brake', 'full_loop_tunnel_normal'),
                                 ('static', 'full_loop_tunnel_normal'), ('pedestrian', 'full_loop_tunnel_dynamic'),
                                 ('vehicle', 'full_loop_tunnel_vehicle')):
            scene = self.scene(variant, mission)
            scene.cfg['expert_max_cross_track_error_m'] = 1.5
            scene._configure_variant()
            self.assertEqual(scene.cfg['expert_max_cross_track_error_m'], 10.0, variant)

    def test_pedestrian_uses_the_random_pedestrian_animal_mix_unless_fixed(self):
        mixed = self.scene('pedestrian', 'full_loop_tunnel_dynamic').cfg['tunnel_dynamic']
        self.assertTrue(mixed['random_mix']['enabled'])
        self.assertEqual(mixed['candidate_policy'], 'trial')
        self.assertNotIn('trial_candidate_id', mixed)
        fixed = self.scene('pedestrian', 'full_loop_tunnel_dynamic', fixed_pedestrian=True).cfg['tunnel_dynamic']
        self.assertNotIn('random_mix', fixed)
        self.assertEqual(fixed['trial_candidate_id'], 'x')

    def test_pedestrian_spawn_distance_options_only_touch_the_tunnel_trigger(self):
        default = self.scene('pedestrian', 'full_loop_tunnel_dynamic').cfg['tunnel_dynamic']
        # Launcher default: 2.0 s reaction time for tunnel pedestrians only; margin keeps the scenario value.
        self.assertEqual(default['interaction_trigger'], {'reaction_time_sec': 2.0})
        restored = self.scene('pedestrian', 'full_loop_tunnel_dynamic', pedestrian_reaction_s=1.0)
        self.assertEqual(restored.cfg['tunnel_dynamic']['interaction_trigger'], {'reaction_time_sec': 1.0})
        tuned = self.scene('pedestrian', 'full_loop_tunnel_dynamic', pedestrian_reaction_s=2.5,
                           pedestrian_margin_m=5.).cfg['tunnel_dynamic']['interaction_trigger']
        self.assertEqual(tuned, {'reaction_time_sec': 2.5, 'stopping_safety_margin_m': 5.})

    def test_stopping_distance_grows_with_the_pedestrian_options(self):
        from ..scenarios.full_loop_parts.pedestrian_timing import interaction_timing
        state = {'speed_mps': 3.6}
        base = interaction_timing({}, 40., 5., 2.5, state, 0.)['safe_stop_distance_m']
        more = interaction_timing({'reaction_time_sec': 2.5, 'stopping_safety_margin_m': 5.},
                                  40., 5., 2.5, state, 0.)['safe_stop_distance_m']
        self.assertGreater(more, base * 1.8)

    def test_pedestrian_options_are_rejected_for_other_scenes_and_bad_values(self):
        with patch.object(run_tunnel, 'AIM_ROOT', self.root), patch.object(run_tunnel, 'RUNTIME', self.runtime):
            for argv in (['--variant', 'vehicle', '--pedestrian-reaction-s', '2'],
                         ['--variant', 'pedestrian', '--pedestrian-margin-m', '-1'],
                         ['--variant', 'pedestrian', '--pedestrian-reaction-s', '99']):
                with self.assertRaises(SystemExit):
                    run_tunnel.main(['--config', str(self.base), *argv])

    def test_the_default_reaction_time_reaches_every_pedestrian_run_of_the_rotation(self):
        code, calls = self.run_main(['--variant', 'all', '--runs', '8'])
        self.assertEqual([c[1] for c in calls].count('pedestrian'), 2)
        self.assertTrue(all(c[2]['pedestrian_reaction_s'] == 2.0 for c in calls))

    def test_the_other_scenes_do_not_get_the_pedestrian_timing(self):
        for variant in ('brake', 'static', 'vehicle', 'normal'):
            scene = self.scene(variant)
            self.assertNotIn('reaction_time_sec', scene.cfg['tunnel_dynamic'].get('interaction_trigger', {}))

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
        from ..scenarios.segment_modes.modes import get_mission_mode
        scene.cfg, scene.variant = {}, 'normal'
        scene.mission_mode = get_mission_mode('full_loop_tunnel_normal')
        scene._validation_controller = 'aim2026_roundabout'
        with patch.object(run_tunnel.harness.HighwayScenario, 'prepare_full_loop_route', return_value='route'):
            self.assertEqual(run_tunnel.TunnelScenario.prepare_full_loop_route(scene), 'route')
        self.assertEqual(scene._validation_controller, 'aim2026_tunnel')


if __name__ == '__main__':
    unittest.main()
