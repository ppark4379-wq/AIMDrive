"""Full-course competition launcher: UDP Expert harness on the competition course, tunnel scenes rotating."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

from types import SimpleNamespace
from unittest.mock import Mock

from ..scenarios.full_loop_parts import competition_static
from ..scenarios.full_loop_competition import FullLoopCompetitionScenario, TUNNEL_SCENES
from ..scenarios.segment_modes.scenario import MissionScenario
from ..tools import run_competition_aim2026 as run_competition


class CompetitionScenarioTest(unittest.TestCase):
    def test_the_harness_runs_over_the_competition_course_not_over_a_segment_mission(self):
        scenario = run_competition.CompetitionScenario
        self.assertTrue(issubclass(scenario, FullLoopCompetitionScenario))
        self.assertFalse(issubclass(scenario, MissionScenario))
        for name in ('check_startup_contact', 'get_ego_motion_state', 'record_collision_stop', 'setup_expert_controller'):
            self.assertTrue(hasattr(scenario, name), name)

    def test_grafted_methods_call_the_next_class_of_the_new_chain(self):
        calls = []

        class Base:
            def run(self):
                calls.append('base')

        class Overlay(Base):
            def run(self):
                calls.append('overlay')
                return super().run()

        class Other:
            def run(self):
                calls.append('other')

        grafted = run_competition.graft(Overlay, Other, 'Grafted')
        grafted().run()
        self.assertEqual(calls, ['overlay', 'other'])           # not 'base': the old parent is gone
        self.assertNotIn(Base, grafted.__mro__)
        self.assertTrue(isinstance(vars(Overlay)['run'], types.FunctionType))
        self.assertIs(vars(Overlay)['run'].__closure__[0].cell_contents, Overlay)   # the overlay itself is left alone


class CompetitionLauncherTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / 'config_local.yaml'
        self.config.write_text('Ego: {}\n', encoding='utf-8')

    def run_main(self, argv, outcomes=None):
        calls = []

        def fake(arguments, options=None, record=None, limits=None, scene=None):
            calls.append((arguments, options, limits, scene))
            if record is not None:
                scene_name = TUNNEL_SCENES[(len(calls) - 1) % 4]
                record.update(run_seed=1000 + len(calls), run_instance_id='id{}'.format(len(calls)),
                              tunnel_scene=scene_name, output=str(self.root / 'run{}'.format(len(calls))))
            outcome = outcomes[len(calls) - 1] if outcomes else 0
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        with patch.object(run_competition, 'run_one', fake), patch.object(run_competition, 'RUNTIME', self.root / 'rt'), \
                patch.object(run_competition, 'ensure_static_library'), patch.object(run_competition.time, 'sleep'):
            code = run_competition.main(['--config', str(self.config), '--no-virtual-gps', *argv])
        return code, calls

    def test_one_pass_over_the_four_tunnel_scenes_by_default(self):
        code, calls = self.run_main(['--seed', '2000', '--validation-events'])
        self.assertEqual((code, len(calls)), (0, 4))
        self.assertEqual([call[0][call[0].index('--seed') + 1] for call in calls], ['2000', '2001', '2002', '2003'])
        self.assertIn('--validation-events', calls[0][0])
        self.assertEqual(calls[0][2], dict(round_timeout_s=run_competition.ROUND_TIMEOUT_S, stall_s=run_competition.STALL_S))
        self.assertEqual(calls[0][1], {})                         # no model filter unless asked

    def test_check_validates_every_tunnel_scene_once(self):
        code, calls = self.run_main(['--check'])
        self.assertEqual((code, [call[3] for call in calls]), (0, list(TUNNEL_SCENES)))

    def test_a_chosen_tunnel_scene_is_run_once_unless_more_runs_are_asked(self):
        code, calls = self.run_main(['--tunnel-scene', 'brake'])
        self.assertEqual((code, len(calls), calls[0][1]), (0, 1, {'tunnel_scene': 'legacy'}))
        code, calls = self.run_main(['--tunnel-scene', 'pedestrian', '--runs', '3'])
        self.assertEqual((code, [call[1]['tunnel_scene'] for call in calls]), (0, ['pedestrian'] * 3))

    def test_check_with_a_chosen_scene_validates_only_that_scene(self):
        code, calls = self.run_main(['--check', '--tunnel-scene', 'vehicle'])
        self.assertEqual((code, [call[3] for call in calls]), (0, ['vehicle']))

    def test_only_the_static_scene_needs_the_static_library(self):
        with patch.object(run_competition, 'ensure_static_library') as ensure, \
                patch.object(run_competition, 'run_one', return_value=0), patch.object(run_competition, 'RUNTIME', self.root / 'rt'), \
                patch.object(run_competition.time, 'sleep'):
            for scene, needed in (('static', 1), ('brake', 0), ('pedestrian', 0), ('vehicle', 0)):
                ensure.reset_mock()
                run_competition.main(['--config', str(self.config), '--no-virtual-gps', '--tunnel-scene', scene])
                self.assertEqual(ensure.call_count, needed, scene)
            ensure.reset_mock()
            run_competition.main(['--config', str(self.config), '--no-virtual-gps'])
            self.assertEqual(ensure.call_count, 1)                      # the rotation includes static

    def test_the_competition_morai_model_list_is_handed_to_every_run(self):
        code, calls = self.run_main(['--runs', '2', '--competition-morai'])
        models = run_competition.tunnel.competition_models()
        self.assertEqual((code, [call[1]['installed_models'] for call in calls]), (0, [models, models]))

    def test_competition_morai_hands_the_competition_vehicle_interface_to_every_run(self):
        self.config.write_text('Ego:\n  cmd_layout: front_rear59\n  local_status_bytes: 245\nControl:\n  positive_steer: right\n',
                               encoding='utf-8')
        calls = []

        def fake(arguments, options=None, record=None, limits=None, scene=None):
            calls.append(arguments)
            return 0
        with patch.object(run_competition, 'run_one', fake), patch.object(run_competition, 'RUNTIME', self.root / 'rt'), \
                patch.object(run_competition, 'ensure_static_library'), patch.object(run_competition.time, 'sleep'):
            run_competition.main(['--config', str(self.config), '--no-virtual-gps', '--runs', '1', '--competition-morai'])
        used = Path(calls[0][calls[0].index('--config') + 1])
        document = run_competition.tunnel._load(used)
        self.assertEqual((document['Ego'], document['Control']), (dict(cmd_layout='single_steer55'),
                                                                  dict(positive_steer='left')))
        self.assertIn('local_status_bytes: 245', self.config.read_text(encoding='utf-8'))   # the original stays untouched

    def test_a_failed_run_is_recorded_and_the_next_one_still_starts(self):
        results = self.root / 'results.jsonl'
        code, calls = self.run_main(['--runs', '3', '--results', str(results)],
                                    outcomes=[RuntimeError('EGO collision with NPC'), 0, 0])
        self.assertEqual((code, len(calls)), (1, 3))
        rows = [json.loads(line) for line in results.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['result'] for row in rows], ['collision', 'completed', 'completed'])
        self.assertEqual([row['tunnel_scene'] for row in rows], ['legacy', 'static', 'pedestrian'])

    def test_ctrl_c_stops_the_campaign_after_the_cleanup(self):
        code, calls = self.run_main(['--runs', '4'], outcomes=[0, run_competition.INTERRUPTED, 0, 0])
        self.assertEqual((code, len(calls)), (run_competition.INTERRUPTED, 2))

    def test_options_that_make_no_sense_stop_with_an_explanation(self):
        folder = self.root / 'morai'
        folder.mkdir()
        for argv in (['--runs', '0'], ['--check', '--runs', '2'], ['--round-timeout-s', '-1'],
                     ['--competition-morai', '--morai-assets', str(folder)], ['--morai-assets', str(folder)]):
            with patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
                self.run_main(argv)
            self.assertEqual(raised.exception.code, 2, argv)

    def test_a_missing_static_library_stops_the_course_because_the_rotation_needs_it(self):
        library = self.root / 'no/a/b/approved_candidates.json'
        with patch.object(run_competition.tunnel, 'static_library', return_value=library), \
                patch.object(run_competition.tunnel, 'static_archive', return_value=None), \
                patch('sys.stderr') as stderr, self.assertRaises(SystemExit) as raised:
            run_competition.ensure_static_library(argparse.ArgumentParser())
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('static layout library is not installed', ''.join(str(c.args[0]) for c in stderr.write.call_args_list))


def actor(identity, x, y, kind=2, z=0.):
    return SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=identity), object_type=kind),
                           transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z)))


class TunnelSceneChoiceTest(unittest.TestCase):
    def test_a_chosen_scene_gets_the_next_rotation_number_that_falls_on_it(self):
        for previous in range(0, 12):
            for scene in TUNNEL_SCENES:
                sequence = run_competition.sequence_for_scene(previous, scene)
                self.assertGreater(sequence, previous)
                self.assertLessEqual(sequence - previous, 4)
                self.assertEqual(TUNNEL_SCENES[(sequence - 1) % 4], scene)    # the saved record stays consistent
        self.assertEqual(run_competition.sequence_for_scene(25, 'static'), 26)
        self.assertEqual(run_competition.sequence_for_scene(26, 'static'), 30)

    def test_scene_names_map_brake_to_the_rotations_legacy_scene(self):
        self.assertEqual(run_competition.SCENE_NAMES['brake'], 'legacy')
        self.assertEqual(set(run_competition.SCENE_NAMES.values()), set(TUNNEL_SCENES))


class ParentActorBindingTest(unittest.TestCase):
    """The competition MORAI lists parent-spawned actors by native ID but answers a query with their request alias."""

    def setUp(self):
        self.alias = 'highway_merge_adjacent_0_run_2020_Kia_K5'
        self.own = actor(self.alias, 10., 20.)           # the single-actor answer: alias as ID
        stub = SimpleNamespace(GetActorState=Mock(return_value=self.own))
        self.parent = SimpleNamespace(client_key='kcity', world=SimpleNamespace(_sim_adapter=SimpleNamespace(_actor_stub=stub)))
        self.handle = SimpleNamespace(get_id=lambda: self.alias,
                                      get_object_info=lambda: SimpleNamespace(client_key='kcity'))
        self.context = SimpleNamespace(grpc=self.parent, scenario_actors=[self.handle])

    def baseline(self, inventory):
        return competition_static._preserved_baseline(self.context, inventory, 'ego')

    def test_a_parent_actor_known_only_by_alias_is_bound_by_its_place_and_type(self):
        inventory = {'ego': actor('ego', 0., 0.), '27': actor('27', 10.2, 20.1), '3': actor('3', 50., 50.)}
        self.assertEqual(set(self.baseline(inventory)), {'ego', '3'})    # the parent's own actor retires; foreign ones stay

    def test_the_same_place_with_another_type_or_far_away_is_not_the_parent_actor(self):
        inventory = {'ego': actor('ego', 0., 0.), '27': actor('27', 10., 20., kind=3), '3': actor('3', 30., 20.)}
        with self.assertRaisesRegex(RuntimeError, 'cannot be bound to the setup inventory'):
            self.baseline(inventory)

    def test_two_candidates_at_that_place_are_too_ambiguous_to_bind(self):
        inventory = {'ego': actor('ego', 0., 0.), '27': actor('27', 10., 20.), '28': actor('28', 10.1, 20.)}
        with self.assertRaisesRegex(RuntimeError, 'cannot be bound to the setup inventory'):
            self.baseline(inventory)

    def test_an_alias_that_is_already_a_native_inventory_id_needs_no_pose(self):
        self.alias = '27'
        self.handle.get_id = lambda: '27'
        inventory = {'ego': actor('ego', 0., 0.), '27': actor('27', 999., 999.)}
        self.assertEqual(set(self.baseline(inventory)), {'ego'})


if __name__ == '__main__':
    unittest.main()
