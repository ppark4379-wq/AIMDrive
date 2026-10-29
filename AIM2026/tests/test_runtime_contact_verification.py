"""Distant-spawn artifacts must not suppress real, delayed or unknown contacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from AIM2026.Expert.run_highway import HighwayScenario, _finish_validation
from scenario_runner.core.contact_verification import SpawnContactVerifier


class ContactVerificationTests(unittest.TestCase):
    def setUp(self):
        self.check = SpawnContactVerifier()
        # The last run's contemporaneous ego/target positions (~199 m apart).
        # Its birth timestamps were NOT logged; supply explicit synthetic birth
        # evidence here and separately verify missing evidence never exempts it.
        self.ego = (67.7103195, 59.95084)
        self.target = (64.1652908, -139.2830963)

    def birth(self, identity='npc', ego=None, target=None, time=100.):
        self.check.record_spawn(identity, time, time, ego or self.ego,
                                target or self.target, target or self.target)

    def assess(self, now=100.5, query=.01, targets=None, reports=None):
        return self.check.assess(['npc'] if reports is None else reports, now, query,
                                 self.ego, {'npc': self.target} if targets is None else targets)

    def test_recorded_far_geometry_requires_verified_fresh_birth(self):
        self.assertEqual(self.assess()['npc']['action'], 'stop')
        self.birth()
        result = self.assess()['npc']
        self.assertEqual(result['action'], 'continue_with_review')
        self.assertAlmostEqual(result['distance_m'], 199.2655, places=3)
        self.assertEqual(self.check.review_ids, {'npc'})

    def test_close_contact_stops_even_with_recent_spawn(self):
        target = (self.ego[0], self.ego[1]+5.)
        self.birth(target=target)
        self.assertEqual(self.assess(targets={'npc': target})['npc']['action'], 'stop')

    def test_unknown_actor_is_not_hidden_by_another_far_report(self):
        self.birth()
        result = self.assess(reports=['npc', 'barrier', '-100'])
        self.assertEqual(result['npc']['action'], 'continue_with_review')
        self.assertEqual(result['barrier']['action'], 'stop')
        self.assertEqual(result['-100']['action'], 'stop')

    def test_missing_stale_or_invalid_geometry_stops(self):
        for query, targets in ((.3, {'npc': self.target}), (.01, {}),
                               (.01, {'npc': (float('nan'), 0.)})):
            with self.subTest(query=query, targets=targets):
                self.birth()
                self.assertEqual(self.assess(query=query, targets=targets)['npc']['action'], 'stop')

    def test_old_birth_or_long_history_gap_cannot_dismiss_delayed_collision(self):
        self.birth()
        self.assertEqual(self.assess(now=103.)['npc']['action'], 'stop')
        self.birth()
        self.assertEqual(self.assess()['npc']['action'], 'continue_with_review')
        # At 120 m/s relative speed, 3.51 s permits closing the ~199 m
        # separation AND returning to it; endpoint distance alone is not proof.
        self.assertEqual(self.assess(now=104.)['npc']['action'], 'stop')

    def test_recorded_0437_spawn_pause_keeps_distant_report_deferred(self):
        # Actual last accepted snapshot and stopping snapshot from
        # highway-20261028-043542-553823/runtime_contact_checks.jsonl.
        self.check.continuing['npc'] = dict(
            time=122819.942845124-.0007923179946374148,
            ego=(74.61946868896484, -51.1104736328125),
            target=(75.00238800048828, -125.03260803222656),
            distance=73.9231261606554)
        result = self.check.assess(['npc'], 122820.337872013, .0008330050040967762,
            (74.6384048461914, -54.23075866699219),
            {'npc': (75.01509857177734, -127.70093536376953)})['npc']
        self.assertEqual(result['action'], 'continue_with_review')
        self.assertAlmostEqual(result['separation_lower_bound_m'], 47.9480, places=3)

    def test_far_endpoints_do_not_hide_possible_contact_between_observations(self):
        # With a long enough gap, two far snapshots can bracket a collision.
        self.check.continuing['npc'] = dict(time=100., ego=(0., 0.),
                                            target=(74., 0.), distance=74.)
        result = self.check.assess(['npc'], 101.3, .001, (0., 0.), {'npc': (74., 0.)})['npc']
        self.assertEqual(result['action'], 'stop')
        self.assertEqual(result['reason'], 'contact_possible')

    def test_report_clearance_revokes_exception(self):
        self.birth()
        self.assess()
        self.assess(now=100.6, reports=[])
        self.assertEqual(self.assess(now=100.7)['npc']['action'], 'stop')

    def test_entry_gap_policy_preserves_stop_conditions(self):
        for entry, query, distance, expected in (
                (True, .01, 100.47194361620241, 'continue_with_review'),
                (False, .01, 100.47194361620241, 'stop'),
                (True, .01, 25., 'stop'),
                (True, .3, 100.47194361620241, 'stop')):
            with self.subTest(entry=entry, query=query, distance=distance):
                self.check.continuing['npc'] = dict(time=100., ego=(0., 0.),
                    target=(120.12874328462847, 0.), distance=120.12874328462847)
                result = self.check.assess(['npc', 'wall'], 101.57733668992296,
                    query, (0., 0.), {'npc': (distance, 0.)}, entry_review=entry)
                self.assertEqual(result['npc']['action'], expected)
                self.assertEqual(result['wall']['action'], 'stop')
                if expected == 'continue_with_review':
                    self.assertEqual(result['npc']['reason'], 'entry_distant_report_review')
                    self.assertIn('npc', self.check.review_ids)

    def test_entry_gap_exception_requires_continuing_report(self):
        self.check.record_spawn('npc', 100., 100., (0., 0.), (120., 0.), (120., 0.))
        result = self.check.assess(['npc'], 101.58, .01, (0., 0.),
                                  {'npc': (100., 0.)}, entry_review=True)
        self.assertEqual(result['npc']['action'], 'stop')

    def test_continuing_report_is_rechecked_and_near_contact_stops(self):
        self.birth()
        self.assess()
        self.assertEqual(self.assess(now=100.6)['npc']['action'], 'continue_with_review')
        self.assertEqual(self.assess(now=100.7, targets={'npc': self.ego})['npc']['action'], 'stop')

    def test_spawn_at_wrong_location_does_not_get_exemption(self):
        self.check.record_spawn('npc', 100., 100., self.ego, self.target, self.ego)
        self.assertEqual(self.assess()['npc']['action'], 'stop')

    def test_continuing_report_includes_previous_query_time(self):
        self.birth()
        self.assertEqual(self.assess(query=.1)['npc']['action'], 'continue_with_review')
        self.assertAlmostEqual(self.check.continuing['npc']['time'], 100.4)

    @staticmethod
    def actor(identity, point):
        return SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=identity)),
                               transform=SimpleNamespace(location=SimpleNamespace(x=point[0], y=point[1])))

    def test_runtime_wrapper_preserves_raw_evidence_and_unknown_contacts(self):
        self.birth()
        with TemporaryDirectory() as directory:
            scene = object.__new__(HighwayScenario)
            scene.output = Path(directory)
            scene._contact_drive_active = True
            scene._spawn_contact_verifier = self.check
            world = Mock()
            world.get_all_actors_state.return_value = SimpleNamespace(states=[
                self.actor('Ego', self.ego), self.actor('npc', self.target)])
            scene.grpc = SimpleNamespace(world=world)
            state = dict(raw_collision_objects=['Ego', 'npc', 'wall'], collision_objects=['npc', 'wall'])
            with patch('AIM2026.Expert.run_highway.time.monotonic', return_value=100.5):
                result = scene._verify_runtime_contacts(state)
            self.assertEqual(result['collision_objects'], ['wall'])
            self.assertEqual(result['raw_collision_objects'], ['Ego', 'npc', 'wall'])
            self.assertEqual(state['collision_objects'], ['npc', 'wall'])
            evidence = json.loads((scene.output/'runtime_contact_checks.jsonl').read_text())
            self.assertEqual(evidence['decisions']['npc']['action'], 'record_only')
            self.assertEqual(evidence['decisions']['npc']['assessment_action'], 'continue_with_review')
            self.assertEqual(evidence['decisions']['wall']['action'], 'record_only')
            world.get_all_actors_state.side_effect = RuntimeError('RPC unavailable')
            with patch('AIM2026.Expert.run_highway.time.monotonic', return_value=100.6):
                self.assertEqual(scene._verify_runtime_contacts(state)['collision_objects'], ['npc', 'wall'])

    def test_startup_or_strict_mode_keeps_existing_fail_closed_behavior(self):
        scene = object.__new__(HighwayScenario)
        scene._contact_drive_active = False
        scene._spawn_contact_verifier = self.check
        raw = dict(collision_objects=['npc'])
        self.birth()
        self.assertIs(scene._verify_runtime_contacts(raw), raw)
        scene._contact_drive_active = True
        scene._spawn_contact_verifier = None  # --strict-collisions
        self.assertIs(scene._verify_runtime_contacts(raw), raw)

    def test_entry_policy_is_limited_to_driving_on_approach_link(self):
        scene = object.__new__(HighwayScenario)
        scene._spawn_contact_verifier = Mock(births={}, continuing={})
        scene._spawn_contact_verifier.assess.return_value = {}
        for driving, link, expected in ((True, 'A2256W000846', True),
                                        (True, 'A2256W000411', False),
                                        (True, None, False),
                                        (False, 'A2256W000846', False)):
            scene._contact_drive_active = driving
            scene._contact_startup_active = not driving
            scene._verify_runtime_contacts(dict(collision_objects=[], current_link=link))
            self.assertEqual(scene._spawn_contact_verifier.assess.call_args.kwargs['entry_review'], expected)
    def test_spawn_evidence_hook_runs_only_during_drive_and_verifies_actor_id(self):
        with TemporaryDirectory() as directory:
            scene = object.__new__(HighwayScenario)
            scene.output = Path(directory)
            scene._spawn_contact_verifier = self.check
            scene._contact_latest_ego = dict(time=100., xy=self.ego)
            actor = Mock()
            actor.get_id.return_value = 'npc'
            actor.get_actor_state.return_value = self.actor('npc', self.target)
            scene.record_live_spawn_contact_evidence(actor, self.target, 100.)
            actor.get_actor_state.assert_not_called()
            scene._contact_drive_active = True
            with patch('AIM2026.Expert.run_highway.time.monotonic', return_value=100.1):
                scene.record_live_spawn_contact_evidence(actor, self.target, 100.)
            self.assertIn('npc', self.check.births)
            actor.get_id.return_value = 'wrong'
            scene.record_live_spawn_contact_evidence(actor, self.target, 100.)
            self.assertNotIn('wrong', self.check.births)

    def test_completed_drive_with_deferred_report_is_not_a_clean_pass(self):
        self.birth()
        self.assess()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scene = SimpleNamespace(_spawn_contact_verifier=self.check,
                run_instance_id='test', run_seed=7, args=SimpleNamespace(mission='full_loop_highway'),
                driver_output=root/'expert')
            with patch('scenario_runner.validation.finish.finish_validation_run') as finish:
                _finish_validation(scene, root, 'run_returned', None)
            self.assertEqual(finish.call_args.args[2], 'review_required')
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['status'], 'REVIEW_REQUIRED')
            self.assertEqual(report['collision_review']['actor_ids'], ['npc'])


if __name__ == '__main__':
    unittest.main()
