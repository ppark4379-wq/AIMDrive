"""Real map and collision checks for upstream second-merge lane preparation."""
import math
from pathlib import Path
from unittest.mock import patch
import unittest

from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner


class SecondMergePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(root / 'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.planner.set_control_dt(.1)

    def pose(self, lane='408', y=-10., speed=12.):
        link = self.lanes.links['A2256W000' + lane]
        match = self.lanes.project(Point(70., y), link)
        return VehicleState(match.x, match.y, match.yaw, speed)

    def car(self, lane, y, speed, identity=1):
        e = self.pose(lane, y, speed)
        return {'object_info': ObjectInfo(e.position.x, e.position.y, speed, 2,
                    unique_id=identity, size_x=4.5, size_y=2., heading_deg=math.degrees(e.yaw))}

    def step(self, ego, entries=(), allow=True):
        # Deliberately keep the old global route on the right lane.
        match = self.lanes.project(ego.position, self.lanes.links['A2256W000408'])
        reference = [row[0] for row in self.lanes.route_points(match, 240., .5)]
        return self.planner.plan(ego, reference, list(entries), desired_velocity=16.,
                                 allow_lane_change=allow)

    def test_gate_uses_user_point_and_only_designated_links(self):
        for lane, y, expected in [('430', 85., False), ('430', 75., True),
                                  ('408', 3., True), ('408', 0., True),
                                  ('434', -20., True), ('422', -160., True),
                                  ('409', 200., True), ('435', -10., True),
                                  ('431', 75., True), ('423', -160., True),
                                  ('421', -10., False)]:
            with self.subTest(lane=lane, y=y):
                e = self.pose(lane, y)
                match = self.lanes.project(e.position, self.lanes.links['A2256W000'+lane])
                self.assertEqual(self.planner.second_merge_preparation.applies(e, match), expected)

    def test_early_links_choose_only_adjacent_left_and_hold_leftmost(self):
        for lane, target in [('409', '410'), ('410', '418')]:
            with self.subTest(lane=lane):
                self.planner.reset()
                result = self.step(self.pose(lane, 180.))
                self.assertEqual(result.reason, 'highway_second_merge_prepare_left')
                self.assertEqual(result.target_link_id, 'A2256W000'+target)
        self.planner.reset()
        result = self.step(self.pose('418', 180.))
        self.assertEqual(result.reason, 'highway_second_merge_hold_left')
        self.assertEqual(result.target_link_id, 'A2256W000418')

    def test_early_activation_continues_across_old_start_and_resets_off_route(self):
        prep = self.planner.second_merge_preparation
        ego = self.pose('409', 180.)
        prep.update(ego, self.lanes.project(ego.position, self.lanes.links['A2256W000409']), [], [])
        ego = self.pose('420', 85.)
        match = self.lanes.project(ego.position, self.lanes.links['A2256W000420'])
        self.assertTrue(prep.applies(ego, match))
        prep.reset()
        self.assertFalse(prep.applies(ego, match))

    def test_early_preparation_respects_traffic_and_completed_change_cooldown(self):
        ego = self.pose('409', 180.)
        result = self.step(ego, [self.car('410', 180., 12.)])
        self.assertNotEqual(result.highway_state, self.planner.CHANGE_LEFT)
        self.planner.reset()
        self.planner._lane_change_cooldown_origin = Point(ego.position.x, ego.position.y)
        result = self.step(ego)
        self.assertNotEqual(result.highway_state, self.planner.CHANGE_LEFT)
        self.assertTrue(result.debug_details['second_merge_preparation']['cooldown'])

    def test_first_merge_and_handoff_finish_before_another_left_change(self):
        result = self.step(self.pose('411', 220., 10.))
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertEqual(result.target_link_id, 'A2256W000409')
        result = self.step(self.pose('409', 165., 10.))
        self.assertEqual(result.reason, 'entry_merge_complete')
        result = self.step(self.pose('409', 164., 10.))
        self.assertEqual(result.reason, 'entry_merge_following')
        self.assertEqual(result.target_link_id, 'A2256W000409')

    def test_empty_target_starts_exactly_one_left_change(self):
        r = self.step(self.pose())
        self.assertEqual(r.reason, 'highway_second_merge_prepare_left')
        self.assertEqual(r.target_link_id, 'A2256W000434')
        self.assertEqual(r.highway_state, self.planner.CHANGE_LEFT)
        self.assertEqual(r.acc_target_velocity, 12.)
        self.assertEqual(len(r.candidates), 2)

    @patch.object(p, 'SECOND_MERGE_PREP_SLOW_FOR_GAP', True)
    def test_parallel_vehicle_blocks_change_and_creates_gap_smoothly(self):
        ego = self.pose()
        traffic = [self.car('434', -10., 12.)]
        speeds = []
        for _ in range(8):
            r = self.step(ego, traffic)
            self.assertEqual(r.reason, 'highway_second_merge_create_gap')
            self.assertEqual(r.target_link_id, 'A2256W000408')
            self.assertEqual(r.debug_details['second_merge_preparation']['follow_id'], 'npc:1')
            speeds.append(r.acc_target_velocity)
        self.assertLess(speeds[-1], speeds[0])
        self.assertTrue(all(0. <= a-b <= p.SECOND_MERGE_PREP_DECEL*.1+1e-8
                            for a,b in zip(speeds, speeds[1:])))

    def test_gap_opening_starts_without_changing_target_id(self):
        self.step(self.pose(), [self.car('434', -10., 12.)])
        r = self.step(self.pose(speed=8.), [self.car('434', -45., 12.)])
        self.assertEqual(r.reason, 'highway_second_merge_prepare_left')
        self.assertEqual(r.target_link_id, 'A2256W000434')

    def test_fast_rear_is_still_an_admission_hazard(self):
        r = self.step(self.pose(), [self.car('434', 5., 24.)])
        self.assertNotEqual(r.highway_state, self.planner.CHANGE_LEFT)
        self.assertFalse(r.candidates[-1].valid)

    def test_short_dropout_does_not_turn_parallel_traffic_into_empty_lane(self):
        ego = self.pose()
        self.step(ego, [self.car('434', -10.,12.)])
        r = self.step(ego, [])
        self.assertEqual(r.reason, 'highway_second_merge_wait_for_gap')
        self.assertGreaterEqual(r.acc_target_velocity, ego.velocity)
        self.assertNotEqual(r.highway_state, self.planner.CHANGE_LEFT)

    @patch.object(p, 'SECOND_MERGE_PREP_SLOW_FOR_GAP', True)
    def test_gap_creation_completes_before_taper_in_longitudinal_rollout(self):
        # Simple bounded-response longitudinal rollout, not a MORAI drive.
        ego_y, other_y, speed = 0., 0., 12.
        phases = []
        for _ in range(90):
            r = self.step(self.pose(y=ego_y,speed=speed), [self.car('434',other_y,12.)])
            phases.append(r.reason)
            if r.highway_state == self.planner.CHANGE_LEFT:
                self.assertGreater(ego_y, -78.)  # 55 m transition + 10 m before taper.
                self.assertGreater(speed, 2.)
                self.assertEqual(r.target_link_id, 'A2256W000434')
                break
            target = min(16., r.acc_target_velocity)
            speed += max(-.3, min(.15, target-speed))
            ego_y -= speed*.1
            other_y -= 1.2
        else:
            self.fail('Gap creation did not admit the change before the deadline')
        self.assertIn('highway_second_merge_create_gap', phases)

    def test_front_emergency_is_not_rate_limited(self):
        traffic = [self.car('408', -17., 0.), self.car('434', -10., 12., 2)]
        r = self.step(self.pose(), traffic)
        self.assertEqual(r.acc_target_velocity, 0.)
        self.assertNotEqual(r.highway_state, self.planner.CHANGE_LEFT)

    def test_stays_on_left_and_runs_existing_merge_priority_on_left_path(self):
        for lane, y in [('431', 75.), ('435', -20.), ('423', -160.)]:
            self.planner.reset()
            e = self.pose(lane, y)
            with patch.object(self.planner, '_merge_acc_speed_limit', return_value=3.) as merge:
                r = self.step(e)
            self.assertEqual(r.reason, 'highway_second_merge_hold_left')
            self.assertLessEqual(r.acc_target_velocity, 3.)
            self.assertEqual(r.target_link_id, 'A2256W000'+lane)
            self.assertTrue(r.active)
            self.assertLess(r.path[0].distance(e.position), .6)
            self.assertLess(merge.call_args.kwargs['reference_path'][0].distance(e.position), .6)

    def test_middle_lane_is_not_completion_and_targets_only_adjacent_left(self):
        for lane, y, target in [('430', 75., '431'), ('434', -10., '435')]:
            with self.subTest(lane=lane):
                self.planner.reset()
                result = self.step(self.pose(lane, y))
                self.assertEqual(result.reason, 'highway_second_merge_prepare_left')
                self.assertEqual(result.target_link_id, 'A2256W000'+target)
                self.assertEqual(result.highway_state, self.planner.CHANGE_LEFT)
                self.assertFalse(self.lanes.links[result.target_link_id].excluded)

    def test_blocked_left_from_middle_keeps_speed_and_rechecks_after_car_passes(self):
        ego = self.pose('434', -10., 15.)
        result = self.step(ego, [self.car('435', -10., 15.)])
        self.assertEqual(result.reason, 'highway_second_merge_wait_for_gap')
        self.assertEqual(result.target_link_id, 'A2256W000434')
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertFalse(result.candidates[-1].valid)
        result = self.step(ego, [self.car('435', -55., 20.)])
        self.assertEqual(result.target_link_id, 'A2256W000435')
        self.assertEqual(result.highway_state, self.planner.CHANGE_LEFT)

    def test_left_preference_never_overrides_fast_rear_or_taper_deadline(self):
        for lane, y, traffic in [
            ('434', -10., [self.car('435', 5., 24.)]),
            ('434', -120., []), ('422', -160., [])]:
            self.planner.reset()
            result = self.step(self.pose(lane, y), traffic)
            self.assertNotEqual(result.highway_state, self.planner.CHANGE_LEFT)
            self.assertEqual(result.target_link_id, 'A2256W000'+lane)

    def test_two_changes_require_completion_and_thirty_meters_between_them(self):
        first = self.step(self.pose('420', 75.))
        self.assertEqual(first.target_link_id, 'A2256W000430')
        first_path = self.planner.active_candidate
        a, b = first_path.points[20:22]
        mid = VehicleState(a.x, a.y, math.atan2(b.y-a.y, b.x-a.x), 12.)
        self.step(mid)
        self.assertIs(self.planner.active_candidate, first_path)
        completed = self.step(self.pose('430', 15.))
        self.assertNotEqual(completed.highway_state, self.planner.CHANGE_LEFT)
        self.assertTrue(completed.debug_details['second_merge_preparation']['cooldown'])
        too_soon = self.step(self.pose('430', 0.))
        self.assertNotEqual(too_soon.highway_state, self.planner.CHANGE_LEFT)
        second = self.step(self.pose('434', -20.))
        self.assertEqual(second.target_link_id, 'A2256W000435')
        self.assertEqual(second.highway_state, self.planner.CHANGE_LEFT)
        held = self.step(self.pose('435', -80.))
        self.assertEqual(held.reason, 'highway_second_merge_hold_left')
        self.assertEqual(held.target_link_id, 'A2256W000435')

    def test_no_last_second_crossing_or_crossing_when_disallowed(self):
        for lane, y, allow in [('408', -120., True), ('445', -155., True), ('408', -10., False)]:
            self.planner.reset()
            r = self.step(self.pose(lane, y), allow=allow)
            self.assertNotEqual(r.highway_state, self.planner.CHANGE_LEFT)
            self.assertEqual(len(r.candidates), 1)

    def test_committed_path_is_not_replanned_for_gap_preparation(self):
        self.step(self.pose())
        c = self.planner.active_candidate
        a,b = c.points[20:22]
        e = VehicleState(a.x,a.y,math.atan2(b.y-a.y,b.x-a.x),12.)
        r = self.step(e)
        self.assertIs(self.planner.active_candidate, c)
        self.assertEqual(r.target_link_id,'A2256W000434')
        self.assertEqual(r.reason, 'highway_lane_change_in_progress')
        self.assertGreaterEqual(r.acc_target_velocity, 12.)

    @patch.object(p, 'SECOND_MERGE_PREP_SLOW_FOR_GAP', True)
    def test_lateral_reset_clears_preparation_state(self):
        self.step(self.pose(), [self.car('434', -10.,12.)])
        self.assertIsNotNone(self.planner.second_merge_preparation.follow_id)
        self.planner.reset()
        self.assertIsNone(self.planner.second_merge_preparation.wait_speed)
        self.assertIsNone(self.planner.second_merge_preparation.follow_id)

    def test_default_waits_for_parallel_car_without_braking_to_make_gap(self):
        ego=self.pose()
        traffic=[self.car('434',-10.,12.)]
        for _ in range(8):
            result=self.step(ego,traffic)
            self.assertEqual(result.reason,'highway_second_merge_wait_for_gap')
            self.assertEqual(result.target_link_id,'A2256W000408')
            self.assertGreaterEqual(result.acc_target_velocity,ego.velocity)
            self.assertGreater(result.acc_target_velocity,p.SECOND_MERGE_PREP_SPEED_KPH/3.6)
            self.assertFalse(result.debug_details['second_merge_preparation']['adjacent_gap_braking'])

    def test_blocked_optional_change_does_not_remove_current_front_or_merge_limit(self):
        traffic=[self.car('434',-10.,12.)]
        with patch.object(self.planner,'_merge_acc_speed_limit',return_value=5.):
            result=self.step(self.pose(),traffic)
        self.assertEqual(result.acc_target_velocity,5.)
        self.assertEqual(result.target_link_id,'A2256W000408')

    def test_faster_parallel_vehicle_can_pass_then_allow_change_without_yield_braking(self):
        ego_y,other_y,speed=0.,0.,12.
        for _ in range(60):
            result=self.step(self.pose(y=ego_y,speed=speed),[self.car('434',other_y,20.)])
            self.assertGreaterEqual(result.acc_target_velocity,speed)
            if result.highway_state==self.planner.CHANGE_LEFT:
                self.assertEqual(result.target_link_id,'A2256W000434')
                break
            ego_y-=speed*.1
            other_y-=2.
        else:
            self.fail('Did not change after the parallel car passed and opened a safe gap')


if __name__ == '__main__':
    unittest.main()
