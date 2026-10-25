"""Measured first-entry geometry and real collision checks; no simulator sockets."""
import math
import json
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.control.path_controller import PathController
from AIM2026.Expert.src.planning.common.target_speed import target_speed
from AIM2026.Expert.src.planning.highway.driving.obstacle_avoidance import HighwayStaticCorridor


class EntryMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.source = self.lanes.links[p.ENTRY_MERGE_SOURCE]
        self.target = self.lanes.links[p.ENTRY_MERGE_TARGET]

    def ego(self, y=250., speed=10.):
        match = self.lanes.project(Point(63., y), self.source)
        return VehicleState(match.x, match.y, match.yaw, speed)

    def car(self, y, speed, identity=7, lane=None):
        match = self.lanes.project(Point(66., y), lane or self.target)
        obj = ObjectInfo(match.x, match.y, speed, -1, name='ve_tracked_unknown:moving',
                         unique_id=identity, size_x=4.5, size_y=1.8, heading_deg=math.degrees(match.yaw))
        return {'object_info': obj}

    def step(self, ego, traffic=(), desired=16.0):
        match = self.lanes.project(ego.position, self.source)
        reference = [row[0] for row in self.lanes.route_points(match, 120., .5)]
        return self.planner.plan(ego, reference, list(traffic), desired_velocity=desired)

    def pose_on_candidate(self, candidate, index, speed):
        a, b = candidate.points[index:index+2]
        return VehicleState(a.x, a.y, math.atan2(b.y-a.y, b.x-a.x), speed)

    def straight_gap_check(self, gap, rear=False, entry=True, committed=False,
                           source=None, target=None, other_speed=10.):
        # Real swept geometry/prediction with constant-speed cars on a straight.
        candidate = SimpleNamespace(points=[Point(i, 0.) for i in range(101)],
            transition_end=Point(20., 0.), entry_merge=entry,
            source_link_id=source or p.ENTRY_MERGE_SOURCE,
            target_link_id=target or p.ENTRY_MERGE_TARGET)
        x = -(gap+2.25) if rear else p.VEHICLE_LENGTH+2.25+gap
        obj = ObjectInfo(x, 0., other_speed, -1, unique_id=7, size_x=4.5, size_y=1.8)
        with patch.multiple(p, ENTRY_MERGE_FRONT_GAP=2., ENTRY_MERGE_REAR_GAP=8.,
                            LANE_CHANGE_BUMPER_MARGIN=5.):
            return self.planner.lane_change_safety.check(candidate,
                VehicleState(0., 0., 0., 10.), [{'object_info': obj}], [], 10., committed=committed)

    def test_entry_front_gap_is_separate_before_and_during_change(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                self.assertIsNone(self.straight_gap_check(3., committed=committed))
                conflict = self.straight_gap_check(3., entry=False, committed=committed)
                self.assertEqual(conflict['kind'], 'gap')
                self.assertEqual(conflict['required_gap'], 5.)
                self.assertEqual(conflict['gap_policy'], 'lane_change')
                conflict = self.straight_gap_check(1.5, committed=committed)
                self.assertEqual(conflict['required_gap'], 2.)

    def test_entry_rear_margin_and_closing_speed_remain_enforced(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                conflict = self.straight_gap_check(6., rear=True, committed=committed)
                self.assertEqual(conflict['kind'], 'gap')
                self.assertEqual(conflict['required_gap'], 8.)
                self.assertIsNone(self.straight_gap_check(6., rear=True, entry=False, committed=committed))
                self.assertIsNotNone(self.straight_gap_check(8.2, rear=True, other_speed=12., committed=committed))

    def test_other_link_pairs_cannot_opt_into_entry_margin(self):
        for source, target in ((p.ENTRY_MERGE_TARGET, p.ENTRY_MERGE_SOURCE),
                               (p.ENTRY_MERGE_TARGET, 'A2256W000410')):
            conflict = self.straight_gap_check(3., source=source, target=target)
            self.assertEqual(conflict['required_gap'], 5.)
            self.assertEqual(conflict['gap_policy'], 'lane_change')

    def test_entry_margin_does_not_replace_body_collision_test(self):
        for committed in (False, True):
            conflict = self.straight_gap_check(-2., committed=committed)
            self.assertEqual(conflict['kind'], 'collision')

    def test_empty_lane_changes_early_one_lane_and_finishes_before_boundary(self):
        ego = self.ego()
        result = self.step(ego)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)
        candidate = self.planner.active_candidate
        self.assertGreaterEqual(candidate.transition_end.y-p.VEHICLE_LENGTH, p.ENTRY_MERGE_END_Y)
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertAlmostEqual(abs(candidate.offset), p.LANE_WIDTH)

    def test_empty_observation_prepares_speed_before_turn_instead_of_using_road_limit(self):
        ego = self.ego(speed=12.)
        result = self.step(ego, desired=60./3.6)
        self.assertEqual(result.reason, 'entry_merge_approach')
        self.assertIsNone(self.planner.active_candidate)
        self.assertLessEqual(result.acc_target_velocity, p.ENTRY_MERGE_CRUISE_SPEED)
        ego.velocity = result.acc_target_velocity
        result = self.step(ego, desired=60./3.6)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertLessEqual(result.acc_target_velocity, p.ENTRY_MERGE_CRUISE_SPEED)
        self.assertLessEqual(self.planner.active_candidate.transition_distance, 45.)

    def test_occupied_side_keeps_source_then_starts_next_clear_observation(self):
        ego = self.ego()
        blocked = self.step(ego, [self.car(ego.position.y, 10.)])
        self.assertEqual(blocked.target_link_id, p.ENTRY_MERGE_SOURCE)
        self.assertEqual(blocked.reason, 'entry_merge_approach')
        self.assertGreater(blocked.acc_target_velocity, 0.)
        self.assertIsNone(self.planner.active_candidate)
        # Same tracked car has passed; no arbitrary additional waiting counter.
        clear = self.step(ego, [self.car(ego.position.y-40., 16.)])
        self.assertEqual(clear.reason, 'entry_merge_start')

    def test_fast_rear_and_close_front_prevent_start(self):
        for y, speed in ((268., 24.), (235., 0.)):
            self.planner.reset()
            result = self.step(self.ego(), [self.car(y, speed)])
            self.assertNotEqual(result.reason, 'entry_merge_start')
            self.assertIsNone(self.planner.active_candidate)

    def test_blocked_approach_brakes_once_and_reserves_room_for_waiting_launch(self):
        previous = float('inf')
        result = None
        for y in (250., 235., 220., 205., 190., 180., 173., 169., 168.7):
            ego = self.ego(y, min(10., previous))
            result = self.step(ego, [self.car(y, 10.)])
            self.assertEqual(result.target_link_id, p.ENTRY_MERGE_SOURCE)
            self.assertLessEqual(result.acc_target_velocity, previous+1e-9)
            self.assertTrue(all(pt.y > p.ENTRY_MERGE_END_Y for pt in result.path))
            previous = result.acc_target_velocity
        self.assertEqual(previous, 0.)
        ego = self.ego(168.7, 0.)
        for _ in range(4):
            self.assertEqual(self.step(ego, [self.car(168.7, 10.)]).acc_target_velocity, 0.)
        clear = self.step(ego, [self.car(125., 15.)])
        self.assertEqual(clear.reason, 'entry_merge_start')
        self.assertGreater(clear.acc_target_velocity, 0.)

    def test_committed_path_and_speed_do_not_flip_with_source_profile_or_rear_gap(self):
        result = self.step(self.ego())
        candidate = self.planner.active_candidate
        held_speed = result.acc_target_velocity
        # A nearby following car cannot replace the committed lateral target.
        for _ in range(4):
            ego = self.pose_on_candidate(candidate, 5, held_speed)
            result = self.step(ego, [self.car(ego.position.y+30., held_speed)], desired=6.)
            self.assertIs(self.planner.active_candidate, candidate)
            self.assertEqual(result.highway_state, self.planner.CHANGE_LEFT)
            self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)
            self.assertEqual(result.acc_target_velocity, held_speed)

    def test_new_front_obstacle_keeps_emergency_braking_and_lateral_path(self):
        self.step(self.ego())
        candidate = self.planner.active_candidate
        ego = self.pose_on_candidate(candidate, 30, 10.)
        front = candidate.points[40]
        obj = ObjectInfo(front.x, front.y, 0., 1, is_static=True, unique_id=88,
                         size_x=4.5, size_y=2., heading_deg=math.degrees(ego.yaw))
        result = self.step(ego, [{'object_info': obj}])
        self.assertEqual(result.reason, 'entry_merge_front_hazard')
        self.assertLess(result.acc_target_velocity, candidate.maneuver_speed_limit)
        self.assertIs(self.planner.active_candidate, candidate)

    def test_completion_holds_409_but_left_410_is_still_legal(self):
        self.step(self.ego())
        candidate = self.planner.active_candidate
        point = candidate.transition_end
        match = self.lanes.project(point, self.target)
        result = self.step(VehicleState(point.x, point.y, match.yaw, 12.))
        self.assertEqual(result.reason, 'entry_merge_complete')
        self.assertTrue(self.planner.changed_lane_hold)
        self.assertIsNone(self.planner.active_candidate)
        self.assertEqual([(d, lane.id) for d, lane in self.lanes.legal_neighbors(self.target)],
                         [('left', 'A2256W000410')])
        result = self.step(VehicleState(point.x, point.y, match.yaw, 12.))
        self.assertNotEqual(result.target_link_id, p.ENTRY_MERGE_SOURCE)

    def test_launch_gain_is_only_for_verified_entry_and_respects_other_stops(self):
        ego = self.ego(168.7, 0.)
        result = self.step(ego)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertEqual(PathController._entry_merge_launch_acceleration_gain(
            result.acc_target_velocity, ego, result), p.ENTRY_MERGE_LAUNCH_P_GAIN)
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(0., ego, result))
        result.debug_details['entry_merge']['verified'] = False
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(result.acc_target_velocity, ego, result))

    def test_invalid_heading_rejects_candidate_without_crashing(self):
        ego = self.ego()
        ego.yaw += math.pi
        self.planner.target_link_id = self.source.id
        result = self.step(ego)
        self.assertIsNone(self.planner.active_candidate)
        self.assertEqual(result.target_link_id, self.source.id)

    def test_static_fallback_searches_only_one_adjacent_lane(self):
        corridor = HighwayStaticCorridor(self.planner)
        corridor.executor = Mock()
        with patch('AIM2026.Expert.src.planning.highway.driving.obstacle_avoidance.HighwayCorridorSearch'):
            corridor.candidates(self.ego(), self.source, [], [])
        jobs = corridor.executor.submit.call_args.args[1]
        self.assertEqual([(d, [lane.id for lane in chain]) for _, d, chain in jobs],
                         [('left', [self.source.id, self.target.id])])

    def test_final_speed_holds_merge_plan_but_signal_stop_still_wins(self):
        ego = self.ego()
        result = self.step(ego)
        owner = SimpleNamespace(
            _merge_ego_curvature_speed_floor=lambda v, *a: v,
            _hold_merge_ego_curvature_speed=lambda v, *a: v,
            _second_merge_slowdown_band=lambda v, *a: v,
            _legacy_acc_entries=lambda *a: [],
            _smooth_static_target_speed=lambda v, *a: v,
            adaptive_cruise_control=SimpleNamespace(check_object=lambda *a: None,
                get_target_velocity=lambda v, target: target),
            traffic_light_stop_controller=SimpleNamespace(get_target_velocity=lambda **k: float('inf')),
            path_manager=SimpleNamespace(current_waypoint=0))
        args = (owner, ego, result.path, result.path, result, 3., False, float('inf'), [], [], None)
        self.assertAlmostEqual(target_speed(*args)[0], result.acc_target_velocity)
        owner.traffic_light_stop_controller.get_target_velocity = lambda **k: 0.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(target_speed(*args)[0], 0.)

    def test_recorded_moving_lead_matches_speed_before_turning(self):
        # 19:22 run: 411 car at 249.58, 409 lead 16.8 m ahead at ~10 m/s.
        ego = VehicleState(62.14, 249.58, math.radians(-89.), 11.88)
        lead = ObjectInfo(66., 232.8, 10., -1, unique_id=1,
                          size_x=4.55, size_y=1.98, heading_deg=-89.)
        result = self.step(ego, [{'object_info': lead}], desired=16.67)
        self.assertEqual(result.reason, 'entry_merge_approach')
        self.assertGreater(result.acc_target_velocity, 8.)
        self.assertLessEqual(result.acc_target_velocity, 10.)
        matched_speed = result.acc_target_velocity
        ego.velocity = matched_speed
        result = self.step(ego, [{'object_info': lead}], desired=16.67)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertEqual(result.acc_target_velocity, matched_speed)

    def test_side_by_side_pair_opens_gap_before_committing(self):
        ego = self.ego(speed=10.)
        result = self.step(ego, [self.car(ego.position.y-6., 10.)])
        self.assertEqual(result.reason, 'entry_merge_approach')
        self.assertGreater(result.acc_target_velocity, 0.)
        # Gap tuning changes how much slower we follow; preserve the behavior
        # (make room before steering), not a speed from the former 6 m setting.
        self.assertLess(result.acc_target_velocity, ego.velocity)
        self.assertIsNone(self.planner.active_candidate)

    def test_following_merge_distance_uses_maneuver_speed_not_road_limit(self):
        ego = VehicleState(62.275043, 236.620823, -1.5748076, 5.87942)
        lead = ObjectInfo(66.090681, 221.058725, 10.312314, -1, unique_id=1,
            size_x=4.759416, size_y=2.052044, heading_deg=-90.51246)
        result = self.step(ego, [{'object_info': lead}], desired=16.67)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertGreater(self.planner.active_candidate.transition_distance, 35.)
        self.assertLess(self.planner.active_candidate.transition_distance, 50.)

    def test_recorded_too_close_stopped_pose_does_not_force_tight_escape(self):
        # Last recorded frame from highway-20261025-192257-669031.
        ego = VehicleState(62.483734, 195.167789, -1.5618123, 0.)
        obstacle = ObjectInfo(62.5851145, 185.3102074, 0., -1, unique_id=9,
            is_static=True, size_x=4.5559579, size_y=1.9826465, heading_deg=-87.8007164)
        entries = [{'object_info': obstacle}]
        result = self.step(ego, entries, desired=16.67)
        self.assertEqual(result.reason, 'entry_merge_wait_gap')
        self.assertIsNone(self.planner.active_candidate)
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_source_obstacle_reserves_room_to_turn_while_waiting(self):
        ego = self.ego(215., 8.)
        obstacle = ObjectInfo(62.5851145, 185.3102074, 0., -1, unique_id=9,
            is_static=True, size_x=4.5559579, size_y=1.9826465, heading_deg=-87.8007164)
        result = self.step(ego, [{'object_info': obstacle}, self.car(215., 10.)])
        self.assertIsNone(self.planner.active_candidate)
        self.assertLess(result.acc_target_velocity, 3.)
        # After the adjacent traffic passes, the earlier waiting position has
        # space for a normal adjacent maneuver instead of a 10 m steering lunge.
        ego = self.ego(210., 0.)
        result = self.step(ego, [{'object_info': obstacle}, self.car(160., 12.)])
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertGreaterEqual(self.planner.active_candidate.transition_distance, 22.)

    def test_recorded_far_future_gap_does_not_restart_horizon_after_turning(self):
        # 19:39 run: scalar VE speed dipped to 8.56 while the measured lead
        # displacement remained ~10.84 m/s. Its gap only failed >5 s ahead,
        # after the remaining lane change and settling were already complete.
        ego = VehicleState(62.2359194, 247.5388844, -1.5515809855, 9.8798286)
        lead = ObjectInfo(66.0463332, 230.5294316, 10.4397755, -1, unique_id=1,
            size_x=4.8552989, size_y=2.096462, heading_deg=-89.57325)
        start = self.step(ego, [{'object_info': lead}], desired=16.6666667)
        self.assertEqual(start.reason, 'entry_merge_start')
        ego = VehicleState(64.7536316, 222.4997837, -1.4207883182, 9.6254253)
        lead = ObjectInfo(66.3699201, 203.9325891, 8.5621853, -1, unique_id=1,
            size_x=4.9902127, size_y=2.1060324, heading_deg=-87.74738)
        result = self.step(ego, [{'object_info': lead}], desired=16.6666667)
        self.assertEqual(result.reason, 'entry_merge_in_progress')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLessEqual(abs(result.acc_target_velocity-start.acc_target_velocity), .3)

    def test_recorded_source_motion_label_flip_does_not_toggle_braking(self):
        # Seed 110: same parked source object oscillated stationary/uncertain.
        # Both hypotheses remain collision-tested; only the unrelated following
        # gap on the abandoned source lane must not depend on that label.
        ego = VehicleState(62.4228719, 217.7436461, -1.5571606, 7.2633129)
        lead = ObjectInfo(66.2974143, 202.4780031, 10.1177959, -1, unique_id=1,
            size_x=4.8655411, size_y=2.0780327, heading_deg=-90.33597)
        start = self.step(ego, [{'object_info': lead}], desired=16.6666667)
        self.assertEqual(start.reason, 'entry_merge_start')
        ego = VehicleState(62.3837347, 216.0792399, -1.5585827, 7.3155181)
        lead = ObjectInfo(66.2071769, 199.9279838, 9.4417772, -1, unique_id=1,
            size_x=4.8964597, size_y=2.0772943, heading_deg=-90.48444)
        for uncertain in (True, False, True, False):
            obstacle = ObjectInfo(62.8500123, 186.2960385, 10.3594961 if uncertain else 0.,
                -1, unique_id=12, is_static=not uncertain, size_x=3.9504797,
                size_y=1.9675912, heading_deg=-90.29412)
            obstacle.motion_uncertain = uncertain
            result = self.step(ego, [{'object_info': lead}, {'object_info': obstacle}])
            self.assertEqual(result.reason, 'entry_merge_in_progress')
            self.assertGreater(result.acc_target_velocity, 0.)
            self.assertLessEqual(abs(result.acc_target_velocity-start.acc_target_velocity), .3)

    def test_uncertain_source_object_still_brakes_for_actual_body_collision(self):
        self.step(self.ego())
        candidate = self.planner.active_candidate
        ego = self.pose_on_candidate(candidate, 30, 10.)
        pt = candidate.points[40]
        obj = ObjectInfo(pt.x, pt.y, 9., -1, unique_id=88, size_x=4.9,
                         size_y=2.1, heading_deg=math.degrees(ego.yaw))
        obj.motion_uncertain = True
        result = self.step(ego, [{'object_info': obj}])
        self.assertEqual(result.reason, 'entry_merge_front_hazard')
        self.assertLess(result.acc_target_velocity, candidate.maneuver_speed_limit)

    def test_recorded_provisional_moving_box_keeps_valid_path_after_jitter(self):
        # A supplied positive speed must not force a short static-escape path.
        # The next observed pose must still pass the committed swept-body check.
        ego = VehicleState(62.3924269, 210.1578745, -1.5712373, 7.0462889)
        lead = ObjectInfo(66.1673734, 194.4425118, 10.2780619, -1, unique_id=1,
                         size_x=4.9833413, size_y=2.0946472, heading_deg=-90.742899)
        obj = ObjectInfo(62.3551717, 185.2316747, 9.5386543, -1, unique_id=13,
                        size_x=4.902994, size_y=2.0781453, heading_deg=-91.008631)
        obj.motion_uncertain = True
        start = self.step(ego, [{'object_info': lead}, {'object_info': obj}], desired=16.6666667)
        self.assertEqual(start.reason, 'entry_merge_start')
        candidate = self.planner.active_candidate
        self.assertTrue(candidate.valid)
        self.assertEqual(obj.velocity, 9.5386543)
        self.assertTrue(obj.motion_uncertain)
        self.assertAlmostEqual(obj.size_y, 2.0781453)
        ego = VehicleState(62.5054851, 208.6749258, -1.5652693, 7.0859109)
        obj.position = Point(62.5775319, 184.6135545)
        obj.size_x, obj.size_y, obj.velocity = 4.9959004, 2.0931845, 7.2105503
        result = self.step(ego, [{'object_info': lead}, {'object_info': obj}])
        self.assertEqual(result.reason, 'entry_merge_in_progress')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLessEqual(abs(result.acc_target_velocity-start.acc_target_velocity), .3)

    def test_completed_entry_uses_supplied_speed_but_still_stops_for_close_body(self):
        self.planner.target_link_id = self.target.id
        self.planner.changed_lane_hold = self.planner.entry_merge.completed = True
        ego = VehicleState(66.4799601, 138.637763, -1.5691475, 12.791818)
        obj = ObjectInfo(66.8830638, 107.4754374, 11.79942, -1, unique_id=16,
                        size_x=4.859129, size_y=2.105798, heading_deg=-90.28469)
        obj.motion_uncertain = True
        result = self.step(ego, [{'object_info': obj}], desired=16.66667)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertEqual(self.planner._front_lead_speed,obj.velocity)
        self.assertEqual(result.debug_details['merge_safety']['tracks'],1)
        # Actual body overlap must still trigger emergency braking.
        self.planner.reset()
        self.planner.target_link_id = self.target.id
        self.planner.changed_lane_hold = self.planner.entry_merge.completed = True
        obj.position = Point(66.6, 135.)
        result = self.step(ego, [{'object_info': obj}], desired=16.66667)
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertNotIn('uncertain_front_acc_only', result.debug_details['merge_safety'])

    def test_recorded_source_provisional_speed_dip_does_not_restore_following_gap(self):
        ego = VehicleState(62.4228719, 217.7436461, -1.564693, 7.7731456)
        lead = ObjectInfo(66.2056891, 201.9460266, 10.7811174, -1, unique_id=1,
                         size_x=4.6786965, size_y=2.0414443, heading_deg=-90.12375)
        start = self.step(ego, [{'object_info': lead}], desired=16.666667)
        self.assertEqual(start.reason, 'entry_merge_start')
        ego = VehicleState(63.779571, 199.948262, -1.4135387, 8.266)
        lead.position = Point(66.3165564, 177.6403338)
        lead.velocity = 11.7705812
        obj = ObjectInfo(62.7473094, 185.2562772, 1.5632603, -1, unique_id=10,
                        size_x=4.6996261, size_y=2.0327818, heading_deg=-88.06855)
        obj.motion_uncertain = True
        result = self.step(ego, [{'object_info': lead}, {'object_info': obj}])
        self.assertEqual(result.reason, 'entry_merge_in_progress')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLessEqual(abs(result.acc_target_velocity-start.acc_target_velocity), .3)
        self.assertEqual(obj.velocity, 1.5632603)

    def test_completed_entry_does_not_remove_other_lanes_from_merge_guard(self):
        ego = VehicleState(66.4799601, 138.637763, -1.5691475, 12.791818)
        self.planner.target_link_id = self.target.id
        self.planner.changed_lane_hold = self.planner.entry_merge.completed = True
        far = self.car(107., 11., identity=16)
        crossing = self.car(132., 11., identity=17, lane=self.source)
        far['object_info'].motion_uncertain = True
        crossing['object_info'].motion_uncertain = True
        result = self.step(ego, [far, crossing], desired=16.66667)
        guard = result.debug_details['merge_safety']
        self.assertEqual(guard['tracks'], 2)
        self.assertEqual(set(self.planner.merge_safety.tracks),{'npc:16','npc:17'})

    def test_completed_entry_keeps_front_acc_across_successor_link_boundary(self):
        self.planner.target_link_id = self.target.id
        self.planner.changed_lane_hold = self.planner.entry_merge.completed = True
        ego = VehicleState(66.63, 107.18, math.radians(-89.94), 14.11)
        obj = ObjectInfo(66.9775689, 76.6875191, 11., -1, unique_id=16,
                        size_x=5., size_y=2.1, heading_deg=-90.31518)
        obj.motion_uncertain = True
        result = self.step(ego, [{'object_info': obj}], desired=16.66667)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertEqual(self.planner._front_lead_speed,obj.velocity)
        self.assertEqual(result.debug_details['merge_safety']['tracks'],1)

    def test_five_kinematic_rollouts_with_real_controller(self):
        # Offline bicycle model: verifies path/controller continuity, not MORAI physics.
        config = json.loads((Path(__file__).resolve().parents[1]/'Expert/src/config/config.json').read_text())
        for y, speed in ((250., 10.), (235., 12.), (205., 8.), (180., 4.), (168.7, 0.)):
            with self.subTest(y=y, speed=speed):
                self.planner.reset()
                ego = self.ego(y, speed)
                controller = PathController(config)
                completed = False
                held = None
                with contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(600):
                        result = self.step(ego)
                        if held is None and result.reason == 'entry_merge_approach':
                            self.assertIsNone(self.planner.active_candidate)
                        else:
                            self.assertIn(result.reason, ('entry_merge_start', 'entry_merge_in_progress', 'entry_merge_complete'))
                            held = result.acc_target_velocity if held is None else held
                            self.assertAlmostEqual(result.acc_target_velocity, held)
                        self.assertGreater(result.acc_target_velocity, 0.)
                        if result.reason == 'entry_merge_complete':
                            completed = True
                            self.assertGreater(ego.position.y-p.VEHICLE_LENGTH, p.ENTRY_MERGE_END_Y)
                            break
                        # Use the configured physical limits; remove only simulator sign convention.
                        acc, _, steer = controller._calculate_control(ego, result.path, result.acc_target_velocity, result)
                        steer /= controller.steering_sign
                        dt = 1./30.
                        ego.velocity = max(0., ego.velocity + (3.*acc if acc >= 0 else 6.*acc)*dt)
                        ego.yaw += ego.velocity/3.*math.tan(steer)*dt
                        ego.position = Point(ego.position.x+ego.velocity*math.cos(ego.yaw)*dt,
                                             ego.position.y+ego.velocity*math.sin(ego.yaw)*dt)
                self.assertTrue(completed, (ego.position.x, ego.position.y))

    def test_dense_traffic_waits_with_turning_room_then_completes_without_rebraking(self):
        # Four cars close the target gap while a parked source car enters the
        # 30 m perception ROI. Its first observations have uncertain motion.
        config = json.loads((Path(__file__).resolve().parents[1]/'Expert/src/config/config.json').read_text())
        controller = PathController(config)
        ego = self.ego(250., 11.)
        stopped = self.car(185., 0., identity=99, lane=self.source)['object_info']
        stopped.is_static = True
        seen_at = None
        waited = completed = False
        held = None
        dt = 1./30.
        with contextlib.redirect_stdout(io.StringIO()):
            for frame in range(900):
                now = frame*dt
                true_traffic = [self.car(y-11.*now, 11., identity=i)
                                for i, y in enumerate((240., 265., 290., 315.))]
                observed = [e for e in true_traffic
                            if -30. <= e['object_info'].position.y-ego.position.y <= 20.]
                if ego.position.y-stopped.position.y <= 30.:
                    seen_at = now if seen_at is None else seen_at
                    seen = self.car(185., 9. if now-seen_at < .5 else 0.,
                                    identity=99, lane=self.source)['object_info']
                    seen.motion_uncertain = now-seen_at < .5
                    seen.is_static = not seen.motion_uncertain
                    observed.append({'object_info': seen})
                result = self.step(ego, observed)
                if held is None:
                    waited |= result.reason == 'entry_merge_wait_gap'
                    if result.reason == 'entry_merge_start':
                        held = result.acc_target_velocity
                        self.assertTrue(waited)
                        self.assertGreater(ego.position.y, 200.)
                else:
                    self.assertIn(result.reason, ('entry_merge_in_progress', 'entry_merge_complete'))
                    self.assertGreater(result.acc_target_velocity, 0.)
                    self.assertLessEqual(abs(result.acc_target_velocity-held), p.ENTRY_MERGE_FOLLOW_ACCEL*dt+1e-6)
                    held = result.acc_target_velocity
                # Use all simulated objects here, including those outside ROI.
                for obj in [e['object_info'] for e in true_traffic]+[stopped]:
                    self.assertFalse(self.planner._obb_collision(
                        ego.position, ego.yaw, obj.position, math.radians(obj.heading_deg), obj))
                if result.reason == 'entry_merge_complete':
                    completed = True
                    self.assertGreater(ego.position.y-p.VEHICLE_LENGTH, p.ENTRY_MERGE_END_Y)
                    break
                acc, _, steer = controller._calculate_control(ego, result.path, result.acc_target_velocity, result)
                steer /= controller.steering_sign
                if held is not None:
                    self.assertGreaterEqual(acc, -.01)
                ego.velocity = max(0., ego.velocity+(3.*acc if acc >= 0 else 6.*acc)*dt)
                ego.yaw += ego.velocity/3.*math.tan(steer)*dt
                ego.position = Point(ego.position.x+ego.velocity*math.cos(ego.yaw)*dt,
                                     ego.position.y+ego.velocity*math.sin(ego.yaw)*dt)
        self.assertTrue(completed)


if __name__ == '__main__':
    unittest.main()
