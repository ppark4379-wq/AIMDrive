"""Highway static avoidance without changes to merge/roundabout scenarios."""

import math
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning import parameters as p
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.obstacle_motion import ObstacleMotion
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.planning.highway_lane_planning.structures import HighwayLaneLink
from expert.src.planning.lattice_planning.planning_process import LatticePlanningProcess
from expert.src.vehicle_state import VehicleState


def entry(x, lane=0, speed=0.0, static=False, uid=1, width=2.0):
    obj = ObjectInfo(x, 2000.0 + lane * 3.5, speed, 2, is_static=static,
                     unique_id=uid, size_x=4.5, size_y=width, heading_deg=0.0)
    return {'object_info': obj, 'local_position': Point(x, lane * 3.5)}


def straight_map():
    lane_map = HighwayLaneMap.__new__(HighwayLaneMap)
    lane_map.links = {}
    for key, lane, number, left, right in (
        ('center', 0, 3, 'left', 'right'), ('left', 1, 2, None, 'center'),
        ('right', -1, 4, 'center', None),
    ):
        points = [Point(-200.0, 2000.0 + lane * 3.5), Point(500.0, 2000.0 + lane * 3.5)]
        lane_map.links[key] = HighwayLaneLink(dict(idx=key, ego_lane=number,
            left_lane_change_dst_link_idx=left, right_lane_change_dst_link_idx=right,
            can_move_left_lane=left is not None, can_move_right_lane=right is not None),
            0, points, [0.0, 700.0], False)
    lane_map.successors = {key: () for key in lane_map.links}
    lane_map.max_section_index = 0
    return lane_map


class ObstacleMotionTest(unittest.TestCase):
    def test_explicit_static_is_immediate_but_moving_label_is_not(self):
        tracker = ObstacleMotion()
        objects = [entry(80, static=True), entry(90, speed=5, static=True, uid=2)]
        tracker.update(objects, 0.1)
        self.assertTrue(tracker.is_static(objects[0]['object_info']))
        self.assertFalse(tracker.is_static(objects[1]['object_info']))

    def test_moving_vehicle_requires_two_seconds_and_releases_on_restart(self):
        for dt in (0.02, 0.1):
            with self.subTest(dt=dt):
                tracker = ObstacleMotion()
                tracker.update([entry(80, speed=15)], dt)
                for _ in range(round(p.STOP_CONFIRM_TIME / dt)):
                    stopped = entry(80)
                    tracker.update([stopped], dt)
                    self.assertFalse(tracker.is_static(stopped['object_info']))
                tracker.update([stopped], dt)
                self.assertTrue(tracker.is_static(stopped['object_info']))
                moving = entry(80.1, speed=1.0)
                tracker.update([moving], dt)
                self.assertFalse(tracker.is_static(moving['object_info']))

    def test_initial_stationary_npc_confirms_quickly_without_an_ego_stop(self):
        tracker = ObstacleMotion()
        for _ in range(6):
            stopped = entry(80, uid=None)
            tracker.update([stopped], 0.1)
        self.assertTrue(tracker.is_static(stopped['object_info']))
        self.assertEqual(1, len(tracker.tracks))

    def test_missing_frames_do_not_count_as_observed_stopping_time(self):
        tracker = ObstacleMotion()
        tracker.update([entry(80, speed=10)], 0.1)
        for _ in range(15):
            tracker.update([entry(80)], 0.1)
        for _ in range(5):
            tracker.update([], 0.1)
        stopped = entry(80)
        tracker.update([stopped], 0.1)
        self.assertFalse(tracker.is_static(stopped['object_info']))
        self.assertEqual(0.0, tracker.states[id(stopped['object_info'])]['seconds'])

    def test_position_motion_and_reset_clear_stationary_history(self):
        tracker = ObstacleMotion()
        stopped = entry(80)
        for _ in range(6):
            tracker.update([stopped], 0.1)
        self.assertTrue(tracker.is_static(stopped['object_info']))
        stopped['object_info'].position[0] += 1.0
        tracker.update([stopped], 0.1)
        self.assertFalse(tracker.is_static(stopped['object_info']))
        tracker.reset()
        self.assertFalse(tracker.tracks)


class HighwayStaticAvoidanceTest(unittest.TestCase):
    def setUp(self):
        self.map = straight_map()
        self.planner = HighwayLanePlanner(lane_map=self.map)
        self.ego = VehicleState(0.0, 2000.0, 0.0, 20.0)
        self.reference = [Point(x, 2000.0) for x in range(161)]

    def plan(self, entries, speed=20.0):
        return self.planner.plan(self.ego, self.reference, entries, desired_velocity=speed)

    def assert_change_at_route_speed(self, result):
        self.assertIn(result.highway_state, ('CHANGE_LEFT', 'CHANGE_RIGHT'))
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))

    def test_highway_process_passes_explicit_static_obstacles_to_planner(self):
        process = LatticePlanningProcess(highway_lane_planner=self.planner)
        result = process._run_highway_zone(self.ego, self.reference,
                                          [entry(80, static=True)], desired_velocity=20.0)
        self.assert_change_at_route_speed(result)
        self.assertEqual('STATIC_EXPLICIT', result.debug_details['obstacle_motion'][0]['state'])

    def test_static_with_less_than_three_seconds_ttc_can_change_without_braking(self):
        result = self.plan([entry(50, static=True)])
        self.assertLess(result.ttc, 3.0)
        self.assert_change_at_route_speed(result)

    def test_too_close_static_rejects_paths_that_sweep_through_it(self):
        result = self.plan([entry(18, static=True)])
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertTrue(result.acc_override)
        self.assertLess(result.acc_target_velocity, self.ego.velocity)
        self.assertTrue(all(not c.valid for c in result.candidates[1:]))

    def test_fast_rear_blocks_one_lane_and_other_lane_is_used(self):
        result = self.plan([entry(80, static=True), entry(-60, lane=1, speed=40, uid=2)])
        self.assert_change_at_route_speed(result)
        self.assertEqual('right', result.target_link_id)

    def test_fast_rear_in_both_lanes_keeps_acc_in_source_lane(self):
        result = self.plan([entry(60, static=True), entry(-60, 1, 40, uid=2),
                            entry(-60, -1, 40, uid=3)])
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertEqual('center', result.target_link_id)
        self.assertTrue(result.acc_override)

    def test_safe_following_car_does_not_prevent_lane_change(self):
        result = self.plan([entry(70, static=True), entry(-40, 1, 18, uid=2),
                            entry(0, -1, 20, uid=3)])
        self.assert_change_at_route_speed(result)
        self.assertEqual('left', result.target_link_id)

    def test_same_speed_rear_with_clear_body_gap_does_not_need_twenty_meters(self):
        result = self.plan([entry(70, static=True), entry(-18, 1, 20, uid=2),
                            entry(0, -1, 20, uid=3)])
        self.assert_change_at_route_speed(result)
        self.assertEqual('left', result.target_link_id)

    def test_smaller_rear_margin_still_rejects_fast_closing_car(self):
        result = self.plan([entry(70, static=True), entry(-18, 1, 35, uid=2),
                            entry(0, -1, 20, uid=3)])
        self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_requested_deceleration_is_checked_against_the_rear_car(self):
        result = self.plan([entry(80, static=True), entry(-60, 1, 25, uid=2),
                            entry(0, -1, 20, uid=3)], speed=8.0)
        self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_adjacent_front_static_blocks_that_lane(self):
        result = self.plan([entry(80, static=True), entry(35, 1, static=True, uid=2)])
        self.assert_change_at_route_speed(result)
        self.assertEqual('right', result.target_link_id)

    def test_static_body_is_checked_even_without_a_lane_match(self):
        wide = entry(70, lane=2, static=True, width=25.0)
        self.assertIsNone(self.map.match(wide['object_info'].position))
        result = self.plan([wide])
        self.assertTrue(math.isfinite(result.debug_details['static_obstacle_distance']))
        self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_shoulder_static_does_not_trigger_avoidance(self):
        result = self.plan([entry(70, lane=2, static=True)])
        self.assertEqual('KEEP_LANE', result.highway_state)
        self.assertFalse(result.acc_override)

    def test_confirmed_static_can_override_lane_change_cooldown(self):
        self.planner._lane_change_cooldown_origin = Point(0.0, 2000.0)
        self.assert_change_at_route_speed(self.plan([entry(70, static=True)]))

    def test_slower_lead_is_overtaken_before_acc_forces_a_slowdown(self):
        self.assert_change_at_route_speed(self.plan([entry(60, speed=15)]))

    def test_already_following_at_lead_speed_uses_requested_speed_to_overtake(self):
        self.ego.velocity = 12.0
        self.assert_change_at_route_speed(self.plan([entry(45, speed=12)], speed=20))

    def test_same_speed_traffic_does_not_cause_unnecessary_changes(self):
        result = self.plan([entry(50, speed=20)])
        self.assertEqual('KEEP_LANE', result.highway_state)

    def test_rear_on_connected_predecessor_is_checked(self):
        target = self.map.links['left']
        target.points = [Point(-10, 2003.5), Point(500, 2003.5)]
        target.cumulative_s = [0.0, 510.0]
        target.length = 510.0
        self.map.links['previous_left'] = HighwayLaneLink(dict(idx='previous_left', ego_lane=2),
            0, [Point(-200, 2003.5), Point(-10, 2003.5)], [0.0, 190.0], False)
        self.map.successors['previous_left'] = ('left',)
        result = self.plan([entry(70, static=True), entry(-25, 1, 30, uid=2),
                            entry(0, -1, 20, uid=3)])
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertTrue(any('blocked-rear' in reason for reason in result.debug_details['lane_checks']))

    def test_distant_rear_conflict_is_seen_beyond_legacy_point_four_seconds(self):
        target = self.map.links['left']
        candidate = self.planner._build_candidate(self.ego, target,
            self.map.project(self.ego.position, target), 'left', self.map.links['center'])
        entries = [entry(-95, 1, 50, uid=2)]
        conflict = self.planner.lane_change_safety.check(candidate, self.ego, entries,
            self.planner._match_objects(entries), 20.0)
        self.assertIsNotNone(conflict)
        self.assertTrue(conflict['rear'])
        self.assertGreater(conflict['time'], p.PREDICTION_HORIZON)

    def test_static_maneuver_finishes_at_speed_with_a_safe_rear_vehicle(self):
        obstacles = [entry(55, static=True), entry(-40, 1, 18, uid=2),
                     entry(0, -1, 20, uid=3)]
        result = self.plan(obstacles)
        self.assert_change_at_route_speed(result)
        path = list(result.path)
        cumulative = self.planner._cumulative_distance(path)
        self.planner.set_control_dt(0.1)
        for step in range(1, 31):
            t = step * 0.1
            point, yaw = self.planner._pose_at_distance(path, cumulative, 20.0 * t)
            self.ego = VehicleState(point.x, point.y, yaw, 20.0)
            objects = [entry(55, static=True), entry(-40 + 18*t, 1, 18, uid=2),
                       entry(20*t, -1, 20, uid=3)]
            result = self.plan(objects)
            self.assertFalse(result.acc_override, (step, result.reason, result.debug_details))
            for obstacle in objects:
                obj = obstacle['object_info']
                self.assertFalse(self.planner._obb_collision(point, yaw, obj.position, 0.0, obj))
        self.assertEqual('KEEP_LANE', result.highway_state)
        self.assertEqual('left', result.target_link_id)

    def test_stopped_lead_is_reclassified_then_avoided_when_lane_opens(self):
        self.planner.set_control_dt(0.1)
        blockers = [entry(0, 1, 20, uid=2), entry(0, -1, 20, uid=3)]
        self.plan([entry(60, speed=15)] + blockers)
        for _ in range(20):
            result = self.plan([entry(60)] + blockers)
            self.assertEqual('STOPPING', result.debug_details['obstacle_motion'][0]['state'])
        result = self.plan([entry(60)])
        self.assertEqual('STATIC_CONFIRMED', result.debug_details['obstacle_motion'][0]['state'])
        self.assert_change_at_route_speed(result)

    def test_new_front_obstacle_during_change_is_rechecked(self):
        result = self.plan([entry(80, static=True)])
        point, yaw = self.planner._pose_at_distance(result.path,
            self.planner._cumulative_distance(result.path), 25.0)
        self.ego = VehicleState(point.x, point.y, yaw, 20.0)
        lane = 1 if result.target_link_id == 'left' else -1
        result = self.plan([entry(80, static=True), entry(40, lane, static=True, uid=2)])
        self.assertTrue(result.acc_override)
        self.assertLess(result.acc_target_velocity, 20.0)
        self.assertIn(result.highway_state, ('CHANGE_LEFT', 'CHANGE_RIGHT'))

    def test_follow_only_section_still_brakes_for_static_without_discarding_a_safe_change(self):
        result = self.planner.plan(self.ego, self.reference, [entry(35, static=True)],
                                   desired_velocity=20.0, allow_lane_change=False)
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertIs(self.reference, result.path)
        self.assertLess(result.acc_target_velocity, 20.0)

    def test_new_fast_rear_before_entry_can_return_to_source_lane(self):
        result = self.plan([entry(80, static=True), entry(0, -1, 20, uid=3)])
        self.assertEqual('left', result.target_link_id)
        point, yaw = self.planner._pose_at_distance(result.path,
            self.planner._cumulative_distance(result.path), 12.0)
        self.ego = VehicleState(point.x, point.y, yaw, 20.0)
        result = self.plan([entry(80, static=True), entry(-10, 1, 40, uid=2)])
        self.assertTrue(result.debug_details.get('returned_to_source_for_rear_vehicle'))
        self.assertEqual('center', result.target_link_id)

    def test_same_physical_static_does_not_freeze_after_restart(self):
        stopped = entry(60)
        for _ in range(17):
            self.planner.obstacle_motion.update([stopped], self.planner.control_dt)
        self.assertTrue(self.planner.obstacle_motion.is_static(stopped['object_info']))
        result = self.plan([entry(60, speed=20)])
        self.assertEqual('MOVING', result.debug_details['obstacle_motion'][0]['state'])
        self.assertEqual('KEEP_LANE', result.highway_state)


if __name__ == '__main__':
    unittest.main()
