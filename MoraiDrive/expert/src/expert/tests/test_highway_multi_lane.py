"""Adjacent lanes stay first choice; two-leg escapes check all crossed traffic."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import unittest

from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.vehicle_state import VehicleState


class HighwayMultiLaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lane_map = HighwayLaneMap(str(Path(__file__).resolve().parents[3] /
                                         'map_data/R_KR_PG_KATRI_2025'))

    def scene(self, via_s=110.0):
        planner = HighwayLanePlanner(lane_map=copy.deepcopy(self.lane_map))
        lane = planner.lane_map.links['A2256W000411']
        pos, yaw, _ = planner.lane_map.point_at_s(lane, 10.0)
        ego = VehicleState(pos.x, pos.y, yaw, 20.0)
        origin = planner.lane_map.project(pos, lane)
        path = [row[0] for row in planner.lane_map.route_points(origin, 260.0, 0.5)]
        entries = [self.object(planner, ego, 'A2256W000411', 55.0, 15.0, 1)]
        if via_s is not None:
            entries.append(self.object(planner, ego, 'A2256W000409', via_s, 0.0, 2))
        return planner, ego, entries, path

    @staticmethod
    def object(planner, ego, link_id, s, speed, uid):
        pos, yaw, _ = planner.lane_map.point_at_s(planner.lane_map.links[link_id], s)
        obj = ObjectInfo(pos.x, pos.y, speed, 2 if speed == 0.0 else 1,
                         is_static=speed == 0.0, size_x=4.5, size_y=2.0,
                         heading_deg=math.degrees(yaw), unique_id=uid)
        return dict(object_info=obj, local_position=(pos-ego.position).rotate(-ego.yaw))

    def test_adjacent_lane_is_preferred_even_if_far_lane_is_empty(self):
        for via_s in (None, 140.0):
            planner, ego, entries, path = self.scene(via_s)
            result = planner.plan(ego, path, entries, desired_velocity=22.0)
            self.assertEqual('CHANGE_LEFT', result.highway_state)
            self.assertEqual('A2256W000409', result.target_link_id)
            self.assertNotIn('multi_lane_change', result.debug_details)
            self.assertFalse(any(getattr(c, 'lane_change_count', 1) == 2 for c in result.candidates))

    def test_two_lanes_only_when_adjacent_path_is_blocked(self):
        planner, ego, entries, path = self.scene()
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
        self.assertEqual('A2256W000410', result.target_link_id)
        self.assertEqual('A2256W000409', result.debug_details['multi_lane_change']['via'])
        self.assertTrue(all(not c.valid for c in result.candidates
                            if c.direction != 'keep' and getattr(c, 'lane_change_count', 1) == 1))
        self.assertTrue(planner.active_candidate.lane_change_verified)
        self.assertGreaterEqual(result.acc_target_velocity, 22.0)

    def test_safe_right_neighbor_wins_over_a_two_lane_left_escape(self):
        planner, _, _, _ = self.scene()
        lane = planner.lane_map.links['A2256W000409']
        pos, yaw, _ = planner.lane_map.point_at_s(lane, 10.0)
        ego = VehicleState(pos.x, pos.y, yaw, 20.0)
        origin = planner.lane_map.project(pos, lane)
        path = [row[0] for row in planner.lane_map.route_points(origin, 260.0, 0.5)]
        entries = [self.object(planner, ego, lane.id, 55.0, 15.0, 1),
                   self.object(planner, ego, 'A2256W000410', 110.0, 0.0, 2)]
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('CHANGE_RIGHT', result.highway_state)
        self.assertEqual('A2256W000411', result.target_link_id)
        self.assertFalse(any(getattr(c, 'lane_change_count', 1) == 2 for c in result.candidates))

    def test_occupied_intermediate_lane_is_not_skipped(self):
        planner, ego, entries, path = self.scene()
        entries.append(self.object(planner, ego, 'A2256W000409', 10.0, 20.0, 3))
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('ACC_FOLLOW', result.highway_state, result.debug_details)
        self.assertFalse(any(c.valid for c in result.candidates if c.direction != 'keep'))

    def test_final_lane_fast_rear_blocks_the_whole_maneuver(self):
        planner, ego, entries, path = self.scene()
        # Ego and rear start nearly alongside in longitudinal distance; the
        # far lane is not occupied until the second leg several seconds later.
        entries.append(self.object(planner, ego, 'A2256W000410', 0.0, 30.0, 3))
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('ACC_FOLLOW', result.highway_state, result.debug_details)

    def test_far_rear_is_checked_at_second_entry_time_not_just_now(self):
        for behind_start, blocked in ((45.0, False), (80.0, True)):
            planner, ego, entries, path = self.scene()
            rear = self.object(planner, ego, 'A2256W000410', 0.0, 40.0, 3)
            obj = rear['object_info']
            angle = math.radians(obj.heading_deg)
            obj.position = Point(obj.position.x - behind_start * math.cos(angle),
                                 obj.position.y - behind_start * math.sin(angle))
            rear['local_position'] = (obj.position-ego.position).rotate(-ego.yaw)
            result = planner.plan(ego, path, entries + [rear], desired_velocity=22.0)
            self.assertEqual('ACC_FOLLOW' if blocked else 'CHANGE_LEFT', result.highway_state,
                             result.debug_details)
            if blocked:
                candidate = next(c for c in result.candidates if getattr(c, 'lane_change_count', 1) == 2)
                self.assertTrue(candidate.prediction_conflict['rear'])
                self.assertGreater(candidate.prediction_conflict['time'], 3.0)

    def test_forbidden_second_boundary_and_excluded_final_lane_are_preserved(self):
        for excluded in (False, True):
            planner, ego, entries, path = self.scene()
            if excluded:
                planner.lane_map.links['A2256W000410'].excluded = True
            else:
                planner.lane_map.links['A2256W000409'].can_move_left = False
            result = planner.plan(ego, path, entries, desired_velocity=22.0)
            self.assertEqual('ACC_FOLLOW', result.highway_state)
            self.assertNotIn('multi_lane_change', result.debug_details)

    def test_intermediate_arrival_does_not_finish_or_wait_for_cooldown(self):
        planner, ego, entries, path = self.scene()
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        candidate = planner.active_candidate
        point = candidate.via_candidate.transition_end
        projection = planner.lane_map.project(point, planner.lane_map.links[candidate.via_link_id])
        ego = VehicleState(point.x, point.y, projection.yaw, 22.0)
        # Advance the original lead to the same elapsed time.
        lead = entries[0]['object_info']
        origin = planner.lane_map.project(lead.position, planner.lane_map.links['A2256W000411'])
        lead.position, angle, _ = planner.lane_map.advance_default(origin, 15.0 * 2.6)
        lead.heading_deg = math.degrees(angle)
        for entry in entries:
            entry['local_position'] = (entry['object_info'].position-ego.position).rotate(-ego.yaw)
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
        self.assertEqual('A2256W000410', result.target_link_id)
        self.assertIsNone(planner._lane_change_cooldown_origin)
        self.assertGreaterEqual(result.acc_target_velocity, 22.0)

    def test_new_final_lane_traffic_can_cancel_second_leg_without_stopping(self):
        planner, ego, entries, path = self.scene()
        planner.plan(ego, path, entries, desired_velocity=22.0)
        # The old intermediate obstruction moves away; a new far-lane vehicle
        # appears before ego has crossed the first boundary.
        entries = entries[:1] + [self.object(planner, ego, 'A2256W000410', 95.0, 0.0, 3)]
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
        self.assertEqual('A2256W000409', result.target_link_id)
        self.assertTrue(result.debug_details['multi_lane_held_intermediate'])
        self.assertGreaterEqual(result.acc_target_velocity, 22.0)

    def test_cancelled_second_leg_returns_from_measured_pose_without_a_path_jump(self):
        planner, ego, entries, path = self.scene()
        planner.plan(ego, path, entries, desired_velocity=22.0)
        candidate = planner.active_candidate
        point = candidate.via_candidate.transition_end
        projection = planner.lane_map.project(point, planner.lane_map.links[candidate.via_link_id])
        ego = VehicleState(point.x - 0.5 * math.sin(projection.yaw),
                           point.y + 0.5 * math.cos(projection.yaw), projection.yaw, 22.0)
        # The old via-lane obstruction is now moving away, so the first lane
        # is a verified fallback while a new obstacle blocks the second lane.
        entries = [self.object(planner, ego, 'A2256W000411', 94.0, 25.0, 1),
                   self.object(planner, ego, 'A2256W000409', 110.0, 25.0, 2),
                   self.object(planner, ego, 'A2256W000410', 95.0, 0.0, 3)]
        result = planner.plan(ego, path, entries, desired_velocity=22.0)
        self.assertTrue(result.debug_details.get('multi_lane_held_intermediate'), result.debug_details)
        self.assertEqual('A2256W000409', result.target_link_id)
        self.assertLess(result.path[0].distance(ego.position), 1e-8)
        self.assertGreaterEqual(result.acc_target_velocity, 22.0)

    def test_two_leg_path_completes_with_real_controller_and_no_collision(self):
        planner, ego, entries, path = self.scene()
        lead = entries[0]['object_info']
        origin = planner.lane_map.project(lead.position, planner.lane_map.links['A2256W000411'])
        config = Path(__file__).resolve().parents[1] / 'src/config/config.json'
        controller = PathController(json.loads(config.read_text(encoding='utf-8')))
        dt = 0.05
        planner.set_control_dt(dt)
        minimum_speed = ego.velocity
        changing = []
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(160):
                lead.position, angle, _ = planner.lane_map.advance_default(origin, 15.0 * step * dt)
                lead.heading_deg = math.degrees(angle)
                for entry in entries:
                    entry['local_position'] = (entry['object_info'].position-ego.position).rotate(-ego.yaw)
                result = planner.plan(ego, path, entries, desired_velocity=22.0)
                changing.append(result.highway_state == 'CHANGE_LEFT')
                target = min(22.0, result.acc_target_velocity)
                self.assertGreater(target, 0.05, (step, result.reason, result.debug_details))
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steer = controller._calculate_control(ego, result.path, target, result)
                speed = max(0.0, ego.velocity + (3.0 if pedal >= 0 else 5.0) * pedal * dt)
                ego = VehicleState(ego.position.x + speed * math.cos(ego.yaw) * dt,
                                   ego.position.y + speed * math.sin(ego.yaw) * dt,
                                   ego.yaw + speed / 3.0 * math.tan(steer/controller.steering_sign) * dt,
                                   speed)
                minimum_speed = min(minimum_speed, speed)
                for entry in entries:
                    obj = entry['object_info']
                    self.assertFalse(planner._obb_collision(ego.position, ego.yaw, obj.position,
                        math.radians(obj.heading_deg), obj), (step, result.reason, obj.unique_id))
        self.assertTrue(planner.changed_lane_hold)
        self.assertLess(planner.lane_map.match(ego.position, heading=ego.yaw).lane_link.lane_number, 4)
        self.assertGreater(minimum_speed, 19.0)
        self.assertTrue(changing[0])
        first_completed = changing.index(False)
        self.assertFalse(any(changing[first_completed:]), 'No stop/start between the two legs')

    def test_reset_and_follow_only_do_not_keep_a_two_lane_goal(self):
        planner, ego, entries, path = self.scene()
        planner.plan(ego, path, entries, desired_velocity=22.0)
        planner.reset()
        self.assertIsNone(planner.active_candidate)
        result = planner.plan(ego, path, entries, desired_velocity=22.0, allow_lane_change=False)
        self.assertNotIn(result.highway_state, ('CHANGE_LEFT', 'CHANGE_RIGHT'))


if __name__ == '__main__':
    unittest.main()
