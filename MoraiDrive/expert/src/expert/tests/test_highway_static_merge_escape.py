"""Regression geometry from the 2026-09-23 stopped-at-merge driving log/video."""

import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning import parameters as p
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.vehicle_state import VehicleState


class StaticMergeEscapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.map = HighwayLaneMap(str(cls.root.parents[1] / 'map_data/R_KR_PG_KATRI_2025'))
        cls.config = json.loads((cls.root / 'src/config/config.json').read_text(encoding='utf-8'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.map)
        self.ego = VehicleState(62.159622, 251.022369, math.radians(-89.0495), 0.0)
        # Actual ROS obstacle pose/size, read without changing simulator state.
        self.blocker = ObjectInfo(61.904037, 238.801269, 0.0, 2, is_static=True,
                                  size_x=4.69, size_y=2.2, heading_deg=-90.0, unique_id=41)

    def plan(self, others=(), reference_path=None, prediction_path=None):
        match = self.map.match(self.ego.position, heading=self.ego.yaw)
        path = (reference_path if reference_path is not None else
                [pose[0] for pose in self.map.route_points(match, 170.0, 0.5)])
        entries = []
        for obj in [self.blocker] + list(others):
            entries.append(dict(object_info=obj, local_position=(obj.position-self.ego.position).rotate(-self.ego.yaw)))
        return self.planner.plan(self.ego, path, entries, desired_velocity=27.78,
                                 prediction_path=prediction_path)

    def test_completed_escape_accelerates_with_global_route_still_in_source_lane(self):
        self.assertEqual('CHANGE_LEFT', self.plan().highway_state)
        # Logged pose where the maneuver finished, followed by the stopped pose.
        # The global route remains in lane 5 after the local escape into lane 4.
        for x, y, heading, speed in ((66.27, 237.76, -88.10, 3.29),
                                      (66.26, 237.31, -88.63, 0.0)):
            self.ego = VehicleState(x, y, math.radians(heading), speed)
            source = self.map.project(self.ego.position, self.map.links['A2256W000411'])
            source_path = [pose[0] for pose in self.map.route_points(source, 170.0, 0.5)]
            for _ in range(3):
                result = self.plan(reference_path=source_path[:50], prediction_path=source_path)
                self.assertTrue(self.planner.changed_lane_hold)
                self.assertIsNone(self.planner.active_candidate)
                self.assertEqual('A2256W000409', result.target_link_id)
                self.assertEqual(27.78, min(27.78, result.acc_target_velocity), result.debug_details)
                self.assertIsNone(result.debug_details['merge_safety']['conflict'])

    def test_completed_escape_still_brakes_for_a_real_target_lane_hazard(self):
        self.plan()
        self.ego = VehicleState(66.27, 237.76, math.radians(-88.10), 3.29)
        self.plan()  # Finish the maneuver before the new vehicle appears.
        source = self.map.project(self.ego.position, self.map.links['A2256W000411'])
        source_path = [pose[0] for pose in self.map.route_points(source, 170.0, 0.5)]
        target_front = ObjectInfo(66.1, 230.5, 0.0, 1, size_x=4.65, size_y=2.1,
                                  heading_deg=-90.0, unique_id=57)
        result = self.plan([target_front], reference_path=source_path[:50], prediction_path=source_path)
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual('npc:57', result.debug_details['merge_safety']['conflict']['id'])

    def test_logged_stopped_pose_can_escape_when_target_lane_is_clear(self):
        result = self.plan()
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertEqual('A2256W000409', result.target_link_id)
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertTrue(result.debug_details['highway_static_escape']['compact'])

    def test_passing_neighbor_allows_rolling_wait_then_escape(self):
        self.ego = VehicleState(62.18, 244.9, math.radians(-89.69), 6.0)
        self.blocker = ObjectInfo(62., 215., 0., 2, is_static=True,
                                 size_x=4.69, size_y=2.2, heading_deg=-90., unique_id=41)
        controller = PathController(self.config)
        dt = .05
        self.planner.set_control_dt(dt)
        waited = changed = False
        minimum_speed = self.ego.velocity
        # A stale lateral preview must not be the path used while waiting in
        # the source lane. It points toward the passing car in the logged case.
        preview = [Point(62.18,244.9), Point(66.,237.), Point(66.5,130.)]
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(100):
                passing = ObjectInfo(66., 237.-7.2*step*dt, 7.2, 1,
                    size_x=4.5, size_y=2., heading_deg=-90., unique_id=29)
                result = self.plan([passing], reference_path=preview, prediction_path=preview)
                self.assertGreater(result.acc_target_velocity, .05, result.debug_details)
                if result.highway_state == 'ACC_FOLLOW':
                    waited = True
                    self.assertEqual('highway_static_wait_for_gap', result.reason)
                    self.assertLess(abs(result.path[3].x-62.18), .5)
                changed |= result.highway_state == 'CHANGE_LEFT'
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steering = controller._calculate_control(self.ego, result.path,
                    min(27.78, result.acc_target_velocity), result)
                speed = max(0., self.ego.velocity+(3. if pedal >= 0 else 5.)*pedal*dt)
                yaw = self.ego.yaw
                self.ego = VehicleState(self.ego.position.x+speed*math.cos(yaw)*dt,
                    self.ego.position.y+speed*math.sin(yaw)*dt,
                    yaw+speed/3.*math.tan(steering/controller.steering_sign)*dt, speed)
                minimum_speed = min(minimum_speed, speed)
                for obj in (self.blocker, passing):
                    self.assertFalse(self.planner._obb_collision(self.ego.position, self.ego.yaw,
                        obj.position, -math.pi/2, obj))
        self.assertTrue(waited)
        self.assertTrue(changed)
        self.assertGreater(minimum_speed, .5)

    def test_waiting_cannot_force_motion_into_a_close_static_body(self):
        self.ego = VehicleState(62.18,244.9,-math.pi/2,0.)
        self.blocker = ObjectInfo(62.18,237.,0.,2,is_static=True,
            size_x=4.69,size_y=2.2,heading_deg=-90.,unique_id=41)
        side = ObjectInfo(66.,243.,0.,2,is_static=True,
            size_x=4.69,size_y=2.2,heading_deg=-90.,unique_id=42)
        result = self.plan([side])
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertEqual(0., result.acc_target_velocity)

    def test_logged_zero_speed_wait_recovers_on_the_source_lane(self):
        self.ego = VehicleState(62.18,244.9,math.radians(-89.69),0.)
        self.blocker = ObjectInfo(62.,215.,0.,2,is_static=True,
            size_x=4.69,size_y=2.2,heading_deg=-90.,unique_id=41)
        passing = ObjectInfo(66.,237.,7.2,1,size_x=4.5,size_y=2.,
            heading_deg=-90.,unique_id=29)
        preview = [Point(62.18,244.9),Point(66.,237.),Point(66.5,130.)]
        result = self.plan([passing],reference_path=preview,prediction_path=preview)
        self.assertEqual('highway_static_wait_for_gap',result.reason)
        self.assertGreater(result.acc_target_velocity,.05)
        self.assertEqual('A2256W000411',result.target_link_id)

    def test_rolling_entry_starts_avoidance_instead_of_stopping(self):
        self.ego = VehicleState(62.16, 256.0, math.radians(-89.05), 5.0)
        result = self.plan()
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertGreaterEqual(result.acc_target_velocity, self.ego.velocity)

    def test_clear_escape_can_start_above_the_old_six_meter_per_second_cap(self):
        self.ego = VehicleState(62.159622, 256.0, math.radians(-89.05), 6.49)
        result = self.plan()
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertGreater(result.acc_target_velocity, self.ego.velocity)
        candidate = self.planner.active_candidate
        self.assertLessEqual(result.acc_target_velocity**2 * candidate.curvature_cost,
                             p.STATIC_CHANGE_LATERAL_ACCEL + 1e-6)

    def test_acceleration_boost_requires_a_verified_escape_without_a_lower_limit(self):
        result = self.plan()
        speed_limit = result.acc_target_velocity
        controller = PathController(self.config)
        self.assertGreater(controller._highway_escape_acceleration_gain(speed_limit, self.ego, result),
                           controller.pid.p_gain)
        self.assertIsNone(controller._highway_escape_acceleration_gain(speed_limit - 1, self.ego, result))
        self.assertIsNone(controller._highway_escape_acceleration_gain(0.0, self.ego, result))
        result.debug_details['highway_static_escape']['verified'] = False
        self.assertIsNone(controller._highway_escape_acceleration_gain(speed_limit, self.ego, result))

    def test_same_lane_upstream_follower_does_not_brake_ego_after_completion(self):
        self.plan()
        self.ego = VehicleState(66.27, 237.76, math.radians(-88.10), 6.0)
        self.plan()
        rear = ObjectInfo(66.0, 300.0, 20.0, 1, size_x=4.6, size_y=2.0,
                          heading_deg=-90.0, unique_id=90)
        self.assertIsNone(self.map.match(rear.position, heading=-math.pi/2))
        result = self.plan([rear])
        self.assertEqual(27.78, result.acc_target_velocity, result.debug_details)

    def test_distant_static_in_merge_does_not_force_a_slow_escape(self):
        self.ego.velocity = 20.0
        lane = self.map.links['A2256W000411']
        origin = self.map.project(self.ego.position, lane)
        point, yaw, _ = self.map.advance_default(origin, 70.0)
        self.blocker = ObjectInfo(point.x, point.y, 0.0, 2, is_static=True,
                                  size_x=4.69, size_y=2.2, heading_deg=math.degrees(yaw), unique_id=41)
        result = self.plan()
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))
        self.assertFalse(result.debug_details['highway_static_escape']['compact'])

    def test_steering_exception_is_only_for_verified_low_speed_escape(self):
        result = self.plan()
        commands = []
        for compact, speed in ((False, 0.0), (True, 0.0), (True, 20.0)):
            controller = PathController(self.config)
            self.ego.velocity = speed
            result.debug_details['highway_static_escape']['compact'] = compact
            # A path requesting a substantial left steer exposes the cap/rate.
            path = [Point(self.ego.position.x + 2, self.ego.position.y - 2),
                    Point(self.ego.position.x + 20, self.ego.position.y - 20)]
            for _ in range(30):
                _, _, steering = controller._calculate_control(self.ego, path, 5.0, result)
            commands.append(abs(math.degrees(steering)))
        self.assertLessEqual(commands[0], 5.0)
        self.assertGreater(commands[1], 5.0)
        self.assertLessEqual(commands[2], 5.0)

    def test_entry_link_advances_before_old_endpoint_distance_reaches_four_meters(self):
        self.planner.target_link_id = 'A2256W000846'
        self.ego = VehicleState(62.16, 256.0, math.radians(-89.05), 5.0)
        result = self.plan()
        self.assertEqual('A2256W000411', result.current_link_id)
        self.assertEqual('CHANGE_LEFT', result.highway_state)

    def test_fast_rear_blocks_escape_even_when_not_matched_to_a_link(self):
        rear = ObjectInfo(66.0, 330.0, 20.0, 1, size_x=4.6, size_y=2.0,
                          heading_deg=-90.0, unique_id=90)
        self.assertIsNone(self.map.match(rear.position, heading=-math.pi/2))
        self.assertEqual('ACC_FOLLOW', self.plan([rear]).highway_state)
        # The short merge-track hold must not disappear in the escape branch.
        self.assertEqual('ACC_FOLLOW', self.plan().highway_state)

    def test_vehicle_that_clears_before_lateral_entry_does_not_force_a_stop(self):
        rear = ObjectInfo(66.0, 290.0, 20.0, 1, size_x=4.6, size_y=2.0,
                          heading_deg=-90.0, unique_id=90)
        self.assertEqual('CHANGE_LEFT', self.plan([rear]).highway_state)

    def test_fast_rear_must_leave_a_time_gap_after_slow_escape(self):
        # No immediate overlap: the problem is running into a slow ego just
        # after the short maneuver, so check future relative-speed headway too.
        rear = ObjectInfo(66.0, self.ego.position.y + 120.0, 20.0, 1,
                          size_x=4.6, size_y=2.0, heading_deg=-90.0, unique_id=90)
        result = self.plan([rear])
        single_lane = [c for c in result.candidates if c.direction != 'keep'
                       and getattr(c, 'lane_change_count', 1) == 1]
        self.assertTrue(all(not c.valid for c in single_lane))
        self.assertTrue(any((getattr(c, 'prediction_conflict', None) or {}).get('rear')
                            for c in single_lane))
        self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_blocked_target_lane_cannot_be_ignored_to_force_a_change(self):
        target_front = ObjectInfo(66.0, 245.5, 0.0, 1, size_x=4.65, size_y=2.1,
                                  heading_deg=-90.0, unique_id=57)
        self.assertEqual('ACC_FOLLOW', self.plan([target_front]).highway_state)

    def test_short_curve_is_not_approved_at_high_speed_by_only_lowering_target(self):
        self.ego.velocity = 20.0
        result = self.plan()
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertTrue(any((getattr(c, 'prediction_conflict', None) or {}).get('kind') == 'turn_speed'
                            for c in result.candidates))

    def test_controller_follows_escape_without_collision_or_forced_stop(self):
        for y, initial_speed, rear_motion, max_completion_time in (
                (251.022369, 0.0, None, 5.1), (264.7, 6.49, None, None),
                (254.0, 0.5, None, 4.8), (256.0, 6.49, None, 3.1),
                (256.0, 4.8, (50.0, 13.9), None),
                (251.022369, 0.0, (140.0, 20.0), None)):
            with self.subTest(start_y=y, speed=initial_speed, rear=rear_motion):
                self.setUp()
                self.ego = VehicleState(62.159622, y, math.radians(-89.05), initial_speed)
                origin = self.map.match(self.ego.position, heading=self.ego.yaw)
                global_path = [pose[0] for pose in self.map.route_points(origin, 170.0, 0.5)]
                controller = PathController(self.config)
                dt = 0.05
                self.planner.set_control_dt(dt)
                started = False
                resumed = False
                completion_time = None
                speeds = []
                with contextlib.redirect_stdout(io.StringIO()):
                    for step in range(300):
                        others = []
                        if rear_motion is not None:
                            gap, rear_speed = rear_motion
                            others = [ObjectInfo(66.0, y + gap - rear_speed*step*dt, rear_speed, 1,
                                                 size_x=4.6, size_y=2.0, heading_deg=-90.0, unique_id=90)]
                        nearest = min(range(len(global_path)),
                                      key=lambda i: global_path[i].distance(self.ego.position))
                        result = self.plan(others, reference_path=global_path[nearest:nearest+50],
                                           prediction_path=global_path[nearest:])
                        target = min(27.78, result.acc_target_velocity)
                        self.assertGreater(target, 0.05, (step, result.reason, result.debug_details))
                        escape = result.debug_details.get('highway_static_escape', {})
                        if escape.get('compact'):
                            self.assertLessEqual(target, escape['speed_limit'] + 1e-6)
                        if started and not self.planner.changed_lane_hold:
                            self.assertEqual('CHANGE_LEFT', result.highway_state)
                        if self.planner.changed_lane_hold:
                            if completion_time is None:
                                completion_time = step*dt
                            self.assertEqual(27.78, target, (step, result.reason, result.debug_details))
                            resumed |= self.ego.velocity > 6.0
                        controller._configure_runtime(sync_mode=False, control_dt=dt)
                        pedal, _, steering = controller._calculate_control(self.ego, result.path, target, result)
                        # A simple bicycle + pedal model exercises the real PP,
                        # steering limits/rate, and PID instead of teleporting
                        # the vehicle onto the planner's ideal path.
                        speed = max(0.0, self.ego.velocity + (3.0 if pedal >= 0 else 5.0) * pedal * dt)
                        yaw = self.ego.yaw
                        actual_steering = steering / controller.steering_sign
                        self.ego = VehicleState(self.ego.position.x + speed*math.cos(yaw)*dt,
                                                self.ego.position.y + speed*math.sin(yaw)*dt,
                                                yaw + speed/3.0*math.tan(actual_steering)*dt, speed)
                        self.assertFalse(self.planner._obb_collision(self.ego.position, self.ego.yaw,
                            self.blocker.position, -math.pi/2, self.blocker), (step, result.reason))
                        for obj in others:
                            self.assertFalse(self.planner._obb_collision(self.ego.position, self.ego.yaw,
                                obj.position, -math.pi/2, obj), (step, result.reason))
                        started |= result.highway_state == 'CHANGE_LEFT'
                        speeds.append(speed)
                        if self.ego.position.y < self.blocker.position.y - 25 and (rear_motion is None or step*dt >= 9.0):
                            break
                self.assertTrue(started)
                self.assertTrue(resumed)
                self.assertLess(self.ego.position.y, self.blocker.position.y - 25)
                if max_completion_time is not None:
                    self.assertLessEqual(completion_time, max_completion_time)
                if initial_speed > 0:
                    self.assertGreater(min(speeds), 0.5)
