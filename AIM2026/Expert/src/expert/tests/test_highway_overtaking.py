"""Prefer a verified faster lane over permanent ACC behind a slower lead."""

import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from expert.src.localization.point import Point
from expert.src.control.path_controller import PathController
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway.lane_map import HighwayLaneMap
from expert.src.planning.highway.planner import HighwayLanePlanner
from expert.src.vehicle_state import VehicleState
from expert.tests.test_highway_static_avoidance import entry, straight_map


class HighwayOvertakingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lane_map = HighwayLaneMap(str(Path(__file__).resolve().parents[3] /
                                         'map_data/R_KR_PG_KATRI_2025'))

    def real_scene(self, link_id='A2256W000411', lead_speed=12.0):
        planner = HighwayLanePlanner(lane_map=self.lane_map)
        lane = self.lane_map.links[link_id]
        point, yaw, _ = self.lane_map.point_at_s(lane, 10.0)
        ego = VehicleState(point.x, point.y, yaw, 12.0)
        point, yaw, _ = self.lane_map.point_at_s(lane, 55.0)
        obj = ObjectInfo(point.x, point.y, lead_speed, 1, size_x=4.5, size_y=2.0,
                         heading_deg=math.degrees(yaw), unique_id=1)
        lead = dict(object_info=obj, local_position=(obj.position-ego.position).rotate(-ego.yaw))
        origin = self.lane_map.project(ego.position, lane)
        path = [row[0] for row in self.lane_map.route_points(origin, 180.0, 0.5)]
        return planner, ego, lead, path

    def test_merge_go_does_not_prevent_overtaking_a_slower_lead(self):
        for link_id in ('A2256W000411', 'A2256W000409'):
            with self.subTest(lane=link_id):
                planner, ego, lead, path = self.real_scene(link_id)
                result = planner.plan(ego, path, [lead], desired_velocity=20.0)
                self.assertEqual('GO', result.debug_details['merge_safety']['decision'])
                self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
                self.assertFalse(result.acc_override)
                # Old source-lane ACC must not return on the next control tick.
                held = planner.plan(ego, path, [lead], desired_velocity=20.0)
                self.assertEqual('CHANGE_LEFT', held.highway_state)
                self.assertFalse(held.acc_override)

    def logged_following_scene(self, link_id='A2256W000420', ego_s=44.4,
                                gap=12.11, speed=11.38, lead_speed=11.1):
        planner = HighwayLanePlanner(lane_map=self.lane_map)
        planner.changed_lane_hold = True
        lane = self.lane_map.links[link_id]
        point, yaw, _ = self.lane_map.point_at_s(lane, ego_s)
        ego = VehicleState(point.x, point.y, yaw, speed)
        origin = self.lane_map.project(point, lane)
        point, yaw, _ = self.lane_map.advance_default(origin, gap + (4.635 + 4.5)/2)
        obj = ObjectInfo(point.x, point.y, lead_speed, 1, size_x=4.5, size_y=2.0,
                         heading_deg=math.degrees(yaw), unique_id=41)
        lead = dict(object_info=obj, local_position=(obj.position-ego.position).rotate(-ego.yaw))
        path = [row[0] for row in self.lane_map.route_points(origin, 220.0, 0.5)]
        return planner, ego, lead, path

    def test_logged_forty_kph_following_can_overtake_despite_source_lane_yield(self):
        for args in ((), ('A2256W000409', 49.7, 18.9, 15.88, 10.8)):
            with self.subTest(scene=args):
                planner, ego, lead, path = self.logged_following_scene(*args)
                result = planner.plan(ego, path, [lead], desired_velocity=27.78)
                self.assertEqual('YIELD', result.debug_details['merge_safety']['decision'])
                self.assertEqual('npc:41', result.debug_details['merge_safety']['conflict']['id'])
                self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
                self.assertGreater(result.acc_target_velocity, ego.velocity)

    def test_source_lane_yield_does_not_allow_entry_into_a_blocked_left_lane(self):
        for closing_rear in (False, True):
            planner, ego, lead, path = self.logged_following_scene()
            current = self.lane_map.match(ego.position, heading=ego.yaw)
            target = dict(self.lane_map.legal_neighbors(current.lane_link))['left']
            projection = self.lane_map.project(ego.position, target)
            point, yaw, _ = self.lane_map.point_at_s(target, projection.s - 35.0 if closing_rear else projection.s)
            obj = ObjectInfo(point.x, point.y, 30.0 if closing_rear else 11.1, 1,
                             size_x=4.5, size_y=2.0, heading_deg=math.degrees(yaw), unique_id=42)
            result = planner.plan(ego, path, [lead, dict(object_info=obj,
                                  local_position=(obj.position-ego.position).rotate(-ego.yaw))],
                                  desired_velocity=27.78)
            self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_logged_following_overtake_completes_and_accelerates_with_real_controller(self):
        planner, ego, lead, path = self.logged_following_scene()
        origin = self.lane_map.project(lead['object_info'].position, self.lane_map.links['A2256W000420'])
        config_path = Path(__file__).resolve().parents[1] / 'src/config/config.json'
        controller = PathController(json.loads(config_path.read_text(encoding='utf-8')))
        dt = 0.05
        planner.set_control_dt(dt)
        minimum_speed = ego.velocity
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(160):
                point, yaw, _ = self.lane_map.advance_default(origin, 11.1*step*dt)
                obj = lead['object_info']
                obj.position, obj.heading_deg = point, math.degrees(yaw)
                lead['local_position'] = (point-ego.position).rotate(-ego.yaw)
                result = planner.plan(ego, path, [lead], desired_velocity=27.78)
                target = min(27.78, result.acc_target_velocity)
                self.assertGreater(target, 0.05, (step, result.reason, result.debug_details))
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steer = controller._calculate_control(ego, result.path, target, result)
                speed = max(0.0, ego.velocity + (3.0 if pedal >= 0 else 5.0)*pedal*dt)
                yaw = ego.yaw
                ego = VehicleState(ego.position.x + speed*math.cos(yaw)*dt,
                                   ego.position.y + speed*math.sin(yaw)*dt,
                                   yaw + speed/3.0*math.tan(steer/controller.steering_sign)*dt, speed)
                minimum_speed = min(minimum_speed, speed)
                self.assertFalse(planner._obb_collision(ego.position, ego.yaw,
                    obj.position, math.radians(obj.heading_deg), obj), (step, result.reason))
        self.assertTrue(planner.changed_lane_hold)
        self.assertGreater(ego.velocity, 20.0)
        self.assertGreater(minimum_speed, 10.0)
        self.assertLess((obj.position-ego.position).rotate(-ego.yaw).x, -5.0)

    def test_merge_still_checks_fast_target_lane_rear_before_overtaking(self):
        planner, ego, lead, path = self.real_scene()
        rear = ObjectInfo(66.0, ego.position.y + 40.0, 30.0, 1,
                          size_x=4.5, size_y=2.0, heading_deg=-90.0, unique_id=2)
        result = planner.plan(ego, path, [lead, dict(object_info=rear,
                              local_position=(rear.position-ego.position).rotate(-ego.yaw))],
                              desired_velocity=20.0)
        self.assertEqual('ACC_FOLLOW', result.highway_state)

    def test_overtake_keeps_full_prediction_distance_as_the_car_advances(self):
        planner, ego, lead, path = self.real_scene(lead_speed=20.0)
        ego.velocity = 20.0
        first = planner.plan(ego, path, [lead], desired_velocity=22.0)
        self.assertEqual('CHANGE_LEFT', first.highway_state)
        points = list(first.path)
        cumulative = planner._cumulative_distance(points)
        source = self.lane_map.links['A2256W000411']
        for distance in (10.0, 20.0, 30.0, 40.0, 50.0):
            point, yaw = planner._pose_at_distance(points, cumulative, distance)
            ego = VehicleState(point.x, point.y, yaw, 20.0)
            point, yaw, _ = self.lane_map.point_at_s(source, 55.0 + distance)
            lead['object_info'].position = point
            lead['object_info'].heading_deg = math.degrees(yaw)
            lead['local_position'] = (point-ego.position).rotate(-ego.yaw)
            result = planner.plan(ego, path, [lead], desired_velocity=22.0)
            self.assertGreaterEqual(result.acc_target_velocity, 22.0, result.debug_details)
            self.assertTrue(all(candidate.valid for candidate in result.candidates))

    def test_lane_change_disabled_and_no_speed_advantage_still_follow(self):
        for desired, allowed in ((12.0, True), (20.0, False)):
            planner, ego, lead, path = self.real_scene()
            result = planner.plan(ego, path, [lead], desired_velocity=desired, allow_lane_change=allowed)
            self.assertNotIn(result.highway_state, ('CHANGE_LEFT', 'CHANGE_RIGHT'))

    def test_overtake_checks_beyond_completion_through_the_merge_horizon(self):
        planner, ego, lead, path = self.real_scene(lead_speed=18.0)
        ego.velocity = 20.0
        point, yaw, _ = self.lane_map.point_at_s(self.lane_map.links['A2256W000409'], 110.0)
        obstacle = ObjectInfo(point.x, point.y, 0.0, 2, is_static=True,
                              size_x=4.5, size_y=2.0, heading_deg=math.degrees(yaw), unique_id=2)
        result = planner.plan(ego, path, [lead, dict(object_info=obstacle,
                              local_position=(obstacle.position-ego.position).rotate(-ego.yaw))],
                              desired_velocity=25.0)
        self.assertEqual('GO', result.debug_details['merge_safety']['decision'])
        adjacent = next(c for c in result.candidates if c.direction == 'left'
                        and getattr(c, 'lane_change_count', 1) == 1)
        self.assertFalse(adjacent.valid)
        self.assertGreater(adjacent.collision_time, 3.5)
        self.assertLessEqual(adjacent.collision_time, 5.0)
        # The unsafe single-lane path still fails; a separately checked
        # two-leg path can leave that lane before reaching its obstruction.
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertEqual('A2256W000410', result.target_link_id)
        self.assertEqual(2, result.debug_details['multi_lane_change']['count'])

    def test_slow_lead_is_considered_before_closing_within_sixty_five_meters(self):
        planner = HighwayLanePlanner(lane_map=straight_map())
        ego = VehicleState(0.0, 2000.0, 0.0, 12.0)
        path = [Point(x, 2000.0) for x in range(161)]
        result = planner.plan(ego, path, [entry(80, speed=12)], desired_velocity=20.0)
        self.assertIn(result.highway_state, ('CHANGE_LEFT', 'CHANGE_RIGHT'))

    def test_cooldown_allows_continuing_left_but_not_immediately_returning_right(self):
        for left_blocked in (False, True):
            with self.subTest(left_blocked=left_blocked):
                planner = HighwayLanePlanner(lane_map=straight_map())
                ego = VehicleState(0.0, 2000.0, 0.0, 12.0)
                planner.changed_lane_hold = True
                planner._lane_change_cooldown_origin = Point(0.0, 2000.0)
                planner._last_lane_change_direction = planner.CHANGE_LEFT
                objects = [entry(50, speed=12)]
                if left_blocked:
                    objects.append(entry(0, lane=1, speed=12, uid=2))
                result = planner.plan(ego, [Point(x, 2000.0) for x in range(161)],
                                      objects, desired_velocity=20.0)
                self.assertEqual('ACC_FOLLOW' if left_blocked else 'CHANGE_LEFT', result.highway_state)
