import math
from pathlib import Path
import unittest
from unittest.mock import patch

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.lattice_planning.roundabout_entry import RoundaboutEntryPlanner
from expert.src.planning.lattice_planning.planning_process import LatticePlanningProcess
from expert.src.planning.lattice_planning import parameters as p


class RoundaboutEntryTest(unittest.TestCase):
    def setUp(self):
        self.planner = RoundaboutEntryPlanner()
        self.path = [Point(float(x), 0.0) for x in range(81)]
        self.ego = VehicleState(0.0, 0.0, 0.0, 5.0)

    @staticmethod
    def obj(x, y, speed, heading=0.0, identity=1):
        return ObjectInfo(x, y, speed, 1, heading_deg=heading,
                          size_x=4.5, size_y=2.0, unique_id=identity)

    def collision(self, obj, target=5.0):
        route = self.planner._route(self.path)
        tracks = self.planner._tracks(self.ego, [{"object_info": obj}])
        count = int(p.ROUNDABOUT_PREDICTION_HORIZON / p.ROUNDABOUT_PREDICTION_STEP) + 1
        predictions = [[self.planner._object_poses(obj, None, i * p.ROUNDABOUT_PREDICTION_STEP)]
                       for i in range(count)]
        return self.planner._collision(self.ego, route, 0.0, tracks, target, predictions)

    def test_crossing_vehicle_not_overlapping_now_is_predicted(self):
        collision = self.collision(self.obj(15.0, -12.0, 5.0, 90.0))
        self.assertIsNotNone(collision)
        self.assertGreater(collision["collision_time"], 0.0)

    def test_vehicle_leaves_before_ego_arrives_does_not_block(self):
        self.assertIsNone(self.collision(self.obj(20.0, 0.0, 10.0, 90.0)))

    def test_slow_launch_checks_later_arrival(self):
        self.ego.velocity = 0.0
        self.assertGreater(
            self.planner._distance(0.0, 8.33, 3.0),
            self.planner._distance(0.0, 8.33, 3.0, acceleration_scale=0.65),
        )
        self.assertIsNotNone(self.collision(self.obj(12.0, -16.0, 4.0, 90.0), 8.33))

    def test_front_is_selected_by_forward_route_gap(self):
        behind = self.obj(-5.0, 0.0, 8.0, identity=1)
        crossing = self.obj(6.0, 0.0, 3.0, 90.0, identity=2)
        front = self.obj(20.0, 0.0, 4.0, identity=3)
        lead = self.planner._front(self.ego, self.planner._route(self.path), 0.0,
                                   [(behind, None), (crossing, None), (front, None)])
        self.assertEqual(3, lead["id"])

    def test_follow_limit_does_not_switch_to_cruise_when_ego_slows(self):
        front = self.obj(14.0, 0.0, 3.0)
        route = self.planner._route(self.path)
        for speed in (3.0, 3.5, 4.0):
            self.ego.velocity = speed
            lead = self.planner._front(self.ego, route, 0.0, [(front, None)])
            self.assertLess(lead["limit"], 5.0)

    def test_fast_rear_cannot_raise_front_limit(self):
        front = self.obj(12.0, 0.0, 2.0)
        rear = self.obj(-5.0, 0.0, 10.0)
        route = self.planner._route(self.path)
        first = self.planner._front(self.ego, route, 0.0, [(front, None)])
        second = self.planner._front(self.ego, route, 0.0, [(front, None), (rear, None)])
        self.assertEqual(first["limit"], second["limit"])

    def test_committed_rear_threat_does_not_command_braking(self):
        self.planner.committed = True
        rear = self.obj(-8.0, 0.0, 12.0)
        result = self.planner.plan(
            self.ego, self.path, [{"object_info": rear}],
            {"in_approach": True, "past_go_line": True},
        )
        self.assertEqual("rear_pressure", result.debug_details["roundabout_entry"]["phase"])
        self.assertGreater(result.acc_target_velocity, self.ego.velocity)

    def test_collision_beyond_old_five_second_horizon_is_checked(self):
        self.ego.velocity = 0.0
        collision = self.collision(self.obj(25.0, -30.0, 5.0, 90.0), 8.33)
        self.assertIsNotNone(collision)
        self.assertGreater(collision["collision_time"], 5.0)

    def test_short_preview_is_not_treated_as_clear(self):
        self.path = self.path[:12]
        collision = self.collision(self.obj(50.0, 10.0, 0.0))
        self.assertTrue(collision["prediction_incomplete"])

    def test_launch_delay_reduces_predicted_progress_but_not_braking(self):
        self.assertLess(self.planner._distance(0, 8.33, 4., launch_delay=.4),
                        self.planner._distance(0, 8.33, 4.))
        self.assertEqual(self.planner._distance(5, 0, 2., launch_delay=.4),
                         self.planner._distance(5, 0, 2.))

    def test_passed_lead_is_followed_instead_of_frozen_speed_collision_stop(self):
        self.ego.velocity = 0.
        lead = self.obj(16.885, 0., 3.)
        route = self.planner._route(self.path)
        front = self.planner._front(self.ego, route, 0., [(lead, None)])
        self.assertIsNotNone(self.collision(lead, front['limit']))
        with patch.object(p, 'ROUNDABOUT_ENTRY_HOLD_POINT', (0., 0.)), \
                patch.object(p, 'ROUNDABOUT_ENTRY_MERGE_POINT', (15., 0.)):
            result = self.planner.plan(self.ego, self.path, [{'object_info': lead}],
                                       {'in_approach': True, 'past_go_line': False})
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertEqual(result.debug_details['roundabout_entry']['phase'], 'follow')
        self.assertEqual(result.debug_details['roundabout_entry']['front_id'], lead.unique_id)

    def test_lower_entry_speed_is_selected_before_commanding_stop(self):
        self.ego.velocity = 0.
        crossing = self.obj(30., -30., 5., 90.)
        self.assertIsNotNone(self.collision(crossing, p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY))
        with patch.object(p, 'ROUNDABOUT_ENTRY_HOLD_POINT', (0., 0.)), \
                patch.object(p, 'ROUNDABOUT_ENTRY_MERGE_POINT', (15., 0.)):
            result = self.planner.plan(self.ego, self.path, [{'object_info': crossing}],
                                       {'in_approach': True, 'past_go_line': False})
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLess(result.acc_target_velocity, p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY)
        self.assertTrue(result.debug_details['roundabout_entry']['following_speed_adjusted'])
        self.assertIsNone(self.collision(crossing, result.acc_target_velocity))

    def test_close_crossing_car_still_blocks_release_with_departed_lead(self):
        self.ego.velocity = 0.
        lead = self.obj(16.885, 0., 3.)
        crossing = self.obj(4., -2., 2., 90., identity=2)
        with patch.object(p, 'ROUNDABOUT_ENTRY_HOLD_POINT', (0., 0.)), \
                patch.object(p, 'ROUNDABOUT_ENTRY_MERGE_POINT', (15., 0.)):
            result = self.planner.plan(self.ego, self.path,
                                       [{'object_info': lead}, {'object_info': crossing}],
                                       {'in_approach': True, 'past_go_line': False})
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_standstill_launch_keeps_delay_check(self):
        self.ego.velocity = 0.
        with patch.object(self.planner, '_collision', return_value=None) as collision:
            self.planner._checked_collision(self.ego, self.planner._route(self.path), 0., [], 5., [])
        self.assertEqual(collision.call_count, 3)
        self.assertEqual(collision.call_args_list[-1].args[-1], p.ROUNDABOUT_LAUNCH_DELAY)

    def test_stopped_ego_keeps_measured_pose_for_future_collision_checks(self):
        from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
        self.ego = VehicleState(0., .4, math.pi / 3, 0.)
        with patch.object(HighwayLanePlanner, '_obb_collision', return_value=False) as collision:
            self.collision(self.obj(40., 10., 0.), 0.)
        for call in collision.call_args_list:
            point, yaw = call.args[:2]
            self.assertAlmostEqual(point.x, self.ego.position.x)
            self.assertAlmostEqual(point.y, self.ego.position.y)
            self.assertAlmostEqual(yaw, self.ego.yaw)


class RoundaboutMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_dir = Path(__file__).resolve().parents[3] / "map_data" / "R_KR_PG_KATRI_2025"
        cls.global_path = [Point(*map(float, line.split()[:2])) for line in
                           (cls.map_dir / "2026_molit_comp_global_path.txt").read_text().splitlines()
                           if line.strip()]
        cls.lane_map = RoundaboutEntryPlanner(str(cls.map_dir)).lane_map

    def setUp(self):
        self.planner = RoundaboutEntryPlanner()
        self.planner.lane_map = self.lane_map
        self.ego = VehicleState(-99.48, 312.0, 1.53, 4.0)

    def run_scene(self, objects):
        index = min(range(len(self.global_path)), key=lambda i: self.global_path[i].distance(self.ego.position))
        state = LatticePlanningProcess(highway_zone_points=())._roundabout_entry_state(self.ego)
        return self.planner.plan(self.ego, self.global_path[index:index+200],
                                 [{"object_info": obj} for obj in objects], state)

    def test_rear_arc_vehicle_requires_yield_at_new_hold_pose(self):
        self.ego = VehicleState(*p.ROUNDABOUT_ENTRY_HOLD_POINT, math.radians(80), 0.)
        obj = RoundaboutEntryTest.obj(-108.0, 326.5, 6.0, -20.0)
        result = self.run_scene([obj])
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertFalse(self.planner.committed)
        self.assertGreater(result.ttc, 0.0)

    def test_cleared_vehicle_releases_without_extra_wait_cycles(self):
        self.ego = VehicleState(*p.ROUNDABOUT_ENTRY_HOLD_POINT, math.radians(80), 0.)
        incoming = RoundaboutEntryTest.obj(-104.45, 325.64, 5.0)
        self.assertEqual(0.0, self.run_scene([incoming]).acc_target_velocity)
        self.ego.velocity = 0.0
        departed = RoundaboutEntryTest.obj(-89.0, 330.0, 5.0, 50.0)
        result = self.run_scene([departed])
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertFalse(self.planner.committed)

    def test_old_stop_position_proceeds_to_new_hold_without_commitment(self):
        self.ego.velocity = 0.
        for objects in ([], [RoundaboutEntryTest.obj(-108., 326.5, 6., -20.)]):
            result = self.run_scene(objects)
            self.assertGreater(result.acc_target_velocity, 0.)
            self.assertLessEqual(result.acc_target_velocity, p.ROUNDABOUT_APPROACH_VELOCITY)
            self.assertGreater(result.debug_details['roundabout_entry']['hold_gap'], 7.)
            self.assertFalse(self.planner.committed)

    def test_hold_point_is_ego_pose_not_bumper_coordinate(self):
        self.ego = VehicleState(*p.ROUNDABOUT_ENTRY_HOLD_POINT, math.radians(80), 0.)
        result = self.run_scene([])
        self.assertAlmostEqual(result.debug_details['roundabout_entry']['hold_gap'], 0., places=6)
        self.assertEqual(result.acc_target_velocity, p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY)
        self.assertFalse(self.planner.committed)

    def test_passed_go_line_does_not_commit_before_actual_merge(self):
        self.ego = VehicleState(-98.5, 321., math.radians(70), 3.)
        result = self.run_scene([])
        self.assertTrue(result.debug_details['roundabout_entry']['entering'])
        self.assertFalse(self.planner.committed)
        self.assertLess(result.debug_details['roundabout_entry']['hold_gap'], 0.)
        self.assertGreater(result.debug_details['roundabout_entry']['merge_gap'], 0.)

    def test_slow_launch_checks_continue_until_merge_completed(self):
        self.ego = VehicleState(-98.5, 321., math.radians(70), 3.)
        with patch.object(self.planner, '_collision', return_value=None) as collision:
            self.run_scene([])
        self.assertEqual(collision.call_count, 2)
        self.assertEqual(collision.call_args_list[-1].args[-1], 0.)
        self.ego = VehicleState(-92.5, 328.2, math.radians(40), 4.)
        with patch.object(self.planner, '_collision', return_value=None) as collision:
            self.run_scene([])
        self.assertTrue(self.planner.committed)
        self.assertEqual(collision.call_count, 1)

    def test_new_threat_after_release_is_still_checked(self):
        self.ego = VehicleState(*p.ROUNDABOUT_ENTRY_HOLD_POINT, math.radians(80), 0.)
        self.assertGreater(self.run_scene([]).acc_target_velocity, 0.)
        incoming = RoundaboutEntryTest.obj(-108., 326.5, 6., -20.)
        self.assertEqual(self.run_scene([incoming]).acc_target_velocity, 0.)
        self.assertFalse(self.planner.committed)

    def test_immediate_obstacle_before_hold_still_stops(self):
        obstacle = RoundaboutEntryTest.obj(-99.48, 314., 0., 90.)
        self.assertEqual(self.run_scene([obstacle]).acc_target_velocity, 0.)

    def test_entry_respects_lower_path_speed_limit(self):
        self.ego = VehicleState(*p.ROUNDABOUT_ENTRY_HOLD_POINT, math.radians(80), 0.)
        index = min(range(len(self.global_path)), key=lambda i: self.global_path[i].distance(self.ego.position))
        state = LatticePlanningProcess(highway_zone_points=())._roundabout_entry_state(self.ego)
        result = self.planner.plan(self.ego, self.global_path[index:index+200], [], state, 26./3.6)
        self.assertAlmostEqual(result.acc_target_velocity * 3.6, 26.)

    def test_constant_speed_circulating_vehicle_does_not_need_to_brake_for_entry(self):
        from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
        route = self.planner._route(self.global_path)
        npc_link = self.lane_map.links['A2256W000134']
        initial = self.lane_map.project(npc_link.points[0], npc_link)
        for npc_speed in (4., 6., 8.):
            with self.subTest(npc_speed=npc_speed):
                self.setUp()
                self.ego.velocity = 2.
                progress = self.lane_map.project(self.ego.position, route).s
                for index in range(130):
                    center, yaw, _ = self.lane_map.advance_default(initial, npc_speed * index * .1)
                    npc = RoundaboutEntryTest.obj(center.x, center.y, npc_speed, math.degrees(yaw))
                    self.assertFalse(HighwayLanePlanner._obb_collision(
                        self.ego.position, self.ego.yaw, center, yaw, npc), (index, npc_speed))
                    result = self.run_scene([npc])
                    if self.planner.committed:
                        break
                    target = result.acc_target_velocity
                    previous_speed = self.ego.velocity
                    self.ego.velocity = max(0., previous_speed + max(-.25, min(.1, target-previous_speed)))
                    progress += (previous_speed + self.ego.velocity) * .05
                    self.ego.position, self.ego.yaw, _ = self.lane_map.point_at_s(route, progress)
                self.assertTrue(self.planner.committed)

    def test_vehicle_in_arc_but_clear_of_future_ego_does_not_stop(self):
        self.ego.velocity = 0.0
        obj = RoundaboutEntryTest.obj(-95.71, 326.15, 8.0, 25.0)
        result = self.run_scene([obj])
        self.assertGreater(result.acc_target_velocity, 0.0)

    def test_braking_begins_before_stop_line(self):
        self.ego = VehicleState(-99.6, 305.0, 1.53, 7.0)
        obj = RoundaboutEntryTest.obj(-98.0, 326.1, 0.0, 15.0)
        result = self.run_scene([obj])
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertLess(result.acc_target_velocity, 7.0)
        self.assertEqual("yield", result.debug_details["roundabout_entry"]["phase"])

    def test_committed_vehicle_keeps_emergency_collision_check(self):
        self.planner.committed = True
        obj = RoundaboutEntryTest.obj(-99.4, 315.0, 0.0, 88.0)
        result = self.run_scene([obj])
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual("emergency", result.debug_details["roundabout_entry"]["phase"])

    def test_committed_vehicle_does_not_stop_for_safe_new_arrival(self):
        self.planner.committed = True
        self.ego = VehicleState(-91.0, 329.0, 0.7, 5.0)
        obj = RoundaboutEntryTest.obj(-108.0, 326.5, 3.0, -20.0)
        result = self.run_scene([obj])
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertNotEqual("yield", result.debug_details["roundabout_entry"]["phase"])

    def test_map_prediction_follows_curved_lane(self):
        obj = RoundaboutEntryTest.obj(-104.45, 325.64, 5.0)
        match = self.lane_map.match(obj.position, 0.0)
        self.assertEqual("A2256W000134", match.lane_link.id)
        poses = self.planner._object_poses(obj, match, 3.0)
        self.assertTrue(any(abs(angle) > 0.2 for _, angle in poses))
        self.assertTrue(all(point.x > obj.position.x for point, _ in poses))

    def test_teleport_resets_entry_permission(self):
        self.planner.committed = True
        self.planner.entering = True
        self.run_scene([])
        self.assertTrue(self.planner.committed)
        self.ego = VehicleState(0.0, 0.0, 0.0, 0.0)
        self.assertIsNone(self.run_scene([]))
        self.assertFalse(self.planner.committed)
        self.assertFalse(self.planner.entering)

    def test_process_keeps_control_path_and_owns_dynamic_vehicle_acc(self):
        from expert.src.autonomous_driving import AutonomousDriving
        process = LatticePlanningProcess(highway_zone_points=())
        process.roundabout_planner = self.planner
        index = min(range(len(self.global_path)), key=lambda i: self.global_path[i].distance(self.ego.position))
        local_path = self.global_path[index:index+50]
        preview = self.global_path[index:index+120]
        vehicle = RoundaboutEntryTest.obj(-104.45, 325.64, 5.0)
        pedestrian = ObjectInfo(-99.0, 319.0, 0.0, 0)
        static = ObjectInfo(-99.0, 320.0, 0.0, 2, is_static=True)
        entries = [{"object_info": obj} for obj in (vehicle, pedestrian, static)]
        result = process.run(self.ego, local_path, entries, prediction_path=preview)
        self.assertIs(local_path, result.path)
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.lattice_planning_process = process
        self.assertEqual(entries[1:], controller._legacy_acc_entries(self.ego, result, entries))


if __name__ == "__main__":
    unittest.main()
