import math
from pathlib import Path
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.lattice_planning.roundabout_entry import RoundaboutEntryPlanner
from expert.src.planning.lattice_planning.planning_process import LatticePlanningProcess


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
        predictions = [[self.planner._object_poses(obj, None, i * 0.1)] for i in range(51)]
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
        return self.planner.plan(self.ego, self.global_path[index:index+120],
                                 [{"object_info": obj} for obj in objects], state)

    def test_rear_arc_vehicle_requires_yield_at_stop_line(self):
        obj = RoundaboutEntryTest.obj(-108.0, 326.5, 6.0, -20.0)
        result = self.run_scene([obj])
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertFalse(self.planner.committed)
        self.assertGreater(result.ttc, 0.0)

    def test_cleared_vehicle_releases_without_extra_wait_cycles(self):
        incoming = RoundaboutEntryTest.obj(-104.45, 325.64, 5.0)
        self.assertEqual(0.0, self.run_scene([incoming]).acc_target_velocity)
        self.ego.velocity = 0.0
        departed = RoundaboutEntryTest.obj(-89.0, 330.0, 5.0, 50.0)
        result = self.run_scene([departed])
        self.assertGreater(result.acc_target_velocity, 0.0)
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
        self.run_scene([])
        self.assertTrue(self.planner.committed)
        self.ego = VehicleState(0.0, 0.0, 0.0, 0.0)
        self.assertIsNone(self.run_scene([]))
        self.assertFalse(self.planner.committed)

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
