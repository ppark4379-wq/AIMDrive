import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.lattice_planning import parameters
from expert.src.planning.lattice_planning.lattice_planner import LatticePlanner
from expert.src.planning.lattice_planning.planning_process import LatticePlanningProcess
from expert.src.vehicle_state import VehicleState


class LatticePlanningTest(unittest.TestCase):
    def setUp(self):
        self.path = [Point(float(x), 0.0) for x in range(41)]
        self.ego = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        self.static_obstacle = ObjectInfo(
            15.0,
            0.0,
            0.0,
            2,
            is_static=True,
            size_x=1.0,
            size_y=1.0,
        )

    def test_static_obstacle_selects_route_right_candidate(self):
        planner = LatticePlanner()
        result = planner.plan(self.ego, self.path, [self.static_obstacle])

        self.assertTrue(result.active)
        self.assertTrue(result.obstacle_detected)
        self.assertLess(result.selected_offset, 0.0)
        self.assertIsNot(result.path, self.path)
        self.assertEqual(18, len(result.candidates))
        self.assertEqual(9, sum(path.layer == "long" for path in result.candidates))
        self.assertEqual(9, sum(path.layer == "short" for path in result.candidates))
        self.assertTrue(all(hasattr(path, "curvature_cost") for path in result.candidates))
        self.assertGreater(
            max(path.curvature_cost for path in result.candidates if path.valid),
            0.0,
        )

    def test_long_center_obb_sets_no_op_speed_flag(self):
        planner = LatticePlanner()
        result = planner.plan(self.ego, self.path, [self.static_obstacle])

        self.assertTrue(result.long_obstacle_detected)
        self.assertEqual(
            parameters.LONG_OBSTACLE_SPEED_SCALE,
            result.longitudinal_speed_scale,
        )

    def test_layer_lengths_do_not_change_with_vehicle_speed(self):
        planner = LatticePlanner()
        stopped = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        fast = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=20.0)

        stopped_long = planner._generate_candidates(
            stopped,
            self.path,
            layer="long",
            lookahead_distance=parameters.LONG_LOOKAHEAD_DISTANCE,
        )[4]
        fast_long = planner._generate_candidates(
            fast,
            self.path,
            layer="long",
            lookahead_distance=parameters.LONG_LOOKAHEAD_DISTANCE,
        )[4]
        stopped_short = planner._generate_candidates(
            stopped,
            self.path,
            layer="short",
            lookahead_distance=parameters.SHORT_LOOKAHEAD_DISTANCE,
        )[4]
        fast_short = planner._generate_candidates(
            fast,
            self.path,
            layer="short",
            lookahead_distance=parameters.SHORT_LOOKAHEAD_DISTANCE,
        )[4]

        self.assertAlmostEqual(parameters.LONG_LOOKAHEAD_DISTANCE, stopped_long.points[-1].x)
        self.assertAlmostEqual(stopped_long.points[-1].x, fast_long.points[-1].x)
        self.assertAlmostEqual(parameters.SHORT_LOOKAHEAD_DISTANCE, stopped_short.points[-1].x)
        self.assertAlmostEqual(stopped_short.points[-1].x, fast_short.points[-1].x)

    def test_no_obstacle_keeps_reference_path(self):
        planner = LatticePlanner()
        result = planner.plan(self.ego, self.path, [])

        self.assertFalse(result.active)
        self.assertEqual("no_static_obstacle", result.reason)
        self.assertIs(result.path, self.path)

    def test_process_only_runs_inside_configured_zone(self):
        detected = [{"object_info": self.static_obstacle}]
        outside_process = LatticePlanningProcess(zone_bounds=(10.0, 20.0, -5.0, 5.0))
        outside = outside_process.run(self.ego, self.path, detected)
        self.assertFalse(outside.active)
        self.assertEqual("outside_zone", outside.reason)

        inside_process = LatticePlanningProcess(zone_bounds=(-5.0, 5.0, -5.0, 5.0))
        inside = inside_process.run(self.ego, self.path, detected)
        self.assertTrue(inside.active)
        self.assertLess(inside.selected_offset, 0.0)

    def test_npc_does_not_activate_static_obstacle_process(self):
        npc = ObjectInfo(15.0, 0.0, 0.0, 1, is_static=False)
        process = LatticePlanningProcess(zone_bounds=(-5.0, 5.0, -5.0, 5.0))
        result = process.run(self.ego, self.path, [{"object_info": npc}])

        self.assertFalse(result.active)
        self.assertEqual("no_static_obstacle", result.reason)

    def test_measured_static_obstacle_corridor(self):
        process = LatticePlanningProcess()
        inside = VehicleState(x=-60.0, y=-100.0, yaw=0.0, velocity=0.0)
        outside = VehicleState(x=-50.0, y=-100.0, yaw=0.0, velocity=0.0)

        self.assertTrue(process._is_inside_zone(inside))
        self.assertFalse(process._is_inside_zone(outside))

    def test_measured_highway_corridor_follows_curve(self):
        process = LatticePlanningProcess()
        inside = VehicleState(x=67.7, y=-100.0, yaw=0.0, velocity=0.0)
        outside = VehicleState(x=40.0, y=-100.0, yaw=0.0, velocity=0.0)

        self.assertTrue(process._is_inside_highway_zone(inside))
        self.assertFalse(process._is_inside_highway_zone(outside))

    def test_highway_safe_vehicle_uses_cost_based_lattice(self):
        process = LatticePlanningProcess(
            zone_bounds=(100.0, 110.0, 100.0, 110.0),
            highway_zone_points=((-5.0, 0.0), (40.0, 0.0)),
            highway_zone_half_width=3.0,
        )
        npc = ObjectInfo(
            20.0,
            0.0,
            3.0,
            1,
            is_static=False,
            size_x=4.5,
            size_y=2.0,
        )
        result = process.run(
            self.ego,
            self.path,
            [{"object_info": npc, "local_position": Point(20.0, 0.0)}],
        )

        self.assertEqual("highway", result.mode)
        self.assertTrue(result.active)
        self.assertFalse(result.acc_override)
        self.assertEqual("highway_vehicle_lattice", result.reason)
        self.assertEqual(18, len(result.candidates))
        self.assertTrue(
            all(candidate.right_preference_cost == 0.0 for candidate in result.candidates)
        )

    def test_highway_urgent_vehicle_gives_acc_priority(self):
        process = LatticePlanningProcess(
            zone_bounds=(100.0, 110.0, 100.0, 110.0),
            highway_zone_points=((-5.0, 0.0), (40.0, 0.0)),
            highway_zone_half_width=3.0,
        )
        npc = ObjectInfo(
            8.0,
            0.0,
            0.0,
            1,
            is_static=False,
            size_x=4.5,
            size_y=2.0,
        )
        result = process.run(
            self.ego,
            self.path,
            [{"object_info": npc, "local_position": Point(8.0, 0.0)}],
        )

        self.assertEqual("highway", result.mode)
        self.assertFalse(result.active)
        self.assertTrue(result.acc_override)
        self.assertEqual("highway_acc_override", result.reason)
        self.assertIs(result.path, self.path)
        self.assertLessEqual(result.nearest_vehicle_distance, 10.0)

    def test_highway_acc_override_has_release_hysteresis(self):
        process = LatticePlanningProcess(
            zone_bounds=(100.0, 110.0, 100.0, 110.0),
            highway_zone_points=((-5.0, 0.0), (40.0, 0.0)),
            highway_zone_half_width=3.0,
        )

        def run_vehicle(x, velocity):
            npc = ObjectInfo(x, 0.0, velocity, 1, is_static=False)
            return process.run(
                self.ego,
                self.path,
                [{"object_info": npc, "local_position": Point(x, 0.0)}],
            )

        self.assertTrue(run_vehicle(8.0, 0.0).acc_override)
        self.assertTrue(run_vehicle(16.0, 5.0).acc_override)
        self.assertFalse(run_vehicle(25.0, 5.0).acc_override)


if __name__ == "__main__":
    unittest.main()
