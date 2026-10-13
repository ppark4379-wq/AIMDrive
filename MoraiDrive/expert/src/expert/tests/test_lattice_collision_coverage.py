import math
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.lattice_planning.lattice_planner import LatticePlanner
from expert.src.planning.lattice_planning.structures import CandidatePath
from expert.src.vehicle_state import VehicleState


class LatticeCollisionCoverageTest(unittest.TestCase):
    def setUp(self):
        self.planner = LatticePlanner()
        self.ego = VehicleState(0.0, 0.0, 0.0, 5.0)
        self.route = [Point(float(x), 0.0) for x in range(41)]

    @staticmethod
    def obstacle(x, y):
        # Generic perception has no GT class, static status, or tracking ID.
        return ObjectInfo(x, y, 0.0, -1, size_x=1.0, size_y=1.0)

    def test_adjacent_obstacle_alone_does_not_activate_avoidance(self):
        adjacent = self.obstacle(12.0, -4.2)
        result = self.planner.plan(self.ego, self.route, [adjacent])
        self.assertFalse(result.active)
        self.assertFalse(result.obstacle_detected)
        self.assertIs(result.path, self.route)

    def test_candidate_rejects_adjacent_obstacle_outside_gp_trigger_filter(self):
        center = self.obstacle(10.0, 0.0)
        adjacent = self.obstacle(12.0, -4.2)
        center_only = self.planner.plan(self.ego, self.route, [center])
        self.assertTrue(center_only.active)
        self.assertEqual([], self.planner._find_relevant_obstacles(
            self.ego, self.route, [adjacent]))
        self.assertTrue(self.planner.path_has_collision(center_only.path, [adjacent]))

        result = LatticePlanner().plan(self.ego, self.route, [center, adjacent])
        self.assertTrue(result.active)
        self.assertFalse(self.planner.path_has_collision(result.path, [center, adjacent]))
        excluded = next(candidate for candidate in result.candidates
                        if candidate.layer == "short" and
                        candidate.offset == center_only.selected_offset)
        self.assertFalse(excluded.valid)

    def test_rear_overlapping_footprint_not_removed_from_candidate_check(self):
        trigger = self.obstacle(15.0, 0.0)
        rear_overlap = self.obstacle(-0.8, 0.0)
        self.assertEqual([], self.planner._find_relevant_obstacles(
            self.ego, self.route, [rear_overlap]))
        result = self.planner.plan(self.ego, self.route, [trigger, rear_overlap])
        self.assertFalse(result.active)
        self.assertEqual("all_candidates_blocked", result.reason)
        self.assertTrue(result.obstacle_detected)
        self.assertTrue(all(not candidate.valid for candidate in result.candidates))

    def test_collision_between_coarse_endpoints_is_rejected(self):
        planner = LatticePlanner(sample_spacing=2.0, vehicle_length=0.1,
                                 vehicle_width=0.1, collision_margin=0.0)
        candidate = CandidatePath()
        candidate.points = [Point(0.0, 0.0), Point(2.0, 0.0)]
        obstacle = self.obstacle(1.0, 0.0)
        self.assertTrue(all(not planner._obb_collision(point, 0.0, obstacle)
                            for point in candidate.points))
        planner._evaluate_candidates([candidate], [obstacle], add_policy_cost=False)
        self.assertFalse(candidate.valid)

    def test_turning_vehicle_sweep_rejects_corner_obstacle(self):
        path = [Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)]
        obstacle = self.obstacle(4.0, 3.0)
        self.assertTrue(all(not self.planner._obb_collision(
            point, self.planner._path_heading(path, index), obstacle)
            for index, point in enumerate(path)))
        self.assertTrue(self.planner._obb_collision(path[1], math.pi / 4.0, obstacle))
        self.assertTrue(self.planner.path_has_collision(path, [obstacle]))

    def test_translation_margin_covers_gap_between_dense_samples(self):
        planner = LatticePlanner(sample_spacing=0.5, vehicle_length=0.1,
                                 vehicle_width=0.1, collision_margin=0.0)
        path = [Point(0.0, 0.0), Point(1.0, 0.0)]
        # Graze the bottom corner of a diagonal 1 m box. Even the dense 0.25 m
        # poses miss it; the swept translation margin must cover the interval.
        obstacle = self.obstacle(0.425, 0.755)
        obstacle.heading_deg = 45.0
        self.assertTrue(all(not planner._obb_collision(Point(x, 0.0), 0.0, obstacle)
                            for x in (0.0, 0.25, 0.5, 0.75, 1.0)))
        self.assertTrue(planner._obb_collision(Point(0.375, 0.0), 0.0, obstacle))
        self.assertTrue(planner.path_has_collision(path, [obstacle]))
        samples = list(planner._collision_samples(path))
        self.assertLessEqual(max(samples[i + 1][0].distance(samples[i][0])
                                 for i in range(len(samples) - 1)), 0.25)

    def test_clear_path_and_far_footprint_remain_clear(self):
        self.assertFalse(self.planner.path_has_collision(self.route, []))
        self.assertFalse(self.planner.path_has_collision(
            self.route, [self.obstacle(12.0, -8.0)]))
        self.assertTrue(self.planner.path_has_collision([], []))
        self.assertTrue(self.planner.path_has_collision([Point(0.0, 0.0)], []))

    def test_broad_phase_includes_obstacle_diagonal_corner(self):
        planner = LatticePlanner(vehicle_length=0.2, vehicle_width=0.2,
                                 collision_margin=0.0)
        obstacle = self.obstacle(1.08, 1.08)
        obstacle.size_x = obstacle.size_y = 2.0
        path = [Point(0.0, 0.0), Point(0.2, 0.2)]
        # The cost's max-side radius is not a circumscribed circle: using it
        # would skip this actual OBB corner intersection in the broad phase.
        self.assertGreater(path[0].distance(obstacle.position),
                           planner._swept_body_radius(0.0) +
                           planner._obstacle_radius(obstacle))
        self.assertTrue(planner._obb_collision(path[0], math.pi / 4.0, obstacle))
        self.assertTrue(planner.path_has_collision(path, [obstacle]))


if __name__ == "__main__":
    unittest.main()
