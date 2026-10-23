import math
import unittest
from pathlib import Path

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.lattice.centerline_lattice import CenterlineLattice
from expert.src.planning.section_planning import LatticePlanningProcess


MAP_DIR = Path(__file__).resolve().parents[3] / 'map_data' / 'R_KR_PG_KATRI_2025'


class CenterlineLatticeTest(unittest.TestCase):
    def setUp(self):
        self.planner = CenterlineLattice(MAP_DIR)
        self.ego = VehicleState(-60.0, -85.0, math.pi / 2, 0)

    def test_segment_activation_and_direction(self):
        self.assertTrue(self.planner.contains(self.ego))
        self.assertFalse(self.planner.contains(VehicleState(-60, -110, math.pi / 2, 0)))
        self.assertFalse(self.planner.contains(VehicleState(-60, -40, math.pi / 2, 0)))
        self.assertFalse(self.planner.contains(VehicleState(-60, -85, -math.pi / 2, 0)))

    def test_center_crossing_body_rejected(self):
        self.assertTrue(self.planner.center)
        points = [Point(-61.0, -85 + .2 * i) for i in range(50)]
        self.assertFalse(self.planner._body_inside(points, self.ego))

    def test_own_lane_body_allowed(self):
        index = min(range(len(self.planner.center)), key=lambda i: self.planner.center[i].distance(self.ego.position))
        points = self.planner.center[index:index + 20]
        ego = VehicleState(points[0].x, points[0].y, self.planner.planner._path_heading(points, 0), 0)
        self.assertTrue(self.planner._body_inside(points, ego))

    def test_missing_map_blocks_avoidance(self):
        route = [Point(-60, -85 + .5 * i) for i in range(80)]
        entries = [{'object_info': ObjectInfo(-60, -72, 0, 2, is_static=True)}]
        result = CenterlineLattice().run(self.ego, route, entries)
        self.assertFalse(result.active)
        self.assertEqual(0, result.acc_target_velocity)

    def test_zone_precedes_old_overlapping_static_zone(self):
        process = LatticePlanningProcess(map_data_dir=MAP_DIR)
        route = [Point(-60, -85 + .5 * i) for i in range(80)]
        entries = [{'object_info': ObjectInfo(-60, -72, 0, 2, is_static=True)}]
        result = process.run(self.ego, route, entries)
        self.assertIn('centerline_lattice', result.reason)
        self.assertFalse(process.static_avoidance.active)
        for candidate in result.candidates:
            self.assertLessEqual(candidate.offset, 0)
            self.assertTrue(process.centerline_lattice._body_inside(candidate.points, self.ego))
