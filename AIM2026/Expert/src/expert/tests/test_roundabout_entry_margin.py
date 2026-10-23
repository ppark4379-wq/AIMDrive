"""A rear-entry margin uses the existing yield branch and preserves real collisions."""

import math
from pathlib import Path
import unittest
from unittest.mock import patch

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway.planner import HighwayLanePlanner
from expert.src.planning.lattice import parameters as p
from expert.src.planning.roundabout.planner import RoundaboutEntryPlanner
from expert.src.vehicle_state import VehicleState


class RoundaboutEntryMarginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[3] / 'map_data' / 'R_KR_PG_KATRI_2025'
        cls.lane_map = RoundaboutEntryPlanner(str(directory)).lane_map
        cls.path = [Point(*map(float, line.split()[:2])) for line in
                    (directory / '2026_molit_comp_global_path.txt').read_text().splitlines() if line.strip()]

    def setUp(self):
        self.ego = VehicleState(-99.21018982, 318.29766846, math.radians(82.54277), 7.01351833)
        self.rear = ObjectInfo(-109.01864624, 326.96453857, 5.67355368, 1,
                               heading_deg=-23.80322, size_x=4.845, size_y=2.09, unique_id=44)
        self.front = ObjectInfo(-85.28794861, 336.91485596, 5.67529466, 1,
                                heading_deg=67.43376, size_x=4.845, size_y=2.09, unique_id=18)

    def plan(self, margin, objects, entering=False, committed=False):
        planner = RoundaboutEntryPlanner()
        planner.lane_map = self.lane_map
        planner.entering, planner.committed = entering, committed
        index = min(range(len(self.path)), key=lambda i: self.path[i].distance(self.ego.position))
        with patch.object(p, 'ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN', margin):
            return planner.plan(self.ego, self.path[index:index+200],
                                [{'object_info': obj} for obj in objects], {'in_approach': True}, 26./3.6)

    def test_logged_near_entry_uses_existing_yield_branch(self):
        original = self.plan(0., [self.rear, self.front])
        conservative = self.plan(4., [self.rear, self.front])
        self.assertEqual('follow', original.debug_details['roundabout_entry']['phase'])
        self.assertEqual('yield', conservative.debug_details['roundabout_entry']['phase'])
        self.assertLess(conservative.acc_target_velocity, original.acc_target_velocity)
        self.assertEqual(44, conservative.debug_details['roundabout_entry']['blocker_unique_id'])
        self.assertEqual(4., conservative.debug_details['roundabout_entry']['rear_front_margin_m'])

    def test_clear_and_front_only_scenes_keep_their_output(self):
        for objects in ([], [self.front]):
            with self.subTest(count=len(objects)):
                original, conservative = self.plan(0., objects), self.plan(4., objects)
                self.assertEqual(original.debug_details, conservative.debug_details)
                self.assertEqual(original.acc_target_velocity, conservative.acc_target_velocity)

    def test_entry_and_committed_phases_keep_their_output(self):
        for entering, committed in ((True, False), (True, True)):
            with self.subTest(entering=entering, committed=committed):
                original = self.plan(0., [self.rear, self.front], entering, committed)
                conservative = self.plan(4., [self.rear, self.front], entering, committed)
                self.assertEqual(original.debug_details, conservative.debug_details)
                self.assertEqual(original.acc_target_velocity, conservative.acc_target_velocity)

    def test_padding_keeps_physical_body_and_does_not_mutate_input(self):
        planner = RoundaboutEntryPlanner()
        ego = VehicleState(0., 0., 0., 0.)
        route = planner._route([Point(0., 0.), Point(100., 0.)])
        obj = ObjectInfo(-6., 0., 5., 1, size_x=4.5, size_y=2., unique_id=1)
        with patch.object(planner, '_is_rear_entry_vehicle', return_value=True), \
                patch.object(p, 'ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN', 4.), \
                patch.object(HighwayLanePlanner, '_obb_collision', return_value=False) as check:
            planner._collision(ego, route, 0., [(obj, None)], 0., [[[(obj.position, 0.)]]])
        center, shape = check.call_args.args[2], check.call_args.args[4]
        self.assertAlmostEqual(obj.position.x-.5*obj.size_x, center.x-.5*shape.size_x)
        self.assertAlmostEqual(obj.position.x+.5*obj.size_x+4., center.x+.5*shape.size_x)
        self.assertEqual(obj.size_y, shape.size_y)
        self.assertEqual((-6., 4.5), (obj.position.x, obj.size_x))
        with patch.object(planner, '_is_rear_entry_vehicle', return_value=True):
            for x in range(-8, 11):
                obj.position = Point(float(x), 0.)
                with patch.object(p, 'ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN', 0.):
                    old = planner._collision(ego, route, 0., [(obj, None)], 0., [[[(obj.position, 0.)]]])
                with patch.object(p, 'ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN', 4.):
                    new = planner._collision(ego, route, 0., [(obj, None)], 0., [[[(obj.position, 0.)]]])
                if old is not None:
                    self.assertIsNotNone(new, x)


if __name__ == '__main__':
    unittest.main()
