"""Check the rear-only extension in road coordinates and on the bundled map."""

import math
from pathlib import Path
import unittest
from unittest.mock import patch

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway.planner import HighwayLanePlanner
from expert.src.planning.highway.structures import HighwayLaneLink
from expert.src.planning.lattice import parameters as p
from expert.src.planning.roundabout.planner import RoundaboutEntryPlanner
from expert.src.vehicle_state import VehicleState


class RoundaboutTrackingRangeTest(unittest.TestCase):
    def setUp(self):
        self.planner = RoundaboutEntryPlanner()
        self.ego = VehicleState(0., -10., math.pi / 2., 0.)
        self.addCleanup(patch.stopall)
        patch.object(p, 'ROUNDABOUT_TRACK_DISTANCE', 60.).start()
        patch.object(p, 'ROUNDABOUT_REAR_TRACK_DISTANCE', 65.).start()
        patch.object(p, 'ROUNDABOUT_ENTRY_CIRCULATING_LINK_ID', 'merge').start()
        patch.object(p, 'ROUNDABOUT_ENTRY_MERGE_POINT', (0., 0.)).start()
        patch.object(p, 'ROUNDABOUT_ENTRY_HOLD_POINT', (0., -10.)).start()
        # A circulating loop and a separate Ego entry road join at node C.
        specs = (
            ('upstream', 'A', 'B', [(-100., 0.), (-20., 0.)]),
            ('merge', 'B', 'C', [(-20., 0.), (20., 0.)]),
            ('front', 'C', 'D', [(20., 0.), (100., 0.)]),
            ('return', 'D', 'A', [(100., 0.), (100., 100.), (-100., 100.), (-100., 0.)]),
            ('entry', 'E', 'C', [(0., -100.), (0., -10.), (20., 0.)]),
        )
        lane_map = self.planner.lane_map
        for identity, start, end, xy in specs:
            points = [Point(*point) for point in xy]
            lane_map.links[identity] = HighwayLaneLink(
                dict(idx=identity, from_node_idx=start, to_node_idx=end), 0,
                points, HighwayLanePlanner._cumulative_distance(points), False)
        lane_map.successors = {
            link.id: tuple(other.id for other in lane_map.links.values()
                           if other.from_node_id == link.to_node_id)
            for link in lane_map.links.values()
        }

    @staticmethod
    def vehicle(x, y, heading=0., identity=1, speed=8.):
        return ObjectInfo(x, y, speed, 1, heading_deg=heading,
                          size_x=4.5, size_y=2., unique_id=identity)

    def tracked(self, objects):
        return [obj.unique_id for obj, _ in self.planner._tracks(
            self.ego, [{'object_info': obj} for obj in objects])]

    def test_rear_extension_and_front_boundary(self):
        for distance in (59.99, 60., 60.01, 64.99, 65., 65.01):
            x = math.sqrt(distance * distance - 100.)
            with self.subTest(distance=distance):
                rear = self.vehicle(-x, 0., identity=1)
                front = self.vehicle(x, 0., identity=2)
                self.assertEqual(([1] if distance <= 65. else [])
                                 + ([2] if distance <= 60. else []),
                                 self.tracked([rear, front]))

    def test_rear_direction_uses_circulating_lane_not_ego_heading(self):
        rear = self.vehicle(-62., 0.)
        for yaw in (0., math.pi / 2., math.pi, -math.pi / 2.):
            with self.subTest(yaw=yaw):
                self.ego.yaw = yaw
                self.assertEqual([1], self.tracked([rear]))

    def test_entry_queue_wrong_way_and_unmatched_cars_keep_60m(self):
        objects = [self.vehicle(0., -72., 90., identity=1),
                   self.vehicle(-62., 0., 180., identity=2),
                   self.vehicle(-62., 10., identity=3)]
        self.assertEqual([], self.tracked(objects))
        self.assertEqual([4], self.tracked([self.vehicle(0., -69., 90., identity=4)]))

    def test_no_map_preserves_existing_range(self):
        self.planner = RoundaboutEntryPlanner()
        self.assertEqual([1], self.tracked([
            self.vehicle(-50., 0., identity=1), self.vehicle(-62., 0., identity=2)]))

    def test_added_rear_vehicle_reaches_entry_collision_prediction(self):
        rear = self.vehicle(-62., 0., speed=20.)
        path = [Point(0., -10.), Point(0., 0.)] + [Point(float(x), 0.) for x in range(1, 121)]
        state = {'in_approach': True, 'past_go_line': False}
        entries = [{'object_info': rear}]
        with patch.object(p, 'ROUNDABOUT_REAR_TRACK_DISTANCE', 60.):
            previous = self.planner.plan(self.ego, path, entries, state)
        updated = self.planner.plan(self.ego, path, entries, state)
        self.assertGreater(previous.acc_target_velocity, 0.)
        self.assertEqual([], previous.predicted_object_paths)
        self.assertEqual(1, len(updated.predicted_object_paths))
        self.assertEqual(0., updated.acc_target_velocity)
        self.assertEqual(rear.unique_id, updated.debug_details['roundabout_entry']['blocker_unique_id'])


class RoundaboutMapTrackingRangeTest(unittest.TestCase):
    def test_real_map_rear_and_front_in_the_added_distance_band(self):
        map_dir = Path(__file__).resolve().parents[3] / 'map_data' / 'R_KR_PG_KATRI_2025'
        planner = RoundaboutEntryPlanner(str(map_dir))
        ego = VehicleState(*p.ROUNDABOUT_ENTRY_APPROACH_START, math.pi / 2., 0.)
        for link_id, expected in (('A2256W000133', True), ('A2256W000135', False)):
            link = planner.lane_map.links[link_id]
            candidates = [point for point in link.points if 60.1 < point.distance(ego.position) < 64.9]
            self.assertTrue(candidates, link_id)
            for point in candidates:
                with self.subTest(link_id=link_id, point=str(point)):
                    match = planner.lane_map.project(point, link)
                    obj = RoundaboutTrackingRangeTest.vehicle(point.x, point.y, math.degrees(match.yaw))
                    tracks = planner._tracks(ego, [{'object_info': obj}])
                    self.assertEqual(expected, bool(tracks))


if __name__ == '__main__':
    unittest.main()
