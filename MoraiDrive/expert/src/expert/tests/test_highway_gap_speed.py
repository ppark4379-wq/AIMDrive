"""Traffic-gap speed choice and the one-way entrance-lane boundary."""
import math
from pathlib import Path
import unittest

from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState


class HighwayGapSpeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.map = HighwayLaneMap(str(root.parents[1] / 'map_data/R_KR_PG_KATRI_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.map)
        self.planner._requested_speed = 27.78
        self.ego = VehicleState(62.17, 244.68, math.radians(-89.7), 11.)
        self.source = self.map.links['A2256W000411']
        self.target = self.map.links['A2256W000409']

    def scene(self, rear_distance=60., rear_speed=13.89):
        projection = self.map.project(self.ego.position, self.target)
        start, yaw, _ = self.map.point_at_s(self.target, 0.)
        objects = [ObjectInfo(62.2, 186., 0., 2, is_static=True,
                             size_x=4.69, size_y=2.2, heading_deg=-90., unique_id=41)]
        for offset, speed, ident in ((25., 11.11, 51), (-rear_distance, rear_speed, 52)):
            station = projection.s + offset
            if station >= 0.:
                point, angle, _ = self.map.point_at_s(self.target, station)
                x, y = point.x, point.y
            else:
                x, y = start.x + station * math.cos(yaw), start.y + station * math.sin(yaw)
                angle = yaw
            objects.append(ObjectInfo(x, y, speed, 1, size_x=4.5, size_y=2.,
                                      heading_deg=math.degrees(angle), unique_id=ident))
        entries = [dict(object_info=obj,
                        local_position=(obj.position-self.ego.position).rotate(-self.ego.yaw))
                   for obj in objects]
        matches = self.planner._match_objects(entries)
        candidate = self.planner._build_candidate(
            self.ego, self.target, projection, 'left', self.source, transition_distance=55.)
        candidate.static_escape = True
        candidate.compact_static = False
        candidate.speed_limit = float('inf')
        return candidate, entries, matches

    def test_speed_search_accepts_gap_before_rear_vehicle_passes(self):
        candidate, entries, matches = self.scene()
        conflict = self.planner.lane_change_safety.check(candidate, self.ego, entries, matches, 27.78)
        self.assertIsNotNone(conflict)
        self.assertFalse(conflict['rear'])
        self.planner._evaluate_lane_change(candidate, self.ego, entries, matches)
        self.assertTrue(candidate.valid)
        self.assertTrue(candidate.lane_change_verified)
        self.assertGreater(candidate.safety_speed, 0.)
        self.assertLess(candidate.safety_speed, 27.78)
        self.assertIsNone(self.planner.lane_change_safety.check(
            candidate, self.ego, entries, matches, candidate.safety_speed))
        chosen = candidate.safety_speed
        self.planner._evaluate_lane_change(candidate, self.ego, entries, matches, committed=True)
        self.assertEqual(chosen, candidate.safety_speed)
        self.assertIsNone(candidate.prediction_conflict)

    def test_planner_delivers_gap_speed_to_control(self):
        _, entries, _ = self.scene()
        match = self.map.project(self.ego.position, self.source)
        path = [pose[0] for pose in self.map.route_points(match, 170., .5)]
        result = self.planner.plan(self.ego, path, entries, desired_velocity=27.78)
        self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)
        self.assertEqual(self.target.id, result.target_link_id)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLess(result.acc_target_velocity, 27.78)

    def test_speed_search_does_not_cut_in_front_of_close_fast_rear(self):
        candidate, entries, matches = self.scene(rear_distance=30., rear_speed=25.)
        self.planner._evaluate_lane_change(candidate, self.ego, entries, matches)
        self.assertFalse(candidate.valid)
        self.assertFalse(candidate.lane_change_verified)

    def test_boundary_is_directional_and_preserves_other_right_changes(self):
        self.assertIn(('left', self.target), self.map.legal_neighbors(self.source))
        self.assertNotIn(self.source, [link for _, link in self.map.legal_neighbors(self.target)])
        self.assertIn(('left', self.map.links['A2256W000410']), self.map.legal_neighbors(self.target))
        self.assertIn(('right', self.target), self.map.legal_neighbors(self.map.links['A2256W000410']))
        self.assertFalse(self.source.excluded)
        self.assertEqual(self.source.id, self.map.default_successor('A2256W000846'))

    def test_obstacle_after_merge_uses_left_lane_or_acc(self):
        for block_left in (False, True):
            with self.subTest(block_left=block_left):
                self.planner = HighwayLanePlanner(lane_map=self.map)
                point, yaw, _ = self.map.point_at_s(self.target, 50.)
                ego = VehicleState(point.x, point.y, yaw, 0.)
                objects = []
                lanes = [self.target]
                if block_left:
                    lanes += [self.map.links['A2256W000410'], self.map.links['A2256W000418']]
                for i, lane in enumerate(lanes):
                    # Keep every legal left lane occupied beside ego, with a
                    # stopped lead in the current lane. Rightmost lane is empty.
                    pos, angle, _ = self.map.point_at_s(lane, (62. if block_left else 90.) if i == 0 else 50.)
                    objects.append(ObjectInfo(pos.x, pos.y, 0., 2, is_static=True,
                        size_x=4.69, size_y=2.2, heading_deg=math.degrees(angle), unique_id=80+i))
                entries = [dict(object_info=obj,
                    local_position=(obj.position-ego.position).rotate(-ego.yaw)) for obj in objects]
                path = [pose[0] for pose in self.map.route_points(
                    self.map.project(ego.position, self.target), 170., .5)]
                result = self.planner.plan(ego, path, entries, desired_velocity=27.78)
                self.assertNotEqual(self.source.id, result.target_link_id)
                if block_left:
                    self.assertEqual('ACC_FOLLOW', result.highway_state, result.debug_details)
                else:
                    self.assertEqual('CHANGE_LEFT', result.highway_state, result.debug_details)

    def test_abort_cannot_reenter_entrance_lane(self):
        self.planner.source_link_id = self.source.id
        self.planner.target_link_id = self.target.id
        self.planner.state = self.planner.CHANGE_LEFT
        self.assertIsNone(self.planner._safe_return_to_source(self.ego, [], []))
        self.assertEqual(self.target.id, self.planner.target_link_id)


if __name__ == '__main__':
    unittest.main()
