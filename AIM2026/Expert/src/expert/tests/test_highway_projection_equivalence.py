"""Compare optimized geometry queries against the original scalar projection."""

import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from expert.src.localization.point import Point
from expert.src.planning.highway.lane_map import HighwayLaneMap


class ScalarLaneMap(HighwayLaneMap):
    project = HighwayLaneMap._project_scalar


class HighwayProjectionEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        map_dir = Path(__file__).resolve().parents[3] / "map_data" / "R_KR_PG_KATRI_2025"
        cls.fast = HighwayLaneMap(str(map_dir))
        cls.reference = ScalarLaneMap(str(map_dir))

    def assert_same_match(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertIsNotNone(actual)
        self.assertEqual(actual.lane_link.id, expected.lane_link.id)
        self.assertEqual((actual.s, actual.distance, actual.x, actual.y, actual.yaw),
                         (expected.s, expected.distance, expected.x, expected.y, expected.yaw))

    def test_all_map_vertices_and_offset_queries_match_exactly(self):
        rng = random.Random(2126)
        for link_id, link in self.fast.links.items():
            queries = list(link.points)
            queries += [Point(point.x + rng.uniform(-6, 6), point.y + rng.uniform(-6, 6))
                        for point in rng.choices(link.points, k=100)]
            for point in queries:
                self.assert_same_match(self.fast.project(point, link),
                                       self.reference.project(point, self.reference.links[link_id]))

    def test_lane_identity_heading_preference_and_distance_cutoff(self):
        for link_id, link in self.fast.links.items():
            for s in (0., link.length / 2., link.length):
                point, yaw, _ = self.fast.point_at_s(link, s)
                for offset in (-4., -2.400000001, -2.4, 0., 2.4, 2.400000001, 4.):
                    position = Point(point.x - offset * math.sin(yaw),
                                     point.y + offset * math.cos(yaw))
                    for heading, preferred in ((None, ()), (yaw, (link_id,)), (yaw + math.pi, ())):
                        self.assert_same_match(self.fast.match(position, heading, preferred),
                                               self.reference.match(position, heading, preferred))

    def test_duplicate_segments_ties_endpoints_and_live_positions(self):
        lane_map = HighwayLaneMap.__new__(HighwayLaneMap)
        # Two equidistant, differently oriented segments: the first must win.
        link = SimpleNamespace(id="tie", points=[Point(-2, 0), Point(0, 0), Point(0, 0), Point(0, 2)],
                               cumulative_s=[0., 2., 2., 4.])
        lane_map._projection_geometry = {
            link.id: (link, lane_map._compile_projection_geometry(link))
        }
        position = Point(-1, 1)
        for xy in ((-1., 1.), (-4., 0.), (0., 5.), (-1., 1. + 1e-15), (-1., 1. - 1e-15)):
            position[:] = xy
            self.assert_same_match(lane_map.project(position, link), lane_map._project_scalar(position, link))
        position[:] = (-1., 1.)
        self.assertEqual(lane_map.project(position, link).yaw, 0.)

    def test_external_or_replaced_link_does_not_reuse_cached_geometry(self):
        link = next(iter(self.fast.links.values()))
        external = SimpleNamespace(id=link.id, points=[Point(0, 0), Point(10, 0)], cumulative_s=[0., 10.])
        self.assert_same_match(self.fast.project(Point(3, 4), external),
                               self.fast._project_scalar(Point(3, 4), external))


if __name__ == "__main__":
    unittest.main()
