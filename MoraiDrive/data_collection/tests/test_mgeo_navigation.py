import json
import math
import os
import tempfile
import unittest

from mgeo_navigation import (
    MGeoMap, MGeoRouteTracker, GNSSMotionEstimator, stable_mgeo_id,
)


class MGeoMapTest(unittest.TestCase):
    def test_stable_mgeo_id(self):
        self.assertEqual(stable_mgeo_id("C1256W000091"),
                         stable_mgeo_id("C1256W000091"))
        self.assertGreaterEqual(stable_mgeo_id("C1256W000091"), 0)
        self.assertEqual(stable_mgeo_id(None), -1)

    def test_gnss_motion_estimator_uses_course_and_speed(self):
        estimator = GNSSMotionEstimator()
        first = estimator.update(0, 0, 1_000_000_000, 0.25, 0.0)
        self.assertAlmostEqual(first.heading_rad, 0.25)
        second = estimator.update(0, 2, 2_000_000_000, 0.25, 0.0)
        self.assertAlmostEqual(second.heading_rad, math.pi / 2)
        self.assertAlmostEqual(second.speed_mps, 2.0)
        self.assertEqual(second.heading_source, "gnss_course")

    def test_checkpoint_tie_prefers_outgoing_link_start(self):
        links = [
            {"idx": "incoming", "points": [[-10, 0, 0], [0, 0, 0]]},
            {"idx": "outgoing", "points": [[0, 0, 0], [10, 0, 0]]},
        ]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "link_set.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(links, f)
            match = MGeoMap(path).match(0, 0, heading_rad=0)
            self.assertEqual(match.link_id, "outgoing")

    def test_matches_nearest_link_and_heading(self):
        links = [
            {"idx": "east", "points": [[0, 0, 0], [10, 0, 0]]},
            {"idx": "north", "points": [[20, 0, 0], [20, 10, 0]]},
        ]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "link_set.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(links, f)
            mgeo = MGeoMap(path)
            match = mgeo.match(5.0, 0.25)
            self.assertEqual(match.link_id, "east")
            self.assertAlmostEqual(match.distance_m, 0.25)
            self.assertAlmostEqual(match.heading_rad, 0.0)
            self.assertAlmostEqual(match.projected_x, 5.0)

            match = mgeo.match(19.8, 5.0)
            self.assertEqual(match.link_id, "north")
            self.assertAlmostEqual(match.heading_rad, math.pi / 2)

    def test_heading_disambiguates_and_builds_directed_route(self):
        links = [
            {"idx": "east1", "from_node_idx": "n0", "to_node_idx": "n1",
             "points": [[0, 0, 0], [10, 0, 0]], "link_length": 10},
            {"idx": "east2", "from_node_idx": "n1", "to_node_idx": "n2",
             "points": [[10, 0, 0], [20, 0, 0]], "link_length": 10},
            {"idx": "west", "from_node_idx": "n2", "to_node_idx": "n1",
             "points": [[20, 0.1, 0], [10, 0.1, 0]], "link_length": 10},
        ]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "link_set.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(links, f)
            mgeo = MGeoMap(path)
            self.assertEqual(mgeo.match(15, 0.05, 0.0).link_id, "east2")
            route = mgeo.build_checkpoint_route([
                {"name": "start", "x": 1, "y": 0, "yaw_deg": 0},
                {"name": "end", "x": 19, "y": 0, "yaw_deg": 0},
            ])
            self.assertEqual(route["nav_link_ids"], ["east1", "east2"])

            mgeo.links["east2"]["related_signal"] = "left"
            tracker = MGeoRouteTracker(mgeo, route, target_distance_m=5,
                                       command_lookahead_m=25)
            progress = tracker.update(4, 0, 0)
            self.assertEqual(progress.current_link_id, "east1")
            self.assertEqual(progress.command, 1)
            self.assertGreaterEqual(progress.target_x, 9)
            later = tracker.update(14, 0, 0)
            self.assertGreaterEqual(later.route_index, progress.route_index)

            hinted = MGeoRouteTracker(mgeo, route, initial_route_index=1)
            self.assertGreaterEqual(hinted.update(10, 0, 0).route_index, 1)


if __name__ == "__main__":
    unittest.main()
