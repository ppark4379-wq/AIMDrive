import json
import math
import os
import tempfile
import unittest
from dataclasses import dataclass

from mgeo_navigation import (
    MGeoMap, MGeoRouteTracker, GNSSMotionEstimator, stable_mgeo_id,
    load_official_route,
)


@dataclass
class FakeActor:
    obj_id: int
    obj_type: str
    x: float
    y: float


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
            self.assertEqual(progress.command, 4)
            self.assertEqual(progress.gt_command, 4)
            self.assertGreater(progress.near_route_index, progress.route_index)
            self.assertGreaterEqual(progress.target_x, 9)
            self.assertEqual(progress.future_waypoints[0], [0.0, 0.0, 4])
            later = tracker.update(16, 0, 0)
            self.assertGreaterEqual(later.route_index, progress.route_index)
            self.assertEqual(later.command, 1)

            hinted = MGeoRouteTracker(mgeo, route, initial_route_index=1)
            self.assertGreaterEqual(hinted.update(10, 0, 0).route_index, 1)

            hazards = tracker.hazard_actor_ids(
                [FakeActor(10, "vehicle", 10.0, 0.2),
                 FakeActor(20, "pedestrian", 10.0, 1.0),
                 FakeActor(30, "vehicle", -8.0, 0.0)],
                4.0, 0.0, 0.0, progress.route_index, progress.is_junction,
            )
            self.assertEqual(hazards["vehicle"], [10])
            self.assertEqual(hazards["lane_vehicle"], [10])
            self.assertEqual(hazards["pedestrian"], [20])

    def test_official_scenario_route_is_used_without_recalculation(self):
        data_collection_dir = os.path.dirname(os.path.dirname(__file__))
        morai_dir = os.path.dirname(data_collection_dir)
        scenario_dir = os.path.join(morai_dir, "scenario_runner")
        map_dir = os.path.join(
            scenario_dir, "data", "R_KR_PR_K-city_2025"
        )
        route = load_official_route(
            os.path.join(scenario_dir, "config", "full_loop_links.yaml"),
            os.path.join(map_dir, "2026_molit_comp_global_path.txt"),
        )
        self.assertEqual(route["route_source"], "scenario_runner_official")
        self.assertEqual(len(route["nav_link_ids"]), 39)
        self.assertEqual(len(route["nav_waypoints"]), 4430)

        tracker = MGeoRouteTracker(
            MGeoMap(os.path.join(map_dir, "link_set.json")), route
        )
        self.assertEqual(len(tracker.points), 4430)
        index = 2275
        point = route["nav_waypoints"][index]
        following = route["nav_waypoints"][index + 1]
        heading = math.atan2(following[1] - point[1], following[0] - point[0])
        progress = tracker.update(point[0], point[1], heading)
        self.assertEqual(progress.route_index, index)
        self.assertEqual(progress.future_waypoints[0][:2], point[:2])
        self.assertEqual(len(progress.future_waypoints), 50)
        self.assertEqual(
            progress.future_waypoints,
            tracker.get_future_waypoints(progress.route_index, 50),
        )
        for waypoint in progress.future_waypoints:
            self.assertEqual(len(waypoint), 3)
            self.assertIsInstance(waypoint[0], float)
            self.assertIsInstance(waypoint[1], float)
            self.assertIsInstance(waypoint[2], int)

    def _official_loop_tracker(self, initial_route_index=None):
        from pathlib import Path
        scenario = Path(__file__).resolve().parents[2] / 'scenario_runner'
        map_dir = scenario / 'data/R_KR_PR_K-city_2025'
        route = load_official_route(str(scenario / 'config/full_loop_links.yaml'),
                                   str(map_dir / '2026_molit_comp_global_path.txt'))
        return MGeoRouteTracker(MGeoMap(str(map_dir / 'link_set.json')), route,
                                initial_route_index=initial_route_index)

    def test_closed_route_tracks_two_laps_and_wraps_targets(self):
        tracker = self._official_loop_tracker()
        end = len(tracker.points) - 1
        for lap in range(2):
            for index in list(range(0, end, 10)) + [end - 2, end]:
                x, y = tracker.points[index]
                progress = tracker.update(x, y, tracker._point_heading(index))
                self.assertLess(progress.distance_to_route_m, 0.6)
                self.assertEqual(len(progress.future_waypoints), 50)
                if index == 10:
                    self.assertLess(progress.route_index, 20)
            lookahead = tracker.get_future_waypoints(end, 50)
            self.assertEqual(lookahead[0][:2], list(tracker.points[0]))
            self.assertEqual(lookahead[1][:2], list(tracker.points[1]))
            self.assertGreater(progress.target_route_index, 0)
            self.assertLess(progress.target_route_index, 100)

    def test_closed_route_does_not_reset_at_stop_wrong_heading_or_midroute(self):
        tracker = self._official_loop_tracker()
        end = len(tracker.points) - 1
        tracker._route_index = end
        x, y = tracker.points[end]
        for _ in range(5):
            self.assertEqual(tracker.update(x, y, tracker._point_heading(end)).route_index, end)
        x, y = tracker.points[10]
        self.assertEqual(tracker.update(x, y, tracker._point_heading(10) + math.pi).route_index, end)
        tracker._route_index = 2000
        self.assertGreaterEqual(tracker.update(x, y, tracker._point_heading(10)).route_index, 2000)
        tracker._route_index = end
        self.assertLess(tracker.update(x, y, tracker._point_heading(10)).route_index, 20)

    def test_open_route_keeps_end_and_clipped_waypoints(self):
        links = [{'idx': 'open', 'points': [[0, 0, 0], [10, 0, 0], [20, 0, 0]]}]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'link_set.json')
            with open(path, 'w') as f:
                json.dump(links, f)
            tracker = MGeoRouteTracker(MGeoMap(path), {'nav_link_ids': ['open']})
            tracker.update(20, 0, 0)
            progress = tracker.update(0, 0, 0)
            self.assertEqual(progress.route_index, 2)
            self.assertEqual(progress.target_route_index, 2)
            self.assertEqual(len(progress.future_waypoints), 1)


if __name__ == "__main__":
    unittest.main()
