import csv
import os
import time
import unittest

from expert.src.localization.point import Point
from expert.src.path.mapping import NodeTrafficLightStoplineMapper
from expert.src.planning.traffic_light_stop import TrafficLightStopController


SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
GT_BEV_ROOT = os.path.normpath(os.path.join(SRC_ROOT, ".."))
MAP_DIR = os.path.join(GT_BEV_ROOT, "map_data", "R_KR_PG_KATRI_2025")
PATH_CSV = os.path.join(GT_BEV_ROOT, ".runtime", "path.csv")
if not os.path.exists(PATH_CSV):
    PATH_CSV = os.path.join(
        GT_BEV_ROOT, ".runtime", "scenario_runner", "path.csv"
    )


def load_path():
    with open(PATH_CSV, "r", encoding="utf-8", newline="") as path_file:
        return [
            Point(float(row["x"]), float(row["y"]))
            for row in csv.DictReader(path_file)
        ]


class TrafficLightStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = NodeTrafficLightStoplineMapper(MAP_DIR)
        cls.path = load_path()
        cls.controller = TrafficLightStopController(
            path=cls.path,
            stopline_mapper=cls.mapper,
            is_closed_path=True,
        )

    def test_topic_signal_maps_to_expected_stop_node(self):
        targets = self.mapper.get_stop_points("C1256W000016")
        self.assertEqual(1, len(targets))
        self.assertEqual("A1256W000208", targets[0]["idx"])
        self.assertAlmostEqual(-60.15604465093929, targets[0]["point"][0])
        self.assertAlmostEqual(-34.51018160348758, targets[0]["point"][1])

    def test_stop_node_is_on_current_route(self):
        targets = self.controller.get_route_targets("C1256W000016")
        self.assertEqual(1, len(targets))
        self.assertEqual(879, targets[0]["path_index"])
        self.assertAlmostEqual(0.0, targets[0]["lateral_distance"], places=6)

    def test_red_slows_and_stops_while_green_releases(self):
        now = time.monotonic()
        red = ["C1256W000016", 1, 2, now]
        green = ["C1256W000016", 16, 2, now]

        approach_speed = self.controller.get_target_velocity(red, 800, 16.0)
        self.assertLess(approach_speed, 16.0)
        self.assertGreater(approach_speed, 0.0)

        stop_speed = self.controller.get_target_velocity(red, 879, 16.0)
        self.assertEqual(0.0, stop_speed)
        self.assertEqual(16.0, self.controller.get_target_velocity(green, 879, 16.0))

    def test_expert_builds_with_direct_map_configuration(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.config.config import Config

        Config._instance = None
        autonomous_driving = AutonomousDriving()
        self.assertGreater(
            autonomous_driving.traffic_light_stop_controller.route_signal_count,
            0,
        )
        path_manager = autonomous_driving.path_manager
        self.assertEqual(1, len(path_manager.speed_zone_indices))
        zone = path_manager.speed_zone_indices[0]
        self.assertEqual(2251, zone["start_index"])
        self.assertEqual(3500, zone["end_index"])
        self.assertAlmostEqual(100.0 / 3.6, path_manager.speed_limit_profile[2251])
        self.assertAlmostEqual(100.0 / 3.6, path_manager.speed_limit_profile[3500])
        self.assertAlmostEqual(60.0 / 3.6, path_manager.speed_limit_profile[2250])
        self.assertAlmostEqual(60.0 / 3.6, path_manager.speed_limit_profile[3501])
        self.assertAlmostEqual(
            100.0 / 3.6,
            max(path_manager.velocity_profile[2251:3501]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
