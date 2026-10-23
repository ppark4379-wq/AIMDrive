"""Signal-specific policies use the same route stop target and braking curve."""

import csv
import json
import math
from pathlib import Path
import time
import unittest

from expert.src.localization.point import Point
from expert.src.path.mapping import NodeTrafficLightStoplineMapper
from expert.src.planning.traffic.stop_controller import TrafficLightStopController


IGNORED_SIGNAL = "C1256W000123"
LEFT_SIGNAL = "C1256W000052"
ORDINARY_SIGNAL = "C1256W000016"
EXPERT_ROOT = Path(__file__).resolve().parents[3]
MAP_DIR = EXPERT_ROOT / "map_data" / "R_KR_PG_KATRI_2025"
CONFIG_FILE = EXPERT_ROOT / "src" / "expert" / "src" / "config" / "config.json"


class StopMapper:
    def __init__(self):
        self.stops = {
            IGNORED_SIGNAL: [{"idx": "ignored_stop", "point": [30.0, 0.0]}],
            ORDINARY_SIGNAL: [{"idx": "ordinary_stop", "point": [90.0, 0.0]}],
            LEFT_SIGNAL: [{"idx": "left_stop", "point": [150.0, 0.0]}],
        }
        self.stop_node_count = 3

    def __len__(self):
        return len(self.stops)

    def mapped_signal_ids(self):
        return list(self.stops)

    def get_stop_points(self, signal_id):
        return list(self.stops.get(signal_id, []))


def signal(signal_id, status, age=0.0):
    return [signal_id, status, 2, time.monotonic() - age]


def controller(with_policies=True, **kwargs):
    settings = {
        "path": [Point(float(index), 0.0) for index in range(301)],
        "stopline_mapper": StopMapper(),
        "stop_margin": 4.0,
        "comfortable_deceleration": 2.5,
        "max_lookahead": 120.0,
        "signal_timeout": 2.0,
    }
    if with_policies:
        settings.update(
            ignored_signal_ids=[IGNORED_SIGNAL],
            left_arrow_only_signal_ids=[LEFT_SIGNAL],
        )
    settings.update(kwargs)
    return TrafficLightStopController(**settings)


class TrafficLightPolicyTest(unittest.TestCase):
    def test_ignored_signal_never_adds_its_own_stop(self):
        for status in (-2, -1, 0, 1, 4, 5, 16, 32, 33, 36, 48):
            with self.subTest(status=status):
                subject = controller()
                self.assertEqual(
                    12.0,
                    subject.get_target_velocity(signal(IGNORED_SIGNAL, status), 30, 12.0),
                )

    def test_ignored_signal_does_not_disable_other_red_signal(self):
        subject = controller()
        subject.get_target_velocity(signal(IGNORED_SIGNAL, 1), 30, 12.0)
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(ORDINARY_SIGNAL, 1), 90, 12.0)
        )
        self.assertEqual(ORDINARY_SIGNAL, subject.last_decision["signal_id"])

    def test_ordinary_red_uses_closer_stop_than_unknown_left_signal(self):
        subject = controller()
        expected = math.sqrt(2.0 * 2.5 * (10.0 - 4.0))
        actual = subject.get_target_velocity(signal(ORDINARY_SIGNAL, 1), 80, 12.0)
        self.assertAlmostEqual(expected, actual)
        self.assertEqual("ordinary_stop", subject.last_decision["node_id"])

    def test_ordinary_green_still_releases_ordinary_stop(self):
        subject = controller()
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(ORDINARY_SIGNAL, 1), 90, 12.0)
        )
        self.assertEqual(
            12.0, subject.get_target_velocity(signal(ORDINARY_SIGNAL, 16), 90, 12.0)
        )

    def test_left_arrow_combinations_release_required_stop(self):
        # MORAI LG, R|LG, Y|LG and SG|LG all include an illuminated left arrow.
        for status in (32, 33, 36, 48):
            with self.subTest(status=status):
                subject = controller()
                self.assertEqual(
                    12.0,
                    subject.get_target_velocity(signal(LEFT_SIGNAL, status), 150, 12.0),
                )

    def test_every_state_without_left_arrow_stops(self):
        for status in (-2, -1, 0, 1, 4, 5, 16, 20, 64, 128):
            with self.subTest(status=status):
                subject = controller()
                self.assertEqual(
                    0.0,
                    subject.get_target_velocity(signal(LEFT_SIGNAL, status), 150, 12.0),
                )
                self.assertEqual(LEFT_SIGNAL, subject.last_decision["signal_id"])

    def test_left_policy_matches_existing_red_braking_curve(self):
        ordinary_red = controller(with_policies=False)
        subject = controller()
        for waypoint in (110, 125, 130, 140, 146, 150):
            with self.subTest(waypoint=waypoint):
                expected = ordinary_red.get_target_velocity(
                    signal(LEFT_SIGNAL, 1), waypoint, 20.0
                )
                actual = subject.get_target_velocity(signal(LEFT_SIGNAL, 16), waypoint, 20.0)
                self.assertAlmostEqual(expected, actual)
                self.assertEqual("left_stop", subject.last_decision["node_id"])

    def test_left_policy_uses_configured_stop_margin_and_deceleration(self):
        subject = controller(stop_margin=13.0, comfortable_deceleration=1.5)
        actual = subject.get_target_velocity(signal(LEFT_SIGNAL, 16), 130, 12.0)
        self.assertAlmostEqual(math.sqrt(2.0 * 1.5 * (20.0 - 13.0)), actual)
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 16), 137, 12.0)
        )

    def test_required_light_releases_and_stops_again(self):
        subject = controller()
        for status, expected in ((16, 0.0), (33, 12.0), (16, 0.0), (48, 12.0), (4, 0.0)):
            with self.subTest(status=status):
                self.assertEqual(
                    expected,
                    subject.get_target_velocity(signal(LEFT_SIGNAL, status), 150, 12.0),
                )

    def test_missing_input_stops_without_having_seen_red(self):
        for sample in (None, [], [LEFT_SIGNAL]):
            with self.subTest(sample=sample):
                subject = controller()
                self.assertEqual(0.0, subject.get_target_velocity(sample, 150, 12.0))
                self.assertEqual(LEFT_SIGNAL, subject.last_decision["signal_id"])

    def test_other_signal_permission_cannot_release_required_stop(self):
        for signal_id in (ORDINARY_SIGNAL, IGNORED_SIGNAL, "unmapped_signal"):
            for status in (16, 32, 48):
                with self.subTest(signal_id=signal_id, status=status):
                    subject = controller()
                    self.assertEqual(
                        0.0,
                        subject.get_target_velocity(signal(signal_id, status), 150, 12.0),
                    )

    def test_missing_or_other_input_revokes_previous_left_permission(self):
        for next_signal in (None, [], signal(ORDINARY_SIGNAL, 48)):
            with self.subTest(next_signal=next_signal):
                subject = controller()
                self.assertEqual(
                    12.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 48), 150, 12.0)
                )
                self.assertEqual(0.0, subject.get_target_velocity(next_signal, 150, 12.0))

    def test_stale_left_arrow_does_not_grant_permission(self):
        for status in (32, 33, 36, 48):
            with self.subTest(status=status):
                subject = controller()
                self.assertEqual(
                    0.0,
                    subject.get_target_velocity(signal(LEFT_SIGNAL, status, age=10.0), 150, 12.0),
                )

    def test_stale_arrow_revokes_previous_fresh_permission(self):
        subject = controller()
        self.assertEqual(
            12.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 48), 150, 12.0)
        )
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 48, age=10.0), 150, 12.0)
        )

    def test_fresh_simulation_time_overrides_old_wall_arrival(self):
        subject = controller()
        sample = signal(LEFT_SIGNAL, 33, age=100.0) + [10.0, 10.5]
        self.assertEqual(12.0, subject.get_target_velocity(sample, 150, 12.0))

    def test_stale_or_future_simulation_sample_stops(self):
        for source_time, reference_time in ((10.0, 13.0), (11.0, 10.0)):
            with self.subTest(source_time=source_time, reference_time=reference_time):
                subject = controller()
                sample = signal(LEFT_SIGNAL, 33) + [source_time, reference_time]
                self.assertEqual(0.0, subject.get_target_velocity(sample, 150, 12.0))

    def test_left_policy_requires_permission_even_when_stale_hold_is_disabled(self):
        subject = controller(hold_on_stale=False)
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 48, age=10.0), 150, 12.0)
        )

    def test_left_policy_does_not_stop_before_lookahead_or_after_stop_node(self):
        subject = controller(max_lookahead=30.0)
        for waypoint in (100, 151, 200):
            with self.subTest(waypoint=waypoint):
                self.assertEqual(12.0, subject.get_target_velocity(None, waypoint, 12.0))

    def test_closed_route_does_not_stop_for_passed_left_target(self):
        subject = controller(is_closed_path=True)
        self.assertEqual(12.0, subject.get_target_velocity(None, 151, 12.0))

    def test_no_configuration_retains_existing_status_policy(self):
        subject = controller(with_policies=False)
        for status, expected in ((1, 0.0), (16, 12.0), (4, 12.0), (33, 12.0)):
            with self.subTest(status=status):
                self.assertEqual(
                    expected,
                    subject.get_target_velocity(signal(LEFT_SIGNAL, status), 150, 12.0),
                )
        self.assertEqual(12.0, subject.get_target_velocity(None, 150, 12.0))
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(IGNORED_SIGNAL, 1), 30, 12.0)
        )


class ConfiguredTrafficLightPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_FILE.open(encoding="utf-8") as config_file:
            cls.config = json.load(config_file)
        with (MAP_DIR / "node_set.json").open(encoding="utf-8") as node_file:
            cls.nodes = json.load(node_file)
        cls.mapper = NodeTrafficLightStoplineMapper(str(MAP_DIR))
        path_file = EXPERT_ROOT / ".runtime" / "path.csv"
        if not path_file.exists():
            path_file = EXPERT_ROOT / ".runtime" / "scenario_runner" / "path.csv"
        with path_file.open(encoding="utf-8", newline="") as route_file:
            cls.path = [
                Point(float(row["x"]), float(row["y"]))
                for row in csv.DictReader(route_file)
            ]

    def configured_controller(self):
        return TrafficLightStopController(
            path=self.path,
            stopline_mapper=self.mapper,
            is_closed_path=self.config["map"]["is_closed_path"],
            **self.config["planning"]["traffic_light_stop"],
        )

    def test_configuration_scopes_overrides_to_exact_two_signals(self):
        policy = self.config["planning"]["traffic_light_stop"]
        self.assertEqual([IGNORED_SIGNAL], policy["ignored_signal_ids"])
        self.assertEqual([LEFT_SIGNAL], policy["left_arrow_only_signal_ids"])
        self.assertEqual([1], policy["stop_statuses"])

    def test_photo_coordinates_select_expected_nearest_mapped_stop_nodes(self):
        for photo_xy, expected_signal, expected_node in (
            ((-60.77, 73.79), IGNORED_SIGNAL, "A1256W000757"),
            ((-13.13, 98.09), LEFT_SIGNAL, "A1256W000742"),
        ):
            with self.subTest(photo_xy=photo_xy):
                nearest = min(
                    (node for node in self.nodes if node.get("on_stop_line") and node.get("traffic_light_id")),
                    key=lambda node: math.hypot(
                        node["point"][0] - photo_xy[0], node["point"][1] - photo_xy[1]
                    ),
                )
                self.assertEqual(expected_signal, nearest["traffic_light_id"])
                self.assertEqual(expected_node, nearest["idx"])

    def test_both_photo_stop_nodes_are_on_actual_route(self):
        subject = self.configured_controller()
        for signal_id, node_id in (
            (IGNORED_SIGNAL, "A1256W000757"),
            (LEFT_SIGNAL, "A1256W000742"),
        ):
            with self.subTest(signal_id=signal_id):
                targets = subject.get_route_targets(signal_id)
                self.assertTrue(any(target["node_id"] == node_id for target in targets))

    def test_configuration_applies_left_policy_at_real_stop_node(self):
        subject = self.configured_controller()
        target = next(
            target for target in subject.get_route_targets(LEFT_SIGNAL)
            if target["node_id"] == "A1256W000742"
        )
        waypoint = target["path_index"]
        self.assertEqual(
            0.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 16), waypoint, 12.0)
        )
        self.assertEqual("A1256W000742", subject.last_decision["node_id"])
        self.assertEqual(
            12.0, subject.get_target_velocity(signal(LEFT_SIGNAL, 33), waypoint, 12.0)
        )


if __name__ == "__main__":
    unittest.main()
