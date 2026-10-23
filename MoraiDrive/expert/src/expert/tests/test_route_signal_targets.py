"""Protected movement tests use exact ordered routes, independent of runtime files."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from expert.src.planning.route_signal_targets import build_route_signal_targets, signal_permission
from expert.src.planning.traffic_light_stop import TrafficLightStopController


class Mapper:
    def __init__(self, nodes):
        self.nodes = nodes

    def mapped_signal_ids(self):
        return ["T"]

    def get_stop_points(self, signal):
        return [{"idx": row["idx"], "point": row["point"]}
                for row in self.nodes if row.get("traffic_light_id") == signal]

    def __len__(self):
        return 1


class RouteSignalPermissionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.links = []
        for index, movement in enumerate((None, "straight", None, "left")):
            self.links.append({"idx": "L%d" % index, "from_node_idx": "N%d" % index,
                               "to_node_idx": "N%d" % (index + 1), "related_signal": movement,
                               "points": [[x, 0, 0] for x in range(index * 20, (index + 1) * 20 + 1)]})
        self.nodes = [{"idx": "N%d" % index, "point": [index * 20, 0, 0],
                       "on_stop_line": index in (1, 3),
                       "traffic_light_id": "T" if index in (1, 3) else None}
                      for index in range(5)]
        self.route = [row["idx"] for row in self.links]
        self.path = [SimpleNamespace(x=p[0], y=p[1]) for link in self.links for p in link["points"]]
        self.write_map()

    def write_map(self):
        for name, rows in (("link_set.json", self.links), ("node_set.json", self.nodes)):
            with open(os.path.join(self.folder.name, name), "w") as out:
                json.dump(rows, out)

    def document(self, path=None):
        return build_route_signal_targets(self.path if path is None else path,
                                          self.folder.name, self.route)

    def controller(self, document=True, path=None, **kwargs):
        path = self.path if path is None else path
        doc = self.document(path) if document else None
        with contextlib.redirect_stdout(io.StringIO()):
            return TrafficLightStopController(path, Mapper(self.nodes), route_signal_targets=doc,
                                              signal_timeout=0, **kwargs)

    def cap(self, controller, status, waypoint, source=None):
        signal = ["T", status, 2]
        if source is not None:
            signal.extend([0, source, source + 5])
        with contextlib.redirect_stdout(io.StringIO()):
            return controller.get_target_velocity(signal, waypoint, 10)

    def test_straight_green_stops_left_but_releases_straight_at_same_signal_id(self):
        controller = self.controller()
        straight, left = self.document()["targets"]
        self.assertEqual("straight", straight["signal_movement"])
        self.assertEqual("left", left["signal_movement"])
        self.assertEqual(10, self.cap(controller, 16, straight["path_index"]))
        self.assertEqual(0, self.cap(controller, 16, left["path_index"]))
        self.assertEqual("left", controller.last_decision["signal_movement"])
        self.assertFalse(controller.last_decision["signal_permission"])
        self.assertEqual("L3", controller.last_decision["signal_route_link_id"])

    def test_canonical_phase_matrix_at_exact_stopline(self):
        straight, left = self.document()["targets"]
        for movement, target, allowed in (("straight", straight, {16, 48}),
                                           ("left", left, {32, 33, 48})):
            for status in (1, 4, 16, 32, 33, 48):
                with self.subTest(movement=movement, raw_status=status):
                    ctrl = self.controller()
                    self.assertEqual(10 if status in allowed else 0,
                                     self.cap(ctrl, status, target["path_index"]))

    def test_cropped_runtime_path_retains_exact_occurrence_and_following_movement(self):
        cropped = self.path[50:75]
        doc = self.document(cropped)
        self.assertEqual(50, doc["route_slice_start_index"])
        self.assertEqual(1, len(doc["targets"]))
        target = doc["targets"][0]
        self.assertEqual("left", target["signal_movement"])
        ctrl = self.controller(path=cropped)
        self.assertEqual(0, self.cap(ctrl, 16, target["path_index"]))
        self.assertEqual(10, self.cap(ctrl, 33, target["path_index"]))

    def test_cap_decelerates_before_protected_left_and_does_not_stop_immediately(self):
        ctrl = self.controller()
        target = self.document()["targets"][1]
        cap = self.cap(ctrl, 16, target["path_index"] - 10)
        self.assertGreater(cap, 0)
        self.assertLess(cap, 10)
        self.assertEqual(10, self.cap(ctrl, 33, target["path_index"] - 10))

    def test_stale_release_holds_only_the_exact_previously_stopped_occurrence(self):
        ctrl = self.controller()
        ctrl.signal_timeout = 1
        straight, left = self.document()["targets"]
        self.assertEqual(0, self.cap(ctrl, 1, straight["path_index"]))
        self.assertEqual(0, self.cap(ctrl, 16, straight["path_index"], source=10))
        self.assertIsNone(ctrl.last_decision["signal_permission"])
        # A stale packet must not transfer the first node's stop latch to the
        # next route occurrence governed by the same light ID.
        self.assertEqual(10, self.cap(ctrl, 16, left["path_index"], source=10))
        self.assertIsNone(ctrl.last_decision)
        self.assertEqual(0, self.cap(ctrl, 16, left["path_index"]))

    def test_past_stop_node_and_unknown_signal_do_not_cap_another_route_occurrence(self):
        ctrl = self.controller()
        straight, left = self.document()["targets"]
        self.assertEqual(0, self.cap(ctrl, 1, straight["path_index"]))
        self.assertEqual(10, self.cap(ctrl, 48, left["path_index"] + 1))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(10, ctrl.get_target_velocity(["unrelated", 1, 2], 20, 10))

    def test_unrelated_or_missing_packet_cannot_release_current_protected_stop(self):
        ctrl = self.controller()
        left = self.document()["targets"][1]
        index = left["path_index"]
        self.assertEqual(0, self.cap(ctrl, 16, index))
        for packet in (["unrelated", 16, 2], None):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, ctrl.get_target_velocity(packet, index, 10))
            self.assertEqual("T", ctrl.last_decision["signal_id"])
            self.assertIsNone(ctrl.last_decision["signal_permission"])
        self.assertEqual(10, self.cap(ctrl, 32, index))

    def test_future_signal_red_does_not_brake_for_a_different_upcoming_signal(self):
        self.nodes[3]["traffic_light_id"] = "future"
        self.write_map()
        ctrl = self.controller()
        first = self.document()["targets"][0]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(10, ctrl.get_target_velocity(["future", 1, 2], first["path_index"], 10))
        self.assertIsNone(ctrl.last_decision)
        self.assertEqual(0, self.cap(ctrl, 1, first["path_index"]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, ctrl.get_target_velocity(["future", 32, 2], first["path_index"], 10))

    def test_unknown_phase_does_not_release_already_blocked_protected_left(self):
        ctrl = self.controller()
        left = self.document()["targets"][1]
        self.assertEqual(0, self.cap(ctrl, 16, left["path_index"]))
        for status in (0, -1, 20, 36):
            self.assertIsNone(signal_permission(status, "left"))
            self.assertEqual(0, self.cap(ctrl, status, left["path_index"]))
            self.assertIsNone(ctrl.last_decision["signal_permission"])
        self.assertEqual(10, self.cap(ctrl, 32, left["path_index"]))

    def test_unknown_movement_preserves_legacy_status_policy_without_guessing_turn(self):
        self.links[1]["related_signal"] = "right_unprotected"
        self.write_map()
        ctrl = self.controller()
        target = self.document()["targets"][0]
        self.assertIsNone(target["signal_movement"])
        self.assertEqual(0, self.cap(ctrl, 1, target["path_index"]))
        self.assertEqual(10, self.cap(ctrl, 16, target["path_index"]))
        self.assertEqual(10, self.cap(ctrl, 33, target["path_index"]))

    def test_no_route_contract_preserves_legacy_controller(self):
        ctrl = self.controller(document=False)
        self.assertFalse(ctrl.route_permission_bound)
        target = ctrl.get_route_targets("T")[0]
        self.assertEqual(0, self.cap(ctrl, 1, target["path_index"]))
        self.assertEqual(10, self.cap(ctrl, 16, target["path_index"]))

    def test_geometry_mismatch_disconnected_route_and_ambiguous_slice_fail(self):
        bad = list(self.path)
        bad[10] = SimpleNamespace(x=10, y=0.01)
        with self.assertRaisesRegex(ValueError, "unique exact slice"):
            self.document(bad)
        self.links[1]["from_node_idx"] = "not_connected"
        self.write_map()
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            self.document()
        self.links[1]["from_node_idx"] = "N1"
        # Repeat an exact two-point geometry inside one otherwise valid route.
        self.links[0]["points"] = [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [20, 0, 0]]
        self.write_map()
        with self.assertRaisesRegex(ValueError, "unique exact slice"):
            self.document([SimpleNamespace(x=0, y=0), SimpleNamespace(x=1, y=0)])


if __name__ == "__main__":
    unittest.main()
