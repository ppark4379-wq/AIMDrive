import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts import traffic_light_override


def make_context(color=1):
    context = SimpleNamespace(
        cfg={
            "traffic_light_wait_override": {
                "enabled": True,
                "wait_wall_sec": 10.0,
                "stopped_speed_mps": 0.5,
                "minimum_stopline_distance_m": 0.0,
                "maximum_stopline_distance_m": 25.0,
                "query_period_wall_sec": 0.25,
                "retry_period_wall_sec": 1.0,
                "verification_timeout_wall_sec": 2.0,
                "target_color": 16,
                "impulse": True,
                "set_sibling": True,
            }
        },
        grpc=Mock(),
        event=Mock(),
    )
    traffic_light_override.initialize(context)
    context.traffic_light_stops = [{"light_id": "C-test", "route_s": 100.0}]
    context.grpc.get_traffic_light_state_by_id.return_value = {
        "id": "C-test",
        "color": color,
    }
    context.grpc.set_traffic_light_color_verified.return_value = {
        "id": "C-test",
        "target_color": 16,
        "accepted": True,
        "verified": False,
        "before": {"id": "C-test", "color": 1},
        "after": {"id": "C-test", "color": 1},
        "error": None,
    }
    return context


class TrafficLightOverrideTests(unittest.TestCase):
    def test_red_changes_only_after_ten_host_monotonic_seconds(self):
        context = make_context()
        red = {"id": "C-test", "color": 1}
        green = {"id": "C-test", "color": 16}
        context.grpc.get_traffic_light_state_by_id.side_effect = [
            red,
            red,
            red,
            green,
        ]
        with patch.object(
            traffic_light_override.time,
            "monotonic",
            side_effect=[100.0, 109.9, 110.16, 110.42],
        ):
            traffic_light_override.update(context, 20.0, {"speed": 0.0}, 90.0)
            traffic_light_override.update(context, 21.0, {"speed": 0.0}, 90.0)
            context.grpc.set_traffic_light_color_verified.assert_not_called()
            traffic_light_override.update(context, 22.0, {"speed": 0.0}, 90.0)
            self.assertEqual(context.traffic_override_verified, 0)
            traffic_light_override.update(context, 22.1, {"speed": 1.0}, 91.0)

        context.grpc.set_traffic_light_color_verified.assert_called_once_with(
            "C-test", 16, impulse=True, set_sibling=True
        )
        self.assertEqual(context.traffic_override_attempts, 1)
        self.assertEqual(context.traffic_override_verified, 1)
        self.assertIn(("C-test", 100.0), context.traffic_override_completed)
        names = [call.args[0] for call in context.event.call_args_list]
        self.assertEqual(
            names,
            [
                "traffic_wait_started",
                "traffic_light_override",
                "traffic_light_override_verified",
            ],
        )
        for call in context.event.call_args_list[1:]:
            self.assertFalse(call.kwargs["training_visible"])
            self.assertTrue(call.kwargs["intervention"])

    def test_moving_or_green_ego_never_requests_override(self):
        moving = make_context(color=1)
        green = make_context(color=16)
        with patch.object(
            traffic_light_override.time,
            "monotonic",
            side_effect=[100.0, 100.0],
        ):
            traffic_light_override.update(moving, 20.0, {"speed": 1.0}, 90.0)
            traffic_light_override.update(green, 20.0, {"speed": 0.0}, 90.0)
        moving.grpc.set_traffic_light_color_verified.assert_not_called()
        green.grpc.set_traffic_light_color_verified.assert_not_called()
        self.assertIsNone(moving.traffic_wait_state)
        self.assertIsNone(green.traffic_wait_state)

    def test_missing_speed_fails_closed(self):
        context = make_context(color=1)
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 20.0, {}, 90.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()
        context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_morai_bitmask_semantics(self):
        self.assertFalse(traffic_light_override.straight_movement_is_blocked(16))
        self.assertFalse(traffic_light_override.straight_movement_is_blocked(48))
        self.assertTrue(traffic_light_override.straight_movement_is_blocked(1))
        self.assertTrue(traffic_light_override.straight_movement_is_blocked(33))
        self.assertTrue(traffic_light_override.target_is_satisfied(16, 16))
        self.assertTrue(traffic_light_override.target_is_satisfied(48, 16))
        self.assertFalse(traffic_light_override.target_is_satisfied(1, 16))
        self.assertFalse(traffic_light_override.target_is_satisfied(33, 16))

    def test_prepare_binds_exact_stop_light_and_active_route_window(self):
        links = {
            "L1": {"points": [(0, 0, 0), (10, 0, 0)], "to_node_idx": "N1"},
            "L2": {"points": [(10, 0, 0), (20, 0, 0)], "to_node_idx": "N2"},
        }
        nodes = {
            "N1": {"on_stop_line": True, "traffic_light_id": "C-test"},
            "N2": {"on_stop_line": False},
        }
        context = SimpleNamespace(
            cfg={"traffic_light_wait_override": {"enabled": True}},
            route_definition={"route_links": ["L1", "L2"]},
            map_loader=SimpleNamespace(
                get_link=lambda link_id: links[link_id],
                node_set=nodes,
            ),
            route_points=[(0, 0, 0), (10, 0, 0), (20, 0, 0)],
            route_length_m=20.0,
        )
        traffic_light_override.initialize(context)
        traffic_light_override.prepare(context)
        self.assertEqual(len(context.traffic_light_stops), 1)
        self.assertEqual(context.traffic_light_stops[0]["light_id"], "C-test")
        self.assertEqual(context.traffic_light_stops[0]["route_s"], 10.0)

        context._full_route_points = list(context.route_points)
        context._route_prefix_s = 10.0
        context.route_points = context.route_points[1:]
        context.route_length_m = 10.0
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]["route_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
