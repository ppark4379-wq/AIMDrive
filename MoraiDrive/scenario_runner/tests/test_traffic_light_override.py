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
    def test_protected_left_waits_on_straight_green_then_requests_left_green(self):
        context = make_context(color=16)
        context.traffic_light_stops[0].update(
            signal_movement="left", signal_route_link_id="LEFT-EXIT",
        )
        context.grpc.get_traffic_light_state_by_id.side_effect = [
            {"id": "C-test", "color": 16},
            {"id": "C-test", "color": 16},
            {"id": "C-test", "color": 33},
        ]
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111, 112]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
            self.assertEqual(context.traffic_override_verified, 0)
            traffic_light_override.update(context, 12, {"speed": 1}, 91)
        context.grpc.set_traffic_light_color_verified.assert_called_once_with(
            "C-test", 32, impulse=True, set_sibling=True,
        )
        self.assertEqual(context.traffic_override_verified, 1)
        verified = context.event.call_args_list[-1]
        self.assertEqual(verified.kwargs["signal_movement"], "left")
        self.assertEqual(verified.kwargs["verification_semantics"], "protected_left_green_bit")

    def test_same_signal_phase_depends_on_ordered_route_movement(self):
        for movement, color, expected in (
            ("left", 1, 32), ("left", 16, 32),
            ("left", 32, None), ("left", 33, None), ("left", 48, None),
            ("straight", 1, 16), ("straight", 32, 16), ("straight", 33, 16),
            ("straight", 16, None), ("straight", 48, None),
        ):
            with self.subTest(movement=movement, color=color):
                context = make_context(color=color)
                context.traffic_light_stops[0]["signal_movement"] = movement
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                if expected is None:
                    context.grpc.set_traffic_light_color_verified.assert_not_called()
                else:
                    self.assertEqual(context.grpc.set_traffic_light_color_verified.call_args.args,
                                     ("C-test", expected))

    def test_unknown_route_movement_or_signal_color_never_forces_straight_release(self):
        for movement, color in ((None, 1), ("right_unprotected", 1), ("left", 0), ("left", 17)):
            with self.subTest(movement=movement, color=color):
                context = make_context(color=color)
                context.traffic_light_stops[0]["signal_movement"] = movement
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_verified_flag_needs_correct_exact_id_movement_readback(self):
        for after in ({"id": "OTHER", "color": 32}, {"id": "C-test", "color": 16},
                      {"id": "C-test", "color": None}, None):
            with self.subTest(after=after):
                context = make_context(color=16)
                context.traffic_light_stops[0]["signal_movement"] = "left"
                result = context.grpc.set_traffic_light_color_verified.return_value
                result.update(verified=True, after=after)
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                self.assertEqual(context.traffic_override_verified, 0)
                self.assertEqual(context.traffic_override_completed, set())

    def test_verified_result_for_different_requested_color_is_not_used(self):
        context = make_context(color=16)
        context.traffic_light_stops[0]["signal_movement"] = "left"
        context.grpc.set_traffic_light_color_verified.return_value.update(
            verified=True, after={"id": "C-test", "color": 32}, target_color=16,
        )
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
        self.assertEqual(context.traffic_override_verified, 0)

    def test_exact_left_readback_can_verify_immediately(self):
        context = make_context(color=16)
        context.traffic_light_stops[0]["signal_movement"] = "left"
        context.grpc.set_traffic_light_color_verified.return_value.update(
            verified=True, after={"id": "C-test", "color": 32}, target_color=32,
        )
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
        self.assertEqual(context.traffic_override_verified, 1)
        self.assertIsNone(context.traffic_wait_state)

    def test_dynamic_releases_repeated_red_just_past_stop_projection(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.traffic_override_completed.add(("C-test", 100.0))
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_called_once()
        self.assertEqual(context.dynamic_signal_attempts[("C-test", 100.0)], 1)

    def test_dynamic_recovery_does_not_target_light_after_passing_junction(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 103.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()

    def test_other_modes_keep_legacy_completed_stop_behavior(self):
        context = make_context()
        context.traffic_override_completed.add(("C-test", 100.0))
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 99.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()

    def test_dynamic_signal_recovery_retries_are_bounded(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.dynamic_signal_attempts[("C-test", 100.0)] = 3
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            with self.assertRaisesRegex(RuntimeError, "signal recovery exhausted"):
                traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_dynamic_signal_recovery_uses_configured_attempt_limit(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.cfg["traffic_light_wait_override"]["dynamic_maximum_attempts"] = 40
        context.dynamic_signal_attempts[("C-test", 100.0)] = 3
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_called_once()
        self.assertEqual(context.dynamic_signal_attempts[("C-test", 100.0)], 4)

    def test_signal_rpc_error_does_not_abort_or_command_green(self):
        context = make_context()
        context.grpc.get_traffic_light_state_by_id.side_effect = RuntimeError("RPC unavailable")
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 99.0)
        context.grpc.set_traffic_light_color_verified.assert_not_called()

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

        context.cfg['traffic_light_wait_override']['enabled'] = False
        context.cfg['dynamic_obstacle'] = {'pre_signal_guard': {'enabled': True}}
        context.grpc = Mock()
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]['light_id'], 'C-test')
        traffic_light_override.update(context, 0.0, {}, 0.0)
        self.assertEqual(context.grpc.mock_calls, [])

        context._full_route_points = list(context.route_points)
        context._route_prefix_s = 10.0
        context.route_points = context.route_points[1:]
        context.route_length_m = 10.0
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]["route_s"], 0.0)

    def test_prepare_uses_connected_following_link_and_does_not_guess_by_geometry(self):
        links = {
            "APPROACH": {"points": [(0, 0), (10, 0)], "to_node_idx": "STOP",
                         "related_signal": "straight"},
            "EXIT": {"points": [(10, 0), (20, 0)], "from_node_idx": "STOP",
                     "to_node_idx": "END", "related_signal": "left"},
        }
        context = SimpleNamespace(
            cfg={"traffic_light_wait_override": {"enabled": True}},
            route_definition={"route_links": ["APPROACH", "EXIT"]},
            map_loader=SimpleNamespace(get_link=lambda i: links[i], node_set={
                "STOP": {"on_stop_line": True, "traffic_light_id": "C-test"},
                "END": {"on_stop_line": False},
            }),
            route_points=[(0, 0), (10, 0), (20, 0)], route_length_m=20,
        )
        traffic_light_override.initialize(context)
        traffic_light_override.prepare(context)
        stop = context.traffic_light_stops[0]
        self.assertEqual(stop["signal_movement"], "left")
        self.assertEqual(stop["signal_route_link_id"], "EXIT")
        self.assertEqual(stop["signal_movement_binding"], "connected_ordered_route_link")
        links["EXIT"]["related_signal"] = "straight"
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]["signal_movement"], "straight")
        links["EXIT"]["from_node_idx"] = "OTHER-NODE"
        traffic_light_override.prepare(context)
        stop = context.traffic_light_stops[0]
        self.assertIsNone(stop["signal_movement"])
        self.assertEqual(stop["signal_movement_binding"], "following_link_not_connected")


if __name__ == "__main__":
    unittest.main()
