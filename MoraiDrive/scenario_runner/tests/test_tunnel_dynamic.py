"""Tunnel dynamic obstacle setup and lifecycle tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts import tunnel_dynamic
from ..scenarios.full_loop_parts.tunnel_dynamic import (
    arm_tunnel_dynamic,
    prepare_tunnel_dynamic,
    update_tunnel_dynamic,
)


class TunnelDynamicTest(unittest.TestCase):
    def test_enabled_crossing_requires_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        with patch.object(tunnel_dynamic, "TUNNEL_DYNAMIC_HANDLER", None):
            with self.assertRaisesRegex(RuntimeError, "validated placement and lifecycle handler"):
                prepare_tunnel_dynamic(context)
        self.assertIsNone(context.tunnel_dynamic_handler)

    def test_validated_handler_receives_lifecycle_calls(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        handler = Mock()
        self.assertTrue(prepare_tunnel_dynamic(context, handler))
        handler.validate.assert_called_once_with(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed": 5.0}, 35.0)
        handler.arm.assert_called_once_with(context)
        handler.update.assert_called_once_with(context, 1.0, {"speed": 5.0}, 35.0)

    def test_disabled_crossing_does_not_touch_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": False}})
        handler = Mock()
        self.assertFalse(prepare_tunnel_dynamic(context, handler))
        self.assertIsNone(context.tunnel_dynamic_handler)
        handler.validate.assert_not_called()

    def test_clean_road_cannot_enable_crossing(self):
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {"enabled": True}}, clean_road_mode=True
        )
        with self.assertRaisesRegex(ValueError, "clean_road_mode"):
            prepare_tunnel_dynamic(context)

    def test_other_segment_cannot_arm_tunnel_crossing(self):
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {"enabled": True}},
            mission_mode=SimpleNamespace(zones=SimpleNamespace(tunnel=False)),
        )
        self.assertFalse(prepare_tunnel_dynamic(context))
        self.assertIsNone(context.tunnel_dynamic_handler)

    def make_route_context(self):
        waypoints = [
            {"x": 45.0, "y": 2.0, "z": 0.0},
            {"x": 50.0, "y": 2.0, "z": 0.0},
            {"x": 50.0, "y": -2.0, "z": 0.0},
        ]
        actor = Mock()
        actor.get_id.return_value = "ped-1"
        actor.destroy.return_value = True
        grpc = Mock()
        grpc.get_available_pedestrian_models.return_value = ["Man1"]
        grpc.spawn_pedestrian.return_value = actor
        grpc.control_pedestrian.return_value = True
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {
                "enabled": True, "model": "Man1", "path_id": "walk_then_cross",
                "speed_mps": 1.5, "spawn_trigger_before_crossing_m": 30.0,
                "arrival_tolerance_m": 0.5, "max_run_sec": 15.0,
            }},
            route_definition={"zones": {"tunnel": {"dynamic_paths": {
                "walk_then_cross": {"route_path_index": 50, "waypoints": waypoints}
            }}}},
            route_points=[(float(i), 0.0, 0.0) for i in range(101)],
            path_s_at_index=lambda index: float(index),
            tunnel_entry_s=10.0, tunnel_exit_s=90.0,
            grpc=grpc, scenario_actors=[], run_seed=7, run_instance_id="run-7",
        )
        return context, actor

    def test_one_pedestrian_moves_a_to_b_to_c_then_is_removed(self):
        context, actor = self.make_route_context()
        actor.get_actor_state.side_effect = [
            SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=50.0, y=2.0))),
            SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=50.0, y=-2.0))),
        ]
        self.assertTrue(prepare_tunnel_dynamic(context))
        arm_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_phase, "waiting")
        update_tunnel_dynamic(context, 0.0, {}, 19.0)
        context.grpc.spawn_pedestrian.assert_not_called()
        update_tunnel_dynamic(context, 1.0, {}, 20.0)
        self.assertEqual(context.tunnel_dynamic_phase, "moving")
        self.assertEqual(context.tunnel_dynamic_target_index, 1)
        update_tunnel_dynamic(context, 2.0, {}, 21.0)
        self.assertEqual(context.tunnel_dynamic_target_index, 2)
        update_tunnel_dynamic(context, 3.0, {}, 22.0)
        self.assertEqual(context.tunnel_dynamic_phase, "completed")
        self.assertTrue(context.tunnel_dynamic_result["completed"])
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(context.grpc.control_pedestrian.call_count, 3)
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])

    def test_wrong_side_path_fails_before_spawn(self):
        context, _ = self.make_route_context()
        context.route_definition["zones"]["tunnel"]["dynamic_paths"][
            "walk_then_cross"]["waypoints"][2].update(x=52.0, y=2.0)
        with self.assertRaisesRegex(RuntimeError, "must cross the official route"):
            prepare_tunnel_dynamic(context)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_timeout_destroys_owned_pedestrian(self):
        context, actor = self.make_route_context()
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {}, 20.0)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            update_tunnel_dynamic(context, 17.0, {}, 21.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])


if __name__ == "__main__":
    unittest.main()
