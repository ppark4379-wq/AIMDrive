import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from ..scenarios.full_loop_parts import static_blockage_release as release
from ..scenarios.full_loop_parts import static_obstacle


def obstacle_state(actor_id):
    location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=actor_id)),
                           transform=SimpleNamespace(location=location))


def make_context(policy=True, lights=()):
    context = SimpleNamespace(cfg={}, grpc=Mock(), event=Mock(), traffic_light_stops=list(lights))
    release.initialize(context)
    context.static_blockage_release_policy = release.validate_policy(policy)
    context.static_obstacle_expectations = [
        {"actor_id": "7", "model": "WoodBox", "route_s": 110.0},
        {"actor_id": "8", "model": "Bike1", "route_s": 112.0},
    ]
    context.grpc.destroy_obstacle_by_id.return_value = True
    context.grpc.get_all_obstacle_states.return_value = []
    return context


def events(context):
    return [call.args[0] for call in context.event.call_args_list]


def drive(context, samples):
    """samples: (monotonic, elapsed, speed, route_s)"""
    with patch.object(release.time, "monotonic", side_effect=[s[0] for s in samples]), \
            patch.object(static_obstacle.time, "sleep"):
        for _, elapsed, speed, route_s in samples:
            release.update(context, elapsed, {"speed": speed}, route_s)


class StaticBlockageReleaseTests(unittest.TestCase):
    def test_policy_defaults_bounds_and_unknown_keys(self):
        self.assertIsNone(release.validate_policy(None))
        self.assertEqual(release.validate_policy(True), release.DEFAULTS)
        self.assertEqual(release.validate_policy({"wait_wall_sec": 8})["wait_wall_sec"], 8.0)
        with self.assertRaises(ValueError):
            release.validate_policy({"wait_wall_sec": 0.1})
        with self.assertRaises(ValueError):
            release.validate_policy({"release_after": 3})

    def test_unarmed_run_never_touches_obstacles(self):
        context = make_context(policy=None)
        drive(context, [(100.0, 0.0, 0.0, 101.0), (110.0, 10.0, 0.0, 101.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        context.event.assert_not_called()

    def test_stop_close_behind_layout_releases_every_obstacle_and_discloses_it(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.4, 0.1, 101.0)])
        self.assertEqual([call.args[0] for call in context.grpc.destroy_obstacle_by_id.call_args_list],
                         ["7", "8"])
        self.assertTrue(context.static_blockage_released)
        self.assertEqual(events(context), ["static_blockage_release_armed", "static_blockage_wait_started",
                                           "actor_lifecycle_started", "actor_lifecycle_finished",
                                           "static_blockage_released"])
        released = context.event.call_args_list[-1].kwargs
        self.assertTrue(released["intervention"] and released["mission_distribution_changed"])
        self.assertFalse(released["training_visible"])
        self.assertEqual(released["actor_ids"], ["7", "8"])
        self.assertAlmostEqual(released["gap_m"], 9.0)
        self.assertTrue(context.event.call_args_list[3].kwargs["success"])
        # Nothing happens after the release.
        drive(context, [(120.0, 20.0, 0.0, 101.0)])
        self.assertEqual(context.grpc.destroy_obstacle_by_id.call_count, 2)

    def test_wall_time_alone_is_not_enough_when_simulation_runs_slow(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 2.0, 0.0, 101.0)])   # 6.5 s wall, 2 s simulated
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        drive(context, [(119.0, 6.1, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        released = context.event.call_args_list[-1].kwargs
        self.assertAlmostEqual(released["wait_scenario_sec"], 6.1)
        self.assertAlmostEqual(released["scenario_elapsed_sec"], 6.1)

    def test_stop_far_before_layout_is_not_a_blockage_wait(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 90.0), (110.0, 10.0, 0.0, 90.0)])   # 20 m: a stop-line wait
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        self.assertIsNone(context.static_blockage_wait)

    def test_red_stop_line_ahead_explains_the_stop(self):
        context = make_context(lights=[{"light_id": "L1", "route_s": 104.0}])
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 1}
        drive(context, [(100.0, 0.0, 0.0, 101.0), (110.0, 10.0, 0.0, 101.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 16}
        drive(context, [(111.0, 11.0, 0.0, 101.0), (117.5, 17.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)

    def test_release_at_a_used_stop_line_lets_the_signal_override_act_again(self):
        context = make_context(lights=[{"light_id": "L1", "route_s": 101.5}])
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 16}
        context.traffic_override_completed = {("L1", 101.5), ("L9", 300.0)}
        context.traffic_wait_state = {"stop_key": ("L1", 101.5)}
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        self.assertEqual(context.traffic_override_completed, {("L9", 300.0)})
        self.assertIsNone(context.traffic_wait_state)
        self.assertIn("static_blockage_signal_override_rearmed", events(context))

    def test_moving_resets_the_wait(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (104.0, 4.0, 1.5, 101.5),
                        (105.0, 5.0, 0.0, 102.0), (110.0, 10.0, 0.0, 102.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        self.assertIn("static_blockage_wait_reset", events(context))
        drive(context, [(111.5, 11.5, 0.0, 102.0)])
        self.assertTrue(context.static_blockage_released)

    def test_destroy_applied_on_a_later_simulation_step_is_awaited(self):
        context = make_context()
        both = [obstacle_state("7"), obstacle_state("8")]
        context.grpc.get_all_obstacle_states.side_effect = [both, both, [obstacle_state("8")]] + [[]] * 5
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        removed = context.event.call_args_list[-1].kwargs["removed"]
        self.assertTrue(all(row["absent_after_readback"] for row in removed))
        self.assertEqual(removed[0]["inventory_polls"], 5)

    def test_obstacle_still_in_inventory_fails_closed(self):
        context = make_context()
        context.grpc.get_all_obstacle_states.return_value = [obstacle_state("8")]
        with self.assertRaises(RuntimeError):
            drive(context, [(100.0, 0.0, 0.0, 101.0), (107.0, 7.0, 0.0, 101.0)])
        self.assertFalse(context.static_blockage_released)
        finished = [c for c in context.event.call_args_list if c.args[0] == "actor_lifecycle_finished"]
        self.assertFalse(finished[0].kwargs["success"])

    def test_released_layout_is_not_verified_as_vanished(self):
        context = SimpleNamespace(cfg={}, grpc=Mock(), static_blockage_released=True,
                                  static_obstacle_expectations=[{"actor_id": "7"}])
        static_obstacle.verify_static_obstacle(context, 10.0, 100.0)
        context.grpc.get_all_obstacle_states.assert_not_called()

    def test_preset_declares_the_release_policy(self):
        document = {"schema": "morai_static_obstacle_preset_v1", "name": "blocked_01",
                    "map_name": "TestMap",
                    "obstacles": [{"model": "WoodBox", "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                                   "scale": [1.0, 1.0, 1.0], "yaw_deg": 5.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(yaml.safe_dump(document))
            plain = static_obstacle.load_static_obstacle_preset(path, "TestMap", [(0.0, 0.0), (1.0, 2.0)])
            document["blockage_release"] = {"wait_wall_sec": 5.0}
            path.write_text(yaml.safe_dump(document))
            blocked = static_obstacle.load_static_obstacle_preset(path, "TestMap", [(0.0, 0.0), (1.0, 2.0)])
        self.assertIsNone(plain["blockage_release"])
        self.assertEqual(blocked["blockage_release"]["wait_wall_sec"], 5.0)
        self.assertEqual(blocked["blockage_release"]["max_gap_m"], release.DEFAULTS["max_gap_m"])


if __name__ == "__main__":
    unittest.main()
