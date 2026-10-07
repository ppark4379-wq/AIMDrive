"""Vehicle route reuse, spawn windows, lifecycle and official catalog checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ..scenarios.full_loop_parts import vehicle_common
from ..scenarios.full_loop_parts.tunnel_vehicle_dynamic import prepare_tunnel_vehicle
from ..scenarios.full_loop_parts.tunnel_vehicle_catalog import generate_tunnel_vehicle_candidates
from ..tools.generate_tunnel_dynamic_candidates import build_catalog as pedestrian_catalog
from ..tools.generate_tunnel_vehicle_candidates import build_catalog


def state(x, speed=4.0, link="lane"):
    return SimpleNamespace(
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=0.0, z=0.0)),
        velocity=SimpleNamespace(x=speed, y=0.0, z=0.0),
        vehicle_state=SimpleNamespace(current_link_info=SimpleNamespace(id=SimpleNamespace(value=link))),
    )


class TunnelVehicleTest(unittest.TestCase):
    def make_context(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        points = [(float(i), 0.0, 0.0) for i in range(301)]
        sections = {"division_count": 4, "endpoint_margin_m": 10.0}
        actor = Mock()
        actor.destroy.return_value = True
        actor.get_actor_state.return_value = state(75.0, 0.0)
        grpc = Mock()
        grpc.spawn_vehicle.return_value = actor
        cfg = {
            "enabled": True, "trial_candidate_id": "vehicle_slow_01",
            "slow_speed_mps": 4.0, "spawn_trigger_gap_m": 40.0,
            "minimum_spawn_gap_m": 30.0, "spawn_position_tolerance_m": 3.0,
            "arrival_tolerance_m": 1.0, "minimum_observed_travel_m": 5.0,
            "max_run_sec": 20.0,
        }
        context = SimpleNamespace(
            cfg={"tunnel_vehicle_dynamic": cfg, "highway_npc": {"model": "Sedan"},
                 "profile_state_path": str(Path(temporary.name) / "profile.json")},
            scenario_name="full_loop_tunnel_vehicle", clean_road_mode=False,
            route_points=points, tunnel_entry_s=60.0, tunnel_exit_s=200.0,
            route_definition={"zones": {"tunnel": {
                "candidate_catalog": sections, "vehicle_candidate_catalog": {"slow_length_m": 20.0},
                "route_links": ["lane"], "npc_route_links": ["lane", "tail"],
            }}},
            map_loader=SimpleNamespace(link_set={"lane": {}, "tail": {}},
                                       get_link_points=lambda link: points if link == "lane" else [(300.,0.,0.),(301.,0.,0.)]),
            _nearest_waypoint=vehicle_common._nearest_waypoint,
            grpc=grpc, scenario_actors=[], vehicle_models=lambda: ["Sedan"],
            run_seed=1, run_instance_id="vehicle-test",
        )
        context._spawn_highway_vehicle = lambda *a, **kw: vehicle_common._spawn_highway_vehicle(context, *a, **kw)
        return context, actor

    def armed(self):
        context, actor = self.make_context()
        self.assertTrue(prepare_tunnel_vehicle(context))
        handler = context.tunnel_vehicle_handler
        handler.arm(context)
        return context, actor, handler

    def test_official_catalog_shares_anchors_and_keeps_paths_in_sections(self):
        vehicles, pedestrians = build_catalog(), pedestrian_catalog()
        self.assertEqual(len(vehicles["candidates"]), 4)
        for index, path in enumerate(vehicles["candidates"], 1):
            counterpart = next(p for p in pedestrians["candidates"] if p["section_index"] == index)
            self.assertEqual(path["id"], f"vehicle_slow_{index:02d}")
            self.assertEqual(path["section_s_range_m"], counterpart["section_s_range_m"])
            self.assertEqual(path["anchor_s_m"], counterpart["anchor_s_m"])
            self.assertAlmostEqual(path["end_s_m"] - path["start_s_m"], 20.0)
            self.assertEqual(path["lane_offset_m"], 0.0)
            self.assertGreaterEqual(path["start_s_m"], path["section_s_range_m"]["start"])
            self.assertLessEqual(path["end_s_m"], path["section_s_range_m"]["end"])

    def test_reuses_vehicle_route_then_records_motion_and_cleans_up(self):
        context, actor, handler = self.armed()
        handler.update(context, 0.0, {}, 34.0)
        context.grpc.spawn_vehicle.assert_not_called()
        handler.update(context, 1.0, {}, 35.0)
        self.assertEqual(context.tunnel_vehicle_phase, "moving")
        context.grpc.set_vehicle_route.assert_called_once_with(
            actor, ["lane", "tail"], decision_range=30.0,
            route_waypoint_indices={"lane": 75}, label="vehicle_slow_01")
        context.grpc.set_vehicle_velocity.assert_called_with(actor, 4.0)
        self.assertEqual(context.tunnel_vehicle_result["initial_route_gap_m"], 40.0)
        actor.get_actor_state.return_value = state(85.0)
        handler.update(context, 3.5, {}, 45.0)
        actor.get_actor_state.return_value = state(95.0)
        handler.update(context, 6.0, {}, 60.0)
        self.assertEqual(context.tunnel_vehicle_phase, "completed")
        self.assertTrue(context.tunnel_vehicle_result["observed_moving"])
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])
        self.assertTrue(context.tunnel_vehicle_report_path.is_file())

    def test_missed_safe_gap_rejects_spawn_before_rpc(self):
        context, _, handler = self.armed()
        with self.assertRaisesRegex(RuntimeError, "unsafe spawn gap"):
            handler.update(context, 1.0, {}, 46.0)
        context.grpc.spawn_vehicle.assert_not_called()
        self.assertEqual(context.tunnel_vehicle_phase, "failed")

    def test_wrong_actual_spawn_is_destroyed_and_fails(self):
        context, actor, handler = self.armed()
        actor.get_actor_state.return_value = state(80.0)
        with self.assertRaisesRegex(RuntimeError, "spawn verification failed"):
            handler.update(context, 1.0, {}, 35.0)
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])

    def test_rejected_velocity_fails_and_removes_actor(self):
        context, actor, handler = self.armed()
        context.grpc.set_vehicle_velocity.return_value = False
        with self.assertRaisesRegex(RuntimeError, "velocity rejected"):
            handler.update(context, 1.0, {}, 35.0)
        actor.destroy.assert_called_once()

    def test_timeout_removes_stalled_actor(self):
        context, actor, handler = self.armed()
        handler.update(context, 1.0, {}, 35.0)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            handler.update(context, 22.0, {}, 35.0)
        actor.destroy.assert_called_once()

    def test_instant_endpoint_jump_cannot_complete(self):
        context, actor, handler = self.armed()
        handler.update(context, 1.0, {}, 35.0)
        actor.get_actor_state.return_value = state(95.0)
        with self.assertRaisesRegex(RuntimeError, "without verified travel"):
            handler.update(context, 1.1, {}, 35.0)
        self.assertFalse(context.tunnel_vehicle_result["completed"])

    def test_wrong_live_route_fails_and_removes_actor(self):
        context, actor, handler = self.armed()
        handler.update(context, 1.0, {}, 35.0)
        actor.get_actor_state.return_value = state(80.0, link="foreign")
        with self.assertRaisesRegex(RuntimeError, "left its assigned route"):
            handler.update(context, 2.0, {}, 36.0)
        actor.destroy.assert_called_once()

    def test_other_modes_cannot_enable_vehicle_trial(self):
        for mode in ("full_loop", "full_loop_tunnel_dynamic", "full_loop_static"):
            context, _ = self.make_context()
            context.scenario_name = mode
            self.assertFalse(prepare_tunnel_vehicle(context))
            self.assertIsNone(context.tunnel_vehicle_handler)
            context.grpc.spawn_vehicle.assert_not_called()

    def test_invalid_config_fails_before_simulator_calls(self):
        for key, value in (("trial_candidate_id", "vehicle_stopped_01"),
                           ("slow_speed_mps", float("nan")),
                           ("minimum_spawn_gap_m", 45.0)):
            context, _ = self.make_context()
            context.cfg["tunnel_vehicle_dynamic"][key] = value
            with self.assertRaises(ValueError):
                prepare_tunnel_vehicle(context)
            self.assertEqual(context.grpc.mock_calls, [])

    def test_catalog_rejects_paths_larger_than_sections(self):
        context, _ = self.make_context()
        zone = context.route_definition["zones"]["tunnel"]
        with self.assertRaisesRegex(ValueError, "fit inside"):
            generate_tunnel_vehicle_candidates(context.route_points, 60., 200.,
                zone["candidate_catalog"], {"slow_length_m": 31.0})


if __name__ == "__main__":
    unittest.main()
