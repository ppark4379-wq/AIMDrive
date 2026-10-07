"""Whole-tunnel lifecycle, real map topology and continuous actor checks."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import yaml
from ..scenarios.full_loop_parts import vehicle_common
from ..scenarios.full_loop_parts.tunnel_vehicle_dynamic import (
    prepare_tunnel_vehicle,
    TunnelVehicleTraffic,
)
from ..scenarios.full_loop_parts.tunnel_vehicle_catalog import (
    build_behavior_catalog,
    validate_vehicle_route,
    UnsupportedVehicleTopology,
)
from ..scenarios.segment_modes.modes import MISSION_MODES
from ..tools.generate_tunnel_dynamic_candidates import (
    build_catalog as pedestrian_catalog,
)
from ..tools.generate_tunnel_vehicle_candidates import build_catalog

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / "config/full_loop.yaml").read_text(encoding="utf-8-sig"))[
    "scenarios"
]["full_loop"]


def state(x, speed=4.0, link="lane", y=0.0, yaw=0.0, actor_id="npc-1"):
    return NS(
        actor_info=NS(id=NS(value=actor_id)),
        transform=NS(location=NS(x=x, y=y, z=0.0), rotation=NS(z=yaw)),
        velocity=NS(x=speed, y=0.0, z=0.0),
        vehicle_state=NS(current_link_info=NS(id=NS(value=link))),
    )


class TunnelVehicleTest(unittest.TestCase):
    def make_context(self, behavior="vehicle_slow", seed=1, cut_in=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        points = [(float(i), 0.0, 0.0) for i in range(301)]
        lane = points[:161]
        tail = points[160:]
        incoming = [(float(i), 3.5, 0.0) for i in range(137, -1, -1)]
        outlet = [(float(i), 3.5, 0.0) for i in range(0, -51, -1)]
        links = {
            "lane": {"from_node_idx": "a", "to_node_idx": "b"},
            "tail": {"from_node_idx": "b", "to_node_idx": "c"},
            "opposite": {"from_node_idx": "d", "to_node_idx": "e"},
            "outlet": {"from_node_idx": "e", "to_node_idx": "f"},
        }
        geometry = {"lane": lane, "tail": tail, "opposite": incoming, "outlet": outlet}
        if cut_in:
            links["adjacent"] = {
                "from_node_idx": "g",
                "to_node_idx": "h",
                "can_move_left_lane": True,
                "left_lane_change_dst_link_idx": "lane",
            }
            geometry["adjacent"] = [(float(i), -3.0, 0.0) for i in range(161)]
        actor = Mock()
        actor.get_id.return_value = "npc-1"
        actor.destroy.return_value = True
        grpc = Mock()
        grpc.spawn_vehicle.return_value = actor
        grpc.destroy_vehicle_by_id.side_effect = lambda actor_id: actor.destroy()
        grpc.get_ego_motion_state.return_value = {"x": 0.0, "y": 0.0, "speed": 0.0}
        cfg = copy.deepcopy(BASE["tunnel_vehicle_dynamic"])
        cfg.update(
            enabled=True, behavior=behavior, random_model=True, models=["Sedan", "Van"]
        )
        context = NS(
            cfg={
                "tunnel_vehicle_dynamic": cfg,
                "tunnel_sudden_brake": copy.deepcopy(BASE["tunnel_sudden_brake"]),
                "highway_npc": {"model": "Sedan"},
                "profile_state_path": str(Path(tmp.name) / "profile.json"),
            },
            scenario_name="full_loop_tunnel_vehicle",
            clean_road_mode=False,
            route_points=points,
            tunnel_entry_s=0.0,
            tunnel_exit_s=137.0,
            route_definition={
                "zones": {
                    "tunnel": {
                        "candidate_catalog": {
                            "division_count": 4,
                            "endpoint_margin_m": 10.0,
                        },
                        "route_links": ["lane"],
                        "npc_route_links": ["lane", "tail"],
                        "oncoming_route_links": ["opposite", "outlet"],
                        "cut_in_route_links": (
                            ["adjacent", "lane", "tail"] if cut_in else []
                        ),
                    }
                }
            },
            map_loader=NS(link_set=links, get_link_points=lambda link: geometry[link]),
            grpc=grpc,
            scenario_actors=[],
            vehicle_models=lambda: ["Sedan", "Van"],
            run_seed=seed,
            random=random.Random(seed),
            run_instance_id=f"vehicle-test-{seed}",
            now=0.0,
        )
        context._scenario_now = lambda: context.now
        context._spawn_highway_vehicle = (
            lambda *a, **kw: vehicle_common._spawn_highway_vehicle(context, *a, **kw)
        )

        def spawn(**kw):
            p = context.tunnel_vehicle_plan["spawn_target"]
            actor.get_actor_state.return_value = state(
                p["x"], link=p["link"], y=p["y"], yaw=p["yaw_deg"], speed=kw["velocity"]
            )
            return actor

        grpc.spawn_vehicle.side_effect = spawn
        return context, actor

    def armed(self, behavior="vehicle_slow", seed=1, cut_in=False):
        context, actor = self.make_context(behavior, seed, cut_in)
        prepare_tunnel_vehicle(context)
        handler = context.tunnel_vehicle_handler
        handler.arm(context)
        self.start(context, handler)
        return context, actor, handler

    def start(self, context, handler, ego_x=None, ego_speed=None):
        if ego_x is None:
            ego_x = (
                40.0
                if context.tunnel_vehicle_plan["behavior"] == "vehicle_oncoming"
                else 0.0
            )
        if ego_speed is None:
            ego_speed = (
                21.6
                if context.tunnel_vehicle_plan["behavior"] == "vehicle_oncoming"
                else 36.0
            )
        handler.update(
            context, context.now, {"x": ego_x, "y": 0.0, "speed": ego_speed}, ego_x
        )

    def tick(
        self,
        c,
        a,
        h,
        time,
        x,
        ego_x,
        speed=4.0,
        link="lane",
        y=0.0,
        yaw=0.0,
        ego_speed=36.0,
    ):
        c.now = time
        if x > 160 and link == "lane":
            link = "tail"
        a.get_actor_state.return_value = state(x, speed, link, y, yaw)
        h.update(c, time, {"x": ego_x, "y": 0.0, "speed": ego_speed}, ego_x)

    def test_long_stationary_ego_has_no_npc_or_startup_timeout(self):
        for behavior in (
            "vehicle_slow",
            "vehicle_sudden_brake",
            "vehicle_stopped",
            "vehicle_oncoming",
        ):
            with self.subTest(behavior=behavior):
                c, a = self.make_context(behavior)
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                for t in (0, 60, 300, 600):
                    c.now = t
                    h.update(c, t, {"x": 0.0, "y": 0.0, "speed": 0.0}, 0.0)
                c.grpc.spawn_vehicle.assert_not_called()
                a.set_pause.assert_not_called()
                self.assertEqual(c.tunnel_vehicle_phase, "waiting_for_ego")
                if behavior != "vehicle_oncoming":
                    self.start(c, h)
                    self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])
                    self.assertEqual(c.tunnel_vehicle_spawned_at, 600)

    def test_progress_gate_requires_actual_motion_and_runtime_pose_follows_ego(self):
        c, a = self.make_context()
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h, ego_x=0.0, ego_speed=5.4)
        self.start(c, h, ego_x=3.0, ego_speed=0.0)
        c.grpc.spawn_vehicle.assert_not_called()
        self.start(c, h, ego_x=3.0, ego_speed=5.4)
        self.assertEqual(c.tunnel_vehicle_plan["spawn_target"]["x"], 43.0)
        self.assertEqual(c.tunnel_vehicle_result["initial_route_gap_m"], 40.0)
        self.assertLess(c.tunnel_vehicle_plan["speed_mps"], 1.5)
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])

    def test_slow_ego_and_npc_actually_encounter_before_exit(self):
        c, a = self.make_context()
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h, ego_speed=7.2)
        npc_speed = c.tunnel_vehicle_plan["speed_mps"]
        self.assertLess(npc_speed, 2.0)
        self.assertGreaterEqual(
            npc_speed, c.cfg["tunnel_vehicle_dynamic"]["minimum_npc_speed_mps"]
        )
        for t in range(2, 44, 2):
            self.tick(
                c, a, h, t, 40 + npc_speed * t, 2 * t, speed=npc_speed, ego_speed=7.2
            )
            if c.tunnel_vehicle_result["encounter_verified"]:
                break
        self.assertTrue(c.tunnel_vehicle_result["encounter_verified"])
        sample = c.tunnel_vehicle_result["encounter_observation"]
        self.assertLess(sample["npc_s_m"], c.tunnel_exit_s)
        self.assertLessEqual(sample["route_gap_m"], 20.0)
        a.destroy.assert_not_called()
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_npc_does_not_run_away_when_ego_slows_before_encounter(self):
        c, a, h = self.armed()
        self.tick(c, a, h, 1, 44, 2, ego_speed=3.6)
        self.assertLessEqual(c.grpc.set_vehicle_velocity.call_args.args[1], 1.0)
        self.tick(c, a, h, 2, 44.6, 2, speed=0.6, ego_speed=0.0)
        self.assertEqual(c.grpc.set_vehicle_velocity.call_args.args[1], 0.0)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 3, 44.6, 2, speed=0.0, ego_speed=18.0)
        self.assertGreater(c.grpc.set_vehicle_velocity.call_args.args[1], 0.0)

    def test_oncoming_waits_for_eta_then_physically_passes_with_same_actor(self):
        c, a = self.make_context("vehicle_oncoming")
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        for t, x in ((0, 0), (5, 10), (10, 20)):
            c.now = t
            self.start(c, h, ego_x=x, ego_speed=7.2)
            c.grpc.spawn_vehicle.assert_not_called()
        c.now = 15
        self.start(c, h, ego_x=30.0, ego_speed=7.2)
        timing = c.tunnel_vehicle_result["spawn_timing"]
        self.assertLessEqual(abs(timing["eta_delta_sec"]), 1.5)
        self.assertGreater(timing["npc_eta_sec"], 0)
        self.tick(c, a, h, 27, 55, 54, speed=6, link="opposite", y=3.5, yaw=180)
        self.assertLess(
            abs(c.tunnel_vehicle_result["last_observation"]["route_gap_m"]), 2
        )
        self.tick(c, a, h, 29, 43, 58, speed=6, link="opposite", y=3.5, yaw=180)
        self.assertTrue(c.tunnel_vehicle_result["passed"])
        a.destroy.assert_not_called()
        self.tick(c, a, h, 43, -41, 86, speed=6, link="outlet", y=3.5, yaw=180)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_called_once()

    def test_multiple_independent_traffic_actors_can_coexist(self):
        c, _ = self.make_context()
        c.cfg["tunnel_vehicle_dynamic"]["additional_behaviors"] = ["vehicle_oncoming"]
        prepare_tunnel_vehicle(c)
        traffic = c.tunnel_vehicle_handler
        actors = {}

        def spawn(**kwargs):
            owner = next(
                owner
                for _, owner in traffic.entries
                if owner.tunnel_vehicle_plan["behavior"] == kwargs["label"]
            )
            pose = owner.tunnel_vehicle_plan["spawn_target"]
            actor = Mock()
            actor_id = f"npc-{len(actors) + 1}"
            actor.get_id.return_value = actor_id
            actor.get_actor_state.return_value = state(
                pose["x"],
                kwargs["velocity"],
                pose["link"],
                pose["y"],
                pose["yaw_deg"],
                actor_id,
            )
            actors[actor_id] = actor
            return actor

        c.grpc.spawn_vehicle.side_effect = spawn
        c.grpc.destroy_vehicle_by_id.side_effect = (
            lambda actor_id: actors[actor_id].destroy() or True
        )
        traffic.arm(c)
        self.start(c, traffic, ego_speed=7.2)
        self.assertEqual(len(c.scenario_actors), 1)
        for t in (5, 10, 15):
            c.now = t
            actors["npc-1"].get_actor_state.return_value = state(
                40 + c.tunnel_vehicle_plan["speed_mps"] * t,
                c.tunnel_vehicle_plan["speed_mps"],
                actor_id="npc-1",
            )
            self.start(c, traffic, ego_x=2 * t, ego_speed=7.2)
        self.assertEqual(len(c.scenario_actors), 2)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 2)
        for actor in actors.values():
            actor.destroy.assert_not_called()
        traffic.record_interruption(c, KeyboardInterrupt())
        self.assertEqual(c.scenario_actors, [])
        self.assertEqual(c.grpc.destroy_vehicle_by_id.call_count, 2)

    def test_ego_goal_observes_concurrent_traffic_until_each_cleanup(self):
        events, entries = [], []
        for behavior in ("vehicle_slow", "vehicle_oncoming"):
            owner = NS(
                tunnel_vehicle_phase="moving",
                tunnel_vehicle_plan={"behavior": behavior},
                tunnel_vehicle_result={"spawned": True},
                tunnel_vehicle_runtime_timeout=3,
                tunnel_vehicle_started_at=0,
                tunnel_vehicle_last_ego={"x": 137, "y": 0, "speed": 0},
                tunnel_vehicle_last_ego_s=137,
            )
            handler = Mock()
            handler._now.return_value = 0

            def update(context, *args, behavior=behavior):
                if context.tunnel_vehicle_phase == "completed":
                    return
                events.append(behavior)
                if behavior == "vehicle_oncoming" or events.count("vehicle_slow") == 2:
                    context.tunnel_vehicle_phase = "completed"

            handler.update.side_effect = update
            entries.append((handler, owner))
        with patch(
            "scenario_runner.scenarios.full_loop_parts.tunnel_vehicle_dynamic.time.sleep"
        ):
            TunnelVehicleTraffic(entries).finish_at_ego_goal(NS(cfg={}))
        self.assertEqual(events, ["vehicle_slow", "vehicle_oncoming", "vehicle_slow"])
        for handler, owner in entries:
            self.assertEqual(owner.tunnel_vehicle_phase, "completed")
            handler.record_interruption.assert_not_called()

    def test_common_cp14_start_and_cp15_end(self):
        ped, veh = (
            MISSION_MODES["full_loop_tunnel_dynamic"],
            MISSION_MODES["full_loop_tunnel_vehicle"],
        )
        self.assertEqual(ped.start, veh.start)
        self.assertEqual(ped.end, veh.end)
        self.assertEqual(veh.start.index, 3768)
        self.assertEqual(veh.end.index, 4045)

    def test_real_map_plans_share_pedestrian_anchors_not_short_paths(self):
        vehicles, pedestrians = build_catalog(), pedestrian_catalog()
        self.assertEqual(
            set(vehicles["behaviors"]),
            {
                "vehicle_slow",
                "vehicle_sudden_brake",
                "vehicle_stopped",
                "vehicle_oncoming",
            },
        )
        self.assertIn("vehicle_cut_in", vehicles["unavailable_behaviors"])
        for s in vehicles["sections"]:
            p = next(
                p
                for p in pedestrians["candidates"]
                if p["section_index"] == s["section_index"]
            )
            self.assertAlmostEqual(s["anchor_s_m"], p["anchor_s_m"])
        oncoming = vehicles["behaviors"]["vehicle_oncoming"]
        self.assertEqual(
            oncoming["route_links"], ["A2256W000131", "A2256W000129", "A2256W000028"]
        )
        self.assertTrue(
            all(
                e["tangent_dot"] < -0.5 and abs(e["lateral_m"]) >= 2
                for e in oncoming["direction_evidence"]
            )
        )
        self.assertTrue(
            all(
                p["lifecycle"] == "each_actor_through_continuation_or_teardown"
                for p in vehicles["behaviors"].values()
            )
        )

    def test_slow_runtime_spawn_keeps_same_actor_across_sections(self):
        c, a, h = self.armed()
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        self.assertEqual(c.tunnel_vehicle_result["initial_route_gap_m"], 40.0)
        for t, x in ((5, 60), (10, 80), (15, 100), (20, 120), (25, 140)):
            self.tick(c, a, h, t, x, x - 15)
            a.destroy.assert_not_called()
        self.assertEqual(c.tunnel_vehicle_phase, "moving")
        self.tick(c, a, h, 40, 190, 137)
        h.finish(c)
        a.destroy.assert_called_once()
        self.assertEqual(c.scenario_actors, [])
        self.assertTrue(c.tunnel_vehicle_result["same_actor_continuity"])
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        report = json.loads(c.tunnel_vehicle_report_path.read_text())
        self.assertEqual(report["actor_id"], "npc-1")
        self.assertTrue(report["cleanup_verified"])

    def test_set_ai_rejection_does_not_remove_observed_moving_vehicle(self):
        c, a = self.make_context()
        c.grpc.set_vehicle_ai.return_value = False
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h)
        self.assertEqual(c.tunnel_vehicle_phase, "moving")
        self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["velocity"], 4.0)
        c.grpc.set_vehicle_ai.assert_not_called()
        a.set_pause.assert_called_once_with(False)
        self.tick(c, a, h, 2, 48, 8)
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])
        self.assertIsNone(
            c.tunnel_vehicle_result["motion_commands"]["ai_enable_accepted"]
        )
        a.destroy.assert_not_called()

    def test_tunnel_commands_route_unpause_then_speed_without_set_ai(self):
        c, a = self.make_context()
        commands = []
        c.grpc.set_vehicle_route.side_effect = (
            lambda *args, **kw: commands.append("route") or True
        )
        c.grpc.set_vehicle_speed_limit.side_effect = (
            lambda *args, **kw: commands.append("speed_limit") or True
        )
        c.grpc.set_vehicle_velocity.side_effect = (
            lambda *args, **kw: commands.append("velocity") or True
        )
        a.set_pause.side_effect = (
            lambda value: commands.append("unpause" if value is False else "pause")
            or True
        )
        prepare_tunnel_vehicle(c)
        c.tunnel_vehicle_handler.arm(c)
        self.start(c, c.tunnel_vehicle_handler)
        self.assertEqual(commands, ["route", "unpause", "speed_limit", "velocity"])
        c.grpc.set_vehicle_ai.assert_not_called()

    def test_runtime_stalled_startup_fails_despite_successful_command_acks(self):
        c, a = self.make_context()
        c.grpc.spawn_vehicle.side_effect = None
        a.get_actor_state.return_value = state(40, speed=0.0)
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        c.grpc.spawn_vehicle.assert_not_called()
        self.start(c, h)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        with self.assertRaisesRegex(RuntimeError, "startup motion not observed"):
            self.tick(c, a, h, 2.1, 40, 10, speed=0.0)
        self.assertFalse(c.tunnel_vehicle_result["startup_motion_verified"])
        a.destroy.assert_called_once()

    def test_startup_displacement_can_verify_motion_without_speed_telemetry(self):
        c, a = self.make_context()
        c.grpc.spawn_vehicle.side_effect = None
        a.get_actor_state.return_value = state(40, speed=0.0)
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h)
        self.tick(c, a, h, 0.2, 40.8, 2, speed=0.0)
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])
        self.assertGreaterEqual(c.tunnel_vehicle_result["startup_displacement_m"], 0.5)

    def test_stopped_spawns_moving_and_unpaused_without_set_ai(self):
        c, a, h = self.armed("vehicle_stopped")
        self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["velocity"], 9.0)
        c.grpc.set_vehicle_ai.assert_not_called()
        a.set_pause.assert_called_once_with(False)
        c.grpc.stop_vehicle.assert_not_called()
        self.assertEqual(c.tunnel_vehicle_phase, "cruising_to_stop")
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])

    def test_sdk_destroy_none_is_not_treated_as_failed_removal(self):
        c, a, h = self.armed()
        a.destroy.return_value = None  # Real SDK Actor.destroy() returns None.
        c.grpc.destroy_vehicle_by_id.side_effect = None
        c.grpc.destroy_vehicle_by_id.return_value = True
        self.tick(c, a, h, 5, 60, 45)
        self.tick(c, a, h, 40, 190, 137)
        h.finish(c)
        c.grpc.destroy_vehicle_by_id.assert_called_once_with("npc-1")
        a.destroy.assert_not_called()
        self.assertTrue(c.tunnel_vehicle_result["cleanup_verified"])

    def test_legacy_helper_spawn_defaults_are_unchanged(self):
        c, a = self.make_context()
        prepare_tunnel_vehicle(c)
        c.highway_fixed_model = "Sedan"
        c._spawn_highway_vehicle(
            c.grpc.make_transform(40, 0, 0, 0), ["lane", "tail"], 40, "legacy", 4.0
        )
        self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["velocity"], 0.0)
        a.set_pause.assert_called_once_with(False)
        c.grpc.set_vehicle_ai.assert_called_once_with(a, True)

    def test_same_actor_brake_stop_hold_resume_exit(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.assertEqual(c.tunnel_vehicle_plan["event_section"], 3)
        self.tick(c, a, h, 4, 70, 45, speed=9.0)
        self.assertEqual(c.tunnel_vehicle_phase, "braking")
        self.tick(c, a, h, 5, 71, 47, speed=0.0, ego_speed=20.0)
        self.assertEqual(c.tunnel_vehicle_phase, "stopped")
        a.destroy.assert_not_called()
        self.tick(c, a, h, 6, 71, 48, speed=0.0, ego_speed=10.0)
        self.assertEqual(c.tunnel_vehicle_phase, "stopped")
        self.tick(c, a, h, 7.1, 71, 48, speed=0.0, ego_speed=0.0)
        self.assertEqual(c.tunnel_vehicle_phase, "departing")
        self.tick(c, a, h, 9, 85, 50, speed=10.0)
        self.tick(c, a, h, 15, 140, 100, speed=10.0)
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        self.assertTrue(c.tunnel_vehicle_result["actor_active"])
        a.destroy.assert_not_called()
        self.tick(c, a, h, 20, 190, 137, speed=10.0)
        h.finish(c)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_called_once()
        self.assertTrue(c.tunnel_vehicle_result["brake"]["resume_motion_verified"])

    def test_brake_forced_stop_fallback_and_emergency_resumes_same_actor(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 4, 70, 45, speed=9.0)
        self.tick(c, a, h, 5.2, 71, 46, speed=2.0)
        c.grpc.stop_vehicle.assert_called_once_with(a)
        self.tick(c, a, h, 6, 72, 67, speed=2.0)
        self.assertTrue(c.tunnel_vehicle_result["brake"]["emergency_resume"])
        self.assertEqual(c.tunnel_vehicle_phase, "departing")
        a.destroy.assert_not_called()
        c.now = 15
        a.get_actor_state.return_value = state(140, 10.0)
        with self.assertRaisesRegex(RuntimeError, "complete brake"):
            h.finish(c)

    def test_unsafe_brake_gap_never_commands_brake(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 4, 70, 60, speed=9.0)
        c.grpc.set_vehicle_speed.assert_not_called()
        self.assertEqual(c.tunnel_vehicle_phase, "closing_gap")
        with self.assertRaisesRegex(RuntimeError, "missed safe brake"):
            self.tick(c, a, h, 10, 100, 95, speed=7.0)

    def park(self, c, a, h):
        x = c.tunnel_vehicle_plan["event_region"]["anchor_s_m"]
        self.tick(c, a, h, 5, x, x - 30, speed=9.0)
        self.assertEqual(c.tunnel_vehicle_phase, "stopping")
        self.tick(c, a, h, 5.5, x, x - 25, speed=0.0)
        self.assertTrue(c.tunnel_vehicle_result["stopped_verified"])
        return x

    def test_stopped_drives_to_event_and_remains_after_pass_until_teardown(self):
        c, a, h = self.armed("vehicle_stopped")
        self.assertEqual(c.tunnel_vehicle_plan["spawn_target"]["x"], 40)
        x = self.park(c, a, h)
        self.tick(c, a, h, 8, x, x + 16, speed=0.0)
        self.assertEqual(c.tunnel_vehicle_phase, "parked_after_pass")
        self.assertTrue(c.tunnel_vehicle_result["ego_passed"])
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        self.assertTrue(c.tunnel_vehicle_result["actor_active"])
        a.destroy.assert_not_called()
        h.finish(c)
        a.destroy.assert_not_called()
        h.teardown(c)
        a.destroy.assert_called_once()
        self.assertFalse(c.tunnel_vehicle_result["actor_active"])
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_stopped_vehicle_accidental_motion_fails(self):
        c, a, h = self.armed("vehicle_stopped")
        x = self.park(c, a, h)
        with self.assertRaisesRegex(RuntimeError, "Stopped vehicle moved"):
            self.tick(c, a, h, 6, x + 2, 20, speed=1.0)
        a.destroy.assert_called_once()

    def test_actual_oncoming_pass_retains_actor_until_outside_entrance(self):
        c, a, h = self.armed("vehicle_oncoming")
        self.tick(c, a, h, 5, 97, 30, speed=6.0, link="opposite", y=3.5, yaw=180.0)
        self.tick(c, a, h, 10, 67, 83, speed=6.0, link="opposite", y=3.5, yaw=180.0)
        self.assertTrue(c.tunnel_vehicle_result["passed"])
        self.assertTrue(c.tunnel_vehicle_result["actor_active"])
        a.destroy.assert_not_called()
        self.tick(c, a, h, 20, 7, 120, speed=6.0, link="opposite", y=3.5, yaw=180.0)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 28, -41, 137, speed=6.0, link="outlet", y=3.5, yaw=180.0)
        a.destroy.assert_called_once()
        self.assertTrue(c.tunnel_vehicle_result["completed"])
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_resume_acks_without_actual_motion_fail(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 4, 70, 45, speed=9.0)
        self.tick(c, a, h, 5, 71, 47, speed=0.0, ego_speed=20.0)
        self.tick(c, a, h, 7.1, 71, 48, speed=0.0, ego_speed=0.0)
        self.assertEqual(c.tunnel_vehicle_phase, "departing")
        with self.assertRaisesRegex(RuntimeError, "resumed movement not observed"):
            self.tick(c, a, h, 9.2, 71, 48, speed=0.0, ego_speed=0.0)
        a.destroy.assert_called_once()

    def test_oncoming_same_direction_or_same_lane_route_rejected(self):
        for key in ("same_direction", "same_lane"):
            c, _ = self.make_context("vehicle_oncoming")
            if key == "same_direction":
                c.route_definition["zones"]["tunnel"]["oncoming_route_links"] = [
                    "lane",
                    "tail",
                ]
            else:
                old = c.map_loader.get_link_points
                c.map_loader.get_link_points = lambda link: (
                    [(p[0], 0.0, p[2]) for p in old(link)]
                    if link in ("opposite", "outlet")
                    else old(link)
                )
            with self.assertRaisesRegex(ValueError, "direction/lane"):
                prepare_tunnel_vehicle(c)
            c.grpc.spawn_vehicle.assert_not_called()

    def test_cut_in_missing_topology_fails_before_morai(self):
        c, _ = self.make_context("vehicle_cut_in")
        with self.assertRaisesRegex(
            UnsupportedVehicleTopology, "no permitted adjacent"
        ):
            prepare_tunnel_vehicle(c)
        self.assertEqual(c.grpc.mock_calls, [])

    def test_legal_cut_in_changes_route_without_teleport_or_respawn(self):
        c, a, h = self.armed("vehicle_cut_in", cut_in=True)
        self.assertEqual(c.grpc.set_vehicle_route.call_args.args[1], ["adjacent"])
        self.tick(c, a, h, 5, 70, 45, speed=7.0, link="adjacent", y=-3.0)
        self.assertEqual(c.tunnel_vehicle_phase, "changing_lane")
        self.tick(c, a, h, 5.5, 72, 47, speed=7.0, link="adjacent", y=-1.8)
        self.tick(c, a, h, 6, 75, 50, speed=7.0, link="lane", y=0.0)
        self.assertEqual(c.tunnel_vehicle_phase, "target_lane_cruise")
        self.assertTrue(c.tunnel_vehicle_result["lane_change_verified"])
        self.tick(c, a, h, 16, 140, 110, speed=7.0)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 24, 190, 137, speed=7.0)
        h.finish(c)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        self.assertEqual(c.grpc.set_vehicle_route.call_count, 2)
        a.set_transform.assert_not_called()

    def test_cut_in_link_ack_without_lateral_motion_cannot_complete(self):
        c, a, h = self.armed("vehicle_cut_in", cut_in=True)
        self.tick(c, a, h, 5, 70, 45, link="adjacent", y=-3.0)
        self.tick(c, a, h, 6, 72, 47, link="lane", y=-3.0)
        self.assertFalse(c.tunnel_vehicle_result.get("lane_change_verified", False))
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.tick(c, a, h, 16, 80, 55, link="adjacent", y=-3.0)

    def test_seed_replays_behavior_section_and_model(self):
        def run(seed):
            c, a = self.make_context(seed=seed)
            c.cfg["tunnel_vehicle_dynamic"]["random_behavior"] = True
            prepare_tunnel_vehicle(c)
            c.tunnel_vehicle_handler.arm(c)
            p = c.tunnel_vehicle_plan
            return p["behavior"], p["event_section"], p["model"]

        choices = [run(seed) for seed in range(12)]
        self.assertGreater(len(set(choices)), 3)
        self.assertEqual(choices, [run(seed) for seed in range(12)])
        self.assertNotIn("vehicle_cut_in", [c[0] for c in choices])

    def test_model_seed_is_independent_of_weather_history_rng_draws(self):
        contexts = []
        for count in (0, 100):
            c, _ = self.make_context(seed=123)
            prepare_tunnel_vehicle(c)
            for _ in range(count):
                c.random.random()
            c.tunnel_vehicle_handler.arm(c)
            contexts.append(c)
        self.assertEqual(
            contexts[0].tunnel_vehicle_plan["model"],
            contexts[1].tunnel_vehicle_plan["model"],
        )

    def test_invalid_config_connectivity_and_unavailable_model(self):
        for key, value in (
            ("slow_speed_mps", float("nan")),
            ("initial_spawn_gap_m", 10.0),
            ("event_sections", [1]),
            ("max_run_sec", 1.0),
        ):
            c, _ = self.make_context()
            c.cfg["tunnel_vehicle_dynamic"][key] = value
            with self.assertRaises(ValueError):
                prepare_tunnel_vehicle(c)
            self.assertEqual(c.grpc.mock_calls, [])
        c, _ = self.make_context()
        c.map_loader.link_set["tail"]["from_node_idx"] = "broken"
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            prepare_tunnel_vehicle(c)
        c, a = self.make_context()
        c.cfg["tunnel_vehicle_dynamic"]["model"] = "Missing"
        prepare_tunnel_vehicle(c)
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            c.tunnel_vehicle_handler.arm(c)
        c.grpc.spawn_vehicle.assert_not_called()
        self.assertEqual(c.tunnel_vehicle_phase, "failed")

    def test_identity_teleport_wrong_link_nonfinite_and_direction_fail(self):
        for kind in ("identity", "teleport", "link", "nonfinite", "direction"):
            with self.subTest(kind=kind):
                c, a, h = self.armed()
                c.now = 0.1
                samples = {
                    "identity": state(40, actor_id="other"),
                    "teleport": state(100),
                    "link": state(40, link="foreign"),
                    "nonfinite": state(float("nan")),
                    "direction": state(40, yaw=180),
                }
                a.get_actor_state.return_value = samples[kind]
                with self.assertRaises(RuntimeError):
                    h.update(c, 0.1, {"x": 0.0, "y": 0.0, "speed": 0.0}, 0.0)
                self.assertFalse(c.tunnel_vehicle_result["completed"])
                a.destroy.assert_called_once()

    def test_off_route_ego_rejects_runtime_spawn(self):
        c, a = self.make_context()
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        with self.assertRaisesRegex(RuntimeError, "Ego position/progress"):
            h.update(c, 0, {"x": 20.0, "y": 10.0, "speed": 36.0}, 20.0)
        c.grpc.spawn_vehicle.assert_not_called()

    def test_spawn_xyz_and_current_link_are_strict(self):
        for bad in (state(45), state(40, link="tail")):
            c, a = self.make_context()
            c.grpc.spawn_vehicle.side_effect = None
            a.get_actor_state.return_value = bad
            prepare_tunnel_vehicle(c)
            c.tunnel_vehicle_handler.arm(c)
            with self.assertRaises(RuntimeError):
                self.start(c, c.tunnel_vehicle_handler)
            a.destroy.assert_called_once()

    def test_route_and_velocity_rejection_clean_up_tracked_actor(self):
        for method in ("set_vehicle_route", "set_vehicle_velocity"):
            c, a = self.make_context()
            getattr(c.grpc, method).return_value = False
            prepare_tunnel_vehicle(c)
            c.tunnel_vehicle_handler.arm(c)
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                self.start(c, c.tunnel_vehicle_handler)
            a.destroy.assert_called_once()
            self.assertEqual(c.scenario_actors, [])

    def test_timeout_interrupt_and_cleanup_rejection_are_reported(self):
        c, a, h = self.armed()
        c.now = 181
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            h.update(c, 181, {}, 0.0)
        a.destroy.assert_called_once()
        c, a, h = self.armed()
        a.destroy.return_value = False
        h.record_interruption(c, KeyboardInterrupt())
        report = json.loads(c.tunnel_vehicle_report_path.read_text())
        self.assertEqual(report["failure_reason"], "KeyboardInterrupt")
        self.assertFalse(report["cleanup_verified"])
        self.assertIn(a, c.scenario_actors)

    def test_ego_goal_before_verified_travel_does_not_pass(self):
        c, a, h = self.armed()
        with self.assertRaisesRegex(RuntimeError, "exit/travel"):
            h.finish(c)
        self.assertFalse(c.tunnel_vehicle_result["completed"])

    def test_ego_goal_waits_for_same_slow_actor_to_exit(self):
        c, a, h = self.armed()
        self.tick(c, a, h, 5, 60, 45)
        self.tick(c, a, h, 6, 64, 135)

        def advance(_):
            c.now += 5
            a.get_actor_state.return_value = state(
                min(190, 60 + (c.now - 5) * 4),
                link="tail" if 60 + (c.now - 5) * 4 > 160 else "lane",
            )

        with patch(
            "scenario_runner.scenarios.full_loop_parts.tunnel_vehicle_dynamic.time.sleep",
            side_effect=advance,
        ):
            h.finish_at_ego_goal(c)
        self.assertTrue(c.tunnel_vehicle_result["completed"])
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_full_route_projection_retains_prefix(self):
        c, a = self.make_context()
        c.mission_route = NS(
            full_points=[(float(i), 0.0, 0.0) for i in range(-50, 301)], prefix_s=50.0
        )
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h)
        self.assertEqual(c.tunnel_vehicle_result["initial_route_gap_m"], 40.0)
        self.tick(c, a, h, 2, 48, 8)
        self.assertEqual(c.tunnel_vehicle_result["last_observation"]["npc_s_m"], 48.0)

    def test_cli_vehicle_options_once_gate_and_manual_controller(self):
        from .. import main_runner

        for argv in (
            ["runner", "--scenario", "full_loop_tunnel_vehicle"],
            [
                "runner",
                "--scenario",
                "full_loop_static",
                "--once",
                "--tunnel-vehicle-random",
            ],
        ):
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main_runner.main()
        global_cfg, zone_cfg, cfg = main_runner.load_configuration()
        for option in (
            ["--tunnel-vehicle-behavior", "vehicle_slow"],
            ["--tunnel-vehicle-random"],
        ):
            argv = [
                "runner",
                "--scenario",
                "full_loop_tunnel_vehicle",
                "--once",
                "--manual-ego",
                "--seed",
                "123",
                "--check",
            ] + option
            with patch("sys.argv", argv), patch.object(
                main_runner,
                "load_configuration",
                return_value=(global_cfg, zone_cfg, copy.deepcopy(cfg)),
            ), patch.object(main_runner, "run_offline_check") as check, patch.object(
                main_runner, "MoraiGrpcClient"
            ) as client:
                main_runner.main()
            client.assert_not_called()
            selected = check.call_args.args[1]
            self.assertEqual(selected["drive_control_mode"], "morai_keyboard")
            self.assertEqual(selected["random_seed"], 123)
            self.assertFalse(selected["sync_runtime"]["enabled"])

    def test_other_modes_do_not_enable_vehicle_handler(self):
        for mode in (
            "full_loop",
            "full_loop_tunnel_dynamic",
            "full_loop_static",
            "full_loop_highway",
        ):
            c, _ = self.make_context()
            c.scenario_name = mode
            self.assertFalse(prepare_tunnel_vehicle(c))
            self.assertEqual(c.grpc.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
