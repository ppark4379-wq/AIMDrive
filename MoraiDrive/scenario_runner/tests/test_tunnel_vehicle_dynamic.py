"""Whole-tunnel lifecycle, real map topology and continuous actor checks."""

import copy
import json
import math
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
    load_catalog_inputs,
)
from ..tools.generate_tunnel_vehicle_candidates import build_catalog
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import interpolate_on_polyline, project_distance_on_polyline

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / "config/full_loop.yaml").read_text(encoding="utf-8-sig"))[
    "scenarios"
]["full_loop"]


def state(x, speed=4.0, link="lane", y=0.0, yaw=0.0, actor_id="npc-1"):
    return NS(
        actor_info=NS(id=NS(value=actor_id)),
        transform=NS(location=NS(x=x, y=y, z=0.0), rotation=NS(z=yaw)),
        velocity=NS(x=speed * 3.6, y=0.0, z=0.0),
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
        # Existing lifecycle fixtures specify exact coordinates; randomized cases below
        # enable the production range and verify geometry/ETA using the resulting plan.
        cfg["spawn_forward_offset_range_m"] = [0.0, 0.0]
        cfg["sudden_brake"]["departure_wait_enabled"] = False
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
                "vehicle_oncoming",
                "vehicle_cut_in",
            },
        )
        self.assertNotIn("vehicle_cut_in", vehicles["unavailable_behaviors"])
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

    def test_real_map_oncoming_native_request_reaches_existing_cleanup_before_route_end(self):
        metadata, points, zone = load_catalog_inputs()
        loader = MGeoMapLoader(str(ROOT / "data/R_KR_PR_K-city_2025"))
        for competition in (False, True):
            with self.subTest(competition=competition):
                c, a = self.make_context("vehicle_oncoming", seed=2709)
                c.route_points, c.map_loader = points, loader
                c.tunnel_entry_s = metadata["tunnel_entry_s_m"]
                c.tunnel_exit_s = metadata["tunnel_exit_s_m"]
                c.route_definition = {"zones": {"tunnel": copy.deepcopy(zone)}}
                if competition:
                    c.scenario_name = "full_loop_competition"
                    c.cfg["competition_tunnel_scene"] = "vehicle"
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                c.tunnel_vehicle_runtime_timeout = c.cfg["tunnel_vehicle_dynamic"]["max_run_sec"]
                plan = copy.deepcopy(c.tunnel_vehicle_plan)
                rng_before = c.random.getstate()

                def spawn(**kw):
                    p = c.tunnel_vehicle_plan["spawn_target"]
                    observed = state(p["x"], speed=kw["velocity"], link=p["link"],
                                     y=p["y"], yaw=p["yaw_deg"])
                    observed.transform.location.z = p["z"]
                    a.get_actor_state.return_value = observed
                    return a

                c.grpc.spawn_vehicle.side_effect = spawn
                ex, ey, _, _ = interpolate_on_polyline(points, c.tunnel_entry_s + 10.)
                h._spawn(c, {"x": ex, "y": ey, "speed": 21.6})
                request = c.grpc.set_vehicle_route.call_args
                route, indices = request.args[1], request.kwargs["route_waypoint_indices"]
                last = route[-1]
                self.assertEqual(last, "A2256W000028")
                last_points = loader.get_link_points(last)
                self.assertEqual(indices[last], len(last_points) - 1)
                old_end_s = project_distance_on_polyline(
                    c.tunnel_vehicle_cleanup_points, *last_points[0][:2])
                actual_end_s = project_distance_on_polyline(
                    c.tunnel_vehicle_cleanup_points, *last_points[indices[last]][:2])
                self.assertLess(old_end_s, plan["cleanup_route_s_m"])
                self.assertGreater(actual_end_s, plan["cleanup_route_s_m"])
                self.assertEqual(route, plan["route_links"][plan["spawn_target"]["route_index"]:])
                self.assertEqual(c.tunnel_vehicle_plan, plan)
                self.assertEqual(c.random.getstate(), rng_before)
                self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["model_name"], plan["model"])
                self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["velocity"], plan["speed_mps"])
                self.assertEqual(c.tunnel_vehicle_result["motion_commands"]["route_endpoint"]["waypoint"],
                                 len(last_points) - 1)

    def test_native_slow_brake_and_oncoming_preserve_plan_and_request_full_last_link(self):
        for behavior in ("vehicle_slow", "vehicle_sudden_brake", "vehicle_oncoming"):
            for competition in (False, True):
                with self.subTest(behavior=behavior, competition=competition):
                    c, a = self.make_context(behavior, seed=71)
                    if competition:
                        c.scenario_name = "full_loop_competition"
                        c.cfg["competition_tunnel_scene"] = "vehicle"
                        c.competition_tunnel_start_s = 0.
                    prepare_tunnel_vehicle(c)
                    h = c.tunnel_vehicle_handler
                    h.arm(c)
                    c.tunnel_vehicle_runtime_timeout = c.cfg["tunnel_vehicle_dynamic"]["max_run_sec"]
                    plan = copy.deepcopy(c.tunnel_vehicle_plan)
                    before_rng = c.random.getstate()
                    h._spawn(c, {"x": 0., "y": 0., "speed": 36.})
                    request = c.grpc.set_vehicle_route.call_args
                    route, indices = request.args[1], request.kwargs["route_waypoint_indices"]
                    self.assertEqual(route, plan["route_links"])
                    self.assertEqual(indices[route[0]], plan["spawn_target"]["waypoint"])
                    self.assertEqual(indices[route[-1]], len(c.map_loader.get_link_points(route[-1])) - 1)
                    self.assertEqual(c.tunnel_vehicle_plan, plan)
                    self.assertEqual(c.random.getstate(), before_rng)
                    self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["model_name"], plan["model"])
                    c.grpc.set_vehicle_velocity.assert_called_once_with(a, plan["speed_mps"])
                    c.grpc.set_vehicle_speed_limit.assert_called_once_with(a, plan["speed_mps"], enabled=True)

    def test_oncoming_native_endpoint_lifetime_allows_cleanup_before_actor_disappears(self):
        for competition in (False, True):
            with self.subTest(competition=competition):
                c, a = self.make_context("vehicle_oncoming")
                if competition:
                    c.scenario_name = "full_loop_competition"
                    c.cfg["competition_tunnel_scene"] = "vehicle"
                    c.competition_tunnel_start_s = 0.
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                self.start(c, h)
                request = c.grpc.set_vehicle_route.call_args
                last_link = request.args[1][-1]
                endpoint = c.map_loader.get_link_points(last_link)[
                    request.kwargs["route_waypoint_indices"].get(last_link, 0)]
                # MORAI may drop the actor as soon as its requested route ends.
                # The old implicit waypoint 0 was x=0, before cleanup at x=-40.
                self.assertEqual(c.map_loader.get_link_points(last_link)[0][0], 0.)
                observed = {"actor": a.get_actor_state.return_value}
                a.get_actor_state.side_effect = lambda: (
                    observed["actor"] if observed["actor"].transform.location.x > endpoint[0] else None)
                for t, x, ego_x, link in ((5., 97., 30., "opposite"),
                                          (10., 67., 83., "opposite"),
                                          (20., 7., 120., "opposite"),
                                          (28., -41., 137., "outlet")):
                    c.now = t
                    observed["actor"] = state(x, speed=6., link=link, y=3.5, yaw=180.)
                    h.update(c, t, {"x": ego_x, "y": 0., "speed": 36.}, ego_x)
                self.assertTrue(c.tunnel_vehicle_result["passed"])
                self.assertTrue(c.tunnel_vehicle_result["completed"])
                self.assertTrue(c.tunnel_vehicle_result["cleanup_verified"])
                self.assertEqual(c.tunnel_vehicle_result["cleanup_reason"],
                                 "outside_tunnel_continuation_and_ego_clearance")
                c.grpc.destroy_vehicle_by_id.assert_called_once_with("npc-1")
                self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_bad_actual_native_endpoint_is_rejected_before_spawn(self):
        for bad_points in ([], [(0., 3.5, 0.)],
                           [(0., 3.5, 0.), (float("nan"), 3.5, 0.)],
                           [(0., 3.5, 0.), (-50., 3.5)],
                           [(0., 3.5, 0.), (-40., 3.5, 0.)],
                           [(0., 3.5, 0.), (-10., 3.5, 0.)]):
            with self.subTest(points=bad_points):
                c, _ = self.make_context("vehicle_oncoming")
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                original = c.map_loader.get_link_points
                c.map_loader.get_link_points = lambda link: bad_points if link == "outlet" else original(link)
                with self.assertRaisesRegex(RuntimeError, "native route endpoint"):
                    h._spawn(c, {"x": 0., "y": 0., "speed": 36.})
                c.grpc.spawn_vehicle.assert_not_called()
                c.grpc.set_vehicle_route.assert_not_called()
                self.assertEqual(c.scenario_actors, [])
                self.assertFalse(c.tunnel_vehicle_result["completed"])

    def test_native_assigned_route_cannot_stop_at_a_prefix_before_cleanup(self):
        c, _ = self.make_context("vehicle_oncoming")
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        c.tunnel_vehicle_plan["route_links"] = ["opposite"]
        with self.assertRaisesRegex(RuntimeError, "native route endpoint"):
            h._spawn(c, {"x": 0., "y": 0., "speed": 36.})
        c.grpc.spawn_vehicle.assert_not_called()
        c.grpc.set_vehicle_route.assert_not_called()

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

    def waiting_brake(self):
        c, a = self.make_context("vehicle_sudden_brake")
        c.cfg["tunnel_vehicle_dynamic"]["sudden_brake"]["departure_wait_enabled"] = True
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h)
        self.assertEqual(c.tunnel_vehicle_phase, "waiting_for_departure")
        self.assertEqual(c.grpc.spawn_vehicle.call_args.kwargs["velocity"], 0)
        self.assertTrue(c.grpc.spawn_vehicle.call_args.kwargs["pause"])
        a.set_pause.assert_called_once_with(True)
        c.grpc.set_vehicle_velocity.assert_called_with(a, 0)
        return c, a, h

    def test_brake_departure_wait_does_not_consume_startup_deadline(self):
        c, a, h = self.waiting_brake()
        for t in (1, 10, 30):
            self.tick(c, a, h, t, 40, 0, speed=0, ego_speed=0)
            self.assertEqual(c.tunnel_vehicle_phase, "waiting_for_departure")
            self.assertFalse(c.tunnel_vehicle_result["startup_motion_verified"])
            self.assertFalse(
                c.tunnel_vehicle_result["brake"].get("brake_commanded", False)
            )
            self.assertNotIn(
                False, [call.args[0] for call in a.set_pause.call_args_list]
            )
        self.tick(c, a, h, 31, 40, 5, speed=0, ego_speed=14.4)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        a.set_pause.assert_called_with(False)
        c.grpc.set_vehicle_velocity.assert_called_with(a, 4)
        self.assertEqual(c.tunnel_vehicle_motion_monitor_started, 31)
        self.tick(c, a, h, 31.25, 40, 5, speed=0, ego_speed=14.4)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        self.tick(c, a, h, 31.5, 41, 6, speed=4, ego_speed=14.4)
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])
        self.assertEqual(c.tunnel_vehicle_phase, "cruising")
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_not_called()

    def test_delayed_departure_then_observed_brake_hold_resume_uses_same_actor(self):
        c, a, h = self.waiting_brake()
        self.tick(c, a, h, 1, 40, 2, speed=0, ego_speed=36)
        self.assertEqual(c.tunnel_vehicle_phase, "waiting_for_departure")
        self.tick(c, a, h, 2, 40, 5, speed=0, ego_speed=14.4)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        self.assertEqual(
            c.tunnel_vehicle_result["brake"]["departure"]["release_reason"], "ego_gap"
        )
        for i in range(1, 15):
            t = 2 + i / 4
            x = 40 + i
            self.tick(c, a, h, t, x, x - 35, speed=4, ego_speed=14.4)
            if c.tunnel_vehicle_phase == "braking":
                break
        self.assertEqual(c.tunnel_vehicle_phase, "braking")
        c.brake_test_time, c.brake_test_x = c.now, c.tunnel_vehicle_last_sample["x"]
        self.hold_brake(c, a, h)
        self.tick_after_brake(c, a, h, 8.5, 5.125, gap=22, speed=4)
        metrics = c.tunnel_vehicle_result["brake"]
        self.assertTrue(metrics["resume_motion_verified"])
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        commands = metrics["commands"]
        self.assertEqual((commands[0]["speed_mps"], commands[-1]["speed_mps"]), (4., 4.))
        self.assertTrue(all(cmd["speed_mps"] == .5 for cmd in commands[1:-1]))
        self.assertGreater(len(commands), 3)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        self.tick(c, a, h, 45, 190, 137, speed=4)
        self.assertTrue(c.tunnel_vehicle_result["completed"])
        a.destroy.assert_called_once()

    def test_departure_ttc_and_expert_response_require_actual_approach(self):
        for kind in ("ttc", "expert_response"):
            c, a = self.make_context("vehicle_sudden_brake")
            cfg = c.cfg["tunnel_vehicle_dynamic"]
            cfg["sudden_brake"]["departure_wait_enabled"] = True
            if kind == "ttc":
                cfg["sudden_brake"]["departure_trigger_gap_m"] = 20
            else:
                cfg["spawn_forward_offset_range_m"] = [10, 10]
            prepare_tunnel_vehicle(c)
            h = c.tunnel_vehicle_handler
            h.arm(c)
            self.start(c, h)
            x = c.tunnel_vehicle_plan["spawn_s_m"]
            # Slowing at the start cannot release the NPC without any approach.
            self.tick(c, a, h, 1, x, 0, speed=0, ego_speed=32.4)
            self.assertEqual(c.tunnel_vehicle_phase, "waiting_for_departure")
            self.tick(
                c,
                a,
                h,
                2,
                x,
                11 if kind == "ttc" else 5,
                speed=0,
                ego_speed=36 if kind == "ttc" else 14.4,
            )
            self.assertEqual(c.tunnel_vehicle_phase, "starting")
            expected = (
                "ego_ttc" if kind == "ttc" else "observed_ego_response_after_approach"
            )
            self.assertEqual(
                c.tunnel_vehicle_result["brake"]["departure"]["release_reason"],
                expected,
            )

    def test_departure_rejected_pause_or_unpause_cleans_owned_actor(self):
        c, a = self.make_context("vehicle_sudden_brake")
        c.cfg["tunnel_vehicle_dynamic"]["sudden_brake"]["departure_wait_enabled"] = True
        a.set_pause.return_value = False
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        with self.assertRaisesRegex(RuntimeError, "departure pause rejected"):
            self.start(c, h)
        a.destroy.assert_called_once()
        c.grpc.set_vehicle_route.assert_not_called()
        c, a, h = self.waiting_brake()
        a.set_pause.return_value = False
        with self.assertRaisesRegex(RuntimeError, "departure commands rejected"):
            self.tick(c, a, h, 1, 40, 5, speed=0, ego_speed=14.4)
        a.destroy.assert_called_once()

    def test_departure_requires_observed_stationary_npc_and_bounded_wait(self):
        for kind in ("drift", "speed", "timeout"):
            c, a, h = self.waiting_brake()
            with self.assertRaisesRegex(
                RuntimeError, "before Ego departure trigger|wait timed out"
            ):
                self.tick(
                    c,
                    a,
                    h,
                    46 if kind == "timeout" else 1,
                    40.6 if kind == "drift" else 40,
                    0,
                    speed=0.3 if kind == "speed" else 0,
                    ego_speed=0,
                )
            a.destroy.assert_called_once()

    def test_spawn_adapter_passes_pause_to_world_atomically(self):
        from ..adapters.morai_grpc import MoraiGrpcClient

        client = MoraiGrpcClient.__new__(MoraiGrpcClient)
        client.world = Mock()
        for paused in (False, True):
            client.world.spawn_vehicle.reset_mock()
            client.spawn_vehicle(
                "pose", "Sedan", "brake", multi_ego=False, pause=paused
            )
            kwargs = client.world.spawn_vehicle.call_args.kwargs
            self.assertEqual(kwargs["velocity"], 0)
            self.assertEqual(kwargs.get("pause", False), paused)
            self.assertFalse(kwargs["multi_ego"])

    def stable_brake(self, gap=25):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 1, 44, 15, speed=4.0, ego_speed=28.8)
        self.tick(c, a, h, 1.25, 45, 45 - gap, speed=4.0, ego_speed=18.0)
        for i in range(6, 25):
            t = i / 4.0
            x = 40 + 4 * t
            self.tick(c, a, h, t, x, x - gap, speed=4.0, ego_speed=14.4)
            if c.tunnel_vehicle_phase == "braking":
                break
        self.assertEqual(c.tunnel_vehicle_phase, "braking")
        c.brake_test_time, c.brake_test_x = c.now, c.tunnel_vehicle_last_sample["x"]
        return c, a, h

    def tick_after_brake(
        self, c, a, h, offset, npc_offset, gap=25, speed=0.5, ego_speed=0
    ):
        x = c.brake_test_x + npc_offset
        self.tick(
            c,
            a,
            h,
            c.brake_test_time + offset,
            x,
            x - gap,
            speed=speed,
            ego_speed=ego_speed,
        )

    def hold_brake(self, c, a, h):
        self.tick_after_brake(c, a, h, 0.25, 0.125, gap=22, ego_speed=7.2)
        self.assertEqual(c.tunnel_vehicle_phase, "low_speed_hold")
        for i in range(2, 34):
            offset = i / 4.0
            self.tick_after_brake(
                c, a, h, offset, 0.125 + 0.5 * (offset - 0.25), gap=22
            )
        self.assertEqual(c.tunnel_vehicle_phase, "departing")

    def test_npc_velocity_kmh_normalizes_against_observed_position_speed(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 1, 44, 10, speed=4, ego_speed=14.4)
        observation = c.tunnel_vehicle_result["last_observation"]
        self.assertEqual(observation["raw_velocity_magnitude"], 14.4)
        self.assertEqual(observation["state_velocity_unit"], "kmh")
        self.assertEqual(observation["speed_mps"], 4)
        self.assertEqual(observation["position_speed_mps"], 4)
        self.assertEqual(
            c.tunnel_vehicle_result["brake"]["follow_filter_last"][
                "relative_speed_mps"
            ],
            0,
        )
        c.cfg["tunnel_vehicle_dynamic"]["npc_state_velocity_unit"] = "mps"
        sample = state(44, speed=4)
        sample.velocity.x = 4  # Explicit alternate SDK telemetry units.
        a.get_actor_state.return_value = sample
        self.assertEqual(h._read_state(c)["speed_mps"], 4)

    def test_recorded_kcity_safe_following_replays_to_brake_trigger(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/tunnel_brake_following_kmh.json").read_text(
                encoding="utf-8"
            )
        )
        for run in fixture["runs"]:
            with self.subTest(report=run["source_report"]):
                c, a = self.make_context("vehicle_sudden_brake")
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                t0, npc_s, gap, ego_speed, raw = run["samples"][0]
                self.start(c, h, ego_x=npc_s - gap, ego_speed=ego_speed * 3.6)
                for t, npc_s, gap, ego_speed, raw in run["samples"]:
                    self.tick(
                        c,
                        a,
                        h,
                        t - t0,
                        npc_s,
                        npc_s - gap,
                        speed=raw / 3.6,
                        ego_speed=ego_speed * 3.6,
                    )
                    if c.tunnel_vehicle_phase == "braking":
                        break
                self.assertEqual(c.tunnel_vehicle_phase, "braking")
                metrics = c.tunnel_vehicle_result["brake"]
                self.assertGreater(metrics["expert_follow_gap_m"], 35)
                self.assertAlmostEqual(metrics["pre_brake_npc_speed_mps"], raw / 3.6)
                self.assertTrue(3.5 < metrics["pre_brake_npc_speed_mps"] < 4.5)
                self.assertGreaterEqual(metrics["follow_stable_duration_sec"], 1.5)
                self.assertLess(c.tunnel_vehicle_last_sample["npc_s_m"], 70)
                c.grpc.set_vehicle_velocity.assert_called_with(a, 0.5)
                a.destroy.assert_not_called()

    def test_following_remains_settled_after_ego_recovers_small_speed_drop(self):
        c, a = self.make_context("vehicle_sudden_brake")
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h, ego_speed=16.2)
        self.tick(c, a, h, 1, 44, 15, speed=4, ego_speed=16.2)
        for i in range(5, 25):
            t = i / 4.0
            self.tick(
                c,
                a,
                h,
                t,
                40 + 4 * t,
                40 + 4 * t - 25,
                speed=4,
                ego_speed=14.4 if i < 9 else 15.3,
            )
            if c.tunnel_vehicle_phase == "braking":
                break
        self.assertEqual(c.tunnel_vehicle_phase, "braking")
        metrics = c.tunnel_vehicle_result["brake"]
        self.assertTrue(metrics["ego_deceleration_observed"])
        self.assertLess(metrics["ego_speed_drop_mps"], 0.5)

    def test_stable_follow_outside_old_gap_window_triggers(self):
        c, a, h = self.stable_brake(gap=12)
        self.assertEqual(c.tunnel_vehicle_result["brake"]["expert_follow_gap_m"], 12)
        c.grpc.set_vehicle_velocity.assert_called_with(a, 0.5)
        a.destroy.assert_not_called()

    def test_sudden_brake_waits_until_ego_can_catch_fixed_cruise(self):
        c, a = self.make_context("vehicle_sudden_brake")
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        for i in range(1, 5):
            h.update(c, i, {"x": 0.0, "y": 0.0, "speed": 10.8}, 0.0)
        c.grpc.spawn_vehicle.assert_not_called()
        h.update(c, 5, {"x": 0.0, "y": 0.0, "speed": 18.0}, 0.0)
        c.grpc.spawn_vehicle.assert_called_once()
        c.grpc.set_vehicle_velocity.assert_called_with(a, 4.0)

    def test_hold_actual_speed_escape_recovers_without_destroy(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, 0.25, 0.125, gap=22, ego_speed=7.2)
        self.tick_after_brake(c, a, h, 0.5, 0.5, gap=22, speed=1.5)
        self.assertEqual(c.tunnel_vehicle_phase, "low_speed_hold")
        c.grpc.set_vehicle_velocity.assert_called_with(a, 0.5)
        self.tick_after_brake(c, a, h, 0.75, 0.875, gap=22, speed=1.5)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertFalse(c.tunnel_vehicle_result["brake"].get("hold_verified", False))
        c.grpc.set_vehicle_velocity.assert_called_with(a, 4.0)
        a.destroy.assert_not_called()

    def test_single_07025_overshoot_reapplies_command_then_requires_fresh_eight_second_hold(self):
        c, a, h = self.stable_brake()
        original_cfg, original_plan = copy.deepcopy(c.cfg), copy.deepcopy(c.tunnel_vehicle_plan)
        self.tick_after_brake(c, a, h, .25, .125, gap=22, ego_speed=7.2)
        before = c.grpc.set_vehicle_velocity.call_count
        self.tick_after_brake(c, a, h, .5, .300625, gap=22, speed=.7025)
        metrics = c.tunnel_vehicle_result["brake"]
        self.assertEqual(c.tunnel_vehicle_phase, "low_speed_hold")
        self.assertFalse(metrics.get("failure_pending", False))
        self.assertEqual(metrics["actual_hold_duration_sec"], 0.)
        self.assertGreater(c.grpc.set_vehicle_velocity.call_count, before)
        c.grpc.set_vehicle_velocity.assert_called_with(a, .5)
        c.grpc.set_vehicle_speed_limit.assert_called_with(a, .5, enabled=True)
        self.tick_after_brake(c, a, h, .75, .425625, gap=22, speed=.5)
        self.assertEqual(metrics["hold_start"], c.brake_test_time + .75)
        for i in range(4, 35):
            offset = i / 4.
            self.tick_after_brake(c, a, h, offset, .425625 + .5 * (offset - .75), gap=22)
        self.assertEqual(c.tunnel_vehicle_phase, "low_speed_hold")
        self.assertFalse(metrics.get("hold_verified", False))
        self.assertEqual(metrics["actual_hold_duration_sec"], 7.75)
        self.tick_after_brake(c, a, h, 8.75, 4.425625, gap=22)
        self.assertEqual(c.tunnel_vehicle_phase, "departing")
        self.assertTrue(metrics["hold_verified"])
        self.assertEqual(metrics["hold_end"] - metrics["hold_start"], 8.)
        self.assertEqual(metrics["actual_hold_duration_sec"], 8.)
        self.assertEqual(metrics["hold_restart_count"], 1)
        self.assertGreater(metrics["trough_hold_refresh_count"], 1)
        self.assertEqual(c.cfg, original_cfg)
        self.assertEqual(c.tunnel_vehicle_plan, original_plan)
        self.assertTrue(all(call.args[0] is a for call in c.grpc.set_vehicle_velocity.call_args_list))
        self.assertTrue(all(call.args[0] is a for call in c.grpc.set_vehicle_speed_limit.call_args_list))
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_not_called()

    def test_continuous_small_overshoot_cannot_relax_original_trough_threshold(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, .25, .125, gap=22, ego_speed=7.2)
        self.tick_after_brake(c, a, h, .5, .300625, gap=22, speed=.7025)
        self.tick_after_brake(c, a, h, .75, .47625, gap=22, speed=.7025)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertEqual(c.tunnel_vehicle_result["failure_reason"], "NPC did not maintain trough speed")
        self.assertFalse(c.tunnel_vehicle_result["brake"].get("hold_verified", False))
        self.assertFalse(c.tunnel_vehicle_result["interaction_complete"])
        self.assertFalse(c.tunnel_vehicle_result["completed"])
        c.grpc.set_vehicle_velocity.assert_called_with(a, 4.)
        a.destroy.assert_not_called()

    def test_late_reconfirmation_cannot_claim_continuous_observed_trough(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, .25, .125, gap=22, ego_speed=7.2)
        self.tick_after_brake(c, a, h, .5, .300625, gap=22, speed=.7025)
        # A good sample arriving after the confirmation budget cannot prove
        # what happened in the unobserved interval.
        self.tick_after_brake(c, a, h, 1., .550625, gap=22, speed=.5)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertEqual(c.tunnel_vehicle_result["failure_reason"], "NPC did not maintain trough speed")
        self.assertFalse(c.tunnel_vehicle_result["brake"].get("hold_verified", False))

    def test_trough_refresh_command_rejection_is_observed_failure_not_success(self):
        for method in ("set_vehicle_velocity", "set_vehicle_speed_limit"):
            with self.subTest(method=method):
                c, a, h = self.stable_brake()
                self.tick_after_brake(c, a, h, .25, .125, gap=22, ego_speed=7.2)
                getattr(c.grpc, method).side_effect = [False, True]
                self.tick_after_brake(c, a, h, .5, .300625, gap=22, speed=.7025)
                self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
                self.assertIn("command rejected", c.tunnel_vehicle_result["failure_reason"])
                self.assertTrue(c.tunnel_vehicle_result["failed"])
                self.assertFalse(c.tunnel_vehicle_result["brake"].get("hold_verified", False))
                self.assertFalse(c.tunnel_vehicle_result["completed"])
                c.grpc.set_vehicle_velocity.assert_called_with(a, 4.)
                a.destroy.assert_not_called()

    def test_stable_follow_gap_report_and_immediate_trough_command(self):
        c, a, h = self.stable_brake()
        metrics = c.tunnel_vehicle_result["brake"]
        self.assertEqual(metrics["expert_follow_gap_m"], 25.0)
        self.assertEqual(metrics["expert_follow_gap_upper_m"], 27.0)
        self.assertGreaterEqual(metrics["follow_stable_duration_sec"], 1.5)
        self.assertGreaterEqual(len(metrics["follow_window_samples"]), 7)
        self.assertEqual(metrics["pre_brake_npc_speed_mps"], 4.0)
        self.assertEqual(metrics["brake_command_speed_mps"], 0.5)
        c.grpc.set_vehicle_velocity.assert_called_with(a, 0.5)
        c.grpc.set_vehicle_speed_limit.assert_called_with(a, 0.5, enabled=True)
        self.assertIsNone(c.tunnel_vehicle_plan["event_section"])
        self.assertEqual(
            [r["section_index"] for r in c.tunnel_vehicle_plan["brake_regions"]], [2, 3]
        )
        a.destroy.assert_not_called()

    def test_closing_gap_and_fixed_20_to_35m_do_not_trigger_or_adjust_cruise(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.assertEqual(c.tunnel_vehicle_plan["speed_mps"], 4.0)
        for i in range(1, 33):
            t = i / 4.0
            self.tick(c, a, h, t, 40 + 4 * t, 6 * t, speed=4.0, ego_speed=21.6)
            self.assertEqual(c.tunnel_vehicle_phase, "cruising")
        self.assertFalse(c.tunnel_vehicle_result["brake"].get("brake_commanded", False))
        c.grpc.set_vehicle_velocity.assert_called_once_with(a, 4.0)
        a.destroy.assert_not_called()

    def test_speed_convergence_without_gap_settle_does_not_trigger(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 1, 44, 15, speed=4.0, ego_speed=28.8)
        for i in range(5, 17):
            t = i / 4.0
            x = 40 + 4 * t
            gap = 24 if i % 2 else 26
            self.tick(c, a, h, t, x, x - gap, speed=4.0, ego_speed=14.4)
        self.assertEqual(c.tunnel_vehicle_phase, "cruising")
        self.assertEqual(
            c.tunnel_vehicle_result["brake"]["follow_stable_duration_sec"], 0.0
        )

    def test_equal_speeds_without_observed_catchup_and_ego_slowdown_do_not_trigger(
        self,
    ):
        c, a, h = self.armed("vehicle_sudden_brake")
        for i in range(1, 13):
            t = i / 4.0
            x = 40 + 4 * t
            self.tick(c, a, h, t, x, x - 40, speed=4, ego_speed=14.4)
        self.assertFalse(c.tunnel_vehicle_result["brake"]["catchup_observed"])
        self.assertEqual(c.tunnel_vehicle_phase, "cruising")

    def test_follow_settle_resets_on_unstable_or_missing_sample(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 1, 44, 15, speed=4, ego_speed=28.8)
        self.tick(c, a, h, 1.25, 45, 20, speed=4, ego_speed=14.4)
        self.tick(c, a, h, 1.5, 46, 21, speed=4, ego_speed=14.4)
        self.tick(c, a, h, 1.75, 47, 22, speed=4, ego_speed=14.4)
        self.tick(c, a, h, 3, 52, 27, speed=4, ego_speed=14.4)
        self.assertEqual(
            c.tunnel_vehicle_result["brake"]["follow_stable_duration_sec"], 0.0
        )
        self.assertEqual(c.tunnel_vehicle_phase, "cruising")

    def test_same_actor_actual_drop_eight_second_hold_resume_and_exit(self):
        c, a, h = self.stable_brake()
        self.hold_brake(c, a, h)
        metrics = c.tunnel_vehicle_result["brake"]
        self.assertTrue(metrics["deceleration_verified"])
        self.assertGreaterEqual(metrics["measured_deceleration_mps2"], 2.0)
        self.assertEqual(metrics["minimum_npc_speed_mps"], 0.5)
        self.assertEqual(metrics["hold_start"], c.brake_test_time + 0.25)
        self.assertEqual(metrics["hold_end"], c.brake_test_time + 8.25)
        self.assertEqual(metrics["actual_hold_duration_sec"], 8.0)
        self.assertEqual(metrics["resume_command_speed_mps"], 4.0)
        commands = metrics["commands"]
        self.assertEqual((commands[0]["label"], commands[-1]["label"]), ("sudden_drop", "resume"))
        self.assertGreater(len(commands), 2)
        self.assertTrue(all(cmd["speed_mps"] == .5 for cmd in commands[:-1]))
        self.assertEqual(commands[-1]["speed_mps"], 4.)
        a.destroy.assert_not_called()
        self.tick_after_brake(c, a, h, 8.5, 5.125, gap=22, speed=4.0)
        self.assertTrue(metrics["resume_motion_verified"])
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        self.tick(c, a, h, 35, 150, 110, speed=4)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 45, 190, 137, speed=4)
        h.finish(c)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_called_once()

    def test_no_resume_before_eight_seconds_of_observed_trough(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, 0.25, 0.125, gap=22, ego_speed=7.2)
        for i in range(2, 33):
            offset = i / 4.0
            self.tick_after_brake(
                c, a, h, offset, 0.125 + 0.5 * (offset - 0.25), gap=22
            )
        self.assertEqual(c.tunnel_vehicle_phase, "low_speed_hold")
        self.assertFalse(
            c.tunnel_vehicle_result["brake"].get("resumed_after_event", False)
        )
        c.grpc.set_vehicle_velocity.assert_called_with(a, 0.5)

    def test_emergency_recovers_same_actor_and_fails_only_after_outside_cleanup(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, 0.25, 0.125, gap=6, ego_speed=3.6)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertTrue(c.tunnel_vehicle_result["failed"])
        self.assertFalse(c.tunnel_vehicle_result["interaction_complete"])
        a.set_pause.assert_called_with(False)
        c.grpc.set_vehicle_velocity.assert_called_with(a, 4.0)
        a.destroy.assert_not_called()
        self.tick_after_brake(c, a, h, 0.5, 1.125, gap=7, speed=4)
        self.assertTrue(c.tunnel_vehicle_result["brake"]["recovery_motion_verified"])
        self.tick(c, a, h, 25, 140, 100, speed=4)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 35, 180, 167, speed=4)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        a.destroy.assert_not_called()  # Standalone still requires 30 m clearance.
        with self.assertRaisesRegex(RuntimeError, "emergency gap"):
            self.tick(c, a, h, 38, 190, 137, speed=4)
        a.destroy.assert_called_once()
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_later_missing_state_preserves_original_failed_brake_reason(self):
        c, a, h = self.stable_brake()
        self.tick_after_brake(c, a, h, .25, .125, gap=6, ego_speed=3.6)
        original_failure = c.tunnel_vehicle_result["failure_reason"]
        a.get_actor_state.return_value = None
        c.now += .25
        with self.assertRaisesRegex(RuntimeError, "state unavailable"):
            h.update(c, c.now, {"x": 50., "y": 0., "speed": 0.}, 50.)
        result = c.tunnel_vehicle_result
        self.assertEqual(result["failure_reason"], original_failure)
        self.assertEqual(result["brake"]["failure_reason"], original_failure)
        self.assertIn("state unavailable", result["secondary_failure_reason"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["interaction_complete"])
        self.assertTrue(result["cleanup_verified"])
        self.assertEqual(c.tunnel_vehicle_phase, "failed")
        saved = json.loads(c.tunnel_vehicle_report_path.read_text())
        self.assertEqual(saved["failure_reason"], original_failure)
        self.assertIn("state unavailable", saved["secondary_failure_reason"])

    def test_command_acks_without_actual_drop_recover_and_mark_failed(self):
        c, a, h = self.stable_brake()
        for i in range(1, 7):
            self.tick_after_brake(c, a, h, i / 4.0, i, speed=4, ego_speed=14.4)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertFalse(
            c.tunnel_vehicle_result["brake"].get("deceleration_verified", False)
        )
        self.assertIn("actual deceleration", c.tunnel_vehicle_result["failure_reason"])
        a.destroy.assert_not_called()

    def test_brake_requires_space_for_actual_drop_hold_and_resume(self):
        c, a, h = self.armed("vehicle_sudden_brake")
        self.tick(c, a, h, 1, 44, 15, speed=4, ego_speed=28.8)
        for i in range(41, 51):
            t = i / 4.0
            x = 96 + (t - 10.25) * 0.1
            self.tick(c, a, h, t, x, x - 25, speed=4, ego_speed=14.4)
        self.assertFalse(c.tunnel_vehicle_result["brake"].get("brake_commanded", False))
        a.destroy.assert_not_called()

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

    def test_resume_acks_without_actual_motion_recover_and_mark_failed(self):
        c, a, h = self.stable_brake()
        self.hold_brake(c, a, h)
        deadline = c.tunnel_vehicle_plan["sudden_brake_settings"][
            "resume_confirm_timeout_sec"
        ]
        for i in range(1, int(deadline * 4) + 1):
            self.tick_after_brake(c, a, h, 8.25 + i / 4.0, 4.125, gap=22)
        self.assertEqual(c.tunnel_vehicle_phase, "recovering_failed")
        self.assertFalse(c.tunnel_vehicle_result["brake"]["resume_motion_verified"])
        self.assertIn("resumed movement", c.tunnel_vehicle_result["failure_reason"])
        a.destroy.assert_not_called()

    def test_resume_allows_observed_cruise_acceleration_from_live_trace(self):
        c, a, h = self.stable_brake()
        self.hold_brake(c, a, h)
        for after, travel, speed in (
            (0.5, 0.8, 1.6),
            (1, 1.8, 2.0),
            (2, 4.3, 3.0),
            (3, 7.6, 3.6),
        ):
            self.tick_after_brake(
                c, a, h, 8.25 + after, 4.125 + travel, gap=22, speed=speed
            )
            self.assertFalse(
                c.tunnel_vehicle_result["brake"].get("failure_pending", False)
            )
        self.assertTrue(c.tunnel_vehicle_result["brake"]["resume_motion_verified"])
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        a.destroy.assert_not_called()

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

    def test_cut_in_missing_source_geometry_fails_before_morai(self):
        c, _ = self.make_context("vehicle_cut_in")
        c.route_definition["zones"]["tunnel"]["oncoming_route_links"] = []
        with self.assertRaisesRegex(
            UnsupportedVehicleTopology, "source corridor missing"
        ):
            prepare_tunnel_vehicle(c)
        self.assertEqual(c.grpc.mock_calls, [])

    def test_cut_in_opposite_geometry_same_heading_no_route_at_spawn(self):
        c, a, h = self.armed("vehicle_cut_in")
        p = c.tunnel_vehicle_plan
        self.assertEqual(p["source_corridor_links"], ["opposite", "outlet"])
        self.assertEqual(p["route_links"], [])
        self.assertAlmostEqual(p["spawn_target"]["y"], 3.5)
        self.assertAlmostEqual(p["spawn_target"]["yaw_deg"], 0)
        c.grpc.set_vehicle_route.assert_not_called()
        self.assertTrue(c.grpc.spawn_vehicle.call_args.kwargs["multi_ego"])
        c.grpc.set_vehicle_direct_control.assert_called_once_with(a, True)
        c.grpc.control_vehicle_direct.assert_called_with(a, 0.0, 7.0)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        self.tick(c, a, h, 1, 47, 8, speed=7, link="opposite", y=3.5)
        self.assertEqual(c.tunnel_vehicle_phase, "source_corridor_cruise")
        c.grpc.set_vehicle_route.assert_not_called()
        a.set_transform.assert_not_called()

    def begin_cut_in(self):
        c, a, h = self.armed("vehicle_cut_in")
        self.tick(c, a, h, 1, 47, 8, speed=7, link="opposite", y=3.5)
        self.tick(c, a, h, 5, 70, 45, speed=7, link="opposite", y=3.5)
        self.assertEqual(c.tunnel_vehicle_phase, "changing_lane")
        self.assertLess(c.grpc.control_vehicle_direct.call_args.args[1], 0)
        c.grpc.set_vehicle_route.assert_not_called()
        return c, a, h

    def test_cut_in_direct_merge_verified_before_same_actor_route_handoff(self):
        c, a, h = self.begin_cut_in()
        self.assertNotIn("route_endpoint", c.tunnel_vehicle_result["motion_commands"])
        self.tick(c, a, h, 6, 77, 52, speed=7, link="opposite", y=2, yaw=-10)
        c.grpc.set_vehicle_route.assert_not_called()
        self.tick(c, a, h, 7, 84, 59, speed=7, link="opposite", y=0.3, yaw=-3)
        c.grpc.set_vehicle_route.assert_not_called()
        self.tick(c, a, h, 7.5, 87.5, 62.5, speed=7, link="", y=0)
        self.assertEqual(c.tunnel_vehicle_phase, "route_handoff")
        self.assertFalse(c.tunnel_vehicle_result.get("lane_change_verified", False))
        c.grpc.set_vehicle_route.assert_called_once()
        self.assertEqual(c.grpc.set_vehicle_route.call_args.args[1], ["lane", "tail"])
        self.assertEqual(set(c.grpc.set_vehicle_route.call_args.kwargs["route_waypoint_indices"]), {"lane"})
        c.grpc.set_vehicle_direct_control.assert_called_with(a, False, speed_mps=7.0)
        self.tick(c, a, h, 7.75, 89.25, 64, speed=7, link="lane")
        self.assertTrue(c.tunnel_vehicle_result["lane_change_verified"])
        self.assertTrue(c.tunnel_vehicle_result["interaction_complete"])
        self.tick(c, a, h, 16, 140, 110, speed=7)
        a.destroy.assert_not_called()
        self.tick(c, a, h, 24, 190, 137, speed=7)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        c.grpc.set_vehicle_route.assert_called_once()
        a.set_transform.assert_not_called()
        a.destroy.assert_called_once()

    def test_direct_cut_in_controller_with_bicycle_motion_model(self):
        c, a, h = self.armed("vehicle_cut_in")
        x, y, yaw, ego_x = 40.0, 3.5, 0.0, 0.0
        dt = 0.05
        cfg = c.cfg["tunnel_vehicle_dynamic"]["cut_in"]
        for i in range(1, 201):
            command = c.grpc.control_vehicle_direct.call_args.args
            steer, speed = command[1], command[2]
            delta = steer * math.radians(cfg["max_steer_deg"])
            yaw += speed / cfg["wheel_base_m"] * math.tan(delta) * dt
            x += speed * math.cos(yaw) * dt
            y += speed * math.sin(yaw) * dt
            ego_x += 10 * dt
            self.tick(
                c,
                a,
                h,
                i * dt,
                x,
                ego_x,
                speed=speed,
                link="lane" if y < 1.75 else "opposite",
                y=y,
                yaw=math.degrees(yaw),
            )
            if c.tunnel_vehicle_phase == "route_handoff":
                break
        self.assertEqual(c.tunnel_vehicle_phase, "route_handoff")
        self.assertLessEqual(abs(y), cfg["target_lateral_tolerance_m"])
        self.assertLessEqual(
            abs(math.degrees(yaw)), cfg["target_heading_tolerance_deg"]
        )
        self.assertGreater(
            c.tunnel_vehicle_result["lane_change_lateral_displacement_m"], 2.9
        )
        c.grpc.set_vehicle_route.assert_called_once()
        a.set_transform.assert_not_called()
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_direct_vehicle_adapter_targets_actor_and_verifies_control_mode(self):
        from ..adapters.morai_grpc import MoraiGrpcClient

        client = MoraiGrpcClient.__new__(MoraiGrpcClient)
        actor = Mock()
        actor.set_control_mode.return_value = True
        actor.get_control_mode.return_value = 3
        actor.control.return_value = True
        enums = NS(
            VEHICLE_CONTROL_AUTO_MODE=3,
            VEHICLE_CONTROL_CRUISE_MODE=6,
            LONG_CMD_TYPE_SPEED=2,
            EGO_CRUISE_TYPE_CONSTANT=2,
            GEAR_MODE_D=4,
        )
        with patch.dict("sys.modules", {"proto.morai.actor.actor_enum_pb2": enums}):
            self.assertTrue(client.set_vehicle_direct_control(actor, True))
            actor.set_control_mode.assert_called_with(3)
            actor.set_cruise_mode.assert_called_with(enable=False)
            actor.set_vehicle_gear.assert_called_once_with(4)
            self.assertTrue(client.control_vehicle_direct(actor, -0.2, 7.0))
            actor.control.assert_called_once_with(
                long_cmd_type=2,
                throttle=0.0,
                brake=0.0,
                steer=-0.2,
                velocity=25.2,
                acceleration=0.0,
                frame=0,
            )
            self.assertFalse(client.control_vehicle_direct(actor, 1.1, 7))
            self.assertFalse(client.control_vehicle_direct(actor, 0.0, float("nan")))
            actor.get_control_mode.return_value = 6
            self.assertFalse(client.set_vehicle_direct_control(actor, True))
            self.assertTrue(
                client.set_vehicle_direct_control(actor, False, speed_mps=7.0)
            )
            actor.set_cruise_mode.assert_called_with(
                enable=True, cruise_type=2, link_speed_ratio=0, constant_velocity=25.2
            )
            actor.set_control_mode.assert_called_with(6)
        actor.set_ai.assert_not_called()

    def test_cut_in_keeps_sending_control_while_startup_motion_pending(self):
        c, a, h = self.armed("vehicle_cut_in")
        c.grpc.control_vehicle_direct.reset_mock()
        self.tick(c, a, h, 0.2, 40, 1, speed=0, link="", y=3.5)
        self.assertEqual(c.tunnel_vehicle_phase, "starting")
        c.grpc.control_vehicle_direct.assert_called_once()
        self.assertEqual(c.grpc.control_vehicle_direct.call_args.args[0], a)
        c.grpc.set_vehicle_route.assert_not_called()
        a.destroy.assert_not_called()

    def test_multi_ego_adapter_does_not_bypass_rejected_mode_or_gear(self):
        from ..adapters.morai_grpc import MoraiGrpcClient

        client = MoraiGrpcClient.__new__(MoraiGrpcClient)
        enums = NS(
            VEHICLE_CONTROL_AUTO_MODE=3,
            VEHICLE_CONTROL_CRUISE_MODE=6,
            EGO_CRUISE_TYPE_CONSTANT=2,
            GEAR_MODE_D=4,
        )
        actor = Mock()
        actor.set_control_mode.return_value = False
        with patch.dict("sys.modules", {"proto.morai.actor.actor_enum_pb2": enums}):
            self.assertFalse(client.set_vehicle_direct_control(actor, True))
            actor.get_control_mode.assert_not_called()
            actor.set_vehicle_gear.assert_not_called()
            actor.set_control_mode.return_value = True
            actor.get_control_mode.return_value = 3
            actor.set_vehicle_gear.return_value = False
            self.assertFalse(client.set_vehicle_direct_control(actor, True))
            actor.set_vehicle_gear.return_value = True
            actor.set_cruise_mode.return_value = False
            self.assertFalse(client.set_vehicle_direct_control(actor, True))
        actor.control.assert_not_called()

    def test_cut_in_link_ack_without_lateral_motion_cannot_complete(self):
        c, a, h = self.begin_cut_in()
        self.tick(c, a, h, 6, 77, 52, speed=7, link="lane", y=3.5)
        c.grpc.set_vehicle_route.assert_not_called()
        self.assertFalse(c.tunnel_vehicle_result.get("lane_change_verified", False))
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.tick(c, a, h, 16, 100, 75, speed=7, link="lane", y=3.5)

    def test_cut_in_control_or_handoff_rejection_cleanup_owned_actor(self):
        for method in ("set_vehicle_direct_control", "control_vehicle_direct"):
            c, a = self.make_context("vehicle_cut_in")
            getattr(c.grpc, method).return_value = False
            prepare_tunnel_vehicle(c)
            h = c.tunnel_vehicle_handler
            h.arm(c)
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                self.start(c, h)
            a.destroy.assert_called_once()
            c.grpc.set_vehicle_route.assert_not_called()
        c, a, h = self.begin_cut_in()
        self.tick(c, a, h, 7, 84, 59, speed=7, link="lane", y=0)
        c.grpc.set_vehicle_direct_control.return_value = False
        with self.assertRaisesRegex(RuntimeError, "handoff rejected"):
            self.tick(c, a, h, 7.5, 87.5, 62.5, speed=7, link="lane", y=0)
        a.destroy.assert_called_once()

    def test_cut_in_wrong_heading_and_departure_rejected(self):
        for yaw, y in ((180, 3.5), (0, 7)):
            c, a, h = self.armed("vehicle_cut_in")
            with self.assertRaises(RuntimeError):
                self.tick(c, a, h, 1, 47, 8, speed=7, link="opposite", y=y, yaw=yaw)
            a.destroy.assert_called_once()

    def test_oncoming_cut_in_mutual_exclusion_and_seeded_random_filter(self):
        for behavior, extra in (
            ("vehicle_cut_in", "vehicle_oncoming"),
            ("vehicle_oncoming", "vehicle_cut_in"),
        ):
            c, a = self.make_context(behavior)
            c.cfg["tunnel_vehicle_dynamic"]["additional_behaviors"] = [extra]
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                prepare_tunnel_vehicle(c)
            c.grpc.spawn_vehicle.assert_not_called()
        for seed in range(10):
            c, a = self.make_context(seed=seed)
            c.cfg["tunnel_vehicle_dynamic"].update(
                random_behavior=True,
                behaviors=[
                    "vehicle_slow",
                    "vehicle_sudden_brake",
                    "vehicle_oncoming",
                    "vehicle_cut_in",
                ],
                additional_behaviors=["vehicle_oncoming"],
            )
            prepare_tunnel_vehicle(c)
            self.assertNotEqual(c.tunnel_vehicle_plan["behavior"], "vehicle_cut_in")

    def test_random_spawn_offset_replays_and_is_not_redrawn_while_waiting(self):
        def run(seed):
            c, _ = self.make_context(seed=seed)
            c.cfg["tunnel_vehicle_dynamic"]["spawn_forward_offset_range_m"] = [
                0.0,
                10.0,
            ]
            prepare_tunnel_vehicle(c)
            offset = c.tunnel_vehicle_plan["spawn_forward_offset_m"]
            h = c.tunnel_vehicle_handler
            h.arm(c)
            for t in (0, 10, 30):
                c.now = t
                self.start(c, h, ego_speed=0)
                self.assertEqual(
                    c.tunnel_vehicle_plan["spawn_forward_offset_m"], offset
                )
                c.grpc.spawn_vehicle.assert_not_called()
            self.start(c, h)
            timing = c.tunnel_vehicle_result["spawn_timing"]
            self.assertGreaterEqual(timing["initial_gap_m"], 40)
            self.assertLessEqual(timing["initial_gap_m"], 50.5)
            return offset, c.tunnel_vehicle_plan["spawn_s_m"]

        choices = [run(seed) for seed in range(6)]
        self.assertEqual(choices, [run(seed) for seed in range(6)])
        self.assertGreater(len({item[1] for item in choices}), 1)

    def test_oncoming_random_spawn_moves_inward_and_recomputes_eta(self):
        c, a = self.make_context("vehicle_oncoming")
        c.cfg["tunnel_vehicle_dynamic"]["spawn_forward_offset_range_m"] = [10.0, 10.0]
        prepare_tunnel_vehicle(c)
        p = c.tunnel_vehicle_plan
        self.assertEqual(p["nominal_spawn_s_m"], 127)
        self.assertEqual(p["spawn_s_m"], 117)
        self.assertEqual(p["spawn_target"]["y"], 3.5)
        self.assertEqual(p["spawn_target"]["yaw_deg"], 180)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h, ego_x=43, ego_speed=21.6)
        timing = c.tunnel_vehicle_result["spawn_timing"]
        anchor = p["interaction_s_m"]
        # This fixture has integer-metre MGeo waypoints; the anchor snaps to one.
        self.assertAlmostEqual(timing["npc_eta_sec"], (117 - round(anchor)) / 6)
        self.assertLessEqual(abs(timing["eta_delta_sec"]), 1.5)
        self.assertEqual(timing["applied_spawn_forward_offset_m"], 10)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_random_brake_spawn_preserves_deceleration_hold_resume_space(self):
        from ..scenarios.full_loop_parts.tunnel_vehicle_brake import (
            required_brake_space,
        )

        c, a = self.make_context("vehicle_sudden_brake")
        c.cfg["tunnel_vehicle_dynamic"]["spawn_forward_offset_range_m"] = [10.0, 10.0]
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h, ego_x=35, ego_speed=18)
        p = c.tunnel_vehicle_plan
        self.assertGreaterEqual(p["applied_spawn_forward_offset_m"], 0)
        self.assertLess(p["applied_spawn_forward_offset_m"], 10)
        self.assertGreaterEqual(
            p["exit_s_m"] - p["spawn_s_m"],
            required_brake_space(p["sudden_brake_settings"], 4),
        )
        self.assertGreaterEqual(c.tunnel_vehicle_result["initial_route_gap_m"], 40)
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)

    def test_invalid_spawn_offset_ranges_reject_before_morai(self):
        for value in ([10, 0], [-1, 10], [0, float("nan")], [10], "10"):
            c, _ = self.make_context()
            c.cfg["tunnel_vehicle_dynamic"]["spawn_forward_offset_range_m"] = value
            with self.assertRaisesRegex(ValueError, "spawn_forward_offset_range_m"):
                prepare_tunnel_vehicle(c)
            self.assertEqual(c.grpc.mock_calls, [])

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
        self.assertEqual(
            {choice[0] for choice in choices},
            {"vehicle_slow", "vehicle_sudden_brake", "vehicle_oncoming"},
        )
        self.assertEqual({choice[2] for choice in choices}, {"Sedan", "Van"})

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

    def test_tunnel_model_pool_excludes_live_drone_and_aerial_aliases(self):
        available = [
            "Sedan",
            "Van",
            "AR_DJMini3pro_2022",
            "ar_djmavic_2024",
            "Test_DRONE",
            "Prototype_eVTOL",
            "SE_CT797F_2009",
        ]
        selected = set()
        for seed in range(12):
            c, _ = self.make_context(seed=seed)
            c.vehicle_models = lambda: list(available)
            prepare_tunnel_vehicle(c)
            c.tunnel_vehicle_handler.arm(c)
            selected.add(c.tunnel_vehicle_plan["model"])
            pool = c.tunnel_vehicle_result["model_pool"]
            self.assertEqual(pool["candidates"], ["Sedan", "Van"])
            self.assertEqual(set(pool["excluded"]), set(available) - {"Sedan", "Van"})
            self.assertEqual(c.vehicle_models(), available)
            c.grpc.spawn_vehicle.assert_not_called()
        self.assertEqual(selected, {"Sedan", "Van"})

    def test_drone_cannot_bypass_filter_through_configured_or_fixed_model(self):
        for fixed in (False, True):
            c, _ = self.make_context()
            c.vehicle_models = lambda: ["Sedan", "AR_DJMini3pro_2022"]
            cfg = c.cfg["tunnel_vehicle_dynamic"]
            cfg["models"] = ["Sedan", "AR_DJMini3pro_2022"]
            if fixed:
                cfg["model"] = "AR_DJMini3pro_2022"
            prepare_tunnel_vehicle(c)
            if fixed:
                with self.assertRaisesRegex(RuntimeError, "model excluded"):
                    c.tunnel_vehicle_handler.arm(c)
            else:
                c.tunnel_vehicle_handler.arm(c)
                self.assertEqual(c.tunnel_vehicle_plan["model"], "Sedan")
            c.grpc.spawn_vehicle.assert_not_called()

    def test_all_aerial_model_pool_fails_before_spawn(self):
        c, _ = self.make_context()
        c.vehicle_models = lambda: ["AR_DJMini3pro_2022"]
        c.cfg["tunnel_vehicle_dynamic"]["models"] = []
        prepare_tunnel_vehicle(c)
        with self.assertRaisesRegex(RuntimeError, "models remain after exclusion"):
            c.tunnel_vehicle_handler.arm(c)
        c.grpc.spawn_vehicle.assert_not_called()

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

    def test_spawn_empty_link_waits_for_native_metadata_without_respawn(self):
        c, a = self.make_context()
        c.grpc.spawn_vehicle.side_effect = None
        a.get_actor_state.return_value = state(40, speed=4, link="")
        prepare_tunnel_vehicle(c)
        h = c.tunnel_vehicle_handler
        h.arm(c)
        self.start(c, h)
        self.assertFalse(c.tunnel_vehicle_result["spawn_verified"])
        self.assertFalse(c.tunnel_vehicle_result["startup_motion_verified"])
        self.assertTrue(c.tunnel_vehicle_result["actual_spawn"]["link_pending"])
        self.tick(c, a, h, .5, 42, 1, link="")
        self.assertFalse(c.tunnel_vehicle_result["spawn_verified"])
        self.tick(c, a, h, 1, 44, 2)
        self.assertTrue(c.tunnel_vehicle_result["spawn_verified"])
        self.assertTrue(c.tunnel_vehicle_result["startup_motion_verified"])
        self.assertEqual(c.grpc.spawn_vehicle.call_count, 1)
        a.destroy.assert_not_called()

    def test_missing_spawn_link_has_bounded_wait_and_still_checks_geometry(self):
        for bad in ("timeout", "position", "heading"):
            with self.subTest(bad=bad):
                c, a = self.make_context()
                c.grpc.spawn_vehicle.side_effect = None
                a.get_actor_state.return_value = state(
                    45 if bad == "position" else 40, link="",
                    yaw=180 if bad == "heading" else 0)
                prepare_tunnel_vehicle(c)
                h = c.tunnel_vehicle_handler
                h.arm(c)
                if bad == "timeout":
                    self.start(c, h)
                    with self.assertRaisesRegex(RuntimeError, "reported link unavailable"):
                        self.tick(c, a, h, 2.1, 40, 2, link="")
                else:
                    with self.assertRaises(RuntimeError):
                        self.start(c, h)
                self.assertFalse(c.tunnel_vehicle_result["spawn_verified"])
                self.assertFalse(c.tunnel_vehicle_result["completed"])
                a.destroy.assert_called_once()

    def test_random_vehicle_placeholder_is_excluded_from_tunnel_pool(self):
        c, _ = self.make_context()
        c.vehicle_models = lambda: ["Sedan", "RandomVehicle"]
        c.cfg["tunnel_vehicle_dynamic"]["models"] = []
        prepare_tunnel_vehicle(c)
        c.tunnel_vehicle_handler.arm(c)
        self.assertEqual(c.tunnel_vehicle_plan["model"], "Sedan")
        self.assertIn("RandomVehicle", c.tunnel_vehicle_result["model_pool"]["excluded"])

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
