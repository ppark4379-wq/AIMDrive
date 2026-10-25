"""Seeded tunnel mixes, model availability, and sequential actor lifecycle."""

import contextlib
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import main
from ..scenarios.full_loop_parts.tunnel_dynamic import (
    TunnelDynamicHandler, arm_tunnel_dynamic, prepare_tunnel_dynamic, update_tunnel_dynamic,
)
from ..scenarios.full_loop_parts.dynamic_obstacle import _timing_snapshot
from ..scenarios.full_loop_parts.tunnel_dynamic_catalog import (
    generate_tunnel_candidates, select_tunnel_mix,
)


BEHAVIORS = ["straight_same_direction", "straight_opposite", "cross_left_to_right",
             "cross_right_to_left", "diagonal_cross"]


def state_at(point):
    return SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(
        x=point["x"], y=point["y"])))


class TunnelRandomMixTest(unittest.TestCase):
    def make_context(self, seed=12345):
        grpc = Mock()
        grpc.get_available_pedestrian_models.return_value = ["Woman1", "Man1", "Outside"]
        grpc.control_pedestrian.return_value = True
        grpc.get_all_pedestrian_states.return_value = []
        spec = {
            "division_count": 4, "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5, "opposite_lane_1_offset_m": 2.5,
            "opposite_lane_2_offset_m": 0.0, "straight_length_m": 15.0,
            "cross_delta_s_m": 3.0, "diagonal_delta_s_m": 8.0,
            "approved_ids": [], "disabled_ids": [],
        }
        return SimpleNamespace(
            cfg={
                "drive_control_mode": "morai_keyboard",
                "dynamic_obstacle_models": ["Man1", "Woman1", "Unavailable"],
                "tunnel_dynamic": {
                    "enabled": True, "model": "Man1", "speed_mps": 2.5,
                    "arrival_tolerance_m": 0.5, "max_run_sec": 20.0,
                    "random_mix": {"enabled": True, "events_per_section": 1,
                                   "allow_behavior_repeat": False, "behaviors": BEHAVIORS},
                },
            },
            route_definition={"zones": {"tunnel": {"candidate_catalog": spec}}},
            route_points=[(float(i), 0.0, 10.0) for i in range(201)],
            tunnel_entry_s=10.0, tunnel_exit_s=190.0,
            grpc=grpc, run_seed=seed, run_instance_id=f"mix-{seed}", scenario_actors=[],
        )

    def candidates(self, context):
        return generate_tunnel_candidates(
            context.route_points, context.tunnel_entry_s, context.tunnel_exit_s,
            context.route_definition["zones"]["tunnel"]["candidate_catalog"])

    def test_seed_reproduces_behavior_and_model_with_available_pool(self):
        schedules = []
        for reverse_availability in (False, True):
            context = self.make_context()
            if reverse_availability:
                context.grpc.get_available_pedestrian_models.return_value.reverse()
            prepare_tunnel_dynamic(context)
            arm_tunnel_dynamic(context)
            plans = context.tunnel_dynamic_plans
            self.assertEqual([p["section_index"] for p in plans], [1, 2, 3, 4])
            self.assertEqual(len({p["behavior"] for p in plans}), 4)
            self.assertEqual(context.tunnel_dynamic_model_pool, ["Man1", "Woman1"])
            self.assertTrue(all(p["model"] in ("Man1", "Woman1") for p in plans))
            schedules.append([(p["path_id"], p["model"]) for p in plans])
        self.assertEqual(schedules[0], schedules[1])

    def test_different_seeds_vary_the_scene(self):
        schedules = set()
        for seed in range(5):
            context = self.make_context(seed)
            prepare_tunnel_dynamic(context)
            arm_tunnel_dynamic(context)
            schedules.add(tuple((p["path_id"], p["model"])
                                for p in context.tunnel_dynamic_plans))
        self.assertGreater(len(schedules), 1)

    def test_review_statuses_and_unique_assignment_are_respected(self):
        context = self.make_context()
        candidates = self.candidates(context)
        approved_ids = {"straight_same_direction_01", "straight_opposite_02",
                        "cross_left_to_right_03", "cross_right_to_left_04"}
        for item in candidates:
            item["status"] = "approved" if item["id"] in approved_ids else "disabled"
        selected = select_tunnel_mix(candidates, random.Random(1), BEHAVIORS)
        self.assertEqual({item["id"] for item in selected}, approved_ids)
        with self.assertRaisesRegex(ValueError, "No eligible tunnel mix"):
            select_tunnel_mix(self.candidates(context), random.Random(1), BEHAVIORS)
        context.cfg["drive_control_mode"] = "aimdrive_expert"
        with self.assertRaisesRegex(ValueError, "No eligible tunnel mix"):
            prepare_tunnel_dynamic(context)

    def test_backtracking_finds_feasible_constrained_schedule(self):
        context = self.make_context()
        candidates = self.candidates(context)
        allowed = {"straight_same_direction_01", "straight_opposite_01",
                   "straight_same_direction_02", "cross_left_to_right_03",
                   "cross_right_to_left_04"}
        for item in candidates:
            item["status"] = "approved" if item["id"] in allowed else "disabled"
        for seed in range(10):
            selected = select_tunnel_mix(candidates, random.Random(seed), BEHAVIORS)
            self.assertEqual(selected[0]["id"], "straight_opposite_01")
            self.assertEqual(selected[1]["id"], "straight_same_direction_02")

    def test_expert_trial_mix_uses_candidates_but_excludes_disabled_paths(self):
        context = self.make_context()
        context.cfg["drive_control_mode"] = "aimdrive_expert"
        context.cfg["tunnel_dynamic"]["candidate_policy"] = "trial"
        spec = context.route_definition["zones"]["tunnel"]["candidate_catalog"]
        spec["disabled_ids"] = ["straight_same_direction_01"]
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        self.assertEqual(len(context.tunnel_dynamic_plans), 4)
        self.assertNotIn("straight_same_direction_01",
                         [p["path_id"] for p in context.tunnel_dynamic_plans])
        self.assertTrue(all(p["review_status"] == "candidate"
                            for p in context.tunnel_dynamic_plans))
        self.assertEqual(spec["approved_ids"], [])

    def test_expert_single_candidate_works_only_with_trial_policy(self):
        context = self.make_context()
        context.cfg["drive_control_mode"] = "aimdrive_expert"
        context.cfg["tunnel_dynamic"]["trial_candidate_id"] = "straight_opposite_02"
        with self.assertRaisesRegex(RuntimeError, "requires trial policy"):
            prepare_tunnel_dynamic(context)
        context.cfg["tunnel_dynamic"]["candidate_policy"] = "trial"
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_plan["path_id"], "straight_opposite_02")

    def test_unavailable_model_pool_fails_before_spawning(self):
        context = self.make_context()
        context.cfg["tunnel_dynamic"]["random_mix"]["models"] = ["Unavailable"]
        prepare_tunnel_dynamic(context)
        with self.assertRaisesRegex(RuntimeError, "No configured tunnel pedestrian model"):
            arm_tunnel_dynamic(context)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_four_actors_complete_without_ego_slowdown_and_report_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.make_context()
            context.cfg["profile_state_path"] = str(Path(directory) / "profile.json")
            prepare_tunnel_dynamic(context)
            arm_tunnel_dynamic(context)
            elapsed = 0.0
            for index, plan in enumerate(context.tunnel_dynamic_plans):
                self.assertEqual(context.tunnel_dynamic_event_index, index)
                self.assertEqual(context.tunnel_dynamic_phase, "waiting")
                self.assertEqual(context.scenario_actors, [])
                actor = Mock()
                actor.get_id.return_value = f"actor-{index}"
                actor.destroy.return_value = True
                context.grpc.spawn_pedestrian.return_value = actor
                ego_s = plan["interaction_s"] - (13.0 if plan["behavior"].startswith("straight_") else 10.0)
                update_tunnel_dynamic(context, elapsed, {"speed_mps": 4.0}, ego_s)
                self.assertEqual(context.scenario_actors, [actor])
                start, end = plan["waypoints"]
                duration = plan["path_length_m"] / plan["speed_mps"]
                for step in range(1, math.ceil(duration) + 1):
                    ratio = min(1.0, step / duration)
                    actor.get_actor_state.return_value = state_at({
                        "x": start["x"] + ratio * (end["x"] - start["x"]),
                        "y": start["y"] + ratio * (end["y"] - start["y"]),
                    })
                    update_tunnel_dynamic(context, elapsed + step, {"speed_mps": 4.0}, ego_s)
                elapsed += math.ceil(duration) + 1
                actor.destroy.assert_called_once()
                self.assertEqual(context.scenario_actors, [])
            self.assertEqual(context.tunnel_dynamic_phase, "completed")
            self.assertEqual(context.grpc.spawn_pedestrian.call_count, 4)
            context.grpc.set_pedestrian_waypoint.assert_not_called()
            labels = [call.kwargs["label"] for call in context.grpc.spawn_pedestrian.call_args_list]
            self.assertEqual(len(set(labels)), 4)
            report = json.loads(context.tunnel_dynamic_report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["completed"])
            self.assertFalse(report["ego_response_affects_completion"])
            self.assertFalse(report["interaction_verified"])
            self.assertEqual(report["run_seed"], 12345)
            self.assertTrue(all(item["completed"] for item in report["events"]))
            self.assertTrue(any(item["interaction_required"] for item in report["events"]))
            self.assertTrue(all(not item["interaction_verified"] for item in report["events"]))
            self.assertTrue(all(item["interaction_assessment"] == "not_observed"
                                for item in report["events"] if item["interaction_required"]))

    def test_timeout_cleans_actor_and_does_not_advance(self):
        context = self.make_context()
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        actor = Mock()
        actor.get_id.return_value = "first-actor"
        actor.destroy.return_value = True
        context.grpc.spawn_pedestrian.return_value = actor
        update_tunnel_dynamic(context, 0.0, {"speed_mps": 4.0}, context.tunnel_dynamic_plan["interaction_s"] - 13.0)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            update_tunnel_dynamic(context, 21.0, {}, 0.0)
        actor.destroy.assert_called_once()
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertEqual(context.tunnel_dynamic_event_index, 0)
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)

    def arm_candidate(self, candidate_id="straight_same_direction_01"):
        context = self.make_context()
        context.cfg["tunnel_dynamic"]["trial_candidate_id"] = candidate_id
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        return context

    def test_faster_ego_spawns_further_from_the_same_interaction(self):
        # At 26 m the slow car is outside its window; the faster car is
        # 0.25 s behind the pedestrian's 3 s approach to the anchor.
        for speed, expected_spawn in ((4.0, False), (8.0, True)):
            context = self.arm_candidate()
            update_tunnel_dynamic(context, 1.0, {"speed_mps": speed}, 14.0)
            self.assertEqual(context.grpc.spawn_pedestrian.called, expected_spawn)
            if expected_spawn:
                trigger = context.tunnel_dynamic_result["trigger"]
                self.assertAlmostEqual(trigger["remaining_m"], 26.0)
                self.assertAlmostEqual(trigger["npc_eta_sec"], 3.0)
                self.assertAlmostEqual(trigger["eta_delta_sec"], .25)
                self.assertEqual(context.tunnel_dynamic_result["trigger_s"], 14.0)

    def test_stopped_ego_waits_even_inside_the_distance_window(self):
        context = self.arm_candidate()
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 0.0}, 32.0)
        self.assertEqual(context.tunnel_dynamic_phase, "waiting")
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_too_early_eta_waits_then_spawns_when_eta_matches(self):
        context = self.make_context()
        context.cfg["tunnel_dynamic"].update(
            trial_candidate_id="cross_left_to_right_02",
            interaction_trigger={"interaction_lead_sec": 5.0},
        )
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 4.0}, 57.0)
        context.grpc.spawn_pedestrian.assert_not_called()
        update_tunnel_dynamic(context, 2.0, {"speed_mps": 4.0}, 68.0)
        context.grpc.spawn_pedestrian.assert_called_once()

    def test_unsafe_or_missed_window_does_not_spawn(self):
        for speed, ego_s, reason in (
            (4.0, 32.0, "safe_stop"),
            (0.0, 42.0, "missed_trigger"),
        ):
            with self.subTest(speed=speed, ego_s=ego_s):
                context = self.arm_candidate()
                with self.assertRaisesRegex(RuntimeError, reason):
                    update_tunnel_dynamic(context, 1.0, {"speed_mps": speed}, ego_s)
                context.grpc.spawn_pedestrian.assert_not_called()
                self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_centerline_path_keeps_its_late_arrival_guard(self):
        context = self.arm_candidate("straight_opposite_02")
        plan = context.tunnel_dynamic_plan
        # Just outside stopping distance, but Ego reaches the intersection
        # anchor before this centerline walker can get there.
        with self.assertRaisesRegex(RuntimeError, "NPC would arrive after Ego"):
            update_tunnel_dynamic(context, 1.0, {"speed_mps": 2.0}, plan["interaction_s"] - 5.8)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_crossing_uses_actual_intersection_on_a_bent_route(self):
        context = SimpleNamespace(
            route_points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
                          (20.0, 10.0, 0.0), (30.0, 10.0, 0.0)],
            tunnel_entry_s=0.0, tunnel_exit_s=34.0,
        )
        points = [{"x": 5.0, "y": 3.0, "z": 0.0},
                  {"x": 15.0, "y": -2.0, "z": 0.0}]
        interaction = TunnelDynamicHandler()._interaction(context, points)
        self.assertEqual(interaction["interaction_kind"], "route_crossing")
        self.assertAlmostEqual(interaction["interaction_point"]["x"], 31.0 / 3.0)
        self.assertAlmostEqual(interaction["interaction_point"]["y"], 1.0 / 3.0)
        self.assertAlmostEqual(interaction["interaction_s"], 10.0 + math.sqrt(200.0) / 30.0)
        self.assertAlmostEqual(interaction["start_to_interaction_m"], math.sqrt(125.0) * 8.0 / 15.0)

    def test_straight_opposite_uses_anchor_and_not_spawn_start(self):
        context = self.arm_candidate("straight_opposite_02")
        plan = context.tunnel_dynamic_plan
        self.assertEqual(plan["interaction_kind"], "anchor")
        self.assertEqual(plan["interaction_s"], 80.0)
        self.assertEqual(plan["start_to_interaction_m"], 7.5)
        self.assertEqual(plan["waypoints"][0]["x"], 87.5)

    def test_trigger_inherits_general_dynamic_settings_and_validates_overrides(self):
        context = self.make_context()
        context.cfg["dynamic_obstacle"] = {"reaction_time_sec": 2.0, "maximum_eta_delta_sec": 4.0}
        context.cfg["tunnel_dynamic"]["interaction_trigger"] = {"maximum_eta_delta_sec": 2.0}
        prepare_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_trigger_cfg["reaction_time_sec"], 2.0)
        self.assertEqual(context.tunnel_dynamic_trigger_cfg["maximum_eta_delta_sec"], 2.0)
        for override in ({"comfortable_deceleration_mps2": 0.0},
                         {"maximum_eta_delta_sec": -1.0},
                         {"minimum_ego_speed_mps": float("nan")},
                         {"minimum_trigger_distance_m": True},
                         {"maximum_trigger_distance_m": 1.0, "minimum_trigger_distance_m": 2.0}):
            with self.subTest(override=override):
                context.cfg["tunnel_dynamic"]["interaction_trigger"] = override
                with self.assertRaises((ValueError, RuntimeError)):
                    prepare_tunnel_dynamic(context)

    def test_general_dynamic_keeps_its_existing_distance_floor(self):
        timing = _timing_snapshot(
            SimpleNamespace(cfg={}),
            {"route_s": 100.0, "start_to_conflict_m": 2.5,
             "speed_mps": 2.5, "trigger_distance_m": 25.0},
            {"speed_mps": 5.0}, 80.0,
        )
        self.assertEqual(timing["trigger_distance_m"], 25.0)
        self.assertEqual(timing["eta_delta_sec"], 3.0)
        self.assertAlmostEqual(timing["safe_stop_distance_m"], 8.0 + 25.0 / 7.0)

    def test_cli_random_mix_and_zero_seed_reach_configuration(self):
        args = ["runner", "--scenario", "full_loop_tunnel_dynamic", "--once", "--manual-ego",
                "--tunnel-random-mix", "--seed", "0", "--check"]
        with patch("sys.argv", args), patch(
                "scenario_runner.main_runner.load_configuration", return_value=({}, {}, {})), patch(
                "scenario_runner.main_runner.run_offline_check") as check:
            main()
        config = check.call_args.args[1]
        self.assertEqual(config["random_seed"], 0)
        self.assertTrue(config["tunnel_dynamic"]["random_mix"]["enabled"])
        self.assertEqual(config["drive_control_mode"], "morai_keyboard")

    def test_cli_tunnel_trials_use_expert_without_manual_flag(self):
        for flags in (["--tunnel-random-mix"],
                      ["--tunnel-candidate-id", "straight_opposite_02"]):
            with self.subTest(flags=flags):
                args = ["runner", "--scenario", "full_loop_tunnel_dynamic", "--once",
                        "--no-sync-runtime", "--check", *flags]
                with patch("sys.argv", args), patch(
                        "scenario_runner.main_runner.load_configuration",
                        return_value=({}, {}, {"drive_control_mode": "aimdrive_expert"})), patch(
                        "scenario_runner.main_runner.run_offline_check") as check:
                    main()
                config = check.call_args.args[1]
                self.assertEqual(config["drive_control_mode"], "aimdrive_expert")
                self.assertEqual(config["tunnel_dynamic"]["candidate_policy"], "trial")
                self.assertFalse(config["sync_runtime"]["enabled"])

    def test_cli_candidate_and_random_mix_are_exclusive(self):
        args = ["runner", "--tunnel-random-mix", "--tunnel-candidate-id", "straight_opposite_02"]
        with patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
