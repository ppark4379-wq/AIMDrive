"""Tunnel regression checks: profiles."""

import contextlib, io, json, math, random, tempfile, unittest, copy, hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from ..main_runner import main
from ..scenarios.full_loop_parts.tunnel_dynamic import TunnelDynamicHandler, arm_tunnel_dynamic, prepare_tunnel_dynamic, update_tunnel_dynamic
from ..scenarios.full_loop_parts.dynamic_obstacle import _timing_snapshot
from ..scenarios.full_loop_parts.tunnel_dynamic_catalog import generate_tunnel_candidates, select_tunnel_mix
from contextlib import redirect_stdout, ExitStack
from ..adapters.expert_controller import ExpertController, expert_max_velocity_contract, resolve_expert_max_velocity, validate_max_velocity_kmh
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.segment_modes import events
from ..scenarios.full_loop_parts.tunnel_static_parts import trial as trial_cli
from ..tools.tunnel_static import batch


# Seeded tunnel mixes, model availability, and sequential actor lifecycle.

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


# Tunnel speed comes from the unchanged Expert source, with no fixed cap.

class TunnelSpeedInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expert = self.root/"fake_expert"
        self.config_path = self.expert/"src/config/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.config = {"map":{"name":"original", "data_dir":"original-map", "use_mgeo_path":True,
                              "traffic_light_control":True,"is_closed_path":True},
                       "planning":{"velocity_profile":{"max_velocity":37,"window_size":50,"road_friction":.15,
                                                           "terminal_deceleration":2.,"speed_limit_zones":[]}},
                       "control":{"controller_gain":123}}
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def controller(self, speed=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return ExpertController(str(self.expert),str(self.root/"runtime"),"test-map",str(self.root/"map"),
                                    [[0.,0.],[1.,0.]],max_velocity_kmh=speed,**kwargs)

    def test_none_inherits_non_60_original_and_preserves_the_original_field_and_file(self):
        before = self.config_path.read_bytes()
        controller = self.controller()
        runtime = json.loads(Path(controller.runtime_config_path).read_text())
        self.assertEqual(runtime["planning"],self.config["planning"])
        self.assertIs(type(runtime["planning"]["velocity_profile"]["max_velocity"]),int)
        self.assertEqual(controller.max_velocity_kmh,37.)
        self.assertTrue(controller.speed_contract["inherited"])
        self.assertEqual(controller.speed_contract["expert_config_sha256"],hashlib.sha256(before).hexdigest())
        self.assertEqual(runtime["control"],self.config["control"])
        self.assertEqual(self.config_path.read_bytes(),before)

    def test_explicit_other_zone_override_keeps_existing_export_semantics(self):
        controller = self.controller(60.,velocity_profile_window_size=15)
        runtime = json.loads(Path(controller.runtime_config_path).read_text())
        self.assertEqual(runtime["planning"]["velocity_profile"]["max_velocity"],60.)
        self.assertEqual(runtime["planning"]["velocity_profile"]["window_size"],15)
        self.assertEqual(runtime["planning"]["velocity_profile"]["road_friction"],.15)
        self.assertEqual(runtime["control"],self.config["control"])
        self.assertFalse(controller.speed_contract["inherited"])
        self.assertEqual(json.loads(self.config_path.read_text()),self.config)

    def test_each_invocation_reads_current_original_and_pins_its_source(self):
        first = expert_max_velocity_contract(self.expert)
        self.config["planning"]["velocity_profile"]["max_velocity"] = 43
        self.write_config()
        second = expert_max_velocity_contract(self.expert)
        self.assertEqual(first["resolved_max_velocity_kmh"],37.)
        self.assertEqual(second["resolved_max_velocity_kmh"],43.)
        self.assertNotEqual(first["expert_config_sha256"],second["expert_config_sha256"])
        self.assertEqual(resolve_expert_max_velocity(self.expert),43.)

    def test_inherited_and_explicit_values_require_positive_finite_numbers(self):
        for value in (0,-1,float("nan"),float("inf"),True,"bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):validate_max_velocity_kmh(value)
                self.config["planning"]["velocity_profile"]["max_velocity"] = value
                self.write_config()
                with self.assertRaises(ValueError):self.controller()
        self.assertEqual(validate_max_velocity_kmh(37),37.)
        self.assertEqual(validate_max_velocity_kmh(100),100.)

    def context(self,mode,speed="missing"):
        cfg={"mission_mode":mode,"profile_state_path":str(self.root/"profile.json")}
        if speed != "missing":cfg["expert_max_velocity_kmh"]=speed
        return SimpleNamespace(cfg=cfg,global_cfg={"paths":{"expert_root":str(self.expert),"runtime_root":str(self.root),
                                                           "mgeo_root":str(self.root/"map")}},
                               route_points=[[0.,0.],[1.,0.]],start_link="start",end_link="end")

    def test_core_inherits_missing_speed_and_keeps_explicit_values(self):
        for mode,speed,expected in (("full_loop_tunnel_static",None,None),
                                   ("full_loop_tunnel_static","missing",None),
                                   ("full_loop_static",60,60.),("full_loop_dynamic",37,37.),
                                   ("full_loop_roundabout","missing",None)):
            with self.subTest(mode=mode,speed=speed),patch("scenario_runner.core.full_loop_drive_base.ExpertController") as constructor:
                context=self.context(mode,speed)
                FullLoopDriveBase.setup_expert_controller(context)
                self.assertEqual(constructor.call_args.kwargs["max_velocity_kmh"],expected)

    def test_tunnel_events_keep_all_guard_settings_without_clamping_speed(self):
        context=self.context("full_loop_tunnel_static",None)
        with redirect_stdout(io.StringIO()):events.prepare_mission(context)
        self.assertIsNone(context.cfg["expert_max_velocity_kmh"])
        self.assertEqual(context.cfg["timeout_sec"],180.)
        self.assertEqual(context.cfg["expert_max_cross_track_error_m"],5.)
        self.assertEqual(context.cfg["off_route_timeout_sec"],3.)
        self.assertEqual(context.cfg["progress_watchdog"],{"enabled":True,"timeout_sec":45.,
                                                         "startup_grace_sec":30.,"minimum_progress_m":2.})
        context=self.context("full_loop_tunnel_static",37)
        with redirect_stdout(io.StringIO()):events.prepare_mission(context)
        self.assertEqual(context.cfg["expert_max_velocity_kmh"],37)

    def test_normal_events_do_not_change_normal_speed(self):
        for name in ("full_loop_static","full_loop_dynamic","full_loop_roundabout"):
            context=self.context(name,60)
            before=copy.deepcopy(context.cfg)
            with redirect_stdout(io.StringIO()):events.prepare_mission(context)
            self.assertEqual(context.cfg["expert_max_velocity_kmh"],before["expert_max_velocity_kmh"])

    def trial_args(self,extra=()):
        candidate=self.root/"candidate.json"
        candidate.write_text(json.dumps({"count":1,"obstacles":[{"actor_kind":"obstacle","model":"Box"}]}))
        return trial_cli.build_parser().parse_args(["--candidate",str(candidate),"--output",str(self.root/"trial"),*extra])

    def test_trial_cli_defaults_to_inheritance_and_accepts_original_speed_above_15(self):
        args=self.trial_args()
        self.assertIsNone(args.expert_max_velocity_kmh)
        trial_cli.validate_trial_settings(args)
        args=self.trial_args(("--expert-max-velocity-kmh","37"))
        trial_cli.validate_trial_settings(args)
        args.expert_max_velocity_kmh=float("nan")
        with self.assertRaises(ValueError):trial_cli.validate_trial_settings(args)

    def run_mock_trial(self,args,preview_effect=None):
        grpc=Mock()
        grpc.world=object()
        grpc.get_available_obstacle_models.return_value=["Box"]
        grpc.get_available_surround_vehicle_models.return_value=[]
        callback=Mock(return_value=lambda *a:None)
        configuration={"paths":{"expert_root":str(self.expert),"mgeo_root":str(self.root/"map"),
                                "runtime_root":str(self.root)},"grpc":{}}
        base={"expert_max_velocity_kmh":60.,"profile_state_path":"legacy.json","random_seed":99}
        with ExitStack() as stack:
            stack.enter_context(patch.object(trial_cli,"load_configuration",return_value=(configuration,{},base)))
            stack.enter_context(patch.object(trial_cli,"MGeoMapLoader"))
            stack.enter_context(patch.object(trial_cli,"validate_candidate"))
            stack.enter_context(patch.object(trial_cli,"_controller_publishers",return_value=[]))
            stack.enter_context(patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient",return_value=grpc))
            for name in ("block_and_park", "park_ego"):
                stack.enter_context(patch.object(trial_cli, name))
            stack.enter_context(patch.object(trial_cli, "prepare_common_start", autospec=True))
            stack.enter_context(patch.object(trial_cli,"make_drive_callback",callback))
            stack.enter_context(patch.object(trial_cli,"preview_candidate",side_effect=preview_effect,
                return_value={"status":"valid_driving_trial","expert_verified":True,"live_avoidance_verified":True}))
            report=trial_cli.run(args)
        return report,callback.call_args.args[3]

    def test_cli_delivers_inheritance_seed_and_persistent_profile_to_callback(self):
        profile=self.root/"shared/full_loop_tunnel_static_profile.json"
        args=self.trial_args(("--run-seed","731","--profile-state-path",str(profile),"--console-output"))
        report,cfg=self.run_mock_trial(args)
        self.assertIsNone(cfg["expert_max_velocity_kmh"])
        self.assertEqual(cfg["random_seed"],731)
        self.assertEqual(cfg["profile_state_path"],str(profile.resolve()))
        self.assertTrue(cfg["tunnel_static_profile_state_path_explicit"])
        self.assertTrue(cfg["tunnel_static_console_output"])
        self.assertEqual(report["expert_speed_source"]["resolved_max_velocity_kmh"],37.)
        self.assertTrue(report["expert_speed_source"]["inherited"])

    def test_legacy_cli_keeps_seed_and_history_untouched_when_not_supplied(self):
        _,cfg=self.run_mock_trial(self.trial_args())
        self.assertEqual(cfg["random_seed"],99)
        self.assertEqual(cfg["profile_state_path"],"legacy.json")
        self.assertNotIn("tunnel_static_profile_state_path_explicit",cfg)

    def test_changed_original_config_during_trial_cannot_be_reported_passed(self):
        def changed(*args,**kwargs):
            self.config["planning"]["velocity_profile"]["max_velocity"]=43
            self.write_config()
            return {"status":"valid_driving_trial","expert_verified":True,"live_avoidance_verified":True}
        with self.assertRaisesRegex(RuntimeError,"speed source changed"):
            self.run_mock_trial(self.trial_args(),changed)
        saved=json.loads((self.root/"trial/manifest.json").read_text())
        self.assertEqual(saved["status"],"failed")

    def test_grouped_batch_omits_inherited_speed_and_preserves_explicit_source_value(self):
        args=SimpleNamespace(max_velocity=None,hold_seconds=5.,time_budget=15.)
        context=SimpleNamespace(args=args)
        command=batch.Batch.trial_command(context,Path("candidate.json"),Path("trial"))
        self.assertNotIn("--expert-max-velocity-kmh",command)
        args.max_velocity=37.
        command=batch.Batch.trial_command(context,Path("candidate.json"),Path("trial"))
        self.assertEqual(command[command.index("--expert-max-velocity-kmh")+1],"37.0")
        self.assertIsNone(batch.build_parser().get_default("max_velocity"))


if __name__ == "__main__":
    unittest.main()
