import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scenario_runner" / "tools" / "generate_static_scenario_suite.py"
SPEC = importlib.util.spec_from_file_location("static_scenario_batch", MODULE_PATH)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)

REVIEW_PATH = ROOT / "scenario_runner" / "tools" / "review_static_scenario_candidate.py"
REVIEW_SPEC = importlib.util.spec_from_file_location("static_scenario_review", REVIEW_PATH)
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)


class StaticScenarioBatchTest(unittest.TestCase):
    def policy(self):
        return {
            "models": ["A", "B", "C", "D"],
            "scale_overrides": {"C": [1.0, 1.0, 1.0]},
            "fallback_scale": [2.0, 3.0, 2.0],
            "anchor_start": {
                "route_path_index": 100,
                "position": {"x": 0.0, "y": 0.0, "z": 2.0},
                "yaw_deg": 10.0,
            },
            "anchor_end": {
                "route_path_index": 120,
                "position": {"x": 0.0, "y": 10.0, "z": 4.0},
                "yaw_deg": 10.0,
            },
            "map_name": "MAP",
        }

    def test_hundred_requested_layouts_are_unique_and_bounded(self):
        rng = random.Random(20260920)
        signatures = set()
        counts = set()
        selected_models = set()
        for _ in range(100):
            count = rng.randint(1, 3)
            obstacles = batch.generate_layout(
                self.policy(), rng, count, 0.6, 0.0, 0.0
            )
            signature = batch.layout_signature(obstacles)
            self.assertNotIn(signature, signatures)
            signatures.add(signature)
            counts.add(count)
            selected_models.update(item["model"] for item in obstacles)
            self.assertEqual(len({item["model"] for item in obstacles}), count)
            for item in obstacles:
                self.assertGreaterEqual(item["position"]["y"], 0.0)
                self.assertLessEqual(item["position"]["y"], 10.0)
                self.assertGreaterEqual(item["position"]["x"], -0.6)
                self.assertLessEqual(item["position"]["x"], 0.6)
                self.assertEqual(item["yaw_deg"], 10.0)
                self.assertEqual(item["scale"], [1.0, 1.0, 1.0])
                self.assertEqual(item["generation"]["scale_policy"], "morai_native_model_scale_v1")
                self.assertEqual(item["generation"]["scale_multiplier"], 1.0)
        self.assertEqual(counts, {1, 2, 3})
        self.assertEqual(selected_models, {"A", "B", "C", "D"})

    def test_required_model_is_always_present_without_duplication(self):
        for count in (1, 2, 3):
            obstacles = batch.generate_layout(
                self.policy(), random.Random(100 + count), count, 0.6, 0.0,
                required_model="D",
            )
            models = [item["model"] for item in obstacles]
            self.assertIn("D", models)
            self.assertEqual(len(models), len(set(models)))
        with self.assertRaisesRegex(ValueError, "outside the generation policy"):
            batch.generate_layout(
                self.policy(), random.Random(1), 2, 0.6, 0.0,
                required_model="missing",
            )

    def test_random_scale_rejected_before_any_trial_or_output(self):
        from unittest.mock import patch
        self.assertEqual(batch.parse_args([]).scale_jitter, 0.0)
        self.assertEqual(batch.parse_args([]).yaw_margin_deg, 0.0)
        for jitter in (0.1, -0.1, float("nan"), float("inf")):
            with self.subTest(jitter=jitter):
                with self.assertRaisesRegex(ValueError, "scale is disabled"):
                    batch.generate_layout(self.policy(), random.Random(1), 2, 0.6, 0, jitter)
                with patch.object(batch, "load_generation_policy") as load_policy:
                    with self.assertRaisesRegex(SystemExit, "scale is disabled"):
                        batch.main(["--scale-jitter", str(jitter)])
                    load_policy.assert_not_called()

    def test_legacy_enlargement_cannot_reach_new_candidates(self):
        policy = self.policy()
        policy["models"] = ["C"]
        policy["scale_overrides"]["C"] = [2.0, 3.0, 2.0]
        original = copy.deepcopy(policy)
        result = batch.generate_layout(policy, random.Random(1), 1, 0.6, 0)
        self.assertEqual(result[0]["scale"], [1.0, 1.0, 1.0])
        self.assertEqual(policy, original)

    def test_native_policy_ignores_legacy_config_and_route_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            route = Path(directory) / "route.yaml"
            config.write_text(yaml.safe_dump({"scenarios": {"full_loop": {
                "static_obstacle_models": ["CargoBox", "WoodBox"],
                "static_obstacle_model_scales": {"CargoBox": [2, 3, 2]},
            }}}))
            route.write_text(yaml.safe_dump({"full_loop": {"zones": {
                "static_obstacle": {"scale": [2, 3, 2], "multi_positions": [
                    self.policy()["anchor_start"], self.policy()["anchor_end"],
                ]},
            }}}))
            originals = (config.read_bytes(), route.read_bytes())
            policy = batch.load_generation_policy(config, route)
            result = batch.generate_layout(policy, random.Random(1), 2, 0.6, 0)
            self.assertTrue(all(item["scale"] == [1, 1, 1] for item in result))
            self.assertEqual((config.read_bytes(), route.read_bytes()), originals)

    def test_candidate_is_not_marked_verified(self):
        obstacles = batch.generate_layout(
            self.policy(), random.Random(1), 2, 0.5, 5.0, 0.0
        )
        document = batch.build_candidate_document(
            "trial", 1, 7, obstacles, self.policy()
        )
        self.assertEqual(document["schema"], "morai_static_obstacle_preset_v1")
        self.assertTrue(
            document["generation"]["candidate_only_until_expert_passes"]
        )
        self.assertNotIn("verification", document)

    def test_repository_paths_are_stored_portably(self):
        repository_file = batch.ROOT / "scenario_runner" / "README.md"
        self.assertEqual(
            batch.portable_path(repository_file),
            "scenario_runner/README.md",
        )
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "external.yaml"
            self.assertTrue(Path(batch.portable_path(external)).is_absolute())

    def test_atomic_yaml_is_replayable_by_existing_schema(self):
        obstacles = batch.generate_layout(
            self.policy(), random.Random(2), 3, 0.5, 5.0, 0.0
        )
        document = batch.build_candidate_document(
            "trial", 2, 1, obstacles, self.policy()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial.yaml"
            batch.write_yaml_atomic(output, document)
            loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["obstacles"], obstacles)
        self.assertEqual(loaded["map_name"], "MAP")

    def test_scenario_only_bridge_mode_skips_capture_sensor_gate(self):
        source = (
            ROOT / "data_collection" / "tools" /
            "01_internal_sensor_record_preflight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--scenario-only", source)
        self.assertIn("rosparam set /use_sim_time false", source)
        self.assertIn("시나리오 검증용 bridge를 계속 유지", source)
        self.assertIn("/ctrl_cmd subscriber 확인", source)
        self.assertIn("ctrl_cmd_missing_count", source)
        self.assertIn('if [[ "$RUN_MODE" == "capture" ]]', source)

        launcher = (
            ROOT / "scenario_runner" / "tools" /
            "run_static_scenario_batch.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("01_internal_sensor_record_preflight.sh --scenario-only", launcher)
        self.assertIn("static_scenario_batch_preflight", launcher)
        self.assertIn("/ctrl_cmd 구독자가 없습니다", launcher)

    def test_closed_loop_evidence_requires_goal_plan_and_tracks_max_cte(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[PLAN] candidate dir=L valid=True cost=1\n"
                "[STATIC] {'selected_routes': ['right_02.csv', 'right_03.csv']}\n"
                "[KCityFullLoop] lap=1 cte=0.42 remain=10\n"
                "[KCityFullLoop] lap=1 cte=2.31 remain=2\n"
                "[KCityFullLoop] GOAL REACHED lap=1\n",
                encoding="utf-8",
            )
            evidence = batch.read_closed_loop_evidence(log)
        self.assertTrue(evidence["goal_reached"])
        self.assertEqual(evidence["valid_plan_samples"], 1)
        self.assertEqual(evidence["cross_track_samples"], 2)
        self.assertAlmostEqual(evidence["max_cross_track_error_m"], 2.31)
        self.assertEqual(
            evidence["selected_routes"],
            ["right_02.csv", "right_03.csv"],
        )
        self.assertEqual(
            batch.classify_selected_routes(evidence["selected_routes"]),
            "alternate",
        )

    def test_accepted_route_target_parser(self):
        self.assertEqual(batch.parse_route_targets("11,11,0"), {
            "short": 11, "alternate": 11, "wide": 0,
        })
        with self.assertRaisesRegex(ValueError, "three"):
            batch.parse_route_targets("11,11")

    def test_live_planner_health_detects_latest_tracking_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'driving', 'error': None}\n"
                "[PLAN] candidate dir=R valid=True cost=1\n"
                "[STATIC] {'phase': 'blocked', "
                "'error': 'tracking_error'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_invalid_plan"])
        self.assertIn("tracking_error", health["static_status"])

    def test_live_planner_health_detects_recorded_envelope_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'blocked', "
                "'error': 'recorded_envelope'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_invalid_plan"])

    def test_live_planner_health_detects_persistent_signal_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'follow', 'path_validated': True, "
                "'speed_source': 'signal', "
                "'final_target_velocity': 0.0}\n"
                "[PLAN] candidate dir=? valid=True cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_signal_stop"])
        self.assertFalse(health["persistent_invalid_plan"])

    def test_live_planner_health_clears_after_valid_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'blocked', "
                "'error': 'tracking_error'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n"
                "[STATIC] {'phase': 'driving', 'error': None}\n"
                "[PLAN] candidate dir=L valid=True cost=1\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertFalse(health["persistent_invalid_plan"])
        self.assertFalse(health["persistent_signal_stop"])

    def test_manual_review_can_promote_only_cross_track_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            candidate = suite / "runtime" / "candidate.yaml"
            candidate.parent.mkdir()
            candidate.write_text(yaml.safe_dump({
                "schema": "morai_static_obstacle_preset_v1",
                "name": "candidate",
                "generation": {"candidate_only_until_expert_passes": True},
                "verification": {"expert_completed": True},
                "obstacles": [],
            }), encoding="utf-8")
            manifest_path = suite / "suite_manifest.json"
            manifest_path.write_text(__import__("json").dumps({
                "attempts": [{
                    "attempt": 1,
                    "accepted": False,
                    "candidate": str(candidate),
                    "models": ["ElectricScooter2"],
                    "return_code": 0,
                    "timed_out": False,
                    "rejection_reasons": ["cross_track_limit_exceeded"],
                    "closed_loop_evidence": {
                        "goal_reached": True,
                        "valid_plan_samples": 4,
                        "max_cross_track_error_m": 4.2,
                    },
                }],
                "accepted_count": 0,
                "accepted_model_counts": {"ElectricScooter2": 0},
                "accepted_model_minimum": 1,
                "accepted_model_minimum_met": False,
                "rejected_count": 1,
                "accepted_presets": [],
            }), encoding="utf-8")
            output = review.promote_candidate(
                manifest_path, 1, "tester", "MORAI에서 실선 침범 없음 확인"
            )
            promoted = yaml.safe_load(output.read_text(encoding="utf-8"))
            manifest = __import__("json").loads(
                manifest_path.read_text(encoding="utf-8")
            )
        self.assertEqual(
            promoted["source"], "manually_reviewed_static_expert_verified"
        )
        self.assertFalse(promoted["verification"]["automatic_gate_passed"])
        self.assertEqual(manifest["manual_accepted_count"], 1)
        self.assertEqual(manifest["accepted_model_counts"]["ElectricScooter2"], 1)
        self.assertTrue(manifest["accepted_model_minimum_met"])
        self.assertEqual(manifest["rejected_count"], 0)

    def test_manual_review_cannot_override_collision_or_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            candidate = suite / "runtime" / "candidate.yaml"
            candidate.parent.mkdir()
            candidate.write_text("name: candidate\n", encoding="utf-8")
            manifest_path = suite / "suite_manifest.json"
            manifest_path.write_text(__import__("json").dumps({
                "attempts": [{
                    "attempt": 1,
                    "accepted": False,
                    "candidate": str(candidate),
                    "return_code": 1,
                    "timed_out": False,
                    "rejection_reasons": ["runner_return_code"],
                    "closed_loop_evidence": {
                        "goal_reached": False,
                        "valid_plan_samples": 0,
                    },
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot manually override"):
                review.promote_candidate(
                    manifest_path, 1, "tester", "보기에 괜찮음"
                )


class LaneAwarePlacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = batch.load_lane_policy(batch.load_generation_policy())

    def test_models_are_limited_to_the_verified_footprint_catalog(self):
        catalog = batch.load_footprints()["models"]
        self.assertTrue(self.policy["models"])
        self.assertTrue(set(self.policy["models"]) <= set(catalog))
        self.assertIn("Color_Cone", self.policy["models_without_footprint"])
        self.assertNotIn("Color_Cone", self.policy["models"])

    def test_layouts_follow_their_trajectory_target_on_both_sides(self):
        rng = random.Random(3)
        strata = {}
        route, first_index = self.policy["route"]
        yaw = -83.219
        pools = {target: batch.target_models_for(self.policy, target, yaw)
                 for target in batch.AVOIDANCE_TARGETS}
        for attempt in range(160):
            target = batch.AVOIDANCE_TARGETS[attempt % 4]
            count = 1 + attempt % 3 if target != "late" else 2 + attempt % 2
            required = pools[target][attempt % len(pools[target])]
            try:
                obstacles = batch.generate_lane_layout(self.policy, rng, count, target, required, strata)
            except RuntimeError:
                # Two wide right-side boxes cannot stand side by side at the zone start.
                self.assertEqual((target, count), ("late", 3))
                self.assertIn(required, ("NCAP_GVT", "Steel_Barricade"))
                continue
            self.assertEqual(len(obstacles), count)
            self.assertIn(required, [item["model"] for item in obstacles])
            self.assertEqual(len({item["model"] for item in obstacles}), count)
            generation = [item["generation"] for item in obstacles]
            roles = [row["placement_role"] for row in generation]
            names = [row["lateral_stratum"] for row in generation]
            if target == "straight":
                # Every box leaves right_01 (0.3 m) clear; several mix near and far.
                self.assertEqual(set(roles), {"clear_01"})
                self.assertTrue(all(row["right_01_right_clearance_m"] >= batch.MEDIUM_CLEARANCE_M - 1e-3
                                    for row in generation))
                if count > 1:
                    self.assertEqual(set(names), {"right_near", "right_far"})
            elif target == "small":
                self.assertIn("lane_small", roles)
                if count > 1:
                    self.assertIn("clear_02", roles)  # never all boxes on the left
                for row in generation:
                    if row["placement_role"] == "lane_small":
                        self.assertGreaterEqual(row["right_02_clearance_m"], batch.SMALL_CLEARANCE_M - 1e-3)
                    else:
                        self.assertGreaterEqual(row["right_02_right_clearance_m"],
                                                batch.MEDIUM_CLEARANCE_M - 1e-3)
            elif target == "medium":
                self.assertTrue({"right", "centre"} & set(names))
                if count > 1:
                    self.assertGreaterEqual(len(set(names) | {row["lane_fit"] for row in generation}), 2)
            else:
                # Right-side boxes close every recording but right_01 early; the lane box comes last.
                self.assertEqual(roles, ["block_02", "lane_late"] if count == 2
                                 else ["block_far", "block_02", "lane_late"])
                self.assertGreaterEqual(generation[-1]["right_edge_route_lateral_m"],
                                        batch.LATE_LANE_RIGHT_EDGE_MIN_M - 1e-3)
                self.assertGreaterEqual(generation[-1]["longitudinal_fraction"], 0.85)
                blocked = set()
                for row in generation[:-1]:
                    self.assertLessEqual(row["longitudinal_fraction"], 0.12)
                    self.assertGreaterEqual(row["right_01_right_clearance_m"],
                                            batch.MEDIUM_CLEARANCE_M - 1e-3)
                    blocked.update(row["blocks_routes"])
                self.assertLessEqual(set(batch.BLOCKED_ROUTES[:3]), blocked)
                self.assertLessEqual(generation[-2]["right_02_right_clearance_m"],
                                     batch.BLOCKING_MAX_CLEARANCE_M + 1e-3)
            for item, row in zip(obstacles, generation):
                self.assertEqual(item["scale"], [1.0, 1.0, 1.0])
                self.assertGreaterEqual(row["right_edge_route_lateral_m"], batch.RIGHT_ROAD_LIMIT_M - 1e-3)
                if row["placement_role"] in batch.LANE_ROLES:
                    # No lane box forces a swing beyond right_03 (right_04/05 is about 5 m).
                    self.assertGreaterEqual(row["right_03_clearance_m"], batch.MEDIUM_CLEARANCE_M - 1e-3)
                    left_limit = row["lane_left_bound_m"] - batch.LANE_EDGE_MARGIN_M
                    if row["lane_fit"] == "inside":
                        self.assertLessEqual(row["left_edge_route_lateral_m"], left_limit + 1e-3)
                    else:
                        self.assertEqual(row["lane_fit"], "covers_centre_line")
                        index = 660.0 + 20.0 * row["longitudinal_fraction"]
                        self.assertFalse(batch.lateral_window(
                            self.policy, item["model"], index, item["yaw_deg"])["fits"])
                else:
                    self.assertEqual(row["lane_fit"], "right_of_path")
                _, lateral = batch.lane.route_frame(
                    batch.np.array([[item["position"]["x"], item["position"]["y"]]]),
                    route, first_index)
                self.assertAlmostEqual(lateral[0], row["route_lateral_m"], places=3)
                self.assertTrue(660 <= item["route_path_index"] <= 680)
        self.assertEqual({key.split(":")[0] for key in strata}, set(batch.AVOIDANCE_TARGETS))

    def test_strata_follow_the_recorded_route_envelopes(self):
        middle, yaw = 670.0, -83.219
        overhang = [model for model in self.policy["models"]
                    if not batch.lateral_window(self.policy, model, middle, yaw)["fits"]]
        self.assertEqual(sorted(overhang), ["Rockslide", "Steel_Barricade", "concrete-barrier"])
        small = batch.lateral_window(self.policy, "Bollard01", middle, yaw)
        self.assertEqual(batch.feasible_strata(small), ["right", "centre", "left"])
        # right_02 has barely swung out at the zone start, so a bicycle there
        # cannot leave it room; mid-zone it can.
        start = batch.lateral_window(self.policy, "Bike1", 661.0, yaw)
        self.assertNotIn("left", batch.feasible_strata(start))
        self.assertIn("left", batch.feasible_strata(batch.lateral_window(self.policy, "Bike1", 672.0, yaw)))
        # right_03 is still swinging out at the zone start: the lowest right edge is higher there.
        self.assertGreater(start["right_edge_min"], small["right_edge_min"] + 0.5)
        self.assertEqual(set(self.policy["expert_routes"]),
                         {"right_01.csv"} | set(batch.BLOCKED_ROUTES))

    def test_boxes_must_be_separated_along_or_across_the_route(self):
        near = [(660.0, (-0.8, 0.8), -0.5, 0.5), (662.0, (-0.8, 0.8), -0.4, 0.6)]
        self.assertFalse(batch._boxes_separated(near))
        beside = [(660.0, (-0.8, 0.8), -1.5, -0.8), (662.0, (-0.8, 0.8), -0.4, 0.6)]
        self.assertTrue(batch._boxes_separated(beside))
        apart = [(660.0, (-0.8, 0.8), -0.5, 0.5), (665.0, (-0.8, 0.8), -0.4, 0.6)]
        self.assertTrue(batch._boxes_separated(apart))

    def test_attempt_cell_fills_the_fewest_accepted_count_and_target(self):
        accepted = {"{}:{}".format(c, t): 3 for c in (1, 2) for t in batch.AVOIDANCE_TARGETS}
        accepted.update({"1:small": 2, "1:medium": 1, "2:small": 1, "2:medium": 1})
        manifest = {"accepted_cell_counts": accepted,
                    "attempted_cell_counts": {"1:medium": 4, "2:small": 1, "2:medium": 2}}
        self.assertEqual(batch.choose_attempt_cell(manifest, range(1, 3), random.Random(0)),
                         (2, "small"))
        # A late layout needs two boxes, and weights scale the shares.
        empty = {"accepted_cell_counts": {"{}:{}".format(c, t): 10 for c in (1, 2, 3)
                                          for t in batch.AVOIDANCE_TARGETS},
                 "attempted_cell_counts": {}}
        weights = batch.parse_avoidance_weights("1,1,1,100")
        self.assertEqual(batch.choose_attempt_cell(empty, range(1, 4), random.Random(0), weights)[1],
                         "late")
        self.assertNotEqual(batch.choose_attempt_cell(empty, range(1, 2), random.Random(0), weights)[1],
                            "late")

    def test_missing_ego_trace_rejects_the_candidate(self):
        result = batch.evaluate_lane_trace(Path("/nonexistent/trace.csv"), {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["reasons"], ["ego_trace_missing"])
        self.assertEqual(batch.lane_summary(result)["reasons"], ["ego_trace_missing"])

    def test_blocked_layout_covers_every_recorded_bypass(self):
        rng = random.Random(4)
        roles, fractions = batch.layout_roles(batch.BLOCKED_TARGET, 3, rng)
        self.assertEqual(roles, ["block_far", "block_02", "lane_block"])
        self.assertTrue(all(0.0 <= value <= 0.12 for value in fractions))
        with self.assertRaises(ValueError):
            batch.layout_roles(batch.BLOCKED_TARGET, 2, rng)
        models = batch.target_models_for(self.policy, batch.BLOCKED_TARGET, -83.219)
        for attempt in range(6):
            obstacles = batch.generate_lane_layout(
                self.policy, rng, 3, batch.BLOCKED_TARGET, models[attempt % len(models)], {},
                90.0 if attempt % 2 else 0.0)
            blocked = set()
            for item in obstacles:
                blocked.update(item["generation"].get("blocks_routes", []))
            self.assertTrue(set(batch.BLOCKED_ROUTES) <= blocked)
            roles = [item["generation"]["placement_role"] for item in obstacles]
            self.assertEqual(roles.count("lane_block"), 1)

    def test_blockage_release_policy_and_evidence(self):
        from ..scenarios.full_loop_parts import static_blockage_release
        self.assertEqual(static_blockage_release.validate_policy(batch.BLOCKAGE_RELEASE_POLICY),
                         batch.BLOCKAGE_RELEASE_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "attempt.log"
            log.write_text("[STATIC BLOCKAGE] static_blockage_wait_started {}\n")
            self.assertFalse(batch.blockage_release_observed(log))
            log.write_text(log.read_text() + batch.BLOCKAGE_RELEASED_MARKER + " {}\n")
            self.assertTrue(batch.blockage_release_observed(log))
            self.assertFalse(batch.blockage_release_observed(Path(tmp) / "missing.log"))


if __name__ == "__main__":
    unittest.main()
