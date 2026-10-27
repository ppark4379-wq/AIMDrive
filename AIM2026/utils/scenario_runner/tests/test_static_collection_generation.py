import importlib.util
import json
from pathlib import Path
CONSUMER_FIXTURES = Path(__file__).resolve().parent / "fixtures/external_consumers"
import tempfile
from types import SimpleNamespace
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
PLAN_MODULE_PATH = ROOT / "scenario_runner" / "tools" / "static_collection_plan.py"
PLAN_SPEC = importlib.util.spec_from_file_location("static_collection_plan", PLAN_MODULE_PATH)
plan = importlib.util.module_from_spec(PLAN_SPEC)
PLAN_SPEC.loader.exec_module(plan)


class StaticCollectionPlanTest(unittest.TestCase):
    def test_native_scale_policy_ignores_old_overrides_and_nonuniform_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config, route = Path(directory) / "config.yaml", Path(directory) / "route.yaml"
            config.write_text(yaml.safe_dump({"scenarios": {"full_loop": {
                "static_obstacle_models": ["Box", "Bicycle"],
                "static_obstacle_model_scales": {"Bicycle": [1.2, 1.2, 1.2]},
            }}}))
            route.write_text(yaml.safe_dump({"full_loop": {"zones": {
                "static_obstacle": {"scale": [2, 3, 2]},
            }}}))
            policy = plan.load_fixed_scale_policy(config, route)
            self.assertEqual(policy["model_baselines"], {
                "Box": [1., 1., 1.], "Bicycle": [1., 1., 1.],
            })
            self.assertEqual(policy["sources"][0]["sha256"], plan.sha256_file(config))
            self.assertEqual(policy["name"], plan.FIXED_SCALE_POLICY)

    def test_native_scale_policy_rejects_missing_or_invalid_allowed_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config, route = Path(directory) / "config.yaml", Path(directory) / "route.yaml"
            for models in ([], "Box", [None], [""]):
                config.write_text(yaml.safe_dump({"scenarios": {"full_loop": {
                    "static_obstacle_models": models,
                }}}))
                with self.assertRaisesRegex(ValueError, "allowed model names"):
                    plan.load_fixed_scale_policy(config, route)

    def test_actual_scale_is_checked_and_legacy_scaled_candidates_are_reported(self):
        policy = {"name": plan.FIXED_SCALE_POLICY,
                  "model_baselines": {"Box": [1., 1., 1.]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for index, (scale, marker) in enumerate([
                ([1, 1, 1], "misleading_old_marker"),
                ([2.2, 3.3, 2.2], 1.0),
                ([2, 3, 2], 1.0),
            ]):
                path = root / "preset{}.yaml".format(index)
                path.write_text(yaml.safe_dump({
                    "schema": "morai_static_obstacle_preset_v1",
                    "scenario": "full_loop_static",
                    "scale_policy": plan.FIXED_SCALE_POLICY,
                    "verification": {"expert_completed": True, "collision_free": True,
                                     "selected_routes": ["right_01.csv"]},
                    "obstacles": [{"model": "Box", "scale": scale,
                                   "generation": {"scale_multiplier": marker}}],
                }))
                files.append(path)
            original_bytes = [path.read_bytes() for path in files]
            manifest = root / "suite_manifest.json"
            manifest.write_text(json.dumps({"accepted_presets": [str(p) for p in files]}))
            candidates, sources = plan.load_candidates([manifest], scale_policy=policy)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["balance_features"]["scale_bins"], {"middle": 1})
            self.assertEqual(sources[0]["eligible_fixed_scale_presets"], 1)
            self.assertEqual(len(sources[0]["excluded_scale_presets"]), 2)
            self.assertTrue(all(
                row["issues"][0]["reason"] == "obstacle_scale_differs_from_native_model"
                for row in sources[0]["excluded_scale_presets"]))
            self.assertEqual(original_bytes, [path.read_bytes() for path in files])

    def test_fixed_scale_rejects_unknown_and_invalid_obstacles(self):
        policy = {"name": plan.FIXED_SCALE_POLICY, "model_baselines": {"Box": [1., 1., 1.]}}
        for obstacle in ({"model": "Other", "scale": [2, 3, 2]},
                         {"model": "Box", "scale": [2, 3]},
                         {"model": "Box", "scale": [2, float("nan"), 2]}):
            self.assertTrue(plan.fixed_scale_issues({"obstacles": [obstacle]}, policy))

    def test_new_candidates_reject_legacy_or_forged_native_policy(self):
        for policy_name, scales in (
            (plan.LEGACY_FIXED_SCALE_POLICY, [2, 3, 2]),
            (plan.FIXED_SCALE_POLICY, [2, 3, 2]),
            (plan.FIXED_SCALE_POLICY, [1, float("nan"), 1]),
            (plan.FIXED_SCALE_POLICY, [1, float("inf"), 1]),
            (plan.FIXED_SCALE_POLICY, [1, 0, 1]),
            (plan.FIXED_SCALE_POLICY, [1, 1]),
        ):
            with self.subTest(policy=policy_name, scales=scales):
                with self.assertRaises(ValueError):
                    plan.load_candidates([], scale_policy={
                        "name": policy_name, "model_baselines": {"Box": scales},
                    })

    def test_scale_bin_is_not_a_new_plan_selection_objective(self):
        candidate = self.candidate(1, "compact", "same")
        totals = {key: plan.Counter() for key in candidate["balance_features"]}
        before = plan._feature_cost(candidate, totals)
        totals["scale_bins"]["middle"] = 10000
        self.assertEqual(plan._feature_cost(candidate, totals), before)

    def test_build_plan_never_overwrites_existing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            original = b'{"sealed": "existing-plan"}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(FileExistsError, "Existing plan is preserved"):
                plan.build_plan(SimpleNamespace(output=path))
            self.assertEqual(path.read_bytes(), original)

    def test_legacy_scaled_plan_replays_but_native_policy_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            preset = Path(directory) / "scaled.yaml"
            preset.write_text(yaml.safe_dump({"obstacles": [{
                "model": "Box", "scale": [2.2, 3.3, 2.2],
                "generation": {"scale_multiplier": 1.0},
            }]}))
            document = {"schema": plan.SCHEMA, "scenario": "full_loop_static",
                        "ready": True,
                        "targets": {"weather": {"SUNNY": 1}, "time_hour": {"11": 1}},
                        "runs": [{"run": 1, "preset": str(preset),
                                  "sha256": plan.sha256_file(preset),
                                  "weather": "SUNNY", "time_hour": 11}]}
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(document))
            _, legacy = plan.load_checked_plan(path)
            self.assertEqual(len(legacy["runs"]), 1)
            # Existing configured-scale plans replay their own pinned values,
            # regardless of current native-size generation settings.
            document["scale_policy"] = {"name": plan.LEGACY_FIXED_SCALE_POLICY,
                                         "model_baselines": {"Box": [2.2, 3.3, 2.2]}}
            path.write_text(json.dumps(document))
            original_bytes = path.read_bytes(), preset.read_bytes()
            _, configured = plan.load_checked_plan(path)
            self.assertEqual(len(configured["runs"]), 1)
            self.assertEqual(original_bytes, (path.read_bytes(), preset.read_bytes()))
            document["scale_policy"] = {"name": plan.FIXED_SCALE_POLICY,
                                         "model_baselines": {"Box": [1., 1., 1.]}}
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "non-baseline preset"):
                plan.load_checked_plan(path)
            document["scale_policy"]["model_baselines"]["Box"] = [2.2, 3.3, 2.2]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "every model scale to be"):
                plan.load_checked_plan(path)

    def candidate(self, count, route, suffix):
        return {
            "obstacle_count": count,
            "route_class": route,
            "sha256": suffix,
            "challenge_score": float(count),
            "balance_features": {
                "models": {"M{}".format(suffix): 1},
                "longitudinal_bins": {"middle": 1},
                "lateral_bins": {"middle": 1},
                "scale_bins": {"middle": 1},
            },
        }

    def test_exact_count_and_route_quota_is_selected(self):
        expected = {
            (1, "compact"): 13, (1, "wide"): 17,
            (2, "compact"): 15, (2, "wide"): 19,
            (3, "compact"): 16, (3, "wide"): 20,
        }
        candidates = []
        for cell, amount in expected.items():
            for index in range(amount + 2):
                candidates.append(self.candidate(*cell, "{}-{}".format(cell, index)))
        cells, _ = plan.choose_cell_counts(
            candidates, {1: 30, 2: 34, 3: 36},
            {"compact": 44, "wide": 56},
        )
        self.assertIsNotNone(cells)
        self.assertEqual(sum(cells.values()), 100)
        for count, target in ((1, 30), (2, 34), (3, 36)):
            self.assertEqual(sum(cells[(count, route)] for route in plan.ROUTE_KEYS), target)
        for route, target in (("compact", 44), ("wide", 56)):
            self.assertEqual(sum(cells[(count, route)] for count in plan.COUNT_KEYS), target)

    def test_audit36_selection_uses_one_single_object_preset_per_model(self):
        candidates = []
        for model in ("A", "B", "C"):
            for index, cte in enumerate((2.0, 0.5)):
                item = self.candidate(1, "compact", "{}-{}".format(model, index))
                item["models"] = [model]
                item["balance_features"]["models"] = {model: 1}
                item["max_cross_track_error_m"] = cte
                candidates.append(item)
        selected, missing = plan.select_one_candidate_per_model(
            candidates, ["A", "B", "C"]
        )
        self.assertFalse(missing)
        self.assertEqual(len(selected), 3)
        self.assertEqual({item["models"][0] for item in selected}, {"A", "B", "C"})
        self.assertTrue(all(item["max_cross_track_error_m"] == 0.5 for item in selected))
        selected, missing = plan.select_one_candidate_per_model(candidates, ["A", "D"])
        self.assertFalse(selected)
        self.assertEqual(missing, ["D"])

    def test_difficulty_and_split_counts_are_exact(self):
        selected = [self.candidate(1 + index % 3, "compact", str(index)) for index in range(100)]
        for index, item in enumerate(selected):
            item["challenge_score"] = float(index)
        plan.assign_difficulty(selected, {"easy": 30, "medium": 40, "hard": 30})
        plan.assign_splits(selected, {"train": 80, "val": 10, "test": 10}, 7)
        self.assertEqual(plan.distribution(selected, "difficulty"), {
            "easy": 30, "hard": 30, "medium": 40,
        })
        self.assertEqual(plan.distribution(selected, "split"), {
            "test": 10, "train": 80, "val": 10,
        })

    def test_environment_schedule_is_exact_and_has_no_adjacent_repeat(self):
        selected = [self.candidate(1, "compact", str(index)) for index in range(100)]
        for index, item in enumerate(selected):
            item["split"] = (
                "train" if index < 80 else "val" if index < 90 else "test"
            )
        plan.assign_environment(
            selected,
            {"SUNNY": 50, "FOGGY": 50},
            {11: 34, 13: 33, 15: 33},
            260920,
        )
        self.assertEqual(plan.distribution(selected, "weather"), {
            "FOGGY": 50, "SUNNY": 50,
        })
        self.assertEqual(plan.distribution(selected, "time_hour"), {
            11: 34, 13: 33, 15: 33,
        })
        self.assertTrue(all(
            left["weather"] != right["weather"]
            for left, right in zip(selected, selected[1:])
        ))
        self.assertTrue(all(
            left["time_hour"] != right["time_hour"]
            for left, right in zip(selected, selected[1:])
        ))
        by_split = plan.environment_distribution_by_split(selected)
        for split in ("train", "val", "test"):
            self.assertEqual(
                set(by_split[split]["weather"]), {"SUNNY", "FOGGY"}
            )
            self.assertEqual(set(by_split[split]["time_hour"]), {11, 13, 15})
            self.assertEqual(len(by_split[split]["weather_time_pairs"]), 6)

    def test_plan_check_rejects_hash_change_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            presets = []
            rows = []
            for index in range(2):
                preset = root / "p{}.yaml".format(index)
                preset.write_text(yaml.safe_dump({"index": index}), encoding="utf-8")
                presets.append(preset)
                rows.append({
                    "run": index + 1,
                    "preset": str(preset),
                    "sha256": plan.sha256_file(preset),
                    "weather": ("SUNNY" if index == 0 else "FOGGY"),
                    "time_hour": (11 if index == 0 else 13),
                })
            document = {
                "schema": plan.SCHEMA,
                "scenario": "full_loop_static",
                "ready": True,
                "targets": {
                    "weather": {"SUNNY": 1, "FOGGY": 1},
                    "time_hour": {"11": 1, "13": 1},
                },
                "runs": rows,
            }
            path = root / "plan.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            _, loaded = plan.load_checked_plan(path, 2)
            self.assertEqual(len(loaded["runs"]), 2)
            presets[0].write_text("changed: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                plan.load_checked_plan(path, 2)

    def test_native_plan_requires_pinned_model_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = root / "box.yaml"
            preset.write_text(yaml.safe_dump({
                "obstacles": [{"model": "Box", "scale": [1, 1, 1]}],
            }), encoding="utf-8")
            row = {
                "run": 1, "preset": str(preset),
                "sha256": plan.sha256_file(preset),
                "weather": "SUNNY", "time_hour": 11,
                "balance_features": {
                    "models": {"Box": 1},
                    "longitudinal_bins": {"middle": 1},
                    "lateral_bins": {"middle": 1},
                    "scale_bins": {"middle": 1},
                },
            }
            document = {
                "schema": plan.SCHEMA, "scenario": "full_loop_static",
                "ready": True,
                "scale_policy": {"name": plan.FIXED_SCALE_POLICY,
                                 "model_baselines": {"Box": [1, 1, 1]}},
                "targets": {
                    "required_models": ["Box"],
                    "model_minimum_occurrences": 1,
                    "weather": {"SUNNY": 1}, "time_hour": {"11": 1},
                },
                "runs": [row],
            }
            path = root / "plan.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            plan.load_checked_plan(path, 1, require_native_scale=True)
            document["targets"]["model_minimum_occurrences"] = 2
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                plan.load_checked_plan(path, 1, require_native_scale=True)

    def test_new_collection_rejects_legacy_scale_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({
                "schema": plan.SCHEMA,
                "scenario": "full_loop_static",
                "ready": True,
                "targets": {"weather": {}, "time_hour": {}},
                "runs": [],
            }), encoding="utf-8")
            # Historical replay remains available for provenance checks.
            plan.load_checked_plan(path, 0)
            with self.assertRaisesRegex(ValueError, "legacy scaled plans are replay-only"):
                plan.load_checked_plan(path, 0, require_native_scale=True)

    def test_collection_launcher_supports_per_lap_plan(self):
        source = (
            CONSUMER_FIXTURES / "collect_scenario_laps.sh.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("--static-plan", source)
        self.assertIn("static_collection_plan.py", source)
        self.assertIn("--require-native-scale", source)
        self.assertIn('LAP_STATIC_PRESET="$(/usr/bin/python3', source)
        self.assertIn('collector_command+=(--static-preset "$LAP_STATIC_PRESET")', source)
        self.assertIn('--scenario-weather "$LAP_SCENARIO_WEATHER"', source)
        self.assertIn('--scenario-hour "$LAP_SCENARIO_HOUR"', source)
        self.assertIn('failed_environment_mismatch', source)

    def test_footprint_catalog_limits_and_pins_the_required_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config, catalog = Path(directory) / "config.yaml", Path(directory) / "catalog.json"
            config.write_text(yaml.safe_dump({"scenarios": {"full_loop": {
                "static_obstacle_models": ["Box", "Bicycle", "Cone"],
            }}}))
            catalog.write_text(json.dumps({
                "schema": "morai_static_obstacle_footprints_v1",
                "registry": {"sha256": "r"}, "models": {"Box": {}, "Bicycle": {}},
            }))
            policy = plan.load_fixed_scale_policy(config, footprint_catalog=catalog)
            self.assertEqual(sorted(policy["model_baselines"]), ["Bicycle", "Box"])
            self.assertEqual(policy["sources"][1]["sha256"], plan.sha256_file(catalog))
            self.assertEqual(policy["sources"][1]["registry_sha256"], "r")
            document = {"obstacles": [{"model": "Cone", "scale": [1, 1, 1]}]}
            self.assertEqual(plan.fixed_scale_issues(document, policy)[0]["reason"],
                             "model_missing_from_scale_policy")

    def avoidance_candidate(self, count, avoidance, suffix, passed=True):
        item = dict(self.candidate(count, "compact", suffix), preset=suffix + ".yaml")
        item["avoidance_class"] = avoidance
        item["lane_compliance"] = {"passed": passed, "min_marking_clearance_m": 0.1,
                                   "max_right_deviation_m": 1.3}
        return item

    def test_avoidance_axis_keeps_lane_verified_presets_and_fills_the_quota(self):
        candidates = [self.avoidance_candidate(count, avoidance, "{}{}{}".format(count, avoidance, i))
                      for count in plan.COUNT_KEYS for avoidance in plan.AVOIDANCE_KEYS
                      for i in range(20)]
        rejected = [self.avoidance_candidate(1, "small", "fail", passed=False),
                    self.avoidance_candidate(1, "large", "wide")]
        rejected.append(dict(self.candidate(1, "compact", "old"), preset="old.yaml",
                             lane_compliance=None))
        kept, excluded = plan.lane_evidence_candidates(candidates + rejected)
        self.assertEqual(len(kept), len(candidates))
        self.assertEqual(sorted(row["reason"] for row in excluded), [
            "avoidance_class_outside_plan", "lane_compliance_missing_or_failed",
            "lane_compliance_missing_or_failed"])
        quota = {"straight": 20, "small": 20, "medium": 40, "late": 20}
        cells, _ = plan.choose_cell_counts(
            kept, {1: 30, 2: 34, 3: 36}, quota, "avoidance_class", plan.AVOIDANCE_KEYS)
        self.assertIsNotNone(cells)
        for avoidance in plan.AVOIDANCE_KEYS:
            self.assertEqual(sum(cells[(count, avoidance)] for count in plan.COUNT_KEYS),
                             quota[avoidance])
        for count, target in ((1, 30), (2, 34), (3, 36)):
            self.assertEqual(sum(cells[(count, key)] for key in plan.AVOIDANCE_KEYS), target)
        selected = plan.select_candidates(kept, cells, 5, "avoidance_class")
        self.assertEqual(plan.distribution(selected, "avoidance_class"), quota)
        # Too few late presets: the joint quota is infeasible rather than bent.
        scarce = [item for item in kept if item["avoidance_class"] != "late"][:200]
        self.assertIsNone(plan.choose_cell_counts(
            scarce, {1: 30, 2: 34, 3: 36}, quota, "avoidance_class", plan.AVOIDANCE_KEYS)[0])
        plan.assign_difficulty(selected, {"easy": 30, "medium": 40, "hard": 30})
        plan.assign_splits(selected, {"train": 80, "val": 10, "test": 10}, 5, "avoidance_class")
        self.assertEqual(plan.distribution(selected, "split"), {"test": 10, "train": 80, "val": 10})

    def test_released_blockage_replaces_route_evidence_only_when_proven(self):
        model = sorted(plan.load_fixed_scale_policy()["model_baselines"])[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def suite(name, release, observed):
                preset = root / (name + ".yaml")
                document = {
                    "schema": "morai_static_obstacle_preset_v1", "scenario": "full_loop_static",
                    "obstacles": [{"model": model, "scale": [1.0, 1.0, 1.0]}],
                    "verification": {"expert_completed": True, "collision_free": True,
                                     "selected_routes": [], "blockage_release_observed": observed,
                                     "lane_compliance": {"passed": True, "trajectory_class": "straight"}},
                }
                if release:
                    document["blockage_release"] = {"wait_wall_sec": 6.0}
                preset.write_text(yaml.safe_dump(document), encoding="utf-8")
                manifest = root / (name + "_manifest.json")
                manifest.write_text(json.dumps({
                    "name": name, "accepted_presets": [str(preset)],
                    "attempts": [{"attempt": 1, "accepted_preset": str(preset),
                                  "closed_loop_evidence": {"selected_routes": []}}],
                }), encoding="utf-8")
                return manifest

            candidates, _ = plan.load_candidates([suite("blocked", True, True)])
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["route_class"], "compact")
            self.assertEqual(candidates[0]["avoidance_class"], "straight")
            for name, release, observed in (("unproven", True, False), ("plain", False, True)):
                with self.assertRaisesRegex(ValueError, "route evidence missing"):
                    plan.load_candidates([suite(name, release, observed)])

    def test_lane_aware_presets_balance_on_their_lateral_stratum(self):
        document = {"obstacles": [
            {"model": "A", "generation": {"lateral_stratum": "left", "longitudinal_fraction": 0.2}},
            {"model": "B", "generation": {"lateral_offset_m": 0.5, "longitudinal_fraction": 0.9}},
        ]}
        features = plan.candidate_balance_features(document, fixed_scale=True)
        self.assertEqual(features["lateral_bins"], {"left": 1, "high": 1})
        self.assertEqual(features["longitudinal_bins"], {"low": 1, "high": 1})


import copy
import random



BATCH_MODULE_PATH = ROOT / "scenario_runner" / "tools" / "generate_static_scenario_suite.py"
BATCH_SPEC = importlib.util.spec_from_file_location("static_scenario_batch", BATCH_MODULE_PATH)
batch = importlib.util.module_from_spec(BATCH_SPEC)
BATCH_SPEC.loader.exec_module(batch)

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
        repository_file = ROOT / "scenario_runner" / "README.md"
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
            CONSUMER_FIXTURES / "01_internal_sensor_record_preflight.sh.txt"
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
        self.assertIn("preflight_ros_control.sh --scenario-only", launcher)
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

    def test_route_evidence_retains_last_valid_nonempty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_bytes(
                b"{'selected_routes': ['old.csv']}\n"
                b"{'selected_routes': ['right_02.csv', 3]}\n"
                b"{'selected_routes': []}\n"
                b"{'selected_routes': [not_a_literal]}\n\xff"
            )
            self.assertEqual(plan.parse_selected_routes(log), ["right_02.csv", "3"])
            self.assertEqual(batch.read_closed_loop_evidence(log)["selected_routes"],
                             ["right_02.csv", "3"])

    def test_missing_log_keeps_plan_and_generator_error_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.log"
            self.assertEqual(plan.parse_selected_routes(missing), [])
            with self.assertRaises(FileNotFoundError):
                batch.read_closed_loop_evidence(missing)

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
