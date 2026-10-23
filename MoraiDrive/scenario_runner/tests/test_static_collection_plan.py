import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scenario_runner" / "tools" / "static_collection_plan.py"
SPEC = importlib.util.spec_from_file_location("static_collection_plan", MODULE_PATH)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


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
            ROOT / "data_collection" / "internal_sensor_record" / "tools" /
            "collect_scenario_laps.sh"
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


if __name__ == "__main__":
    unittest.main()
