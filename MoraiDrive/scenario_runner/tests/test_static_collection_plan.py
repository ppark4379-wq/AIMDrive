import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scenario_runner" / "tools" / "static_collection_plan.py"
SPEC = importlib.util.spec_from_file_location("static_collection_plan", MODULE_PATH)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class StaticCollectionPlanTest(unittest.TestCase):
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

    def test_collection_launcher_supports_per_lap_plan(self):
        source = (
            ROOT / "data_collection" / "internal_sensor_record" / "tools" /
            "collect_scenario_laps.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--static-plan", source)
        self.assertIn("static_collection_plan.py", source)
        self.assertIn('LAP_STATIC_PRESET="$(/usr/bin/python3', source)
        self.assertIn('collector_command+=(--static-preset "$LAP_STATIC_PRESET")', source)
        self.assertIn('--scenario-weather "$LAP_SCENARIO_WEATHER"', source)
        self.assertIn('--scenario-hour "$LAP_SCENARIO_HOUR"', source)
        self.assertIn('failed_environment_mismatch', source)


if __name__ == "__main__":
    unittest.main()
