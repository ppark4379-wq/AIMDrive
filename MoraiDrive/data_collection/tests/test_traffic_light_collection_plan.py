"""Check balanced run plans without sensors, simulator or dataset writes."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
from traffic_light_collection_plan import (COMBINATIONS, create_plan, create_plan_file,
                                          load_plan, plan_runs, seal_plan, validate_plan)


class TrafficLightPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = create_plan()

    def write(self, plan=None, name="plan.json"):
        path = self.root / name
        path.write_text(json.dumps(self.plan if plan is None else plan), encoding="utf-8")
        return path

    def test_default_is_eighteen_laps_with_independent_six_condition_holdout(self):
        self.assertEqual(len(self.plan["runs"]), 18)
        for split, repeats in (("train", 2), ("val", 1)):
            rows = [row for row in self.plan["runs"] if row["split"] == split]
            counts = Counter((row["weather"], row["time_hour"]) for row in rows)
            self.assertEqual(set(counts), COMBINATIONS)
            self.assertEqual(set(counts.values()), {repeats})
        self.assertEqual([row["lap"] for row in self.plan["runs"] if row["split"] == "val"],
                         list(range(13, 19)))
        for before, after in zip(self.plan["runs"], self.plan["runs"][1:]):
            self.assertNotEqual(before["weather"], after["weather"])
            self.assertNotEqual(before["time_hour"], after["time_hour"])

    def test_load_returns_original_document_and_rows_without_normalization(self):
        self.plan["source_note"] = "교차로 기록"
        path = self.write()
        loaded = load_plan(path, expected_laps=18)
        self.assertEqual(loaded, self.plan)
        rows = plan_runs(loaded)
        self.assertIs(rows[1], loaded["runs"][0])
        self.assertEqual(rows[18]["split"], "val")

    def test_at_least_two_repeats_allow_train_and_val(self):
        for repeats in (2, 3, 4):
            with self.subTest(repeats=repeats):
                self.assertEqual(len(create_plan(repeats)["runs"]), repeats * 6)
        for repeats in (True, False, 0, 1, 2.0, "3", None):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                create_plan(repeats)

    def test_expected_laps_must_match_exactly_without_partial_replay(self):
        path = self.write()
        for requested in (17, 19, True, 18.0, "18", 0):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                load_plan(path, requested)

    def test_each_split_must_balance_even_if_overall_counts_match(self):
        plan = copy.deepcopy(self.plan)
        plan["runs"][11]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "split must balance"):
            validate_plan(plan)

    def test_plan_cannot_exceed_the_collector_run_limit(self):
        with self.assertRaisesRegex(ValueError, "100-run limit"):
            create_plan(17)

    def test_rejects_missing_schema_wrong_scenario_and_not_ready(self):
        for field, value in (("schema", None), ("scenario_name", "full_loop_static"),
                             ("ready", False), ("ready", 1)):
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_empty_nonlist_and_malformed_runs(self):
        for rows in ([], {}, None, "runs", [None], [1]):
            plan = copy.deepcopy(self.plan)
            plan["runs"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_duplicate_noncontiguous_and_wrong_type_laps(self):
        for value in (1, 3, True, 2.0, "2", None):
            plan = copy.deepcopy(self.plan)
            plan["runs"][1]["lap"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_unsupported_and_wrong_type_weather_hour_split(self):
        changes = (("weather", "RAINY"), ("weather", "sunny"), ("weather", True),
                   ("weather", None), ("time_hour", 12), ("time_hour", 11.0),
                   ("time_hour", "11"), ("time_hour", True), ("split", "test"),
                   ("split", ""), ("split", None))
        for field, value in changes:
            plan = copy.deepcopy(self.plan)
            plan["runs"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_repeated_adjacent_weather_or_hour(self):
        for field in ("weather", "time_hour"):
            plan = copy.deepcopy(self.plan)
            plan["runs"][1][field] = plan["runs"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Adjacent"):
                validate_plan(plan)

    def test_rejects_combination_imbalance_even_when_adjacent_change_remains(self):
        plan = copy.deepcopy(self.plan)
        plan["runs"][0]["time_hour"] = 15
        with self.assertRaisesRegex(ValueError, "equal repeats"):
            validate_plan(plan)

    def test_rejects_split_missing_one_condition_and_entire_val_split(self):
        for mode in ("one_condition", "all"):
            plan = copy.deepcopy(self.plan)
            for row in plan["runs"]:
                if row["split"] == "val" and (mode == "all" or row["lap"] == 13):
                    row["split"] = "train"
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "all six"):
                validate_plan(plan)

    def test_rejects_false_summary_counts(self):
        for field, value in (("requested_laps", 19), ("requested_laps", "18"),
                             ("repeats_per_combination", 4), ("repeats_per_combination", True)):
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_duplicate_json_keys_and_nonfinite_fields(self):
        for payload in (json.dumps(self.plan).replace('"schema":', '"schema":"duplicate","schema":', 1),
                        json.dumps(dict(self.plan, extra=float("nan")))):
            path = self.root / "ambiguous.json"
            path.write_text(payload)
            with self.assertRaises(ValueError):
                load_plan(path)

    def test_seal_preserves_exact_bytes_including_bom_and_crlf(self):
        source = self.root / "source.json"
        payload = b"\xef\xbb\xbf" + json.dumps(self.plan, indent=3).replace("\n", "\r\n").encode("utf-8") + b"\r\n"
        source.write_bytes(payload)
        destination = self.root / "nested" / "sealed.json"
        sealed = seal_plan(source, destination, 18)
        self.assertEqual(source.read_bytes(), payload)
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(sealed["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(sealed["bytes"], len(payload))
        self.assertEqual(load_plan(destination, 18), self.plan)

    def test_seal_rejects_invalid_input_before_creating_destination(self):
        source = self.write(dict(self.plan, runs=[]))
        destination = self.root / "no_create" / "sealed.json"
        with self.assertRaises(ValueError):
            seal_plan(source, destination, 18)
        self.assertFalse(destination.parent.exists())

    def test_seal_and_create_never_overwrite_existing_files(self):
        source = self.write()
        destination = self.root / "existing.json"
        destination.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            seal_plan(source, destination, 18)
        with self.assertRaises(FileExistsError):
            create_plan_file(destination)
        self.assertEqual(destination.read_bytes(), b"preserve")
        with self.assertRaises(FileExistsError):
            seal_plan(source, source, 18)
        self.assertEqual(load_plan(source), self.plan)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(INTERNAL / "tools" / "traffic_light_collection_plan.py")]
                              + list(args), capture_output=True, text=True)

    def test_cli_creates_checks_seals_and_resolves_plain_scalar_values(self):
        source = self.root / "cli.json"
        destination = self.root / "sealed.json"
        for args in (("create", "--output", str(source)),
                     ("check", "--plan", str(source), "--laps", "18"),
                     ("seal", "--plan", str(source), "--laps", "18", "--output", str(destination))):
            result = self.cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "PASS")
        self.assertEqual(source.read_bytes(), destination.read_bytes())
        for field, value in (("weather", "FOGGY"), ("time_hour", "15")):
            result = self.cli("resolve", "--plan", str(source), "--lap", "18", "--field", field)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), value)
        result = self.cli("resolve", "--plan", str(source), "--lap", "13", "--field", "json")
        self.assertEqual(json.loads(result.stdout), self.plan["runs"][12])

    def test_cli_reports_bad_requested_lap_without_traceback(self):
        path = self.write()
        for lap in ("0", "19"):
            result = self.cli("resolve", "--plan", str(path), "--lap", lap, "--field", "json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("[FAIL]", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
