"""Offline checks for sealed signal schedules, receipts and whole-lap holdouts."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection" / "internal_sensor_record"
TOOLS = INTERNAL / "tools"
sys.path[:0] = [str(INTERNAL), str(TOOLS), str(PROJECT)]

import validate_sensor_record_campaign as validator
from traffic_light_collection_plan import create_plan, plan_runs


class SignalCampaignPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = create_plan()
        self.rows = plan_runs(self.plan)

    def snapshot(self, campaign):
        path = campaign / "signal_collection_plan.json"
        path.write_text(json.dumps(self.plan), encoding="utf-8")
        metadata = {
            "scenario_name": "full_loop_traffic_light", "requested_laps": 18,
            "signal_collection_plan_evidence": path.name,
            "signal_collection_plan_sha256": validator.digest(path),
        }
        return path, metadata

    def test_snapshot_hash_and_requested_count_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            path, metadata = self.snapshot(campaign)
            rows, actual_path = validator.sealed_signal_collection_plan(campaign, metadata, 18)
            self.assertEqual(rows, self.rows)
            self.assertEqual(actual_path, path)
            with self.assertRaisesRegex(ValueError, "requested"):
                validator.sealed_signal_collection_plan(campaign, metadata, 17)
            path.write_text(json.dumps(self.plan) + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validator.sealed_signal_collection_plan(campaign, metadata, 18)

    def test_snapshot_path_cannot_escape_or_bind_a_different_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            campaign.mkdir()
            path, metadata = self.snapshot(campaign)
            for name in (str(path), "../signal_collection_plan.json"):
                with self.subTest(name=name):
                    altered = dict(metadata, signal_collection_plan_evidence=name)
                    with self.assertRaises(ValueError):
                        validator.sealed_signal_collection_plan(campaign, altered, 18)
            for scenario in ("full_loop_static", "full_loop"):
                with self.subTest(scenario=scenario):
                    with self.assertRaisesRegex(ValueError, "requires"):
                        validator.sealed_signal_collection_plan(
                            campaign, dict(metadata, scenario_name=scenario), 18)
            with self.assertRaisesRegex(ValueError, "combined"):
                validator.sealed_signal_collection_plan(campaign, dict(
                    metadata, static_collection_plan_evidence="static.json"), 18)

    def test_unplanned_campaign_remains_unplanned_and_orphan_hash_is_rejected(self):
        self.assertEqual(validator.sealed_signal_collection_plan(Path("/tmp"), {}, 3), ({}, None))
        with self.assertRaisesRegex(ValueError, "no snapshot"):
            validator.sealed_signal_collection_plan(Path("/tmp"), {
                "signal_collection_plan_sha256": "unused"}, 3)

    def receipts(self, number=1):
        planned = self.rows[number]
        row = dict(lap=str(number), weather=planned["weather"], time_hour=str(planned["time_hour"]))
        setup = {"profile": {"weather": planned["weather"]}, "time_hour": planned["time_hour"]}
        state = {"weather": planned["weather"], "time_hour": planned["time_hour"]}
        return row, setup, state, planned

    def test_actual_environment_requires_every_independent_receipt(self):
        row, setup, state, planned = self.receipts()
        self.assertTrue(validator.signal_plan_environment_matches(row, setup, state, planned))
        for receipt in ("row", "setup", "state"):
            for field, altered in (("weather", "FOGGY"), ("time_hour", 13)):
                with self.subTest(receipt=receipt, field=field):
                    modified = copy.deepcopy((row, setup, state))
                    item = modified[("row", "setup", "state").index(receipt)]
                    if receipt == "setup" and field == "weather":
                        item["profile"][field] = altered
                    else:
                        item[field] = altered
                    self.assertFalse(validator.signal_plan_environment_matches(*modified, planned))
        self.assertFalse(validator.signal_plan_environment_matches(dict(row, lap="2"), setup, state, planned))
        self.assertFalse(validator.signal_plan_environment_matches(dict(row, time_hour="-"), setup, state, planned))
        self.assertTrue(validator.signal_plan_environment_matches({}, {}, {}, None))

    def validated_laps(self):
        return [dict(lap=number, valid=True, weather=row["weather"], time_hour=row["time_hour"],
                     frames=440, checks={"signal_collection_plan_environment": True})
                for number, row in sorted(self.rows.items())]

    def test_distribution_pins_each_combination_in_each_split(self):
        laps = self.validated_laps()
        coverage = validator.signal_plan_coverage(self.rows, laps)
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["train_laps"], list(range(1, 13)))
        self.assertEqual(coverage["val_laps"], list(range(13, 19)))
        self.assertEqual(len(coverage["groups"]), 12)
        for row in coverage["groups"]:
            expected = 2 if row["split"] == "train" else 1
            self.assertEqual(row["validated_count"], expected)
            self.assertEqual(row["planned_count"], expected)
        laps[12]["valid"] = False
        missing = validator.signal_plan_coverage(self.rows, laps)
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["missing_or_invalid_laps"], [13])
        self.assertEqual(sum(row["validated_count"] for row in missing["groups"]), 17)

    def test_coverage_rejects_a_valid_flag_with_wrong_environment_or_missing_receipt_check(self):
        for kind in ("environment", "check"):
            with self.subTest(kind=kind):
                laps = self.validated_laps()
                if kind == "environment":
                    laps[0]["weather"] = "FOGGY"
                else:
                    laps[0]["checks"].clear()
                result = validator.signal_plan_coverage(self.rows, laps)
                self.assertFalse(result["complete"])
                self.assertEqual(result["environment_mismatched_laps"], [1])

    def test_every_pinned_val_lap_is_passed_to_builder(self):
        arguments = validator.signal_plan_split_arguments(self.rows)
        self.assertEqual(arguments, [part for number in range(13, 19)
                                    for part in ("--val-lap", str(number))])
        self.assertEqual(validator.signal_plan_split_arguments({}), [])

    def make_campaign(self, temporary):
        base = Path(temporary) / "sensor_record_auto_v1"
        campaign = base / "campaigns" / "signal"
        campaign.mkdir(parents=True)
        _, metadata = self.snapshot(campaign)
        (campaign / "campaign_plan.json").write_text(json.dumps(metadata))
        header = ("lap\tseed\trun_name\tstatus\tweather\ttime_hour\t"
                  "static_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n")
        ledger = ["%d\t%d\tsignal_lap_%03d\tcaptured\t%s\t%d\tOFF\tOFF\ta\tb\n" %
                  (number, 100 + number, number, row["weather"], row["time_hour"])
                  for number, row in sorted(self.rows.items())]
        (campaign / "runs.tsv").write_text(header + "".join(ledger))
        (campaign / "COLLECTION_COMPLETE.json").write_text(json.dumps({
            "scenario_name": "full_loop_traffic_light", "attempted_laps": 18,
            "captured_laps": 18, "failed_laps": 0,
        }))
        return base, campaign

    def fake_lap(self, row, _scenario, *_args, **kwargs):
        number = int(row["lap"])
        self.assertEqual(kwargs["planned_signal_run"], self.rows[number])
        return dict(lap=number, seed=int(row["seed"]), run_name=row["run_name"],
                    weather=row["weather"], time_hour=int(row["time_hour"]),
                    static_obstacle="OFF", dynamic_obstacle="OFF", frames=440,
                    dataset_build_candidates=416, excluded_frames=24,
                    native_files=3080, native_bytes=100, errors=[],
                    checks={"signal_collection_plan_environment": True}, valid=True)

    def invoke(self, base, campaign, extra=(), validator_side_effect=None):
        command = ["validate_sensor_record_campaign.py", "signal", "--no-nas-copy", *extra]
        with patch.object(validator, "BASE", base), \
                patch.object(validator, "resolve_campaign", return_value=campaign), \
                patch.object(validator, "is_direct_campaign_metadata", return_value=False), \
                patch.object(validator, "complete_offline_review"), \
                patch.object(validator, "validate_lap", side_effect=validator_side_effect or self.fake_lap), \
                patch.object(validator, "probe_analysis_environment", return_value={"valid": True}), \
                patch.object(validator, "build_analysis_environment", return_value={}), \
                patch.object(validator, "admission_recovery_build_arguments", return_value=[]), \
                patch.object(validator, "windows_path", side_effect=str), \
                patch.object(validator.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as built, \
                patch.object(sys, "argv", command), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            validator._main()
        return built

    def test_campaign_build_uses_sealed_val_laps_and_reports_actual_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, campaign = self.make_campaign(temporary)
            built = self.invoke(base, campaign)
            built.assert_called_once()
            command = built.call_args.args[0]
            self.assertEqual(command[-12:], validator.signal_plan_split_arguments(self.rows))
            report = json.loads(next(campaign.glob("validation_*/report.json")).read_text())
            self.assertTrue(report["valid"])
            self.assertTrue(report["signal_collection_plan_coverage"]["complete"])
            self.assertEqual(report["signal_collection_plan_sha256"],
                             validator.digest(campaign / "signal_collection_plan.json"))

    def test_campaign_rejects_custom_holdout_even_when_it_matches_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, campaign = self.make_campaign(temporary)
            with self.assertRaises(SystemExit) as rejected:
                self.invoke(base, campaign, ("--val-lap", "13"))
            self.assertEqual(rejected.exception.code, 2)
            self.assertEqual(list(campaign.glob("validation_*")), [])

    def test_missing_planned_combination_cannot_publish_an_unbalanced_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, campaign = self.make_campaign(temporary)

            def missing(row, *args, **kwargs):
                lap = self.fake_lap(row, *args, **kwargs)
                if lap["lap"] == 13:
                    lap["valid"] = False
                return lap

            with self.assertRaises(SystemExit) as rejected:
                self.invoke(base, campaign, ("--allow-valid-subset",), missing)
            self.assertEqual(rejected.exception.code, 2)
            report = json.loads(next(campaign.glob("validation_*/report.json")).read_text())
            self.assertFalse(report["valid"])
            self.assertEqual(report["signal_collection_plan_coverage"]["missing_or_invalid_laps"], [13])

    def test_signal_validation_failure_preserves_payload_even_with_delete_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, campaign = self.make_campaign(temporary)
            payload = base / "signal_lap_013" / "episode_000" / "raw" / "chunks" / "sensor.bin"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"original sensor evidence")

            def missing(row, *args, **kwargs):
                lap = self.fake_lap(row, *args, **kwargs)
                if lap["lap"] == 13:
                    lap["valid"] = False
                return lap

            with patch.object(validator, "cleanup_failed_attempt", side_effect=AssertionError("deletion invoked")), \
                    self.assertRaises(SystemExit) as rejected:
                self.invoke(base, campaign, ("--delete-invalid-payload",), missing)
            self.assertEqual(rejected.exception.code, 2)
            self.assertEqual(payload.read_bytes(), b"original sensor evidence")
            report = json.loads(next(campaign.glob("validation_*/report.json")).read_text())
            self.assertEqual(report["laps"][12]["payload_cleanup"]["status"], "retained")

    def test_signal_capture_failure_without_cleanup_marker_is_retained_and_blocks_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, campaign = self.make_campaign(temporary)
            ledger = campaign / "runs.tsv"
            ledger.write_text(ledger.read_text().replace(
                "signal_lap_013\tcaptured", "signal_lap_013\tfailed_1"))
            complete = campaign / "COLLECTION_COMPLETE.json"
            completion = json.loads(complete.read_text())
            completion.update(captured_laps=17, failed_laps=1)
            complete.write_text(json.dumps(completion))
            payload = base / "signal_lap_013" / "CAMERA_1" / "sensor.jpg"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"original camera evidence")
            with patch.object(validator, "cleanup_failed_attempt", side_effect=AssertionError("deletion invoked")), \
                    self.assertRaises(SystemExit) as rejected:
                self.invoke(base, campaign)
            self.assertEqual(rejected.exception.code, 2)
            self.assertEqual(payload.read_bytes(), b"original camera evidence")
            report = json.loads(next(campaign.glob("validation_*/report.json")).read_text())
            self.assertFalse(report["valid"])
            self.assertTrue(report["cleanup_evidence_valid"])
            self.assertEqual(report["capture_failed_laps"], [13])
            self.assertEqual(report["laps"][12]["payload_cleanup"]["status"], "retained")


if __name__ == "__main__":
    unittest.main()
