"""Offline contracts for bounded attempts and failed-lap exclusion."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection" / "internal_sensor_record"
TOOLS = INTERNAL / "tools"
sys.path[:0] = [str(TOOLS), str(INTERNAL), str(PROJECT)]

from cleanup_failed_sensor_record_attempt import (
    NATIVE_ROLES,
    cleanup_failed_attempt,
)
from build_sensor_record_training_dataset import select_training_laps
import validate_sensor_record_campaign as validator
from validate_sensor_record_campaign import _write_lap_results_csv


class FailedAttemptCleanupTests(unittest.TestCase):
    def make_failed_run(self, temporary, name="campaign_lap_001"):
        base = Path(temporary) / "sensor_record_auto_v1"
        run = base / name
        native = base / "native_sessions" / name
        coordination = (
            run / "episode_000" / "sensor_record_coordination")
        chunks = run / "episode_000" / "raw" / "chunks"
        coordination.mkdir(parents=True)
        chunks.mkdir(parents=True)
        native.mkdir(parents=True)
        (coordination / "recording_stopped.json").write_text(json.dumps({
            "status": "stopped", "record_on": False, "error": None,
        }))
        (run / "NATIVE_CAPTURE_RESULT.json").write_text(json.dumps({
            "error": "capture failed",
        }))
        (run / "native_setup.json").write_text(json.dumps({
            "native_root": str(native),
        }))
        (run / "supervisor.log").write_text("diagnostic")
        (run / "episode_000" / "raw" / "raw_manifest.json").write_text(
            json.dumps({"status": "failed"}))
        (chunks / "raw.bin").write_bytes(b"x" * 11)
        (native / "SESSION.json").write_text("{}")
        for role in NATIVE_ROLES:
            directory = native / role
            directory.mkdir()
            (directory / "payload.bin").write_bytes(b"y" * 13)
        sibling = base / "unrelated_lap"
        sibling.mkdir()
        (sibling / "keep.bin").write_bytes(b"keep")
        return base, run, native, sibling

    def make_pre_readiness_failure(self, temporary,
                                   name="campaign_lap_001"):
        base = Path(temporary) / "sensor_record_auto_v1"
        run = base / name
        native = base / "native_sessions" / name
        run.mkdir(parents=True)
        native.mkdir(parents=True)
        (run / "NATIVE_CAPTURE_RESULT.json").write_text(json.dumps({
            "error": (
                "RuntimeError('Scenario ROS capture exited before readiness "
                "(exit=1)')"),
            "scenario_exit_code": 1,
        }))
        plan = {"schema": "test", "session": name}
        applied = {"applied": True, "plan": plan}
        (base / (name + "_native_session_plan.json")).write_text(
            json.dumps(plan))
        (base / (name + "_native_session_apply.json")).write_text(
            json.dumps(applied))
        (native / "SESSION.json").write_text(json.dumps(applied))
        for role in NATIVE_ROLES:
            (native / role).mkdir()
        return base, run, native

    def test_cleanup_removes_only_payload_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, run, native, sibling = self.make_failed_run(temporary)
            first = cleanup_failed_attempt(
                run.name, base=base, failure_kind="capture",
                failure_reason="failed_1", require_storage=False)
            second = cleanup_failed_attempt(
                run.name, base=base, failure_kind="capture",
                failure_reason="failed_1", require_storage=False)
            self.assertEqual(first, second)
            self.assertEqual(first["cleanup_status"], "completed")
            self.assertEqual(first["recording_state"], "stopped")
            self.assertTrue(first["recording_stopped"].endswith(
                "sensor_record_coordination/recording_stopped.json"))
            self.assertEqual(first["deleted_files"], len(NATIVE_ROLES) + 1)
            self.assertGreater(first["deleted_bytes"], 0)
            self.assertEqual(
                list((run / "episode_000" / "raw" / "chunks").iterdir()), [])
            for role in NATIVE_ROLES:
                self.assertTrue((native / role).is_dir())
                self.assertEqual(list((native / role).iterdir()), [])
            self.assertTrue((native / "SESSION.json").is_file())
            self.assertEqual((run / "supervisor.log").read_text(), "diagnostic")
            self.assertEqual((sibling / "keep.bin").read_bytes(), b"keep")
            marker = json.loads((run / "FAILED_ATTEMPT.json").read_text())
            self.assertFalse(marker["training_visible"])

    def test_cleanup_refuses_unproven_stop_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, run, _, _ = self.make_failed_run(temporary)
            stopped = (
                run / "episode_000" / "sensor_record_coordination"
                / "recording_stopped.json")
            stopped.write_text(json.dumps({
                "status": "stopped", "record_on": True, "error": None,
            }))
            with self.assertRaises(RuntimeError):
                cleanup_failed_attempt(
                    run.name, base=base, failure_kind="capture",
                    failure_reason="failed_1", require_storage=False)
            stopped.write_text(json.dumps({
                "status": "stopped", "record_on": False, "error": None,
            }))
            chunks = run / "episode_000" / "raw" / "chunks"
            (chunks / "escape").symlink_to(
                base / "unrelated_lap", target_is_directory=True)
            with self.assertRaises(ValueError):
                cleanup_failed_attempt(
                    run.name, base=base, failure_kind="capture",
                    failure_reason="failed_1", require_storage=False)
            with self.assertRaises(ValueError):
                cleanup_failed_attempt(
                    "../escape", base=base, failure_kind="capture",
                    failure_reason="failed_1", require_storage=False)
            self.assertTrue((run / "supervisor.log").is_file())

    def test_cleanup_accepts_proven_failure_before_recording_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, run, native = self.make_pre_readiness_failure(temporary)
            first = cleanup_failed_attempt(
                run.name, base=base, failure_kind="capture",
                failure_reason="failed_1", require_storage=False)
            second = cleanup_failed_attempt(
                run.name, base=base, failure_kind="capture",
                failure_reason="failed_1", require_storage=False)

            self.assertEqual(first, second)
            self.assertEqual(first["cleanup_status"], "completed")
            self.assertEqual(first["recording_state"], "not_started")
            self.assertIsNone(first["recording_stopped"])
            self.assertEqual(first["payload_targets"], [])
            self.assertEqual(first["deleted_files"], 0)
            self.assertEqual(first["deleted_bytes"], 0)
            self.assertFalse(first["training_visible"])
            pre_record = first["pre_record_failure_evidence"]
            self.assertIn(
                "Scenario ROS capture exited before readiness",
                pre_record["capture_error"])
            self.assertTrue(pre_record["capture_result"].endswith(
                "/NATIVE_CAPTURE_RESULT.json"))
            self.assertFalse(
                (run / "episode_000" / "sensor_record_coordination").exists())
            self.assertTrue((native / "SESSION.json").is_file())
            for role in NATIVE_ROLES:
                self.assertTrue((native / role).is_dir())
                self.assertEqual(list((native / role).iterdir()), [])

    def test_cleanup_keeps_missing_stop_fail_closed_when_ambiguous(self):
        cases = ("ready", "coordination", "native_payload")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                base, run, native = self.make_pre_readiness_failure(temporary)
                if case == "ready":
                    ready = run / "episode_000" / "ready.json"
                    ready.parent.mkdir(parents=True)
                    ready.write_text("{}")
                elif case == "coordination":
                    coordination = (
                        run / "episode_000" / "sensor_record_coordination")
                    coordination.mkdir(parents=True)
                    (coordination / "recording_armed.json").write_text("{}")
                else:
                    (native / NATIVE_ROLES[0] / "payload.bin").write_bytes(b"x")

                with self.assertRaises(RuntimeError):
                    cleanup_failed_attempt(
                        run.name, base=base, failure_kind="capture",
                        failure_reason="failed_1", require_storage=False)
                self.assertFalse((run / "FAILED_ATTEMPT.json").exists())


class BoundedCampaignSelectionTests(unittest.TestCase):
    def test_training_selection_excludes_capture_and_validation_failures(self):
        report = {"laps": [
            {"lap": 1, "valid": True, "training_usable": True},
            {"lap": 2, "valid": False, "training_usable": False},
            {"lap": 3, "valid": True, "training_usable": True},
            {"lap": 4, "valid": False, "capture_status": "failed"},
        ]}
        self.assertEqual(
            [row["lap"] for row in select_training_laps(report)], [1, 3])
        with self.assertRaises(ValueError):
            select_training_laps({"laps": report["laps"][:2]})

    def test_csv_records_each_attempt_and_training_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            path = _write_lap_results_csv(campaign, [
                {
                    "lap": 1, "run_name": "lap_001",
                    "capture_status": "succeeded",
                    "validation_status": "passed",
                    "training_usable": True, "errors": [],
                },
                {
                    "lap": 2, "run_name": "lap_002",
                    "capture_status": "failed",
                    "validation_status": "skipped",
                    "training_usable": False, "errors": ["failed_1"],
                },
            ])
            with path.open(newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["training_usable"], "true")
            self.assertEqual(rows[1]["capture_status"], "failed")
            self.assertEqual(rows[1]["reason"], "failed_1")

    def test_collection_shell_keeps_attempt_bound_and_no_auto_resume(self):
        source = (
            TOOLS / "collect_scenario_laps.sh").read_text(encoding="utf-8")
        self.assertIn("for ((lap=1; lap<=LAPS; lap++))", source)
        self.assertIn('cleanup_failed_sensor_record_attempt.py"', source)
        self.assertIn('--run-name "$CURRENT_RUN"', source)
        self.assertIn('--failure-status "failed_$code"', source)
        self.assertIn('--run-root "$CAMPAIGN_ROOT/$CURRENT_RUN"', source)
        self.assertIn("[CONTINUE] lap $lap", source)
        self.assertNotIn('exit "$code"', source)
        self.assertIn('"requested_attempts": attempts', source)
        self.assertIn('"attempted_laps": attempts', source)
        self.assertIn("_retry_", source)

    def test_completion_seals_exact_requested_attempt_count(self):
        source = (
            TOOLS / "collect_scenario_laps.sh").read_text(encoding="utf-8")
        completion_code = source.split('"$RUNS_FILE" <<\'PY\'\n', 1)[1]
        completion_code = completion_code.split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "runs.tsv"
            completion = root / "COLLECTION_COMPLETE.json"
            header = (
                "lap\tseed\trun_name\tstatus\tweather\ttime_hour\t"
                "static_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n"
            )
            rows = []
            for lap in range(1, 101):
                status = "failed_1" if lap in (6, 15) else "captured"
                rows.append(
                    "%d\t%d\tlap_%03d\t%s\t-\t-\t-\t-\ta\tb\n"
                    % (lap, 1000 + lap, lap, status)
                )
            ledger.write_text(header + "".join(rows))
            result = subprocess.run([
                sys.executable, "-B", "-c", completion_code,
                str(completion), "campaign", "100", "full_loop_static",
                "current", str(ledger),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(completion.read_text())
            self.assertEqual(payload["requested_attempts"], 100)
            self.assertEqual(payload["attempted_laps"], 100)
            self.assertEqual(payload["captured_laps"], 98)
            self.assertEqual(payload["failed_laps"], 2)
            self.assertEqual(payload["failed_lap_numbers"], [6, 15])

    def test_validator_accepts_terminal_failures_and_publishes_only_valid_laps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "sensor_record_auto_v1"
            campaign = base / "campaigns" / "mixed_campaign"
            campaign.mkdir(parents=True)
            (campaign / "campaign_plan.json").write_text(json.dumps({
                "requested_laps": 3,
                "scenario_name": "full_loop_static",
            }))
            header = (
                "lap\tseed\trun_name\tstatus\tweather\ttime_hour\t"
                "static_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n"
            )
            rows = (
                "1\t10\tmixed_lap_001\tcaptured\tSUNNY\t11\tCargoBox\tOFF\ta\tb\n"
                "2\t11\tmixed_lap_002\tfailed_1\t-\t-\t-\t-\ta\tb\n"
                "3\t12\tmixed_lap_003\tcaptured\tFOGGY\t13\tDrum\tOFF\ta\tb\n"
            )
            (campaign / "runs.tsv").write_text(header + rows)
            (campaign / "COLLECTION_COMPLETE.json").write_text(json.dumps({
                "scenario_name": "full_loop_static",
                "attempted_laps": 3,
                "captured_laps": 2,
                "failed_laps": 1,
            }))
            failed = base / "mixed_lap_002"
            failed.mkdir()
            (failed / "FAILED_ATTEMPT.json").write_text(json.dumps({
                "schema": "morai_sensor_record_failed_attempt_v1",
                "cleanup_status": "completed",
                "failure_kind": "capture",
                "training_visible": False,
                "deleted_files": 8,
                "deleted_bytes": 100,
            }))

            def fake_validate(row, _scenario):
                number = int(row["lap"])
                return {
                    "lap": number,
                    "seed": int(row["seed"]),
                    "run_name": row["run_name"],
                    "run": str(base / row["run_name"]),
                    "weather": "SUNNY" if number == 1 else "FOGGY",
                    "time_hour": 11 if number == 1 else 13,
                    "static_obstacle": row["static_obstacle"],
                    "dynamic_obstacle": row["dynamic_obstacle"],
                    "frames": 10,
                    "dataset_build_candidates": 8,
                    "excluded_frames": 2,
                    "native_files": 70,
                    "native_bytes": 1000,
                    "errors": [],
                    "checks": {"synthetic": True},
                    "valid": True,
                }

            arguments = [
                "validate_sensor_record_campaign.py", "mixed_campaign",
                "--no-prepare-training", "--no-nas-copy",
            ]
            with patch.object(validator, "BASE", base), \
                    patch.object(validator, "resolve_campaign", return_value=campaign), \
                    patch.object(validator, "complete_offline_review"), \
                    patch.object(validator, "validate_lap", side_effect=fake_validate), \
                    patch.object(validator, "probe_analysis_environment",
                                 return_value={"valid": True}), \
                    patch.object(validator, "windows_path",
                                 side_effect=lambda path: "C:\\\\test\\" + Path(path).name), \
                    patch.object(sys, "argv", arguments):
                validator.main()

            reports = sorted(campaign.glob("validation_*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report["valid"])
            self.assertEqual(report["completed_attempts"], 3)
            self.assertEqual(report["capture_failed_laps"], [2])
            self.assertEqual(report["validation_passed"], 2)
            self.assertEqual(
                [row["lap"] for row in report["laps"]
                 if row["training_usable"]], [1, 3])
            self.assertTrue((campaign / "lap_results.csv").is_file())

    def test_validator_cleans_and_excludes_a_validation_failed_lap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "sensor_record_auto_v1"
            campaign = base / "campaigns" / "validation_failure"
            campaign.mkdir(parents=True)
            (campaign / "campaign_plan.json").write_text(json.dumps({
                "requested_laps": 3,
                "scenario_name": "full_loop_static",
            }))
            header = (
                "lap\tseed\trun_name\tstatus\tweather\ttime_hour\t"
                "static_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n"
            )
            rows = "".join(
                "%d\t%d\tvalidation_lap_%03d\tcaptured\tSUNNY\t11\t"
                "CargoBox\tOFF\ta\tb\n" % (lap, 20 + lap, lap)
                for lap in range(1, 4)
            )
            (campaign / "runs.tsv").write_text(header + rows)
            (campaign / "COLLECTION_COMPLETE.json").write_text(json.dumps({
                "scenario_name": "full_loop_static",
                "attempted_laps": 3,
                "captured_laps": 3,
                "failed_laps": 0,
            }))
            failed_run = base / "validation_lap_002"
            native = base / "native_sessions" / failed_run.name
            coordination = (
                failed_run / "episode_000" / "sensor_record_coordination")
            chunks = failed_run / "episode_000" / "raw" / "chunks"
            coordination.mkdir(parents=True)
            chunks.mkdir(parents=True)
            native.mkdir(parents=True)
            (coordination / "recording_stopped.json").write_text(json.dumps({
                "status": "stopped", "record_on": False, "error": None,
            }))
            (failed_run / "native_setup.json").write_text(json.dumps({
                "native_root": str(native),
            }))
            (chunks / "payload.bin").write_bytes(b"raw")
            for role in NATIVE_ROLES:
                directory = native / role
                directory.mkdir()
                (directory / "payload.bin").write_bytes(b"native")

            def fake_validate(row, _scenario):
                number = int(row["lap"])
                valid = number != 2
                return {
                    "lap": number,
                    "seed": int(row["seed"]),
                    "run_name": row["run_name"],
                    "run": str(base / row["run_name"]),
                    "weather": "SUNNY",
                    "time_hour": 11,
                    "static_obstacle": "CargoBox",
                    "dynamic_obstacle": "OFF",
                    "frames": 10,
                    "dataset_build_candidates": 8,
                    "excluded_frames": 2,
                    "native_files": 70,
                    "native_bytes": 1000,
                    "errors": [] if valid else ["synthetic cadence failure"],
                    "checks": {"synthetic": valid},
                    "valid": valid,
                }

            def temporary_cleanup(*args, **kwargs):
                kwargs["require_storage"] = False
                return cleanup_failed_attempt(*args, **kwargs)

            arguments = [
                "validate_sensor_record_campaign.py", "validation_failure",
                "--no-prepare-training", "--no-nas-copy",
            ]
            with patch.object(validator, "BASE", base), \
                    patch.object(validator, "resolve_campaign", return_value=campaign), \
                    patch.object(validator, "complete_offline_review"), \
                    patch.object(validator, "validate_lap", side_effect=fake_validate), \
                    patch.object(validator, "cleanup_failed_attempt",
                                 side_effect=temporary_cleanup), \
                    patch.object(validator, "probe_analysis_environment",
                                 return_value={"valid": True}), \
                    patch.object(validator, "windows_path",
                                 side_effect=lambda path: "C:\\\\test\\" + Path(path).name), \
                    patch.object(sys, "argv", arguments):
                validator.main()

            report_path = next(campaign.glob("validation_*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertTrue(report["valid"])
            self.assertEqual(report["validation_failed_laps"], [2])
            self.assertEqual(report["completed_valid_laps"], 2)
            failed_lap = report["laps"][1]
            self.assertFalse(failed_lap["training_usable"])
            self.assertEqual(failed_lap["payload_cleanup"]["status"], "completed")
            self.assertTrue((failed_run / "FAILED_ATTEMPT.json").is_file())
            self.assertEqual(list(chunks.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
