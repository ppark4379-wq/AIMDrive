"""Shared drive-loop validation hook with a fake Ego and no MORAI process.

The integrated AIM2026 launchers (highway and roundabout) attach a ValidationRunLog to the scenario the same way:
the drive loop observes the Ego, and the launcher finalizes the run afterwards.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..validation.finish import finish_validation_run
from ..validation.replay import replay_run
from ..validation.run_log import ValidationRunLog
from .validation_fakes import FakeDrive, FakeRecordingDrive, ego


class DriveValidationHookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.package_root = Path(self.directory.name) / "scenario_runner"
        self.validation_root = self.package_root / ".runtime" / "validation_runs"

    def invoke(self, drive, plan, *, enabled=True, mission="full_loop_highway",
               controller="aim2026_highway"):
        drive.plans = [plan]
        cfg = {"max_laps": 1, "check_period_sec": 0.0,
               "progress_watchdog": {"enabled": False}, "off_route_grace_sec": 10.0}
        scenario = drive(Mock(), object(), {"paths": {"mgeo_root": "unused"}}, cfg)
        log = ValidationRunLog(self.validation_root, mission) if enabled else None
        if log is not None:
            scenario._runner_validation_log = log
            scenario._runner_scenario_name = mission
            scenario._validation_controller = controller
            log.enable_incidents(self.package_root / ".runtime" / "incidents", scenario)
        status, error = "run_returned", None
        try:
            scenario.run()
        except BaseException as exc:
            status, error = "exception", exc
            raise
        finally:
            finish_validation_run(log, scenario, status, error)

    def rows(self, run_id):
        path = self.validation_root / run_id / "validation_events.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def summary(self, run_id):
        return json.loads((self.validation_root / run_id / "summary.json").read_text(encoding="utf-8"))

    def bundle(self, run_id):
        bundles = list((self.package_root / ".runtime" / "incidents").glob("incident_" + run_id + "_*"))
        self.assertEqual(len(bundles), 1)
        return bundles[0]

    def test_clean_drive_records_observation_and_empty_event_log(self):
        self.invoke(FakeDrive, {"run_id": "clean-1", "states": [ego([])], "progress": [10.0]})
        self.assertEqual(self.rows("clean-1"), [])
        summary = self.summary("clean-1")
        self.assertEqual(summary["status"], "run_returned")
        self.assertEqual(summary["observation_count"], 1)
        observations = [json.loads(line) for line in (
            self.validation_root / "clean-1" / "ego_observations.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["clock_domain"], "fake_scenario_clock")
        self.assertEqual(replay_run(self.validation_root / "clean-1")["status"], "matched")

    def test_collision_is_logged_before_existing_abort_and_incident_is_diagnosed(self):
        with self.assertRaisesRegex(RuntimeError, "expert collision"):
            self.invoke(FakeDrive, {"run_id": "collision-1", "states": [ego(["npc-7"])], "progress": [10.0]})
        self.assertEqual([row["event_type"] for row in self.rows("collision-1")], ["collision_started"])
        summary = self.summary("collision-1")
        self.assertEqual(summary["status"], "exception")
        self.assertEqual(summary["error_type"], "RuntimeError")
        bundle = self.bundle("collision-1")
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["failure"]["actor_ids"], ["npc-7"])
        self.assertEqual(manifest["controller"], "aim2026_highway")
        self.assertEqual(manifest["reproduction"]["start_from_seed"], "manual_same_command")
        self.assertIn("run_highway.sh", json.loads((bundle / "rerun.json").read_text())["command"])
        diagnosis = json.loads((bundle / "diagnosis.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnosis["source_run_id"], "collision-1")
        self.assertIn("[OBSERVED] runtime.exit", (bundle / "diagnosis.txt").read_text())

    def test_roundabout_variant_incident_names_its_own_rerun_command(self):
        with self.assertRaisesRegex(RuntimeError, "expert collision"):
            self.invoke(FakeDrive, {"run_id": "round-1", "states": [ego(["npc-7"])], "progress": [10.0]},
                        mission="full_loop_roundabout_brake", controller="aim2026_roundabout")
        bundle = self.bundle("round-1")
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["scenario_name"], "full_loop_roundabout_brake")
        self.assertEqual(manifest["reproduction"]["start_from_seed"], "manual_same_command")
        command = json.loads((bundle / "rerun.json").read_text())["command"]
        self.assertIn("run_roundabout.sh --variant brake", command)
        self.assertIn("--validation-events", command)

    def test_roundabout_basic_variant_rerun_command_is_normal(self):
        with self.assertRaisesRegex(RuntimeError, "expert collision"):
            self.invoke(FakeDrive, {"run_id": "round-2", "states": [ego(["npc-7"])], "progress": [10.0]},
                        mission="full_loop_roundabout", controller="aim2026_roundabout")
        command = json.loads((self.bundle("round-2") / "rerun.json").read_text())["command"]
        self.assertIn("run_roundabout.sh --variant normal", command)

    def test_persistent_collision_is_not_duplicated(self):
        self.invoke(FakeRecordingDrive, {"run_id": "persistent-1",
                                         "states": [ego(["npc-7"]), ego(["npc-7"])],
                                         "progress": [0.0, 10.0]})
        self.assertEqual([row["event_type"] for row in self.rows("persistent-1")], ["collision_started"])
        self.assertEqual(self.summary("persistent-1")["observation_count"], 2)

    def test_diagnosis_error_does_not_change_collision_abort(self):
        with patch("scenario_runner.validation.diagnose.diagnose_incident",
                   side_effect=RuntimeError("diagnosis unavailable")):
            with self.assertRaisesRegex(RuntimeError, "expert collision"):
                self.invoke(FakeDrive, {"run_id": "diagnosis-failure-1",
                                        "states": [ego(["npc-7"])], "progress": [10.0]})
        self.bundle("diagnosis-failure-1")
        self.assertEqual(self.summary("diagnosis-failure-1")["status"], "exception")

    def test_missing_collision_data_does_not_clear_until_empty_is_observed(self):
        missing = ego([])
        del missing["collision_objects"]
        self.invoke(FakeRecordingDrive, {"run_id": "missing-1",
                                         "states": [ego(["npc-7"]), missing, ego([])],
                                         "progress": [0.0, 0.0, 10.0]})
        self.assertEqual([row["event_type"] for row in self.rows("missing-1")],
                         ["collision_started", "collision_cleared"])

    def test_drive_without_validation_log_writes_nothing(self):
        self.invoke(FakeDrive, {"run_id": "disabled-1", "states": [ego([])], "progress": [10.0]},
                    enabled=False)
        self.assertFalse(self.validation_root.exists())


if __name__ == "__main__":
    unittest.main()
