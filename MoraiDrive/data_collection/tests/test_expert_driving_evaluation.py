import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "04_evaluate_expert_driving.py"
SPEC = importlib.util.spec_from_file_location("expert_driving_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class ExpertDrivingEvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign = self.base / "campaigns/test_campaign"
        self.campaign.mkdir(parents=True)
        self.ledger([(1, "lap_001", "captured"), (2, "lap_002", "captured")])
        self.episode("lap_001")
        self.episode("lap_002", events=[
            {"event": "progress", "sequence": 10},
            {"event": "collision", "sequence": 11,
             "receive_monotonic_ns": 1234567890123456789,
             "elapsed_seconds": 120.5, "route_s": 1432.5,
             "ego": {"collision_objects": ["NPC_17", 7]}},
            {"event": "collision_cleared", "sequence": 12},
        ])

    def ledger(self, rows):
        text = "lap\tseed\trun_name\tstatus\n"
        text += "".join("{}\t1\t{}\t{}\n".format(*row) for row in rows)
        (self.campaign / "runs.tsv").write_text(text, encoding="utf-8")

    def episode(self, run, completed=True, events=None):
        path = self.base / run / "episode_000"
        path.mkdir(parents=True, exist_ok=True)
        events = events or []
        outcome = {
            "status": "mission_completed" if completed else "diagnostic_window_completed",
            "reason": "runner_goal_reached" if completed else "bounded_window_elapsed",
            "full_mission_completed": completed,
            "collision_summary": {"contact_events": sum(e["event"] == "collision" for e in events)},
        }
        (path / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
        (path / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events), encoding="utf-8")
        return path

    def report(self):
        return evaluation.evaluate_campaign(self.base, "test_campaign")

    def cli(self, campaign="test_campaign"):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), campaign, "--base", str(self.base)],
            cwd=self.base, capture_output=True, text=True)

    def test_clean_and_collision_laps_preserve_evidence(self):
        report = self.report()
        self.assertEqual(report["summary"], {
            "total_laps": 2, "completed_laps": 2, "collision_free_laps": 1,
            "collision_events": 1, "laps_with_errors": 0, "evaluation_complete": True})
        collision = report["laps"][1]["collisions"][0]
        self.assertEqual(collision["collision_objects"], ["NPC_17", "7"])
        self.assertEqual(collision["route_s"], 1432.5)
        self.assertEqual(collision["elapsed_seconds"], 120.5)
        self.assertEqual(collision["event_sequence"], 11)
        self.assertEqual(collision["receive_monotonic_ns"], 1234567890123456789)
        self.assertIsNone(collision["sensor_frame"])

    def test_last_entry_per_lap_is_used_and_laps_are_sorted_numerically(self):
        self.ledger([(10, "lap_010", "captured"), (2, "lap_002", "started"),
                     (1, "old_run", "started"), (2, "lap_002", "captured"),
                     (1, "lap_001", "failed")])
        self.episode("lap_010", completed=False)
        report = self.report()
        self.assertEqual([lap["lap"] for lap in report["laps"]], [1, 2, 10])
        self.assertEqual(report["laps"][0]["collection_status"], "failed")
        self.assertEqual(report["laps"][0]["run_name"], "lap_001")
        self.assertEqual(report["summary"]["completed_laps"], 2)

    def test_bounded_window_is_not_completed_or_collision_free(self):
        self.episode("lap_001", completed=False)
        report = self.report()
        self.assertIs(report["laps"][0]["completed"], False)
        self.assertEqual(report["summary"]["completed_laps"], 1)
        self.assertEqual(report["summary"]["collision_free_laps"], 0)

    def test_contact_events_are_not_deduplicated(self):
        self.episode("lap_001", events=[
            {"event": "collision", "ego": {"collision_objects": ["NPC_1"]}},
            {"event": "collision_cleared"},
            {"event": "collision", "ego": {"collision_objects": ["NPC_1"]}}])
        self.assertEqual(self.report()["laps"][0]["collision_count"], 2)

    def test_missing_events_are_unknown_and_never_collision_free(self):
        (self.base / "lap_001/episode_000/events.jsonl").unlink()
        report = self.report()
        self.assertIsNone(report["laps"][0]["collision_count"])
        self.assertEqual(report["summary"]["completed_laps"], 2)
        self.assertEqual(report["summary"]["collision_free_laps"], 0)
        self.assertFalse(report["summary"]["evaluation_complete"])

    def test_missing_episode_does_not_hide_other_laps(self):
        self.ledger([(1, "missing", "started"), (2, "lap_002", "captured")])
        report = self.report()
        self.assertIsNone(report["laps"][0]["completed"])
        self.assertIsNone(report["laps"][0]["collision_count"])
        self.assertEqual(report["summary"]["collision_events"], 1)

    def test_bad_outcomes_do_not_hide_collision_evidence(self):
        path = self.base / "lap_002/episode_000/outcome.json"
        for content in [None, "{", "[]", "{}", '{"full_mission_completed":"true"}',
                        '{"full_mission_completed":1}']:
            with self.subTest(content=content):
                if content is None:
                    path.unlink()
                else:
                    path.write_text(content, encoding="utf-8")
                lap = self.report()["laps"][1]
                self.assertIsNone(lap["completed"])
                self.assertEqual(lap["collision_count"], 1)
                self.assertTrue(lap["errors"])

    def test_corrupt_jsonl_retains_evidence_but_marks_count_unknown(self):
        path = self.base / "lap_002/episode_000/events.jsonl"
        path.write_text('\n{broken\n[]\n' + path.read_text(encoding="utf-8")
                        + '\n{"event":"collision"}\n{"event":', encoding="utf-8")
        lap = self.report()["laps"][1]
        self.assertIsNone(lap["collision_count"])
        self.assertFalse(lap["events_complete"])
        self.assertEqual(len(lap["collisions"]), 2)
        self.assertEqual(len(lap["errors"]), 3)
        self.assertIn("events.jsonl:2", lap["errors"][0])

    def test_invalid_objects_do_not_split_strings_into_characters(self):
        self.episode("lap_001", events=[
            {"event": "collision", "ego": {"collision_objects": "NPC_17"}}])
        lap = self.report()["laps"][0]
        self.assertEqual(lap["collisions"][0]["collision_objects"], [])
        self.assertTrue(lap["errors"])

    def test_summary_mismatch_prevents_collision_free_claim(self):
        (self.base / "lap_002/episode_000/events.jsonl").write_text("", encoding="utf-8")
        report = self.report()
        self.assertEqual(report["summary"]["collision_free_laps"], 1)
        self.assertIn("disagrees", report["laps"][1]["errors"][0])

    def test_utf8_bom_and_blank_lines_are_accepted(self):
        for path in [self.campaign / "runs.tsv", self.base / "lap_002/episode_000/events.jsonl",
                     self.base / "lap_002/episode_000/outcome.json"]:
            path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8-sig")
        self.assertTrue(self.report()["summary"]["evaluation_complete"])

    def test_invalid_ledgers_fail_with_context(self):
        for content in ["", "lap\trun_name\n1\tx\n", "lap\trun_name\tstatus\n",
                        "lap\trun_name\tstatus\nx\trun\tcaptured\n",
                        "lap\trun_name\tstatus\n0\trun\tcaptured\n",
                        "lap\trun_name\tstatus\n1\trun\n"]:
            with self.subTest(content=content):
                (self.campaign / "runs.tsv").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "runs.tsv"):
                    self.report()

    def test_ids_cannot_be_paths(self):
        for name in ["../escape", "..", "/tmp", "C:\\temp", "a\\b", ""]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    evaluation.evaluate_campaign(self.base, name)
                self.ledger([(1, name, "captured")])
                with self.assertRaises(ValueError):
                    self.report()

    def test_cli_saves_report_and_preserves_inputs(self):
        before = {path: path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Collision-free laps : 1/2", result.stdout)
        self.assertIn("NPC_17", result.stdout)
        self.assertIn("NOT MATCHED YET", result.stdout)
        output = self.campaign / "expert_driving_evaluation_v1/report.json"
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), self.report())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_cli_incomplete_data_writes_report_and_returns_one(self):
        (self.base / "lap_001/episode_000/events.jsonl").unlink()
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("UNKNOWN", result.stdout)
        self.assertIn("evaluation incomplete", result.stdout)
        self.assertTrue((self.campaign / "expert_driving_evaluation_v1/report.json").is_file())

    def test_cli_missing_campaign_returns_two_without_creating_output(self):
        result = self.cli("absent")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Evaluation failed:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.base / "campaigns/absent").exists())


if __name__ == "__main__":
    unittest.main()
