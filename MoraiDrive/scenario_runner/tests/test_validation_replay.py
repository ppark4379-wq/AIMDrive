"""Offline replay and damaged recording tests without Runner dependencies."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ..validation.replay import ReplayError, replay_run
from ..validation.run_log import ValidationRunLog


def ego(collisions):
    return {"x": 1.0, "y": 2.0, "z": 3.0, "yaw_deg": 45.0,
            "speed_mps": 4.0, "collision_objects": collisions}


class ValidationReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "validation_runs"

    def record(self, run_id, samples):
        log = ValidationRunLog(self.root, "full_loop_tunnel_vehicle")
        for index, collisions in enumerate(samples):
            log.observe(run_id, float(index), ego(collisions),
                        clock_domain="host_monotonic", lap_index=0)
        log.finish(run_id, "run_returned")
        return self.root / run_id

    def rewrite_observations(self, directory, rows):
        path = directory / "ego_observations.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["observation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def test_collision_start_persistence_and_clear_match_online_events(self):
        directory = self.record("run-1", [[], ["npc-7"], ["npc-7"], [], None])
        result = replay_run(directory)
        self.assertEqual(result["observation_count"], 5)
        self.assertEqual(result["event_count"], 2)
        online = [json.loads(line) for line in (
            directory / "validation_events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event_type"] for row in online],
                         ["collision_started", "collision_cleared"])
        self.assertEqual([row["timestamp"] for row in online], [1.0, 3.0])
        recorded = [json.loads(line) for line in (
            directory / "ego_observations.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIsNone(recorded[-1]["collision_objects"])
        self.assertEqual(recorded[1]["position_xyz"], [1.0, 2.0, 3.0])

    def test_normal_drive_has_zero_events_and_replays(self):
        directory = self.record("run-2", [[], []])
        self.assertEqual(replay_run(directory)["event_count"], 0)
        copied = self.root.parent / "offline_copy" / "run-2"
        shutil.copytree(directory, copied)
        self.assertEqual(replay_run(copied)["status"], "matched")

    def test_missing_and_tampered_files_fail(self):
        directory = self.record("run-3", [["npc-7"]])
        observations = directory / "ego_observations.jsonl"
        observations.write_text("{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(ReplayError, "checksum mismatch"):
            replay_run(directory)
        observations.unlink()
        with self.assertRaisesRegex(ReplayError, "Missing or unreadable"):
            replay_run(directory)
        other = self.record("run-4", [[]])
        (other / "validation_events.jsonl").unlink()
        with self.assertRaisesRegex(ReplayError, "Missing or unreadable"):
            replay_run(other)

    def test_missing_fields_and_bad_order_fail_even_with_updated_checksum(self):
        directory = self.record("run-5", [[], []])
        rows = [json.loads(line) for line in (
            directory / "ego_observations.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[1]["sequence"] = 0
        self.rewrite_observations(directory, rows)
        with self.assertRaisesRegex(ReplayError, "sequence mismatch"):
            replay_run(directory)
        rows[1]["sequence"] = 1
        del rows[1]["collision_objects"]
        self.rewrite_observations(directory, rows)
        with self.assertRaisesRegex(ReplayError, "fields missing"):
            replay_run(directory)
        rows[1]["collision_objects"] = []
        rows[1]["timestamp"] = -1.0
        self.rewrite_observations(directory, rows)
        with self.assertRaisesRegex(ReplayError, "time order invalid"):
            replay_run(directory)

    def test_online_event_mismatch_fails_even_with_updated_checksum(self):
        directory = self.record("run-6", [["npc-7"]])
        events_path = directory / "validation_events.jsonl"
        row = json.loads(events_path.read_text(encoding="utf-8"))
        row["actor_id"] = "different-npc"
        events_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["event_sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ReplayError, "differ from online"):
            replay_run(directory)

    def test_runs_have_separate_monitor_state(self):
        for run_id in ("run-7", "run-8"):
            directory = self.record(run_id, [["npc-7"]])
            self.assertEqual(replay_run(directory)["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
