"""Offline diagnosis: no MORAI, ROS, Expert, or perception process."""

import json
from pathlib import Path
import tempfile
import unittest

from ..validation.diagnose import diagnose_incident


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


class FailureDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "incident_run-1_000002"
        self.folder.mkdir()
        self.manifest = {
            "schema_version": 1, "incident_id": "run-1_000002", "source_run_id": "run-1",
            "scenario_name": "full_loop_tunnel_vehicle", "controller": "aimdrive_expert",
            "seed": 42, "run_options": ["--validation-events"],
            "effective_scenario_config": {"progress_watchdog": {"enabled": False}},
            "failure": {"event_type": "collision_started", "actor_ids": ["npc-7"],
                        "timestamp": 2.0, "time_basis": "lap_elapsed_sec",
                        "observation_sequence": 2, "position_xyz": [2, 0, 0]},
        }
        write_json(self.folder / "manifest.json", self.manifest)
        self.ego = [self.ego_row(i) for i in range(3)]
        self.ego[-1]["collision_objects"] = ["npc-7"]
        write_rows(self.folder / "ego_trace.jsonl", self.ego)
        write_rows(self.folder / "actors_trace.jsonl", [{"sequence": 2, "timestamp": 2.0,
            "actors": [{"actor_id": "npc-7", "position_xyz": [3, 0, 0]}], "errors": []}])
        write_rows(self.folder / "validation_events.jsonl", [{"event_type": "collision_started",
            "run_id": "run-1", "actor_id": "npc-7", "timestamp": 2.0}])
        write_json(self.folder / "scenario_result.json", {"run_seed": 42, "phase": "moving",
            "spawned": True, "spawn_verified": True, "observed_moving": True, "completed": False})
        write_json(self.folder / "summary.json", {"run_id": "run-1", "status": "exception",
            "error_message": "expert collision"})

    @staticmethod
    def ego_row(i):
        return {"run_id": "run-1", "sequence": i, "timestamp": float(i),
                "captured_monotonic_sec": 100.0 + i, "clock_domain": "host_monotonic",
                "time_basis": "lap_elapsed_sec", "lap_index": 0,
                "position_xyz": [float(i), 0.0, 0.0], "speed_mps": 1.0,
                "route_progress_s_m": float(i), "collision_objects": []}

    def checks(self):
        result = diagnose_incident(self.folder)
        self.assertTrue((self.folder / "diagnosis.json").is_file())
        self.assertIn("자동 진단", (self.folder / "diagnosis.txt").read_text(encoding="utf-8"))
        return {entry["check_id"]: entry for entry in result["checks"]}, result

    def test_collision_is_observed_without_blame_on_planning_or_control(self):
        checks, result = self.checks()
        self.assertEqual(checks["collision.event"]["status"], "OBSERVED")
        self.assertIn("npc-7", checks["collision.event"]["finding"])
        self.assertIn({"file": "actors_trace.jsonl", "line": 1}, checks["collision.event"]["evidence"])
        self.assertEqual(checks["runtime.exit"]["status"], "OBSERVED")
        self.assertEqual(checks["scenario.npc"]["status"], "OBSERVED")
        self.assertEqual(checks["planning.path"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(checks["control.response"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("planning.path", result["cause_candidates"])
        self.assertNotIn("control.response", result["cause_candidates"])

    def test_normal_observations_and_no_collision(self):
        self.manifest["failure"] = {}
        write_json(self.folder / "manifest.json", self.manifest)
        write_rows(self.folder / "validation_events.jsonl", [])
        write_json(self.folder / "summary.json", {"run_id": "run-1", "status": "run_returned"})
        checks, _ = self.checks()
        self.assertEqual(checks["collision.event"]["status"], "NOT_APPLICABLE")
        self.assertFalse(checks["ego.trace_quality"]["anomaly"])

    def test_same_frame_empty_path_is_candidate(self):
        write_rows(self.folder / "planner_trace.jsonl", [{"run_id": "run-1",
            "observation_sequence": 2, "clock_domain": "host_monotonic",
            "planned_trajectory": [], "control_command": {"steer": 0.1}}])
        checks, result = self.checks()
        self.assertEqual(checks["planning.path"]["status"], "SUSPECTED")
        self.assertIn("planning.path", result["cause_candidates"])
        self.assertEqual(checks["control.response"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_mismatched_clock_cannot_blame_planning(self):
        write_rows(self.folder / "planner_trace.jsonl", [{"run_id": "run-1",
            "observation_sequence": 2, "clock_domain": "ros_time",
            "planned_trajectory": []}])
        checks, result = self.checks()
        self.assertEqual(checks["planning.path"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("planning.path", result["cause_candidates"])

    def test_corrupt_and_missing_input_report_data_quality(self):
        (self.folder / "ego_trace.jsonl").write_text('{"sequence":', encoding="utf-8")
        (self.folder / "summary.json").unlink()
        checks, _ = self.checks()
        self.assertTrue(checks["ego.trace_quality"]["anomaly"])
        self.assertEqual(checks["runtime.exit"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(checks["ego.stall"]["status"], "NOT_APPLICABLE")

    def test_existing_watchdog_threshold_only(self):
        self.manifest["effective_scenario_config"]["progress_watchdog"] = {
            "enabled": True, "startup_grace_sec": 0.0,
            "timeout_sec": 2.0, "minimum_progress_m": 2.0}
        write_json(self.folder / "manifest.json", self.manifest)
        for row in self.ego:
            row["route_progress_s_m"] = 0.2
        write_rows(self.folder / "ego_trace.jsonl", self.ego)
        checks, _ = self.checks()
        self.assertEqual(checks["ego.stall"]["status"], "SUSPECTED")
        self.assertIn("2.0", checks["ego.stall"]["rule"])

    def test_manual_run_excludes_planning_and_ve(self):
        self.manifest["controller"] = "morai_keyboard"
        write_json(self.folder / "manifest.json", self.manifest)
        checks, result = self.checks()
        self.assertEqual(checks["planning.path"]["status"], "NOT_APPLICABLE")
        self.assertEqual(checks["perception.ve"]["status"], "NOT_APPLICABLE")
        self.assertEqual(result["cause_candidates"], [])

    def test_ve_explicit_bad_health_requires_matching_frame(self):
        self.manifest["run_options"].append("--ve-objects")
        write_json(self.folder / "manifest.json", self.manifest)
        row = {"run_id": "run-1", "observation_sequence": 2, "clock_domain": "ros_time",
               "ve_input": {"stale": True}}
        write_rows(self.folder / "planner_trace.jsonl", [row])
        checks, _ = self.checks()
        self.assertEqual(checks["perception.ve"]["status"], "INSUFFICIENT_EVIDENCE")
        row["clock_domain"] = "host_monotonic"
        write_rows(self.folder / "planner_trace.jsonl", [row])
        checks, _ = self.checks()
        self.assertEqual(checks["perception.ve"]["status"], "OBSERVED")

    def test_mismatched_summary_and_scenario_seed_not_joined(self):
        write_json(self.folder / "summary.json", {"run_id": "other", "status": "run_returned"})
        write_json(self.folder / "scenario_result.json", {"run_seed": 99, "phase": "completed"})
        checks, _ = self.checks()
        self.assertEqual(checks["runtime.exit"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(checks["scenario.npc"]["status"], "INSUFFICIENT_EVIDENCE")

    def timeout_fixture(self, *, with_pipeline=True):
        reason = "Tunnel vehicle timed out"
        self.manifest["controller"] = "ve_gps_udp"
        self.manifest["failure"] = {"event_type": "scenario_timeout", "actor_ids": [],
            "reason": reason, "timestamp": 2.0, "time_basis": "lap_elapsed_sec",
            "captured_monotonic_sec": 103.0, "observation_sequence": 2,
            "position_xyz": [2, 0, 0], "route_progress_s_m": 50.0}
        self.manifest["effective_scenario_config"]["tunnel_vehicle_dynamic"] = {
            "encounter_gap_m": 20.0, "slow_ego_speed_ratio": 0.6}
        write_json(self.folder / "manifest.json", self.manifest)
        self.ego[-1]["route_progress_s_m"] = 50.0
        self.ego[-1]["speed_mps"] = 0.1
        write_rows(self.folder / "ego_trace.jsonl", self.ego)
        write_rows(self.folder / "validation_events.jsonl", [])
        write_json(self.folder / "scenario_result.json", {"run_seed": 42,
            "phase": "failed", "failure_reason": reason, "encounter_verified": False,
            "plan": {"behavior": "vehicle_slow", "speed_mps": 1.193},
            "last_observation": {"route_gap_m": 28.1},
            "approach_speed_mps": 0.1})
        write_json(self.folder / "summary.json", {"run_id": "run-1",
            "status": "exception", "error_message": reason})
        if with_pipeline:
            write_rows(self.folder / "planning_trace.jsonl", [
                {"run_id": "run-1", "clock_domain": "host_monotonic_ns",
                 "timestamp_ns": 101_000_000_000, "cycle_id": 1,
                 "planning_reason": "jamming_whole_body_collision",
                 "target_velocity_mps": 0.0, "input_objects": []},
                {"run_id": "run-1", "clock_domain": "host_monotonic_ns",
                 "timestamp_ns": 101_500_000_000, "cycle_id": 2,
                 "planning_reason": "jamming_dynamic_yield",
                 "target_velocity_mps": 0.106, "input_objects": [
                     {"track_id": 36, "motion_state": "uncertain"}]},
            ])
            write_rows(self.folder / "control_trace.jsonl", [
                {"run_id": "run-1", "clock_domain": "host_monotonic_ns",
                 "timestamp_ns": 101_700_000_000, "cycle_id": 2,
                 "planning_cycle_id": 2, "target_velocity_mps": 0.106,
                 "final_wire_command": {"accel": 0.0, "brake": 0.709,
                                        "steer_normalized": 0.1},
                 "sent": True, "sent_ns": 101_600_000_000},
            ])

    def test_timeout_reports_observed_commands_and_suspected_deadlock(self):
        self.timeout_fixture()
        checks, report = self.checks()
        self.assertEqual(checks["timeout.event"]["status"], "OBSERVED")
        self.assertEqual(checks["timeout.npc_state"]["status"], "OBSERVED")
        self.assertEqual(checks["timeout.ego_progress"]["status"], "OBSERVED")
        self.assertIn("50.00m", checks["timeout.ego_progress"]["finding"])
        self.assertEqual(checks["timeout.plan_control"]["status"], "OBSERVED")
        self.assertIn("0.106m/s", checks["timeout.plan_control"]["finding"])
        self.assertIn("0.709", checks["timeout.plan_control"]["finding"])
        self.assertEqual(checks["timeout.yield_npc_coupling"]["status"], "SUSPECTED")
        self.assertIn("timeout.yield_npc_coupling", report["cause_candidates"])
        self.assertNotIn("scenario.npc", report["cause_candidates"])
        self.assertEqual(checks["timeout.ve_uncertainty"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(checks["collision.event"]["status"], "NOT_APPLICABLE")
        text = (self.folder / "diagnosis.txt").read_text(encoding="utf-8")
        self.assertIn("교착 가능성", text)
        self.assertIn("0.106m/s", text)

    def test_timeout_without_pipeline_does_not_blame_planner(self):
        self.timeout_fixture(with_pipeline=False)
        checks, report = self.checks()
        self.assertEqual(checks["timeout.plan_control"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(checks["timeout.yield_npc_coupling"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("timeout.yield_npc_coupling", report["cause_candidates"])

    def test_timeout_mismatched_clock_or_summary_cannot_claim_deadlock(self):
        self.timeout_fixture()
        plans = [json.loads(line) for line in (self.folder / "planning_trace.jsonl").read_text().splitlines()]
        plans[1]["clock_domain"] = "ros_time"
        write_rows(self.folder / "planning_trace.jsonl", plans)
        checks, report = self.checks()
        self.assertEqual(checks["timeout.yield_npc_coupling"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("timeout.yield_npc_coupling", report["cause_candidates"])
        write_json(self.folder / "summary.json", {"run_id": "other", "status": "exception",
            "error_message": "Tunnel vehicle timed out"})
        checks, _ = self.checks()
        self.assertEqual(checks["timeout.event"]["status"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
