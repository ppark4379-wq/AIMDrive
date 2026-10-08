"""Narrow review refusal, unchanged facts and interrupted atomic transitions."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..tools import tunnel_static_search_preflight_review as review
from ..tools import tunnel_static_search_runtime as runtime


class PreflightReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-preflight-review-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trial = self.root / "attempts/candidate_345_try_01/trial"
        self.source_check, self.process_check, self.geometry_check = Mock(), Mock(return_value=[]), Mock()
        self.lock_path = self.root / "global.lock"
        runtime.atomic_json(self.root / "source_pins.json", {"schema": "fixture_pins"})
        candidate = {"count": 1, "obstacles": [{"model": "native", "actor_kind": "obstacle"}]}
        candidate_path = self.root / "candidates/candidate.json"
        runtime.atomic_json(candidate_path, candidate)
        runtime.atomic_json(self.trial / "candidate.json", candidate)
        preview = {"status": "failed", "failure_reason": review.REASON, "expert_verified": False,
            "live_avoidance_verified": False, "planned": candidate, "route_screened": False,
            "stationary_samples": 0, "observed": [{"raw_actor_id": "6", "model": "native", "actor_kind": "obstacle",
                "spawn_rpc_error": None, "width_m": 1., "length_m": 2., "center_xyz": [0., 0., 0.], "heading_deg": 0.}],
            "cleanup": {"verified": True, "actors": [{"raw_actor_id": "6", "destroy_called": True}]}}
        runtime.atomic_json(self.trial / "preview.json", preview)
        runtime.atomic_json(self.trial / "manifest.json", {"status": "failed", "failure_reason": review.REASON,
            "expert_verified": False, "live_avoidance_verified": False, "fixture_result": preview})
        parking = {"parked": True, "external_input_disabled": True, "gear_mode": 1, "control_mode": 1, "speed_norm_raw": .001}
        recover = {"status": "recovered", "controller_publishers": [], "preserved_foreign_actor_ids": ["2", "3", "4", "5"], "final_parking": parking}
        runtime.atomic_json(self.trial.parent / "recovery/recover_result.json", recover)
        runtime.atomic_json(self.root / "recovery_final/recover_result.json", recover)
        runtime.atomic_json(self.root / "scene_backup/restore_result.json", {"status": "restored", "parking": parking,
            "actors": [{"id": identity} for identity in ("2", "4", "5")]})
        outcome = {"actual_driving": False, "expert_entered": False, "physical_motion_observed": False,
            "strict_pass": False, "planner_calls": 0, "trajectory_checks": 0, "failure_reason": review.REASON,
            "fixture_cleanup_verified": True, "native_metadata_failure": None, "completion_conditions": {}}
        entry = {"candidate_id": "345", "committed": True, "disposition": "fatal", "recovery_verified": True,
            "trial_dir": str(self.trial), "candidate_sha256": runtime.digest(candidate_path), "outcome": outcome,
            "process": {"returncode": 1, "timed_out": False, "interrupted": False, "forced_kill": False}}
        previous = [{"committed": True, "outcome": {"actual_driving": True, "strict_pass": index < 55}} for index in range(69)]
        previous += [{"committed": True, "outcome": {"actual_driving": False}} for _ in range(9)]
        self.state = {"schema": "aimdrive_tunnel_static_adaptive_search_v1", "status": "blocked", "failure_reason": review.BLOCKED_REASON,
            "inflight": None, "cleanup_verified": True, "scene_restored": True, "actual_drives": 69, "actual_passes": 55,
            "actual_failures": 14, "preflight_rejects": 10, "source_snapshot": {"path": str(self.root / "source_pins.json"),
                "sha256": runtime.digest(self.root / "source_pins.json")}, "attempts": previous + [entry],
            "candidates": [{"id": "345", "status": "selected", "candidate_path": str(candidate_path),
                "candidate_sha256": runtime.digest(candidate_path)}]}
        self.write_state()
        runtime.atomic_json(self.root / "supervisor.json", {"status": "stopped_cleanup_verified", "cleanup_verified": True,
            "finished_unix": 1., "worker_returncode": 2, "forced_kill": False, "interrupted": False, "campaign_root": str(self.root)})

    def write_state(self):
        runtime.atomic_json(self.root / "search_manifest.json", self.state)
        runtime.atomic_json(self.root / "batch_manifest.json", self.state)
        runtime.atomic_json(self.root / "progress.json", {key: self.state[key] for key in (*review.COUNTERS, "status", "failure_reason")})

    def run_review(self, **kwargs):
        return review.review_campaign(self.root, source_check=self.source_check, process_check=self.process_check,
            geometry_check=self.geometry_check, global_lock_path=self.lock_path, **kwargs)

    def test_inspection_is_read_only_and_requires_pinned_origin(self):
        before = {name: (self.root / name).read_bytes() for name in ("search_manifest.json", "batch_manifest.json", "progress.json")}
        result = self.run_review()
        self.assertEqual(result["status"], "eligible")
        self.geometry_check.assert_called_once()
        self.source_check.assert_called_once()
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in before})
        self.assertFalse((self.root / "operator_preflight_reviews").exists())

    def test_apply_archives_fatal_facts_preserves_counters_and_is_idempotent(self):
        before = copy.deepcopy(self.state)
        result = self.run_review(apply=True)
        after = runtime.read_json(self.root / "search_manifest.json")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(after["attempts"], before["attempts"])
        self.assertEqual(after["source_snapshot"], before["source_snapshot"])
        for key in review.COUNTERS:
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["candidates"][0]["status"], "preflight_excluded")
        self.assertEqual(after["status"], "prepared")
        self.assertEqual(after["failure_reason"], "")
        docket = runtime.read_json(result["docket_path"])
        self.assertEqual(docket["original_attempt"], before["attempts"][-1])
        self.assertFalse(docket["actual_approval"])
        self.assertEqual(runtime.read_json(Path(result["docket_path"]).with_name("original_search_manifest.json")), before)
        self.assertEqual(self.run_review(apply=True)["status"], "already_applied")
        self.assertEqual(runtime.read_json(self.root / "search_manifest.json"), after)

    def test_partial_mirror_commit_recovers_from_immutable_transaction(self):
        real_write = runtime.atomic_json
        calls = [0]
        def failing(path, document):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("fixture interruption")
            return real_write(path, document)
        with patch.object(runtime, "atomic_json", side_effect=failing):
            with self.assertRaisesRegex(OSError, "interruption"):
                self.run_review(apply=True)
        self.assertEqual(runtime.read_json(self.root / "search_manifest.json")["status"], "blocked")
        self.assertEqual(self.run_review()["status"], "eligible_pending")
        self.assertEqual(self.run_review(apply=True)["status"], "applied")
        self.assertEqual(runtime.read_json(self.root / "search_manifest.json")["actual_drives"], 69)
        self.assertEqual(self.run_review(apply=True)["status"], "already_applied")

    def test_current_process_or_source_change_refuses_all_writes(self):
        self.process_check.return_value = [123]
        with self.assertRaisesRegex(RuntimeError, "remains"):
            self.run_review(apply=True)
        self.process_check.return_value = []
        self.source_check.side_effect = RuntimeError("Frozen source changed")
        with self.assertRaisesRegex(RuntimeError, "Frozen source"):
            self.run_review(apply=True)
        self.assertFalse((self.root / "operator_preflight_reviews").exists())

    def test_motion_expert_rpc_publisher_timeout_and_counter_errors_are_rejected(self):
        changes = [("expert_entered", True), ("physical_motion_observed", True), ("actual_driving", True),
                   ("planner_calls", 1), ("trajectory_checks", 1), ("fixture_cleanup_verified", False),
                   ("native_metadata_failure", {"reason": "publisher_changed"}),
                   ("failure_reason", "UNAVAILABLE"), ("failure_reason", review.REASON + "; cleanup failed")]
        original = copy.deepcopy(self.state)
        for key, value in changes:
            self.state = copy.deepcopy(original)
            self.state["attempts"][-1]["outcome"][key] = value
            self.write_state()
            with self.assertRaises(RuntimeError, msg=key):
                self.run_review(apply=True)
        self.state = copy.deepcopy(original)
        self.state["attempts"][-1]["process"]["timed_out"] = True
        self.write_state()
        with self.assertRaises(RuntimeError):
            self.run_review(apply=True)
        self.state = copy.deepcopy(original)
        self.state["actual_drives"] = 70
        self.write_state()
        with self.assertRaises(RuntimeError):
            self.run_review(apply=True)

    def test_actual_artifacts_and_other_geometry_origins_refuse_exclusion(self):
        for name in ("drive_trial.json", "planner_diagnostics.json", "ego_trace.csv"):
            path = self.trial / name
            path.write_text("fixture")
            with self.assertRaisesRegex(RuntimeError, "artifacts"):
                self.run_review(apply=True)
            path.unlink()
        self.geometry_check.side_effect = RuntimeError("Observed actor overlaps parked Ego")
        with self.assertRaisesRegex(RuntimeError, "overlaps"):
            self.run_review(apply=True)

    def test_idempotent_review_rejects_changed_counters_outcomes_and_source_reference(self):
        self.run_review(apply=True)
        after = runtime.read_json(self.root / "search_manifest.json")
        for key, value in (("actual_drives", 70), ("source_snapshot", {"sha256": "changed"})):
            modified = copy.deepcopy(after)
            modified[key] = value
            runtime.atomic_json(self.root / "search_manifest.json", modified)
            with self.assertRaisesRegex(RuntimeError, "journal changed"):
                self.run_review(apply=True)
        runtime.atomic_json(self.root / "search_manifest.json", after)
        altered = copy.deepcopy(after)
        altered["attempts"][-1]["outcome"]["strict_pass"] = True
        runtime.atomic_json(self.root / "search_manifest.json", altered)
        with self.assertRaises(RuntimeError):
            self.run_review(apply=True)

    def test_review_budget_is_bounded_and_shared_lock_contention_prevents_review(self):
        self.state["operator_review_geometry_exclusions"] = [{}] * 8
        self.write_state()
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.run_review(apply=True)
        with runtime.ExecutionLock(self.lock_path):
            with self.assertRaisesRegex(RuntimeError, "shared execution lock"):
                self.run_review(apply=True)

    def test_negative_parking_speed_is_not_clean_recovery_evidence(self):
        path = self.root / "recovery_final/recover_result.json"
        report = runtime.read_json(path)
        report["final_parking"]["speed_norm_raw"] = -.001
        runtime.atomic_json(path, report)
        with self.assertRaisesRegex(RuntimeError, "parking"):
            self.run_review(apply=True)

    def test_portable_guardian_is_detected_but_reviewing_supervisor_is_allowed(self):
        proc = self.root / "proc"
        for pid, module in (("901", "scenario_runner.tools.tunnel_static_search_guardian"),
                            ("902", "scenario_runner.tools.supervise_tunnel_static_search")):
            node = proc / pid
            node.mkdir(parents=True)
            (node / "stat").write_text(pid + " (python) " + " ".join(["S", "1", pid] + ["0"] * 16 + ["100"]))
            (node / "cmdline").write_bytes(b"\0".join(value.encode() for value in (
                "python3", "-m", module, "--command-file", str(self.root / "launch_spec.json"))))
            (node / "environ").write_bytes(b"AIMDRIVE_TUNNEL_BATCH_ROOT=" + str(self.root).encode() + b"\0")
        self.assertEqual(review._active_processes(self.root, proc), [901])

    def test_unrelated_unreadable_environ_is_skipped_before_access(self):
        proc = self.root / "unrelated_proc"
        entries = (("901", ["python3", "foreign.py", "scenario_runner.tools.run_tunnel_static_search"]),
                   ("902", ["python3", "-c", "print('scenario_runner.tools.run_tunnel_static_search')"]),
                   ("903", ["bash", "-c", "python3 -m scenario_runner.tools.run_tunnel_static_trial"]),
                   ("904", ["python3", "foreign.py", "aimdrive_tunnel_mass_guardian_20261008.py"]))
        for pid, argv in entries:
            node = proc / pid
            node.mkdir(parents=True)
            (node / "stat").write_text(pid + " (process) " + " ".join(["S", "1", pid] + ["0"] * 16 + ["100"]))
            (node / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv))
        original = Path.read_bytes
        def unreadable(path):
            if path.name == "environ":
                raise PermissionError("fixture nondumpable unrelated process")
            return original(path)
        with patch.object(Path, "read_bytes", unreadable):
            self.assertEqual(review._active_processes(self.root, proc), [])

    def test_potential_exact_worker_unreadable_environ_fails_closed(self):
        proc = self.root / "potential_proc"
        node = proc / "901"
        node.mkdir(parents=True)
        (node / "stat").write_text("901 (python) " + " ".join(["S", "1", "901"] + ["0"] * 16 + ["100"]))
        (node / "cmdline").write_bytes(b"python3\0-u\0-m\0scenario_runner.tools.run_tunnel_static_trial\0")
        original = Path.read_bytes
        def unreadable(path):
            if path.name == "environ":
                raise PermissionError("fixture potential worker")
            return original(path)
        with patch.object(Path, "read_bytes", unreadable):
            with self.assertRaisesRegex(RuntimeError, "Cannot prove campaign process absence"):
                review._active_processes(self.root, proc)


if __name__ == "__main__":
    unittest.main()
