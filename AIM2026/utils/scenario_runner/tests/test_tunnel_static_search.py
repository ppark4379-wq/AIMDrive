"""Tunnel regression checks: search."""

import copy, tempfile, threading, time, unittest, ast, json, os, signal, subprocess, sys, io, random
from pathlib import Path
from unittest.mock import Mock, patch
from ..tools import run_tunnel_static_search as search, tunnel_static_search_runtime as runtime, tunnel_static_search_preflight_review as review
from ..scenarios.full_loop_parts.tunnel_static_parts import runtime, execution
from types import SimpleNamespace
from contextlib import redirect_stdout
from ..scenarios.full_loop_parts import route_profile


# Adaptive accounting, source barriers and bounded cleanup without live APIs.

def candidate(index=0, count=1):
    return {"count": count, "seed": index, "candidate_index": index,
            "obstacles": [{"actor_kind": "obstacle", "model": "native_box", "scale": [1., 1., 1.],
                "station_m": 20. + index + 10. * slot, "lateral_m": .1 * (index % 7), "yaw_deg": 0.,
                "position": {"x": 20. + index + 10. * slot, "y": .1 * (index % 7), "z": 0.},
                "planned_bbox": {"width_m": .5, "length_m": .5}, "placement_region": "own_lane"}
                for slot in range(count)]}


def outcome(kind):
    result = dict(actual_driving=False, expert_entered=False, physical_motion_observed=False,
                  strict_pass=False, fixture_cleanup_verified=True, failure_reason="",
                  retryable_native_metadata_ordering=False, recoverable_geometry_rejection=False)
    if kind == "actual_fail":
        result.update(actual_driving=True, expert_entered=True, physical_motion_observed=True,
                      failure_reason="Trajectory failed the measured original-lane return condition")
    elif kind == "geometry":
        result["recoverable_geometry_rejection"] = True
    elif kind == "ordering":
        result["retryable_native_metadata_ordering"] = True
    elif kind == "publisher":
        result["failure_reason"] = "ROS static metadata publisher changed from its bound native source"
    return result


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-adaptive-search-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        proof = self.root / "source_preview.json"
        runtime.atomic_json(proof, {})
        for name, value in (("acceptance.json", {"models": [{"preview_path": str(proof)}]}),
                            ("runtime.json", {"models": [{"measurement_path": str(proof)}]}),
                            ("approved.json", {"candidates": []}),
                            ("ego.json", {"source_preview_path": str(proof)}),
                            ("batch_manifest.json", {"attempts": []}),
                            ("config.json", {})):
            runtime.atomic_json(self.root / name, value)
        self.args = search.build_parser().parse_args([
            "--output", str(self.root / "campaign"), "--model-acceptance", str(self.root / "acceptance.json"),
            "--runtime-catalog", str(self.root / "runtime.json"), "--baseline-approved", str(self.root / "approved.json"),
            "--ego-template", str(self.root / "ego.json"), "--config", str(self.root / "config.json"),
            "--initial-candidates", "12", "--max-candidates", "30", "--actual-runs", "10"])
        self.context = dict(loader=object(), geometry={"own_points": [[0., 0., 0.], [120., 0., 0.]],
                    "departure_points": [[120., 0., 0.], [140., 0., 0.]], "own_lane_width_m": 3.5,
                    "opposing_lane_width_m": 3.5, "length_m": 120., "source_sha256": "map"},
                    statics={"native_box": {}}, vehicles={"native_car": {}},
                    template={"actual_ego_state": {"historical": True}, "actual_ego_footprint": {"full_3d": True}},
                    nominal_body={"width_m": 2.366}, baseline={"records": [], "clusters": {}},
                    yaw_policy={}, boundary=object())
        self.source = {"fixed": "source"}
        self.required = []
        self.runner = Mock(return_value={"returncode": 1, "timed_out": False, "forced_kill": False})
        self.foreign = Mock(side_effect=AssertionError("Offline stage contacted the simulator"))

    def campaign(self, **kwargs):
        def generate(_loader, _statics, _vehicles, seed, **options):
            document = candidate(options["candidate_index"], options["count"])
            choices = [(kind, name) for kind, models in (("obstacle", _statics), ("vehicle", _vehicles)) for name in models]
            for slot, row in enumerate(document["obstacles"]):
                row["actor_kind"], row["model"] = choices[slot % len(choices)]
            return document
        def screen(document, _loader, **options):
            self.assertEqual(options["actual_ego_footprint"], {"full_3d": True})
            return {"status": "nominal_path_found", "path_found": True, "geometry_checks": {"passed": True},
                    "path": [[float(index), 0.] for index in range(141)]}
        return search.SearchCampaign(self.args, context=copy.deepcopy(self.context),
            source_fn=lambda *_args: copy.deepcopy(self.source), require_fn=lambda snapshot: self.required.append(snapshot),
            generate_fn=generate, screen_fn=screen, runner=self.runner, foreign_fn=self.foreign, **kwargs)

    def initialize_row(self):
        campaign = self.campaign()
        campaign.initialize()
        path = self.root / "candidate.json"
        runtime.atomic_json(path, candidate())
        row = dict(id="0", index=0, count=1, status="selected", candidate_path=str(path),
                   candidate_sha256=runtime.digest(path), models=[{"actor_kind": "obstacle", "model": "native_box"}])
        campaign.state["candidates"] = [row]
        events = []
        campaign.helper = lambda action, directory=None: events.append(action) or {"status": "recovered"}
        return campaign, row, events

    def test_offline_prepare_balances_counts_and_never_executes_live_operations(self):
        campaign = self.campaign()
        with patch.object(search, "provenance_matches", return_value=True):
            state = campaign.run()
        self.assertEqual(state["status"], "prepared")
        self.assertEqual(state["actual_drives"], 0)
        self.assertEqual(state["generation_count"], 12)
        self.assertEqual(set(row["count"] for row in state["candidates"]), {1, 2, 3})
        self.assertLessEqual(abs(sum(row["count"] == 2 for row in state["candidates"]) - 5), 1)
        self.runner.assert_not_called()
        self.foreign.assert_not_called()
        self.assertTrue((campaign.root / "source_pins.json").is_file())
        self.assertTrue((campaign.root / "approved_candidates.json").is_file())
        self.assertEqual(runtime.read_json(campaign.root / "approved_candidates.json")["fresh_approved_count"], 0)

    def test_exact_historical_layout_duplicate_is_rejected_before_screening(self):
        self.context["baseline"]["records"] = [{"candidate": candidate(count=2)}]
        campaign = self.campaign()
        campaign.initialize()
        campaign.generate_batch(1)
        self.assertEqual(campaign.state["candidates"][0]["status"], "layout_duplicate")
        self.assertEqual(campaign.state["nominal_passes"], 0)
        self.runner.assert_not_called()

    def test_generated_candidate_tampering_is_a_source_barrier(self):
        campaign = self.campaign()
        campaign.initialize()
        campaign.generate_batch(1)
        row = campaign.state["candidates"][0]
        runtime.atomic_json(row["candidate_path"], candidate(999))
        with self.assertRaisesRegex(RuntimeError, "candidate changed"):
            campaign.candidate_document(row)
        self.runner.assert_not_called()

    def test_source_mismatch_on_resume_never_executes_recovery_or_foreign_reads(self):
        campaign = self.campaign()
        with patch.object(search, "provenance_matches", return_value=True):
            campaign.run()
        self.args.resume = True
        resumed = self.campaign()
        resumed.require_fn = Mock(side_effect=RuntimeError("Frozen sources changed"))
        with self.assertRaisesRegex(RuntimeError, "Frozen sources"):
            resumed.run()
        self.runner.assert_not_called()
        self.foreign.assert_not_called()

    def test_corrupted_existing_journal_refuses_fresh_start(self):
        campaign = self.campaign()
        campaign.root.joinpath("search_manifest.json").write_text("{partial")
        with self.assertRaisesRegex(RuntimeError, "durable JSON"):
            campaign.run()
        self.runner.assert_not_called()

    def test_nominal_workers_are_bounded_and_timeouts_are_journaled(self):
        campaign = self.campaign()
        campaign.initialize()
        campaign.generate_batch(8)
        lock = threading.Lock()
        overlap = threading.Barrier(2)
        active, maximum = [0], [0]
        def work(*_args, **_kwargs):
            with lock:
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
            overlap.wait(timeout=2.)
            with lock:
                active[0] -= 1
            return {"status": "search_timeout", "path_found": False, "worker_terminated": True}
        campaign.screen_fn = work
        campaign.screen_batch()
        self.assertEqual(maximum[0], 2)
        self.assertTrue(all(row["status"] == "nominal_timeout" for row in campaign.state["candidates"]))
        self.assertTrue(all(row["geometric_impossibility_proven"] is False for row in campaign.state["candidates"]))

    def test_geometric_rejections_are_recovered_then_excluded_without_actual_count(self):
        campaign, row, events = self.initialize_row()
        with patch.object(search, "classify_trial", return_value=outcome("geometry")):
            campaign.trial(row)
        self.assertEqual(events, ["recover"] * 3)
        self.assertEqual(row["status"], "preflight_excluded")
        self.assertEqual(campaign.state["actual_drives"], 0)
        self.assertEqual(campaign.state["preflight_rejects"], 3)
        self.assertTrue(all(entry["recovery_verified"] for entry in campaign.state["attempts"]))
        self.assertFalse(row["geometric_impossibility_proven"])

    def test_structured_ordering_retry_is_bounded_and_failed_actual_drive_counts(self):
        campaign, row, events = self.initialize_row()
        with patch.object(search, "classify_trial", side_effect=[outcome("ordering"), outcome("actual_fail")]):
            campaign.trial(row)
        self.assertEqual(events, ["recover", "recover"])
        self.assertEqual(campaign.state["actual_drives"], 1)
        self.assertEqual(campaign.state["actual_failures"], 1)
        self.assertEqual(campaign.state["actual_passes"], 0)
        self.assertEqual(campaign.state["preflight_rejects"], 1)
        self.assertEqual(row["status"], "actual_fail")

    def test_ordering_exhaustion_counts_no_actual_motion(self):
        campaign, row, events = self.initialize_row()
        with patch.object(search, "classify_trial", return_value=outcome("ordering")):
            with self.assertRaisesRegex(RuntimeError, "infrastructure retry budget"):
                campaign.trial(row)
        self.assertEqual(len(events), 5)
        self.assertEqual(campaign.state["actual_drives"], 0)
        self.assertEqual(row["status"], "native_ordering_exhausted")

    def test_every_generated_proposal_contains_its_scheduled_species_and_kind_mix(self):
        campaign = self.campaign()
        campaign.initialize()
        campaign.generate_batch(12)
        for row in campaign.state["candidates"]:
            self.assertEqual(row["status"], "generated")
            models = {(actor["actor_kind"], actor["model"]) for actor in row["models"]}
            self.assertIn((row["recipe"]["focus_actor_kind"], row["recipe"]["focus_model"]), models)
            for companion in row["recipe"]["companions"]:
                self.assertIn((companion["actor_kind"], companion["model"]), models)
            if row["recipe"]["actor_kind"] == "mixed":
                self.assertEqual({kind for kind, _name in models}, {"obstacle", "vehicle"})
        multi = [row for row in campaign.state["candidates"] if row["count"] > 1]
        self.assertEqual({row["recipe"]["actor_kind"] for row in multi}, {"obstacle", "vehicle", "mixed"})

    def test_missing_count_bucket_does_not_allow_other_counts_to_fill_goal(self):
        campaign = self.campaign()
        campaign.initialize()
        campaign.generate_batch(12)
        campaign.state["candidates"] = [row for row in campaign.state["candidates"] if row["count"] != 3]
        with patch.object(search, "provenance_matches", return_value=True):
            campaign.screen_batch()
        def choose(rows, _geometry, **_options):
            return {"selected": [{"id": row["id"]} for row in rows], "selected_ids": [row["id"] for row in rows],
                    "rejected": [], "feature_records": [{"id": row["id"], "features": {"valid": True,
                        "maximum_abs_lateral_m": 1., "maximum_width_m": 1.}} for row in rows]}
        with patch.object(search, "select_diverse_candidates", side_effect=choose):
            selected = campaign.select()
        counts = {count: sum(row["count"] == count for row in selected) for count in (1, 2, 3)}
        self.assertLessEqual(counts[1], 3)
        self.assertLessEqual(counts[2], 4)
        self.assertEqual(counts[3], 0)
        self.assertLess(len(selected), 10)

    def test_strict_proof_read_failure_does_not_partially_increment_actual_accounting(self):
        campaign, row, _events = self.initialize_row()
        passed = outcome("actual_fail")
        passed["strict_pass"] = True
        entry = {"outcome": passed, "trial_dir": "fixture", "committed": False}
        with patch.object(search, "read_trace", side_effect=ValueError("corrupt CSV")):
            with self.assertRaisesRegex(ValueError, "corrupt CSV"):
                campaign.commit(entry, row)
        self.assertEqual(campaign.state["actual_drives"], 0)
        self.assertEqual(campaign.state["actual_passes"], 0)
        self.assertFalse(entry["committed"])
        self.assertIsNone(row.get("actual_driving"))

    def test_resume_nonzero_exit_cannot_promote_strict_looking_report(self):
        campaign, row, _events = self.initialize_row()
        entry = dict(candidate_id=row["id"], trial_dir="fixture", committed=False,
                     process={"returncode": 1, "timed_out": False, "forced_kill": False})
        campaign.state["attempts"] = [entry]
        passed = outcome("actual_fail")
        passed["strict_pass"] = True
        with patch.object(search, "classify_trial", return_value=passed):
            campaign.reconcile()
        self.assertFalse(entry["outcome"]["strict_pass"])
        self.assertEqual(campaign.state["actual_failures"], 1)
        self.assertEqual(campaign.state["actual_passes"], 0)

    def test_publisher_change_stops_after_mandatory_recovery(self):
        campaign, row, events = self.initialize_row()
        with patch.object(search, "classify_trial", return_value=outcome("publisher")):
            with self.assertRaisesRegex(RuntimeError, "publisher"):
                campaign.trial(row)
        self.assertEqual(events, ["recover"])
        self.assertEqual(len(campaign.state["attempts"]), 1)
        self.assertEqual(campaign.state["actual_drives"], 0)

    def test_unstructured_text_never_authorizes_ordering_retry_or_motion_without_expert(self):
        process = {"timed_out": False, "forced_kill": False}
        unstructured = outcome("unknown")
        unstructured["failure_reason"] = "native static metadata sequence regression"
        self.assertEqual(search.SearchCampaign.disposition(unstructured, process), "fatal")
        moving = outcome("geometry")
        moving["physical_motion_observed"] = True
        self.assertEqual(search.SearchCampaign.disposition(moving, process), "fatal")
        changed = outcome("ordering")
        changed["fixture_cleanup_verified"] = False
        self.assertEqual(search.SearchCampaign.disposition(changed, process), "fatal")

    def test_partial_move_is_restored_with_shared_lock_still_held(self):
        campaign = self.campaign()
        self.args.live = True
        campaign.foreign_fn = lambda: {identity: {"model": identity} for identity in search.FOREIGN_IDS}
        real_lock = runtime.ExecutionLock
        global_lock = self.root / "shared_execution.lock"
        events = []
        def lock_factory(path=None):
            return real_lock(global_lock if path is None else path)
        def helper(action, directory=None):
            with self.assertRaisesRegex(RuntimeError, "shared execution lock"):
                with real_lock(global_lock):
                    self.fail("Restoration ran after releasing the shared simulator lock")
            events.append(action)
            if action == "move":
                raise RuntimeError("fixture partial relocation")
            return {"status": "recovered" if action == "recover" else "restored"}
        campaign.helper = helper
        with patch.object(runtime, "ExecutionLock", side_effect=lock_factory), patch.object(search, "provenance_matches", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "partial relocation"):
                campaign.run()
        self.assertEqual(events, ["recover", "move", "recover", "restore"])
        self.assertTrue(campaign.state["scene_restored"])
        self.assertTrue(campaign.state["cleanup_verified"])
        self.assertEqual(campaign.state["status"], "blocked")

    def test_recovery_failure_still_attempts_foreign_restoration(self):
        campaign, _row, _events = self.initialize_row()
        campaign.live_started = True
        campaign.state["scene_started"] = True
        calls = []
        def helper(action, directory=None):
            calls.append(action)
            if action == "recover":
                raise RuntimeError("recovery cannot verify")
            return {"status": "restored"}
        campaign.helper = helper
        campaign.finalize_live()
        self.assertEqual(calls, ["recover", "restore"])
        self.assertTrue(campaign.state["scene_restored"])
        self.assertFalse(campaign.state["cleanup_verified"])

    def test_adaptive_recipe_prioritizes_remaining_object_count_and_species(self):
        campaign = self.campaign()
        campaign.initialize()
        campaign.state["candidates"] = [dict(id=str(index), count=count, status="actual_fail", actual_driving=True,
                models=[{"actor_kind": "obstacle", "model": "native_box"}])
                for index, count in enumerate([1, 1, 1, 2, 2, 2, 2])]
        recipe = campaign.recipe(43)
        self.assertEqual(recipe["count"], 3)
        self.assertEqual(recipe["focus_actor_kind"], "vehicle")
        self.assertEqual(recipe["focus_model"], "native_car")

    def test_parser_caps_reject_200_drives_and_inadequate_cleanup_reserve(self):
        self.args.actual_runs = 200
        with self.assertRaisesRegex(ValueError, "100 actual"):
            search.validate_args(self.args)
        self.args.actual_runs = 100
        self.args.cleanup_reserve = 60
        with self.assertRaisesRegex(ValueError, "deadlines"):
            search.validate_args(self.args)


# Narrow review refusal, unchanged facts and interrupted atomic transitions.

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


# Source freezing and real subprocess ownership for the search coordinator.

class DurableRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-search-runtime-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_durable_json_refuses_corruption_and_preserves_previous_on_invalid_write(self):
        path = self.root / "nested/state.json"
        runtime.atomic_json(path, {"cursor": 4})
        with self.assertRaises(ValueError):
            runtime.atomic_json(path, {"cursor": float("nan")})
        self.assertEqual(runtime.read_json(path), {"cursor": 4})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        path.write_text("{interrupted", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "durable JSON"):
            runtime.read_json(path, {})
        self.assertEqual(runtime.read_json(self.root / "absent", {"default": True}), {"default": True})

    def test_source_pins_catch_program_evidence_and_production_changes(self):
        program, evidence = self.root / "program.py", self.root / "measurement.json"
        program.write_text("original program\n")
        evidence.write_text('{"measured": 2}')
        with patch("scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.evidence.source_snapshot",
                   return_value={"fixed": "production"}) as production:
            pins = runtime.source_snapshot([program], [evidence])
            runtime.require_sources(pins)
            evidence.write_text('{"measured": 3}')
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                runtime.require_sources(pins)
            evidence.write_text('{"measured": 2}')
            program.write_text("changed program\n")
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                runtime.require_sources(pins)
            program.write_text("original program\n")
            production.return_value = {"fixed": "changed"}
            with self.assertRaisesRegex(RuntimeError, "sources/evidence changed"):
                runtime.require_sources(pins)
        self.assertIn(str(evidence), pins["files"])
        self.assertIn(str(program), pins["files"])

    def test_new_run_pins_console_relay_without_refreshing_old_evidence(self):
        relay_source = self.root / "relay.py"
        relay_source.write_text("original log relay\n")
        with patch("scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.evidence.source_snapshot",
                   return_value={"fixed": "production"}), \
                patch.object(runtime.subprocess_log, "__file__", str(relay_source)):
            pins = runtime.source_snapshot()
            self.assertIn(str(relay_source), pins["files"])
            runtime.require_sources(pins)
            relay_source.write_text("changed log relay\n")
            with self.assertRaisesRegex(RuntimeError, "sources/evidence changed"):
                runtime.require_sources(pins)

    def test_shared_lock_rejects_nested_roots_and_releases_after_exception(self):
        lock = self.root / ".batch_execution.lock"
        with runtime.ExecutionLock(lock):
            with self.assertRaisesRegex(RuntimeError, "shared execution lock"):
                with runtime.ExecutionLock(lock):
                    self.fail("A second live coordinator acquired the same lock")
        with self.assertRaisesRegex(ValueError, "fixture"):
            with runtime.ExecutionLock(lock):
                raise ValueError("fixture")
        with runtime.ExecutionLock(lock):
            pass
        self.assertEqual(runtime.GLOBAL_EXECUTION_LOCK,
                         runtime.PACKAGE_ROOT / ".runtime/tunnel_static_trials/.batch_execution.lock")

    def fake_process(self, *, pid=123, ticks="456", root=None, argv=None, pgid=None):
        proc = self.root / "proc"
        node = proc / str(pid)
        node.mkdir(parents=True, exist_ok=True)
        argv = argv or ["/usr/bin/python3", "-u", "-m", "scenario_runner.tools.run_tunnel_static_trial"]
        (node / "cmdline").write_bytes(b"\0".join(os.fsencode(value) for value in argv) + b"\0")
        (node / "environ").write_bytes(os.fsencode(runtime.ROOT_ENV + "=" + str(root or self.root)) + b"\0")
        (node / "stat").write_text(str(pid) + " (python fixture) " +
                                   " ".join(["S", "1", str(pgid or pid)] + ["0"] * 16 + [ticks]))
        return proc

    def test_every_signal_requires_birth_root_entrypoint_and_own_process_group(self):
        module = "scenario_runner.tools.run_tunnel_static_trial"
        command = ["bash", "run_tunnel_static_trial.sh"]
        proc = self.fake_process()
        identity = runtime.process_identity(123, self.root, module, launch_command=command, proc_root=proc)
        kill = Mock()
        self.assertTrue(runtime.signal_owned_group(identity, signal.SIGINT, proc_root=proc, killpg=kill))
        kill.assert_called_once_with(123, signal.SIGINT)
        for change in ({"ticks": "457"}, {"root": self.root / "foreign"},
                       {"argv": ["python3", "-m", "foreign_worker"]}, {"pgid": 999}):
            self.fake_process(**change)
            kill.reset_mock()
            with self.assertRaises(RuntimeError):
                runtime.signal_owned_group(identity, signal.SIGKILL, proc_root=proc, killpg=kill)
            kill.assert_not_called()

    def test_wrapper_exec_to_expected_module_preserves_same_birth(self):
        command = ["bash", "/tmp/run_tunnel_static_trial.sh", "--fixture", "one"]
        proc = self.fake_process(argv=command)
        identity = runtime.process_identity(123, self.root, "scenario_runner.tools.run_tunnel_static_trial",
                                            launch_command=command, proc_root=proc)
        self.fake_process()
        kill = Mock()
        self.assertTrue(runtime.signal_owned_group(identity, signal.SIGINT, proc_root=proc, killpg=kill))
        kill.assert_called_once()

    def test_runtime_has_no_sdk_ros_or_model_api_calls(self):
        tree = ast.parse(Path(runtime.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [entry.name for node in ast.walk(tree) if isinstance(node, ast.Import) for entry in node.names]
        self.assertFalse(any(any(word in name.lower() for word in ("grpc", "rospy", "openai", "requests")) for name in imports))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id in ("eval", "exec") for node in ast.walk(tree)))


@unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process birth tokens are required")
class RealSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-search-subprocess-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.processes = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.)

    def worker(self, name, source):
        path = self.root / (name + ".py")
        path.write_text(source, encoding="utf-8")
        return [sys.executable, "-u", "-m", name]

    def test_timeout_kills_owned_group_and_preserves_unrelated_process(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        self.processes.append(foreign)
        command = self.worker("owned_timeout_fixture", "import os, signal, subprocess, sys, time\n"
                              "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                              "child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(20)'])\n"
                              "open('child.pid', 'w').write(str(child.pid))\n"
                              "print(os.environ['AIMDRIVE_TUNNEL_BATCH_ROOT'], flush=True)\n"
                              "time.sleep(20)\n")
        starts = []
        result = runtime.bounded_run(command, self.root / "worker.log", .8, campaign_root=self.root,
                                     stage="offline_fixture", grace=.2, cwd=self.root, on_start=starts.append)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["forced_kill"])
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertEqual(len(starts), 1)
        self.assertIsNotNone(starts[0]["process_identity"])
        self.assertEqual(starts[0]["process_identity"], result["process_identity"])
        self.assertIsNone(foreign.poll())
        child = int((self.root / "child.pid").read_text())
        deadline = time.monotonic() + 3.
        while time.monotonic() < deadline and runtime._raw_identity(child) is not None:
            time.sleep(.05)
        self.assertIsNone(runtime._raw_identity(child), "Owned group descendant survived shutdown")
        self.assertFalse((self.root / "__pycache__").exists())

    def test_interrupt_callback_and_checkpoint_are_recorded(self):
        command = self.worker("owned_interrupt_fixture", "import time\ntime.sleep(20)\n")
        checkpoint = self.root / "inflight.json"
        result = runtime.bounded_run(command, self.root / "worker.log", 10., campaign_root=self.root,
                                     stage="trial_fixture", stop_requested=lambda: checkpoint.exists(),
                                     cwd=self.root, grace=.3,
                                     on_start=lambda fields: runtime.atomic_json(checkpoint, fields))
        self.assertTrue(result["interrupted"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(runtime.read_json(checkpoint)["process_identity"], result["process_identity"])
        self.assertIsNone(runtime._raw_identity(result["process_pid"]))

    def test_wrapper_exec_is_proven_with_real_module_handoff(self):
        self.worker("owned_exec_fixture", "import signal,time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(20)\n")
        wrapper = self.root / "launcher.sh"
        wrapper.write_text("#!/usr/bin/env bash\nexec " + sys.executable + " -u -m owned_exec_fixture\n")
        result = runtime.bounded_run(["bash", str(wrapper)], self.root / "wrapper.log", .5,
                                     campaign_root=self.root, stage="wrapper_fixture", grace=.2,
                                     expected_module="owned_exec_fixture", cwd=self.root)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["forced_kill"])
        self.assertIsNone(runtime._raw_identity(result["process_pid"]))


# Pre-drive retries do not consume weather history for the actual run.

def choose_environment(path, seed):
    context=SimpleNamespace(cfg={"profile_state_path":str(path),"profiles":[
        {"name":"sun","weather":"SUNNY"},{"name":"fog","weather":"FOGGY"}],
        "scenario_times":[11,13,15]},random=random.Random(seed))
    context.profile_state_path=lambda:route_profile.profile_state_path(context)
    context.read_profile_state=lambda:route_profile.read_profile_state(context)
    context.validate_profiles=lambda profiles:route_profile.validate_profiles(context,profiles)
    with redirect_stdout(io.StringIO()):
        index,profile,previous=route_profile.select_profile(context)
        context.previous_profile_state=previous
        hour=route_profile.select_scenario_hour(context)
    return {"run_seed":seed,"profile_index":index,"weather":profile["weather"],"time_hour":hour}


class EnvironmentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.profile=self.root/"full_loop_tunnel_static_profile.json"

    def previous_sunny(self):
        # Preserve the exact incoming bytes, not only their interpreted fields.
        original=b'{ "profile_index": 0, "weather": "SUNNY", "time_hour": 13, "run_seed": 99 }\n'
        self.profile.write_bytes(original)
        return original

    def test_owned_failed_preflight_restores_incoming_weather_and_time_exactly(self):
        incoming=self.previous_sunny()
        attempt=choose_environment(self.profile,100)
        self.assertEqual(attempt["weather"],"FOGGY")
        self.assertNotEqual(attempt["time_hour"],13)
        runtime.atomic_json(self.profile,attempt)
        execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),incoming)
        self.assertEqual(choose_environment(self.profile,100),attempt)

    def test_first_run_retry_removes_only_its_own_created_profile(self):
        unrelated=self.root/"full_loop_static_profile.json"
        unrelated.write_bytes(b'unchanged unrelated mode history\n')
        runtime.atomic_json(self.profile,choose_environment(self.profile,100))
        execution.restore_preflight_environment_history(self.profile,None,100)
        self.assertFalse(self.profile.exists())
        self.assertEqual(unrelated.read_bytes(),b'unchanged unrelated mode history\n')
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),[unrelated.name])

    def test_external_seed_change_is_rejected_without_overwrite(self):
        incoming=self.previous_sunny()
        external={"profile_index":1,"weather":"FOGGY","time_hour":15,"run_seed":201}
        runtime.atomic_json(self.profile,external)
        before=self.profile.read_bytes()
        with self.assertRaisesRegex(RuntimeError,"outside this owned preflight"):
            execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),before)

    def test_external_new_first_run_history_is_not_deleted(self):
        runtime.atomic_json(self.profile,{"run_seed":201,"weather":"SUNNY","time_hour":13})
        before=self.profile.read_bytes()
        with self.assertRaisesRegex(RuntimeError,"outside this owned preflight"):
            execution.restore_preflight_environment_history(self.profile,None,100)
        self.assertEqual(self.profile.read_bytes(),before)

    def test_no_change_is_a_read_only_no_op(self):
        incoming=self.previous_sunny()
        execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),incoming)
        self.assertFalse(self.profile.with_suffix(".json.preflight_restore.tmp").exists())

    def test_ordering_retry_recovers_then_restores_so_actual_run_remains_foggy(self):
        fixture=SearchTests(methodName="test_structured_ordering_retry_is_bounded_and_failed_actual_drive_counts")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        incoming=self.previous_sunny()
        fixture.args.run_seed=100
        fixture.args.profile_state_path=self.profile
        campaign,row,_events=fixture.initialize_row()
        events=[]
        environments=[]

        def process(command,*args):
            self.assertEqual(command[command.index("--run-seed")+1],"100")
            self.assertEqual(command[command.index("--profile-state-path")+1],str(self.profile))
            self.assertEqual(self.profile.read_bytes(),incoming)
            environment=choose_environment(self.profile,100)
            environments.append(environment)
            events.append("attempt_%s_%s" % (len(environments),environment["weather"]))
            runtime.atomic_json(self.profile,environment)
            return {"returncode":1 if len(environments)==1 else 0,"timed_out":False,"forced_kill":False}

        def recover(action,directory=None):
            self.assertEqual(action,"recover")
            self.assertEqual(json.loads(self.profile.read_text())["weather"],"FOGGY")
            events.append("recover")
            return {"status":"recovered"}

        success=outcome("actual_fail")
        success.update(strict_pass=True,status="passed",failure_reason="")
        campaign.process=process
        campaign.helper=recover
        with patch.object(execution,"classify_trial",side_effect=[outcome("ordering"),success]),patch.object(
                execution,"read_trace",return_value={"valid":True,"evidence_kind":"completed_actual_trace"}):
            campaign.trial(row)
        self.assertEqual(events,["attempt_1_FOGGY","recover","attempt_2_FOGGY","recover"])
        self.assertEqual(environments[0],environments[1])
        self.assertEqual(campaign.state["actual_drives"],1)
        self.assertEqual(campaign.state["actual_passes"],1)
        self.assertEqual(campaign.state["preflight_rejects"],1)
        self.assertFalse(campaign.state["attempts"][0]["outcome"]["actual_driving"])
        self.assertEqual(campaign.state["attempts"][0]["disposition"],"retry_ordering")
        self.assertEqual(json.loads(self.profile.read_text()),environments[1])


if __name__ == "__main__":
    unittest.main()
