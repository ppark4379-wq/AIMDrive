"""Adaptive accounting, source barriers and bounded cleanup without live APIs."""

import copy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from ..tools import run_tunnel_static_search as search
from ..tools import tunnel_static_search_runtime as runtime


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


if __name__ == "__main__":
    unittest.main()
