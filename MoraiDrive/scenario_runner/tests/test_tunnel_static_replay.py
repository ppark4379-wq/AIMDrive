"""Normal-command selection, repeat semantics and curation guards; no live API."""

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..scenarios.segment_modes import tunnel_static_replay as scenario
from ..tools import replay_tunnel_static_library as replayer
from ..tools.tunnel_static_search_runtime import atomic_json
from ..tools.tunnel_static_search_runtime import read_json


class TunnelStaticReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-static-entry-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = self.root / "approved.json"
        self.sidecar = self.root / "excluded.json"
        self.excluded_id = "%064x" % 999
        atomic_json(self.sidecar, {"schema": "aimdrive_tunnel_excluded_no_required_avoidance_v1",
            "excluded_count": 1, "candidates": [{"candidate_sha256": self.excluded_id}]})
        self.document = {"schema": replayer.LIBRARY_SCHEMA, "active_random_library": True,
            "approved_count": 3, "candidates": [{"candidate_sha256": "%064x" % value} for value in (1, 2, 3)],
            "curation": {"schema": "aimdrive_tunnel_random_library_curation_v1", "excluded_count": 1,
                "exclusion_report_path": str(self.sidecar), "exclusion_report_sha256": replayer.digest(self.sidecar)}}
        atomic_json(self.library, self.document)
        self.args = SimpleNamespace(tunnel_static_library=self.library, tunnel_static_output=self.root / "session",
            tunnel_static_seed=100, check=False, once=False, max_runs=3, no_sync_runtime=True)
        self.global_cfg = {"paths": {"mgeo_root": str(self.root / "map")}}
        self.config = {"auto_repeat": True, "repeat_delay_sec": 0.}
        self.config["profile_state_path"] = str(self.root / "full_loop_profile.json")
        self.policy = {"source_files": {}, "expert_speed": {"max_velocity_kmh": 37.}}
        for name, value in (("describe", self.policy),
                            ("approved_ids", {"%064x" % i for i in (1, 2, 3)}),
                            ("approve", None)):
            patcher = patch.object(scenario.execution_policy, name, return_value=value)
            setattr(self, "policy_" + name, patcher.start())
            self.addCleanup(patcher.stop)
        self.calls = []
        def fake_run(args, *, live_fn=None):
            self.calls.append(args)
            result = {"schema": replayer.SCHEMA, "status": "passed" if args.live else "selected",
                "live_requested": args.live, "replay_actual_pass": args.live, "eligible_count": 3,
                "selected": {"id": "%064x" % (len(self.calls) % 3 + 1), "strata": {"count": 1,
                    "longitudinal_third": 0, "actor_kind": "obstacle", "actual_behavior": "detour"}}}
            if args.live:
                self.assertIsNotNone(live_fn)
            else:
                self.assertIsNone(live_fn)
            atomic_json(args.output, result)
            return result
        self.runner = Mock(side_effect=fake_run)

    def run_scenario(self):
        with patch.object(scenario.replayer, "run", self.runner), patch.object(scenario.time, "sleep"):
            return scenario.run_tunnel_static_scenario(self.args, self.global_cfg, self.config)

    def test_check_selects_once_without_live_delegate_or_live_claim(self):
        self.args.check = True
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.calls[0].live)
        self.assertEqual(self.calls[0].seed, 100)
        self.assertEqual(self.calls[0].mgeo_root, self.root / "map")
        history, _sha = replayer.read_history(self.root / "session/selection_history.json")
        self.assertEqual(history[0]["status"], "selected")
        self.assertFalse(history[0]["replay_actual_pass"])

    def test_live_repeat_uses_independent_configuration_fresh_paths_seeds_and_history(self):
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual([args.seed for args in self.calls], [100, 100, 100])
        self.assertTrue(all(args.expert_max_velocity_kmh is None for args in self.calls))
        self.assertTrue(all(args.profile_state_path == self.root / "full_loop_tunnel_static_profile.json" for args in self.calls))
        self.assertTrue(all(args.live for args in self.calls))
        self.assertEqual(len({args.output for args in self.calls}), 3)
        self.assertIsNone(self.calls[0].history)
        self.assertEqual(self.calls[1].history, self.root / "session/selection_history.json")
        history, _sha = replayer.read_history(self.root / "session/selection_history.json")
        self.assertEqual(len(history), 3)

    def test_once_overrides_repeat_and_max_runs(self):
        self.args.once = True
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(len(self.calls), 1)

    def test_normal_run_uses_entire_library_without_current_policy_approval_filter(self):
        self.policy_approved_ids.return_value = set()
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(args.candidate_ids is None for args in self.calls))
        self.policy_approved_ids.assert_not_called()
        self.policy_approve.assert_not_called()

    def test_one_current_setting_approval_does_not_reduce_random_catalogue(self):
        self.policy_approved_ids.return_value = {"%064x" % 1}
        self.assertEqual(self.run_scenario(), 0)
        self.assertTrue(all(args.candidate_ids is None for args in self.calls))
        self.assertTrue(all(args.selection_pool == "curated_historical_library" for args in self.calls))
        self.policy_approved_ids.assert_not_called()

    def test_revalidation_selects_unapproved_candidates_and_approves_only_a_pass(self):
        self.args.tunnel_static_revalidate = True
        self.args.once = True
        self.policy_approved_ids.return_value = {"%064x" % 1}
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(self.calls[0].candidate_ids, {"%064x" % 2, "%064x" % 3})
        self.assertTrue(self.calls[0].validation_only)
        self.policy_approve.assert_called_once()

    def test_revalidation_failure_is_not_approved_and_next_layout_can_be_tested(self):
        self.args.tunnel_static_revalidate = True
        self.args.once = True
        self.policy_approved_ids.return_value = set()
        self.runner.side_effect = lambda *_args, **_kwargs: {"status":"failed", "replay_actual_pass":False,
            "selected":{"id":"%064x" % 2}, "live_requested":True}
        self.assertEqual(self.run_scenario(), 0)
        self.policy_approve.assert_not_called()

    def test_auto_repeat_false_runs_once(self):
        self.config["auto_repeat"] = False
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(len(self.calls), 1)

    def test_failed_replay_and_interrupt_stop_the_sequence(self):
        self.runner.side_effect = lambda *_args, **_kwargs: {"schema": replayer.SCHEMA, "status": "failed", "replay_actual_pass": False,
            "selected": {"id": "fixture", "strata": {"count": 1, "longitudinal_third": 0, "actor_kind": "obstacle", "actual_behavior": "detour"}}}
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_called_once()
        self.runner.reset_mock()
        self.runner.side_effect = KeyboardInterrupt
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_called_once()

    def test_archived_133_library_and_reintroduced_exclusion_stop_before_replayer(self):
        archive = copy.deepcopy(self.document)
        archive.pop("active_random_library")
        atomic_json(self.library, archive)
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_not_called()
        bad = copy.deepcopy(self.document)
        bad["candidates"][0]["candidate_sha256"] = self.excluded_id
        atomic_json(self.library, bad)
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_not_called()

    def test_curation_report_tamper_stops_before_replayer(self):
        atomic_json(self.sidecar, {"changed": True})
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_not_called()

    def test_hash_valid_malformed_curation_report_fails_cleanly(self):
        for malformed in ([], None, {"schema": "aimdrive_tunnel_excluded_no_required_avoidance_v1",
                                  "excluded_count": 1, "candidates": ["invalid row"]}):
            atomic_json(self.sidecar, malformed)
            document = copy.deepcopy(self.document)
            document["curation"]["exclusion_report_sha256"] = replayer.digest(self.sidecar)
            atomic_json(self.library, document)
            self.assertEqual(self.run_scenario(), 1)
            self.runner.assert_not_called()

    def test_library_changed_during_curation_read_cannot_be_pinned_as_verified(self):
        original = replayer._read
        def changed(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if Path(path) == self.library:
                modified = copy.deepcopy(self.document)
                modified["active_random_library"] = False
                atomic_json(self.library, modified)
            return result
        with patch.object(replayer, "_read", side_effect=changed):
            self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_not_called()

    def test_library_change_after_first_run_prevents_repeat(self):
        previous = self.runner.side_effect
        def changing(args, **kwargs):
            result = previous(args, **kwargs)
            altered = copy.deepcopy(self.document)
            altered["curation"]["edited"] = True
            atomic_json(self.library, altered)
            return result
        self.runner.side_effect = changing
        self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_called_once()

    def test_default_uses_exact_current_121_and_fresh_generated_seed(self):
        self.args.tunnel_static_library = None
        self.args.tunnel_static_seed = None
        self.args.once = True
        default = copy.deepcopy(self.document)
        default["candidates"] = [{"candidate_sha256": "%064x" % index} for index in range(1, 122)]
        default["approved_count"] = 121
        atomic_json(self.library, default)
        with patch.object(scenario, "DEFAULT_LIBRARY", self.library), patch.object(scenario, "_fresh_seed", return_value=123456):
            self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(self.calls[0].library, self.library)
        self.assertEqual(self.calls[0].seed, 123456)

    def test_private_output_history_is_preserved_across_commands(self):
        self.args.once = True
        self.assertEqual(self.run_scenario(), 0)
        self.assertEqual(self.run_scenario(), 0)
        history, _sha = replayer.read_history(self.root / "session/selection_history.json")
        self.assertEqual(len(history), 2)
        self.assertNotEqual(self.calls[0].output, self.calls[1].output)
        self.assertEqual(self.calls[1].history, self.root / "session/selection_history.json")

    def test_check_cannot_claim_a_live_pass(self):
        self.args.check = True
        self.runner.side_effect = lambda *_args, **_kwargs: {"schema": replayer.SCHEMA, "status": "passed", "live_requested": False,
            "replay_actual_pass": True, "selected": {"id": "fixture", "strata": {"count": 1, "longitudinal_third": 0,
                "actor_kind": "obstacle", "actual_behavior": "detour"}}}
        self.assertEqual(self.run_scenario(), 1)
        self.assertFalse((self.root / "session/selection_history.json").exists())

    def test_new_adapter_and_main_facade_have_separate_auditable_pins(self):
        self.args.once = True
        self.assertEqual(self.run_scenario(), 0)
        files = list((self.root / "session").glob("scenario_source_pins_*.json"))
        self.assertEqual(len(files), 1)
        document = read_json(files[0])
        self.assertFalse(document["historical_proof_sources_refreshed"])
        self.assertIn(str(Path(scenario.__file__).resolve()), document["files"])
        self.assertIn(str(scenario.PACKAGE_ROOT / "main_runner.py"), document["files"])

    def test_facade_hash_change_stops_before_live_and_cannot_repeat(self):
        self.args.once = True
        original_digest = replayer.digest
        facade = scenario.PACKAGE_ROOT / "main_runner.py"
        seen = [0]
        def changed(path):
            if Path(path) == facade:
                seen[0] += 1
                if seen[0] > 1:
                    return "changed facade hash"
            return original_digest(path)
        with patch.object(replayer, "digest", side_effect=changed):
            self.assertEqual(self.run_scenario(), 1)
        self.runner.assert_not_called()

    def test_live_delegate_pins_curation_during_guarded_replay(self):
        self.args.once = True
        seen = []
        def selected(args, *, live_fn):
            def native_sources():
                seen.append("native")
            live_fn(self.root / "candidate.json", self.root / "private_trial", require_sources_fn=native_sources,
                    time_budget_sec=15., hold_seconds=5., max_velocity_kmh=10., trial_timeout_sec=240.)
            return {"schema": replayer.SCHEMA, "status": "passed", "replay_actual_pass": True,
                "selected": {"id": "fixture", "strata": {"count": 1, "longitudinal_third": 0,
                    "actor_kind": "obstacle", "actual_behavior": "detour"}}}
        self.runner.side_effect = selected
        def guarded(_candidate, _output, **options):
            options["require_sources_fn"]()
            atomic_json(self.sidecar, {"changed_after_selection": True})
            with self.assertRaisesRegex(RuntimeError, "curation changed"):
                options["require_sources_fn"]()
            return {"outcome": {"strict_pass": True}, "scene_restored": True, "cleanup_verified": True}
        with patch("scenario_runner.tools.run_tunnel_static_search.replay_candidate", side_effect=guarded):
            self.assertEqual(self.run_scenario(), 1)
        self.assertEqual(seen, ["native"])


if __name__ == "__main__":
    unittest.main()
