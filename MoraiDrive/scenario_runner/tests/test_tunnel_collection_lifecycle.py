"""Collection adapters reuse the guarded fixture and preserve its lifecycle.

No simulator, ROS node, external Expert, or actual recorder is started here.
BorrowedTrialClient and MissionScenario's real tick dispatch remain exercised.
"""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..scenarios.segment_modes import tunnel_static_trial as drive
from ..scenarios.segment_modes import scenario as scenario_module
from ..scenarios.segment_modes.modes import TUNNEL_NORMAL_START
from ..scenarios.segment_modes.scenario import MissionScenario
from ..scenarios.full_loop_parts.tunnel_static_parts import trial as owner


class TunnelCollectionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-collection-lifecycle-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode_000"
        self.events = []
        self.original_world = NS(identity="already-owned-fixture")
        harness = self

        class FixtureClient:
            def __init__(self):
                self.client_key = "collection_fixture_owner"
                self.world = harness.original_world
                self.owner_start = Mock(side_effect=lambda _tf: harness.events.append("owner_world_start"))
                self.owner_stop = Mock()
                self.reset_actors = Mock(return_value=True)
                self.resume_world = Mock()
                self.get_ego = Mock(return_value=NS())

            def start_world(self, transform):
                self.owner_start(transform)

            def stop(self):
                self.owner_stop()

            def make_transform(self, x, y, z, yaw):
                return NS(location=NS(x=x, y=y, z=z), yaw=yaw)

        self.grpc = FixtureClient()
        self.gc = {"paths": {"mgeo_root": "fixture-map", "runtime_root": "original-runtime", "expert_root": "shared-expert"}}
        self.sc = {"sync_runtime": {"enabled": True, "other": "retained"},
                   "profile_state_path": str(self.root / "profile.json"),
                   "tunnel_static_profile_state_path_explicit": True,
                   "random_seed": 42, "expert_max_velocity_kmh": None}
        self.policy = {"approach_m": 120., "departure_m": 70.,
                       "planning_road_margin_m": .35, "runtime_road_margin_m": .15}
        self.report = {"planned": {"count": 1}, "measured_route_screen": {
            "road_boundary_policy": self.policy, "measured_ego_body": {"width_m": 2.17}},
            "observed": [{"raw_actor_id": "101", "model": "WoodBox"}]}
        class ContextFixture:
            def __new__(cls, *_args, **_kwargs):
                return harness.trial

            def __init__(self, *_args, **_kwargs):
                pass

        self.context_type = ContextFixture
        self.trial = object.__new__(ContextFixture)
        self.trial.references = {"101": {}}
        self.trial.report = {"status": "armed"}
        self.trial.output_dir = self.root / "guard"
        self.trial.inspect, self.trial.fail = Mock(), Mock()
        self.monitor = NS(start=Mock(side_effect=lambda: self.events.append("monitor_start")),
                          close=Mock(side_effect=lambda: self.events.append("monitor_close")))
        self.boundary_factory = Mock(return_value=NS())
        self.start_transform = self.grpc.make_transform(*TUNNEL_NORMAL_START.xyz,
                                                       TUNNEL_NORMAL_START.yaw_deg)
        self.raw_outcome = {"status": "mission_completed", "full_mission_completed": True}
        self.mark_trial_passed = True
        self.reuse_world = True
        self.actors_destroyed = False
        self.raw_starts = 0
        self.instances = []
        harness = self

        class FakeRawLayer:
            def __init__(self, grpc, loader, gc, sc, *, output, capture_token):
                self.grpc, self.map_loader = grpc, loader
                self.global_cfg, self.cfg = copy.deepcopy(gc), copy.deepcopy(sc)
                self.output = Path(output)
                self.capture_token = capture_token
                self.output.mkdir(parents=True, exist_ok=False)
                self.collector = None
                harness.events.append("raw_constructor")
                harness.instances.append(self)

            def setup(self):
                if not hasattr(self, "tunnel_static_trial_context"):
                    raise AssertionError("Trial guards were not bound before raw setup")
                if self.collector is not None:
                    raise AssertionError("A second collector was started")
                harness.events.append("guard_bound_before_setup")
                if harness.reuse_world:
                    self.grpc.start_world(harness.start_transform)
                harness.raw_starts += 1
                self.collector = object()
                harness.events.extend(["raw_ready", "expert_started"])

            def on_expert_timeline_tick(self, *args):
                harness.events.append("raw_health_tick")
                return super().on_expert_timeline_tick(*args)

            def run(self):
                self.setup()
                self.on_expert_timeline_tick(.1, {"x": 1.}, 5., 100.)
                if harness.mark_trial_passed:
                    self.tunnel_static_trial_context.report["status"] = "passed"
                # The existing recorder seals before outer fixture cleanup.
                harness.events.extend(["expert_stopped", "record_off_ack"])
                (self.output / "raw").mkdir()
                (self.output / "raw/RAW_COMPLETE.json").write_text('{"status":"completed"}')
                harness.events.append("raw_sealed")
                (self.output / "outcome.json").write_text(json.dumps(harness.raw_outcome))
                (self.output / "episode.json").write_text('{}')
                return copy.deepcopy(harness.raw_outcome)

        class FakeRawMissionScenario(FakeRawLayer, MissionScenario):
            pass

        self.factory = FakeRawMissionScenario
        self.sink = {}
        self.configure = Mock(side_effect=self.assert_configured)
        self.addCleanup(patch.stopall)
        patch.object(drive.screening, "_expert_search", return_value=(
            None, NS(from_map_dir=self.boundary_factory), None, None, None)).start()
        patch.object(drive, "TrialRosEvidence", return_value=self.monitor).start()
        patch.object(drive, "TrialContext", self.context_type).start()
        patch.object(scenario_module.tunnel_baseline, "observe", return_value=None).start()
        patch.object(MissionScenario.__mro__[1], "on_expert_timeline_tick", return_value=None).start()
        patch.object(drive, "park", side_effect=lambda _scenario: self.events.append("verified_park")).start()

    def assert_configured(self, scenario):
        self.assertIs(scenario.tunnel_static_trial_context, self.trial)
        self.assertEqual(self.raw_starts, 0)
        self.events.append("configured_with_guards")

    def callback(self):
        return drive.make_drive_callback(self.grpc, NS(), self.gc, self.sc, self.root / "guard",
            scenario_factory=self.factory,
            scenario_kwargs={"output": self.episode, "capture_token": "existing-collector"},
            configure_scenario=self.configure, result_sink=self.sink,
            collection_episode=self.episode)

    def test_collection_factory_reuses_world_binds_guards_and_keeps_runtime_inside_episode(self):
        before_gc, before_sc = copy.deepcopy(self.gc), copy.deepcopy(self.sc)
        result = self.callback()(self.report, [{"raw_id": "101"}])
        scenario = self.instances[0]
        self.assertIs(result, self.trial.report)
        self.assertEqual(self.raw_starts, 1)
        self.assertIs(scenario.grpc.world, self.original_world)
        self.grpc.owner_start.assert_not_called()
        self.grpc.owner_stop.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "outer fixture helper owns"):
            scenario.grpc.stop()
        self.grpc.owner_stop.assert_not_called()
        self.assertNotEqual(scenario.grpc.client_key, self.grpc.client_key)
        self.assertEqual(scenario.capture_token, "existing-collector")
        self.assertEqual(Path(scenario.global_cfg["paths"]["runtime_root"]),
                         self.episode / "tunnel_static/runner_runtime")
        self.assertEqual(Path(scenario.cfg["ego_trace_path"]), self.episode / "tunnel_static/ego_trace.csv")
        self.assertEqual(self.trial.output_dir, self.episode / "tunnel_static")
        self.assertFalse(scenario.cfg["sync_runtime"]["enabled"])
        self.assertEqual(scenario.cfg["sync_runtime"]["other"], "retained")
        self.assertEqual(scenario.cfg["random_seed"], 42)
        self.assertIsNone(scenario.cfg["expert_max_velocity_kmh"])
        self.assertEqual(self.gc, before_gc)
        self.assertEqual(self.sc, before_sc)
        self.assertEqual(self.events[:5], ["raw_constructor", "configured_with_guards", "monitor_start",
                                         "guard_bound_before_setup", "raw_ready"])
        self.trial.inspect.assert_called_once_with(scenario, .1, {"x": 1.})
        self.boundary_factory.assert_called_once_with("fixture-map", approach_m=120., departure_m=70.,
            planning_margin_m=.35, runtime_margin_m=.15, body_policy={"width_m": 2.17})
        self.trial.fail.assert_not_called()
        self.monitor.close.assert_called_once()

    def test_raw_returned_failure_is_propagated_to_trial_even_after_sealing(self):
        self.raw_outcome = {"status": "failed", "full_mission_completed": False, "reason": "collector stopped"}
        with self.assertRaisesRegex(RuntimeError, "did not complete all actual-drive conditions"):
            self.callback()(self.report, [{"raw_id": "101"}])
        self.assertEqual(self.sink["outcome"], self.raw_outcome)
        self.trial.fail.assert_called_once()
        self.assertIsInstance(self.trial.fail.call_args[0][0], RuntimeError)
        self.assertIn("raw_sealed", self.events)
        self.assertNotIn("verified_park", self.events)
        self.monitor.close.assert_called_once()

    def test_collection_success_does_not_override_failed_or_incomplete_tunnel_guard(self):
        self.mark_trial_passed = False
        with self.assertRaisesRegex(RuntimeError, "did not complete all actual-drive conditions"):
            self.callback()(self.report, [{"raw_id": "101"}])
        self.trial.fail.assert_called_once()
        self.monitor.close.assert_called_once()

    def test_factory_cannot_complete_without_reusing_the_owned_world(self):
        self.reuse_world = False
        with self.assertRaisesRegex(RuntimeError, "exact parked fixture world"):
            self.callback()(self.report, [{"raw_id": "101"}])
        self.trial.fail.assert_called_once()
        self.monitor.close.assert_called_once()

    def test_foreign_ownership_mismatch_fails_before_starting_any_collector(self):
        with self.assertRaisesRegex(RuntimeError, "fixture ownership differs"):
            self.callback()(self.report, [{"raw_id": "foreign"}])
        self.assertEqual(self.instances, [])
        self.assertEqual(self.raw_starts, 0)
        self.monitor.start.assert_not_called()
        self.grpc.owner_start.assert_not_called()

    def test_collection_final_brake_requires_current_final_verification_and_observed_zero_speed(self):
        self.grpc._tunnel_static_common_start=True
        self.callback()(self.report, [{"raw_id": "101"}])
        final_brake = self.instances[0].final_brake
        observed = {"accepted_and_control_blocked": True, "observed_parked_and_zero_speed": True,
                    "final_verified":True,"hard_failures":[]}
        with patch.object(drive, "immediate_hold", return_value=observed):
            self.assertIs(final_brake(), observed)
        for bad in (None, {}, dict(observed, accepted_and_control_blocked=False),
                    dict(observed,final_verified=False),
                    dict(observed, observed_parked_and_zero_speed=False)):
            with patch.object(drive, "immediate_hold", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "parking was not verified"):
                    final_brake()
        self.trial.report['termination_hard_failures']=['native actuator request rejected']
        with patch.object(drive,'immediate_hold',return_value=observed), \
                self.assertRaisesRegex(RuntimeError,'parking was not verified'):
            final_brake()

    def test_fixture_owner_cleans_up_only_after_common_raw_seal_and_then_restores_foreign_actors(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts import replay, runtime, safety, layout, live
        from ..scenarios.segment_modes import tunnel_static_replay as facade
        library = self.root / "library.json"
        library.write_text('{}')
        candidate_path = self.root / "candidate.json"
        candidate_path.write_text(json.dumps(self.report["planned"]))
        selected = {"id": "a" * 64, "paths": {"candidate": str(candidate_path)},
                    "file_sha256": {}, "strata": {"count": "1"}}
        verified = {"eligible": [selected], "library_sha256": "library-sha", "eligible_count": 121}
        frozen = {"production": {"runtime_files": {}}}
        restored_ids = []

        def relocate(_grpc, _ego, output, ids, action):
            self.events.append("scene_" + action)
            if action == "move":
                restored_ids.extend(ids)
                return {"status": "moved"}
            self.assertIn("owned_fixture_deleted", self.events)
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output) / "restore.json").write_text('{"status":"restored"}')
            return {"status": "restored"}

        def preview(_grpc, _loader, _candidate, **kwargs):
            kwargs["drive_fn"](self.report, [{"raw_id": "101"}])
            self.assertEqual(json.loads((self.episode / "raw/RAW_COMPLETE.json").read_text())["status"], "completed")
            self.assertIn("record_off_ack", self.events)
            self.events.append("owned_fixture_deleted")
            return {"status": "valid_driving_trial", "cleanup": {"verified": True}}

        with patch.object(facade, "DEFAULT_LIBRARY", library), \
                patch.object(facade, "_curation_pins", return_value={str(library): replay.digest(library)}), \
                patch.object(layout, "tunnel_geometry", return_value={}), \
                patch.object(replay, "verify_library", return_value=verified), \
                patch.object(replay, "balanced_choice", return_value=(selected, {})), \
                patch.object(replay, "current_runtime_sources", return_value={}), \
                patch.object(runtime, "source_snapshot", return_value=frozen), \
                patch.object(runtime, "require_sources") as require_sources, \
                patch.object(runtime, "ExecutionLock") as lock, \
                patch.object(owner, "_controller_publishers", return_value=[]), \
                patch.object(owner, "block_and_park"), \
                patch.object(owner, "prepare_common_start", autospec=True), \
                patch.object(owner, "settle_initial_ego"), \
                patch.object(owner, "align_parked_ego"), \
                patch.object(live, "_park"), \
                patch.object(live, "_inventory", return_value={key: {} for key in ("2", "3", "4", "5")}), \
                patch.object(safety, "relocate", side_effect=relocate), \
                patch.object(owner, "preview_candidate", side_effect=preview):
            result = owner.run_collection_episode(self.grpc, NS(link_set={}), self.gc, self.sc,
                scenario_factory=self.factory,
                scenario_kwargs={"output": self.episode, "capture_token": "existing-collector"},
                configure_scenario=self.configure, output_dir=self.root / "preparation", seed=42)
        self.assertEqual(result["status"], "mission_completed")
        self.assertTrue(result["tunnel_static"]["passed"])
        self.assertEqual(self.grpc.owner_start.call_count, 1)
        self.assertEqual(self.raw_starts, 1)
        self.assertEqual(restored_ids, ["2", "4", "5"])
        self.assertNotIn("3", restored_ids)
        self.assertLess(self.events.index("raw_sealed"), self.events.index("owned_fixture_deleted"))
        self.assertLess(self.events.index("owned_fixture_deleted"), self.events.index("scene_restore"))
        self.assertEqual(require_sources.call_count, 2)
        lock.assert_called_once_with()
        proof = json.loads((self.episode / "tunnel_static/COLLECTION_TUNNEL_COMPLETE.json").read_text())
        self.assertEqual(proof["status"], "passed")
        self.assertTrue(proof["raw_record_complete"])
        self.assertTrue(proof["scene_restored"])


if __name__ == "__main__":
    unittest.main()
