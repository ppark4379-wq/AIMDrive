"""Tunnel regression checks: lifecycle."""

import copy, json, tempfile, unittest, io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from ..scenarios.segment_modes import tunnel_static_trial as drive, scenario as scenario_module, tunnel_static_startup as startup, tunnel_static_trial as trial
from ..scenarios.segment_modes.modes import TUNNEL_NORMAL_START
from ..scenarios.segment_modes.scenario import MissionScenario
from ..scenarios.full_loop_parts.tunnel_static_parts import trial as owner
from contextlib import redirect_stdout
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.full_loop_scenario import FullLoopScenario


# Collection adapters reuse the guarded fixture and preserve its lifecycle.
#
# No simulator, ROS node, external Expert, or actual recorder is started here.
# BorrowedTrialClient and MissionScenario's real tick dispatch remain exercised.

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


# Shared final braking retains physical checks and sticky teardown failures.

STOPPED = {
    'control_mode': 3, 'gear_mode': 4, 'brake': 1., 'throttle': 0.,
    'speed_norm_raw': 0., 'position_xyz': [1., 2., 28.],
    'brake_hold_verified': True,
}


class Clock:
    def __init__(self):
        self.value = 100.

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def pending(reason='pedals or residual motion pending', **values):
    error = RuntimeError(reason)
    error.readback = dict(STOPPED, **values)
    return error


class FinalBrakeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.grpc = NS()
        self.owner = patch.object(startup, '_ROS_BRAKE_HOLD', None)
        self.owner.start()
        self.addCleanup(self.owner.stop)

    def finish(self, observed, *, command_error=None, controllers=None):
        record = {}
        hold = Mock(side_effect=command_error)
        exclusive = Mock(side_effect=controllers)
        reads = Mock(side_effect=observed)
        with patch.object(startup, 'hold_common_brake', hold), \
                patch.object(startup, '_require_no_controller', exclusive), \
                patch.object(startup, 'common_stopped_readback', reads):
            result = startup.finish_common_brake(
                self.grpc, record, now=self.clock.now, sleep=self.clock.sleep,
            )
        return result, record, hold, reads

    def test_final_brake_keeps_common_duration_before_actual_readback(self):
        times = []

        def observed(grpc):
            times.append(self.clock.now())
            return NS(), copy.deepcopy(STOPPED)

        result, record, hold, reads = self.finish(observed)
        self.assertEqual(result, STOPPED)
        self.assertGreaterEqual(times[0], 101.5)
        self.assertGreaterEqual(record['brake_duration_sec'], 1.5)
        self.assertTrue(record['final_verified'])
        self.assertEqual(record['status'], 'verified')
        self.assertEqual([row['status'] for row in record['checks']], ['verified'])
        hold.assert_called_once_with(self.grpc)
        reads.assert_called_once_with(self.grpc)

    def test_kinetic_and_pedal_pending_can_settle_without_changing_thresholds(self):
        record = {}
        reads = Mock(side_effect=[pending(speed_norm_raw=.1), pending(brake=.9),
                                  (NS(), copy.deepcopy(STOPPED))])
        with patch.object(startup, 'hold_common_brake'), \
                patch.object(startup, '_require_no_controller'), \
                patch.object(startup, 'common_stopped_readback', reads):
            result = startup.finish_common_brake(
                self.grpc, record, now=self.clock.now, sleep=self.clock.sleep,
            )
        self.assertEqual(result, STOPPED)
        self.assertEqual(reads.call_count, 3)
        self.assertTrue(record['final_verified'])
        self.assertEqual([row['status'] for row in record['checks']], ['pending', 'pending', 'verified'])
        self.assertEqual(record['checks'][0]['readback']['speed_norm_raw'], .1)
        self.assertEqual(record['checks'][1]['readback']['brake'], .9)
        self.assertLessEqual(self.clock.now(), 104.5)

    def test_native_command_rejection_fails_without_retries_or_stop_claim(self):
        record = {}
        reads = Mock()
        with patch.object(startup, 'hold_common_brake', side_effect=RuntimeError('native command rejected')), \
                patch.object(startup, 'common_stopped_readback', reads), \
                self.assertRaisesRegex(RuntimeError, 'native command rejected'):
            startup.finish_common_brake(self.grpc, record, now=self.clock.now, sleep=self.clock.sleep)
        reads.assert_not_called()
        self.assertEqual(self.clock.now(), 100.)
        self.assertFalse(record['final_verified'])
        self.assertEqual(record['status'], 'failed')

    def test_competing_controller_interrupts_braking_immediately(self):
        record = {}
        reads = Mock()
        with patch.object(startup, 'hold_common_brake'), \
                patch.object(startup, '_require_no_controller', side_effect=RuntimeError('foreign controller')), \
                patch.object(startup, 'common_stopped_readback', reads), \
                self.assertRaisesRegex(RuntimeError, 'foreign controller'):
            startup.finish_common_brake(self.grpc, record, now=self.clock.now, sleep=self.clock.sleep)
        reads.assert_not_called()
        self.assertFalse(record['final_verified'])
        self.assertEqual(self.clock.now(), 100.)

    def test_rpc_read_failure_is_fatal_and_never_treated_as_kinetic_pending(self):
        record = {}
        reads = Mock(side_effect=TimeoutError('native state unavailable'))
        with patch.object(startup, 'hold_common_brake'), \
                patch.object(startup, '_require_no_controller'), \
                patch.object(startup, 'common_stopped_readback', reads), \
                self.assertRaisesRegex(TimeoutError, 'native state unavailable'):
            startup.finish_common_brake(self.grpc, record, now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(reads.call_count, 1)
        self.assertFalse(record['final_verified'])
        self.assertEqual(record['checks'], [])
        self.assertAlmostEqual(self.clock.now(), 101.5)

    def test_nonfinite_pedal_observation_is_fatal_even_with_readback_marker(self):
        for values in ({'brake': None}, {'throttle': None}):
            with self.subTest(values=values):
                clock = Clock()
                record = {}
                reads = Mock(side_effect=pending(**values))
                with patch.object(startup, 'hold_common_brake'), \
                        patch.object(startup, '_require_no_controller'), \
                        patch.object(startup, 'common_stopped_readback', reads), \
                        self.assertRaisesRegex(RuntimeError, 'Nonfinite final actuator'):
                    startup.finish_common_brake(self.grpc, record, now=clock.now, sleep=clock.sleep)
                self.assertEqual(reads.call_count, 1)
                self.assertFalse(record['final_verified'])

    def test_unsettled_physical_readback_expires_within_additional_three_seconds(self):
        record = {}
        reads = Mock(side_effect=pending(speed_norm_raw=.051))
        with patch.object(startup, 'hold_common_brake'), \
                patch.object(startup, '_require_no_controller'), \
                patch.object(startup, 'common_stopped_readback', reads), \
                self.assertRaisesRegex(RuntimeError, 'within three seconds'):
            startup.finish_common_brake(self.grpc, record, now=self.clock.now, sleep=self.clock.sleep)
        self.assertGreater(reads.call_count, 1)
        self.assertAlmostEqual(self.clock.now(), 104.5)
        self.assertFalse(record['final_verified'])
        self.assertTrue(all(row['status'] == 'pending' for row in record['checks']))


class ImmediateCommonStopTests(unittest.TestCase):
    def context(self, *, running=False, status='running', stop_error=None):
        events = []
        controller = NS(process=NS(poll=lambda: None)) if running else None
        context = NS(
            cfg={'mission_mode': trial.MODE},
            grpc=NS(world=object(), _tunnel_static_common_start=True,
                    get_ego=Mock(return_value=NS())),
            expert_controller=controller,
            tunnel_static_trial_context=NS(report={'status': status}, persist=Mock()),
        )

        def stop():
            events.append('stop-expert')
            if stop_error:
                raise stop_error
            context.expert_controller = None

        context.stop_expert_controller = stop
        context.stop_drive_control = lambda: events.append('stop-drive')
        return context, events

    def invoke(self, context, events, *, block_error=None, final_error=None):
        def block(grpc):
            events.append('block-external-input')
            if block_error:
                raise block_error
            return {'commands_accepted': True, 'external_input_disabled': True}

        def finish(grpc, evidence):
            events.append('verified-final-brake')
            if final_error:
                evidence.update(status='failed', final_verified=False, failure_reason=str(final_error))
                raise final_error
            evidence.update(status='verified', final_verified=True, actual_stop=copy.deepcopy(STOPPED))
            return copy.deepcopy(STOPPED)

        with patch.object(startup, 'block_common_input', side_effect=block) as blocked, \
                patch.object(startup, 'finish_common_brake', side_effect=finish):
            outcome = trial.immediate_hold(context)
        return outcome, blocked

    def test_common_completed_cleanup_never_switches_to_keyboard(self):
        for running, status in ((False, 'running'), (True, 'passed')):
            with self.subTest(running=running, status=status):
                context, events = self.context(running=running, status=status)
                outcome, blocked = self.invoke(context, events)
                blocked.assert_not_called()
                self.assertEqual(events, ['stop-expert', 'verified-final-brake'])
                self.assertTrue(outcome['final_verified'])
                self.assertTrue(outcome['accepted_and_control_blocked'])
                self.assertEqual(outcome['observed_gear_mode'], 4)
                self.assertTrue(context.tunnel_static_trial_context.report['final_stop_verified'])

    def test_owned_running_expert_is_input_blocked_before_stop_and_ros_brake(self):
        context, events = self.context(running=True)
        outcome, blocked = self.invoke(context, events)
        self.assertEqual(events, ['block-external-input', 'stop-expert', 'verified-final-brake'])
        blocked.assert_called_once_with(context.grpc)
        self.assertTrue(outcome['final_verified'])
        self.assertEqual(context.tunnel_static_trial_context.report['termination_hard_failures'], [])

    def test_accepted_but_delayed_input_mode_readback_is_not_permanent_failure(self):
        context, events = self.context(running=True)
        delayed = RuntimeError('keyboard mode observation pending')
        delayed.readback_pending = True
        delayed.commands_accepted = True
        delayed.control_mode = 3
        outcome, _ = self.invoke(context, events, block_error=delayed)
        report = context.tunnel_static_trial_context.report
        self.assertTrue(outcome['final_verified'])
        self.assertTrue(outcome['input_block_phase']['observation_pending'])
        self.assertFalse(outcome['input_block_phase']['external_input_disabled'])
        self.assertEqual(report['termination_hard_failures'], [])
        self.assertTrue(report['all_immediate_hold_calls_accepted'])
        self.assertTrue(report['final_stop_verified'])

    def test_rejected_input_block_remains_fatal_after_later_verified_stop(self):
        context, events = self.context(running=True)
        first, _ = self.invoke(context, events, block_error=RuntimeError('native KEY request rejected'))
        self.assertFalse(first['final_verified'])
        second, blocked = self.invoke(context, events)
        report = context.tunnel_static_trial_context.report
        blocked.assert_not_called()
        self.assertTrue(second['final_verified'])
        self.assertTrue(report['final_stop_verified'])
        self.assertFalse(report['all_immediate_hold_calls_accepted'])
        self.assertEqual(report['termination_hard_failures'], ['input_block: native KEY request rejected'])
        self.assertEqual(len(report['immediate_hold_history']), 2)
        self.assertFalse(report['immediate_hold_history'][0]['final_verified'])

    def test_failed_owned_expert_teardown_is_not_masked_by_successful_brake(self):
        context, events = self.context(running=True, stop_error=RuntimeError('owned Expert did not stop'))
        outcome, _ = self.invoke(context, events)
        self.assertFalse(outcome['final_verified'])
        self.assertFalse(outcome['accepted_and_control_blocked'])
        self.assertIn('stop_owned_expert: owned Expert did not stop',
                      context.tunnel_static_trial_context.report['termination_hard_failures'])

    def test_terminal_readback_failure_remains_in_history_after_retry(self):
        context, events = self.context()
        first, _ = self.invoke(context, events, final_error=RuntimeError('final brake did not settle'))
        self.assertFalse(first['final_verified'])
        second, _ = self.invoke(context, events)
        report = context.tunnel_static_trial_context.report
        self.assertTrue(second['final_verified'])
        self.assertFalse(report['all_immediate_hold_calls_accepted'])
        self.assertEqual(report['termination_hard_failures'], ['final_brake: final brake did not settle'])
        self.assertEqual(report['immediate_hold_history'][0]['readback_error'], 'final brake did not settle')

    def test_repeated_successful_park_retains_final_proof_without_keyboard_transition(self):
        context, events = self.context(status='passed')
        with patch.object(startup, 'block_common_input') as blocked, \
                patch.object(startup, 'finish_common_brake', return_value=copy.deepcopy(STOPPED)), \
                patch.object(trial.live, '_park', return_value=(NS(), copy.deepcopy(STOPPED))):
            trial.park(context)
            trial.park(context)
        blocked.assert_not_called()
        report = context.tunnel_static_trial_context.report
        self.assertTrue(report['ego_stopped_and_parked'])
        self.assertTrue(report['all_immediate_hold_calls_accepted'])
        self.assertEqual(len(report['immediate_hold_history']), 2)
        self.assertEqual(report['termination_hard_failures'], [])
        self.assertEqual(context.tunnel_static_trial_context.persist.call_count, 2)

    def test_trial_failure_preserves_primary_guard_reason_after_cleanup_failure(self):
        proof = NS(report={
            'status': 'armed', 'failure_reason': '', 'expert_verified': True,
            'live_avoidance_verified': True, 's_avoidance_verified': True,
        }, persist=Mock())
        trial.TrialContext.fail(proof, RuntimeError('root guard: body left mapped road'))
        trial.TrialContext.fail(proof, RuntimeError('cleanup: final brake readback unavailable'))
        self.assertEqual(proof.report['status'], 'failed')
        self.assertEqual(proof.report['failure_reason'], 'root guard: body left mapped road')
        self.assertEqual(proof.report['subsequent_failures'],
                         ['cleanup: final brake readback unavailable'])
        for key in ('expert_verified', 'live_avoidance_verified', 's_avoidance_verified'):
            self.assertFalse(proof.report[key])
        self.assertEqual(proof.persist.call_count, 2)

    def test_park_never_clears_real_hard_failure_history(self):
        context, events = self.context(status='passed')
        context.tunnel_static_trial_context.report['termination_hard_failures'] = ['native request rejected earlier']
        with patch.object(startup, 'finish_common_brake', return_value=copy.deepcopy(STOPPED)), \
                patch.object(trial.live, '_park', return_value=(NS(), copy.deepcopy(STOPPED))), \
                self.assertRaisesRegex(RuntimeError, 'parking handshake failed'):
            trial.park(context)
        self.assertFalse(context.tunnel_static_trial_context.report['all_immediate_hold_calls_accepted'])
        self.assertEqual(context.tunnel_static_trial_context.report['termination_hard_failures'],
                         ['native request rejected earlier'])


class NormalGoalStopTests(unittest.TestCase):
    context = ImmediateCommonStopTests.context
    invoke = ImmediateCommonStopTests.invoke

    def guarded_context(self):
        context, events = self.context(running=True)
        context.expert_controller.process.pid = 42
        context.tunnel_static_trial_context.report.update(
            last_guard_stage='passed', collision_objects=[], expert_verified=False,
            live_avoidance_verified=False, completed=False,
        )
        return context, events

    def request(self, context):
        with patch.object(trial, 'required', return_value=context.tunnel_static_trial_context):
            trial.request_normal_goal_stop(context, elapsed=40., ego_state={'collision_objects': []},
                                           current_s=190., remaining_s=0.)

    def test_goal_intent_is_one_use_and_does_not_premark_completion(self):
        context, events = self.guarded_context()
        self.request(context)
        report = context.tunnel_static_trial_context.report
        self.assertEqual(report['status'], 'running')
        self.assertFalse(report['expert_verified'])
        self.assertFalse(report['live_avoidance_verified'])
        self.assertFalse(report['completed'])
        outcome, blocked = self.invoke(context, events)
        blocked.assert_not_called()
        self.assertTrue(outcome['normal_goal_stop'])
        self.assertEqual(events, ['stop-expert', 'verified-final-brake'])
        self.assertFalse(report['normal_goal_stop_intent']['pending'])
        self.assertTrue(report['normal_goal_stop_intent']['consumed_as_normal_goal'])
        self.assertEqual(report['status'], 'running')
        # A later unclassified stop cannot replay the consumed GOAL permission.
        context.expert_controller = NS(process=NS(pid=42, poll=lambda: None))
        _, blocked = self.invoke(context, events)
        blocked.assert_called_once_with(context.grpc)

    def test_collision_failed_guard_or_hard_failure_cannot_request_normal_stop(self):
        for values in ({'status': 'failed'}, {'last_guard_stage': 'failed'},
                       {'collision_objects': ['7']}, {'termination_hard_failures': ['native rejected']},
                       {'failed_completion_conditions': ['whole_body_in_original_lane']}):
            with self.subTest(values=values):
                context, events = self.guarded_context()
                context.tunnel_static_trial_context.report.update(values)
                with self.assertRaisesRegex(RuntimeError, 'active guarded collision-free'):
                    self.request(context)
                self.assertNotIn('normal_goal_stop_intent', context.tunnel_static_trial_context.report)
                _, blocked = self.invoke(context, events)
                blocked.assert_called_once_with(context.grpc)

    def test_failed_state_or_changed_pid_overrides_a_previous_goal_intent(self):
        for values in ({'status': 'failed'}, {'collision_objects': ['7']},
                       {'termination_hard_failures': ['native rejected']}, {'different_pid': True}):
            with self.subTest(values=values):
                context, events = self.guarded_context()
                self.request(context)
                if values.get('different_pid'):
                    context.expert_controller.process.pid = 43
                else:
                    context.tunnel_static_trial_context.report.update(values)
                outcome, blocked = self.invoke(context, events)
                blocked.assert_called_once_with(context.grpc)
                self.assertFalse(outcome['normal_goal_stop'])

    def test_interrupt_timeout_and_guard_exception_cancel_pending_goal_permission(self):
        for error in (KeyboardInterrupt(), TimeoutError('timeline timeout'), RuntimeError('watchdog abort')):
            with self.subTest(error=type(error).__name__):
                source, events = self.guarded_context()
                context = MissionScenario(source.grpc, None, {}, source.cfg)
                context.expert_controller = source.expert_controller
                context.tunnel_static_trial_context = source.tunnel_static_trial_context

                def stop():
                    events.append('stop-expert')
                    context.expert_controller = None

                context.stop_expert_controller = stop
                self.request(context)
                with patch.object(FullLoopScenario, 'run_timeline', side_effect=error), \
                        self.assertRaises(type(error)):
                    context.run_timeline()
                intent = context.tunnel_static_trial_context.report['normal_goal_stop_intent']
                self.assertFalse(intent['pending'])
                self.assertIn('cancelled_reason', intent)
                _, blocked = self.invoke(context, events)
                blocked.assert_called_once_with(context.grpc)

    def test_real_goal_timeline_stops_normally_before_completion_validation(self):
        events = []
        grpc = NS(world=object(), _tunnel_static_common_start=True,
                  stop_ego_control=lambda: events.append('common-stop-control'))
        scenario = MissionScenario(grpc, None, {}, {
            'mission_mode': trial.MODE, 'max_laps': 1,
            'progress_watchdog': {'enabled': False}, 'minimum_completion_progress_ratio': .95,
        })
        scenario.goal_x, scenario.goal_y, scenario.route_length_m = 190., 0., 190.
        scenario.runtime_bridge = None
        scenario.tunnel_dynamic_handler = None
        scenario.tunnel_vehicle_handler = None
        scenario.expert_controller = NS(process=NS(pid=42, poll=lambda: None),
            is_running=lambda: True, stop_process=lambda: events.append('stop-expert'))
        scenario.get_ego_motion_state = lambda: {
            'x': 190., 'y': 0., 'yaw_deg': 0., 'speed': 7.5,
            'current_link': 'END', 'collision_objects': [],
        }
        scenario.record_ego_trace = Mock()
        scenario.project_on_route_near_progress = lambda *args, **kwargs: (190., 0.)
        scenario.on_expert_timeline_tick = lambda **kwargs: events.append('guarded-tick')
        report = {'status': 'running', 'last_guard_stage': 'passed', 'collision_objects': [],
                  'expert_verified': False, 'live_avoidance_verified': False}
        proof = NS(report=report, persist=Mock())
        scenario.tunnel_static_trial_context = proof

        def final_brake(grpc, evidence):
            events.append('verified-final-brake')
            self.assertEqual(report['status'], 'running')
            self.assertFalse(report['expert_verified'])
            return copy.deepcopy(STOPPED)

        def finish(current):
            events.append('validate-completion')
            self.assertIsNone(current.expert_controller)
            self.assertEqual(report['status'], 'running')
            self.assertTrue(report['immediate_hold']['normal_goal_stop'])
            report.update(status='passed', expert_verified=True, live_avoidance_verified=True)

        proof.finish = finish
        with patch.object(trial, 'required', return_value=proof), \
                patch.object(startup, 'block_common_input') as blocked, \
                patch.object(startup, 'finish_common_brake', side_effect=final_brake), \
                redirect_stdout(io.StringIO()):
            scenario.run_timeline()
        blocked.assert_not_called()
        self.assertEqual(events, ['guarded-tick', 'stop-expert', 'verified-final-brake',
                                  'common-stop-control', 'validate-completion'])
        self.assertEqual(report['status'], 'passed')

    def test_shared_default_goal_hook_is_a_noop(self):
        scenario = FullLoopDriveBase(None, None, {}, {'mission_mode': 'full_loop_static'})
        self.assertIsNone(scenario.on_goal_reached_before_stop(
            elapsed=40., ego_state={}, current_s=190., remaining_s=0.,
        ))
        self.assertFalse(hasattr(scenario, 'tunnel_static_trial_context'))


if __name__ == "__main__":
    unittest.main()
