# Shared drive-loop validation hook with a fake Ego and no MORAI process.
#
# The integrated AIM2026 launchers (highway and roundabout) attach a ValidationRunLog to the scenario the same way:
# the drive loop observes the Ego, and the launcher finalizes the run afterwards.

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


from types import SimpleNamespace as NS
from unittest.mock import patch
from scenario_runner.scenarios.full_loop_parts import dynamic_obstacle as dynamic
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.core.full_loop_drive_base import FullLoopDriveBase


class DynamicSetupFailureTests(unittest.TestCase):
    def test_stopline_guard_failure_keeps_cleanup_metrics_initialized(self):
        context = object.__new__(FullLoopScenario)
        context.cfg = {}
        context.route_definition = {'zones': {'dynamic_obstacle': {}}}
        context.traffic_light_stops = []
        context.dynamic_obstacle_event_index = 0
        with patch.object(dynamic, '_strict_assessment'), \
             patch.object(dynamic, '_dynamic_cfg', return_value={}), \
             patch.object(dynamic, '_catalogue_entries', return_value=[]), \
             patch.object(dynamic, '_expand_catalogue', return_value=[]), \
             patch.object(dynamic, '_choose_schedule', return_value=[{'id':'event'}]), \
             patch.object(dynamic, '_signal_guard_cfg', return_value={'enabled':True}):
            with self.assertRaisesRegex(RuntimeError,'no verified traffic-light stop'):
                dynamic.spawn_dynamic_obstacle(context)
        self.assertFalse(context.dynamic_validation['goal_reached'])
        with patch.object(FullLoopDriveBase, 'cleanup'), \
             patch.object(FullLoopScenario, 'cleanup_spawned_actors'), \
             patch.object(dynamic, '_persist_progress'):
            context.cleanup()
        self.assertTrue(context.dynamic_validation['cleanup_verified'])
        self.assertFalse(context.dynamic_validation['safety_validation_passed'])


# A successful creation can still return an unusable SpawnPoint runtime ID.

from scenario_runner.scenarios.full_loop_parts import highway


def context(disable_results, create_ids):
    grpc = Mock()
    grpc.create_vehicle_spawn_point.side_effect = create_ids
    grpc.enable_vehicle_spawn_point.side_effect = disable_results
    return NS(cfg=dict(highway_npc=dict(model='car', collection_maximum_total_vehicles=4,
        spawn_points=[dict(id=6, name='competition_spawn_6', maximum_spawn_vehicle=4)],
        spawn_point_lifecycle=dict(create_attempts=3, create_settle_sec=0., recreate_delay_sec=0.))),
        vehicle_models=lambda: ['car'], highway_spawn_points=[], grpc=grpc)


class SpawnStartupTests(unittest.TestCase):
    def test_replaces_created_id_that_rejects_initial_disable(self):
        c = context([False, True], ['7', '13'])
        highway.spawn_highway_npcs(c)
        self.assertTrue(c.highway_spawn_points[0]['retired'])
        self.assertEqual(c.highway_spawn_points[1]['id'], '13')
        self.assertEqual(c.highway_npc_planned_count, 1)
        self.assertFalse(c.highway_spawn_points_active)
        self.assertTrue(all(call.args[0]['pause'] is True
                            for call in c.grpc.create_vehicle_spawn_point.call_args_list))
        self.assertTrue(all(call.args[1] is False
                            for call in c.grpc.enable_vehicle_spawn_point.call_args_list))
        c.grpc.enable_vehicle_spawn_point.reset_mock()
        c.grpc.enable_vehicle_spawn_point.side_effect = None
        c.grpc.enable_vehicle_spawn_point.return_value = True
        self.assertEqual(highway._set_highway_spawn_points_enabled(c, True), 1)
        self.assertEqual(c.grpc.enable_vehicle_spawn_point.call_args.args, ('13', True))

    def test_never_continues_after_exhausted_disable_recovery(self):
        c = context([False, False, False], ['7', '8', '9'])
        with self.assertRaisesRegex(RuntimeError, 'confirm disabled'):
            highway.spawn_highway_npcs(c)
        self.assertEqual(c.grpc.create_vehicle_spawn_point.call_count, 3)
        self.assertTrue(all(p['retired'] for p in c.highway_spawn_points))
        self.assertTrue(all(call.args[1] is False
                            for call in c.grpc.enable_vehicle_spawn_point.call_args_list))

    def test_creation_failure_and_healthy_point_use_same_bounded_budget(self):
        c = context([True], [None, '13'])
        highway.spawn_highway_npcs(c)
        self.assertEqual(c.highway_npc_planned_count, 1)
        self.assertEqual(len(c.highway_spawn_points), 1)


from types import SimpleNamespace

from ..adapters.sync_runtime import SyncRuntimeBridge
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.full_loop_scenario import FullLoopScenario


def sync_ego():
    return SimpleNamespace(position=SimpleNamespace(x=1, y=2, z=3),
                           velocity=SimpleNamespace(x=3, y=4, z=0), heading=90)


class SyncRuntimeTest(unittest.TestCase):
    def test_external_scenario_clock_is_injected_without_collection_dependency(self):
        clock = SimpleNamespace(
            domain="grpc_simulation_timestamp",
            now=Mock(return_value=12.5),
        )
        scenario = FullLoopDriveBase(Mock(), None, {}, {}, clock=clock)
        self.assertIs(scenario.clock, clock)
        self.assertEqual(scenario.clock.domain, "grpc_simulation_timestamp")
        self.assertEqual(scenario.clock.now(), 12.5)

    def test_requires_message_after_active_observed(self):
        bridge = SyncRuntimeBridge({})
        bridge.rospy = Mock()
        bridge.rospy.is_shutdown.return_value = False
        bridge.rospy.get_param.return_value = True
        bridge._on_ego(sync_ego())  # Pre-activation sample must not release the gate.
        sleeps = []
        def advance(_):
            sleeps.append(1)
            if len(sleeps) == 2:
                bridge._on_ego(sync_ego())
        with patch('time.sleep', side_effect=advance):
            bridge.wait_until_ready(Mock(is_running=Mock(return_value=True)))
        self.assertEqual(len(sleeps), 2)
        bridge.rospy.set_param.assert_called_once_with(bridge.ready_param, True)

    def test_timeout_and_dead_expert(self):
        bridge = SyncRuntimeBridge({'startup_timeout_sec': 0})
        bridge.rospy = Mock()
        with self.assertRaises(TimeoutError):
            bridge.wait_until_ready(Mock(is_running=Mock(return_value=True)))
        bridge.rospy.reset_mock()
        with self.assertRaises(RuntimeError):
            bridge.wait_until_ready(Mock(is_running=Mock(return_value=False)))
        bridge.rospy.set_param.assert_not_called()

    def test_ros_fallback_and_stale_rejection(self):
        scenario = FullLoopDriveBase(Mock(), None, {}, {})
        scenario.runtime_bridge = SyncRuntimeBridge({'ego_max_age_sec': 1})
        scenario.grpc.get_ego_motion_state.side_effect = RuntimeError('gRPC unavailable')
        with patch('time.monotonic', return_value=10):
            scenario.runtime_bridge._on_ego(sync_ego())
            self.assertEqual(scenario.get_ego_motion_state()['speed'], 5)
        with patch('time.monotonic', return_value=12):
            with self.assertRaisesRegex(RuntimeError, 'No Ego state'):
                scenario.get_ego_motion_state()
        scenario.grpc.get_ego_motion_state.side_effect = None
        scenario.grpc.get_ego_motion_state.return_value = {'x': 7}
        self.assertEqual(scenario.get_ego_motion_state(), {'x': 7})

    def test_cleanup_order_and_failure(self):
        scenario = FullLoopDriveBase(Mock(), None, {}, {})
        bridge = scenario.runtime_bridge = SyncRuntimeBridge({})
        calls = Mock()
        bridge.rospy = Mock(set_param=calls.ready)
        bridge.subscriber = Mock(unregister=calls.unsubscribe)
        scenario.expert_controller = Mock(stop_process=calls.stop)
        scenario.cleanup()
        self.assertEqual([c[0] for c in calls.mock_calls], ['ready', 'unsubscribe', 'stop'])
        scenario.expert_controller = Mock()
        bridge.rospy.set_param.side_effect = RuntimeError('ROS disconnected')
        with self.assertRaises(RuntimeError):
            scenario.cleanup()
        self.assertIsNone(scenario.expert_controller)

    def test_empty_grpc_response_uses_ros_or_fails(self):
        scenario = FullLoopDriveBase(Mock(), None, {}, {})
        scenario.runtime_bridge = SyncRuntimeBridge({})
        for response in (None, {}):
            with self.subTest(response=response):
                scenario.grpc.get_ego_motion_state.return_value = response
                with patch('time.monotonic', return_value=10):
                    scenario.runtime_bridge._on_ego(sync_ego())
                    self.assertEqual(scenario.get_ego_motion_state()['x'], 1)
                with patch('time.monotonic', return_value=12):
                    with self.assertRaisesRegex(RuntimeError, 'No Ego state'):
                        scenario.get_ego_motion_state()

    def test_run_waits_after_setup_before_timeline(self):
        scenario = FullLoopDriveBase(Mock(), None, {}, {})
        calls = Mock()
        controller = Mock()

        def setup():
            calls.setup()
            scenario.expert_controller = controller

        scenario.setup = setup
        scenario.run_timeline = calls.timeline
        scenario.cleanup = calls.cleanup
        with patch(FullLoopDriveBase.__module__ + '.SyncRuntimeBridge') as cls:
            cls.return_value.start = calls.start
            cls.return_value.wait_until_ready = calls.wait
            scenario.run()
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ['start', 'setup', 'wait', 'timeline', 'cleanup'],
        )
        calls.wait.assert_called_once_with(controller)

    def test_run_cleans_up_when_startup_fails(self):
        scenario = FullLoopDriveBase(Mock(), None, {}, {})
        scenario.setup = Mock()
        scenario.expert_controller = Mock()
        scenario.run_timeline = Mock()
        with patch(FullLoopDriveBase.__module__ + '.SyncRuntimeBridge') as cls:
            cls.return_value.wait_until_ready.side_effect = TimeoutError()
            with self.assertRaises(TimeoutError):
                scenario.run()
            cls.return_value.close.assert_called_once()
        scenario.run_timeline.assert_not_called()

    def test_run_without_sync_runtime_executes_timeline_directly(self):
        scenario = FullLoopDriveBase(
            Mock(), None, {}, {"sync_runtime": {"enabled": False}}
        )
        calls = Mock()
        scenario.setup = calls.setup
        scenario.run_timeline = calls.timeline
        scenario.cleanup = calls.cleanup
        scenario.require_async_ros_control_path = calls.preflight
        with patch(FullLoopDriveBase.__module__ + '.SyncRuntimeBridge') as cls:
            scenario.run()
        cls.assert_not_called()
        self.assertIsNone(scenario.runtime_bridge)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ['preflight', 'setup', 'timeline', 'cleanup'],
        )

    def test_manual_ego_skips_ros_control_preflight(self):
        scenario = FullLoopDriveBase(
            Mock(), None, {}, {
                "sync_runtime": {"enabled": False},
                "drive_control_mode": "morai_keyboard",
            },
        )
        calls = Mock()
        scenario.setup = calls.setup
        scenario.run_timeline = calls.timeline
        scenario.cleanup = calls.cleanup
        scenario.require_async_ros_control_path = calls.preflight
        scenario.run()
        calls.preflight.assert_not_called()
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["setup", "timeline", "cleanup"],
        )

    def test_manual_ego_uses_keyboard_without_starting_expert(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {
            "sync_runtime": {"enabled": False},
            "drive_control_mode": "morai_keyboard",
            "reset_ego_motion_on_start": True,
            "reset_ego_motion_wait_sec": 0,
        })
        scenario.runtime_bridge = None
        scenario.start_tf = object()
        scenario.setup_expert_controller = calls.setup_expert
        with patch("time.sleep"):
            self.assertTrue(scenario.configure_drive_from_start())
        names = [call[0] for call in calls.mock_calls]
        self.assertLess(names.index("grpc.reset_ego_motion"),
                        names.index("grpc.set_ego_control_mode_keyboard"))
        calls.grpc.set_ego_control_mode_auto.assert_not_called()
        calls.setup_expert.assert_not_called()
        self.assertEqual(scenario.route_setup_mode, "morai_keyboard")

    def test_cleanup_before_setup_does_not_touch_morai_actors(self):
        scenario = FullLoopScenario(Mock(), None, {}, {})
        scenario.cleanup_spawned_actors()
        scenario.grpc.reset_actors.assert_not_called()

    def test_async_mode_resets_motion_before_expert_start(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {
            'sync_runtime': {'enabled': False},
            'reset_ego_motion_on_start': True,
            'reset_ego_motion_wait_sec': 0,
        })
        scenario.runtime_bridge = None
        scenario.start_tf = object()
        scenario.expert_controller = Mock(stop_process=calls.stop_old)
        def setup():
            scenario.expert_controller = Mock(start_process=calls.start_new)
        scenario.setup_expert_controller = setup
        with patch('time.sleep'):
            scenario.configure_drive_from_start()
        names = [call[0] for call in calls.mock_calls]
        self.assertLess(names.index('stop_old'), names.index('grpc.set_ego_transform'))
        self.assertLess(names.index('grpc.set_ego_transform'), names.index('grpc.reset_ego_motion'))
        self.assertLess(names.index('grpc.reset_ego_motion'), names.index('start_new'))

    def test_clean_road_skips_spawning_and_events(self):
        for clean in (True, False):
            scenario = FullLoopScenario(Mock(), None, {}, {'clean_road_mode': clean})
            methods = ('load_route_definition', 'prepare_full_loop_route', 'apply_weather',
                       'apply_time', 'resolve_profile_models', 'spawn_static_obstacle',
                       'spawn_dynamic_obstacle', 'spawn_roundabout_npcs', 'spawn_highway_npcs',
                       'arm_tunnel_sudden_brake', 'configure_drive_with_retries', 'persist_profile_state')
            for method in methods:
                setattr(scenario, method, Mock())
            scenario.select_profile = Mock(return_value=(0, {'name': 'test', 'weather': 'SUNNY'}, {}))
            scenario.select_scenario_hour = Mock(return_value=11)
            scenario.start_tf = None
            scenario.static_obstacle_model = scenario.dynamic_obstacle_model = 'test'
            scenario.setup()
            for method in methods[4:10]:
                self.assertEqual(getattr(scenario, method).call_count, 0 if clean else 1)
            if clean:
                scenario.cfg['scenario_time_verify_period_sec'] = 0
                scenario.update_dynamic_obstacle = Mock()
                scenario.activate_highway_spawn_points = Mock()
                scenario.update_tunnel_sudden_brake = Mock()
                scenario.on_expert_timeline_tick(0, {}, 0, 100)
                scenario.update_dynamic_obstacle.assert_not_called()
                scenario.activate_highway_spawn_points.assert_not_called()
                scenario.update_tunnel_sudden_brake.assert_not_called()

    def test_spawn_resets_motion_before_expert_start(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {
            'reset_ego_motion_on_start': True,
            'reset_ego_motion_wait_sec': 0,
        })
        scenario.runtime_bridge = Mock()
        scenario.start_tf = object()
        scenario.expert_controller = Mock(stop_process=calls.stop_old)
        def setup():
            scenario.expert_controller = Mock(start_process=calls.start_new)
        scenario.setup_expert_controller = setup
        with patch('time.sleep'):
            scenario.configure_drive_from_start()
        names = [call[0] for call in calls.mock_calls]
        self.assertLess(names.index('stop_old'), names.index('grpc.set_ego_transform'))
        self.assertLess(names.index('grpc.set_ego_transform'), names.index('grpc.reset_ego_motion'))
        self.assertLess(names.index('grpc.reset_ego_motion'), names.index('start_new'))

    def test_collection_owned_runtime_preserves_proven_ego_call_order(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {
            'reset_ego_motion_on_start': True,
            'reset_ego_motion_wait_sec': 0,
        })
        scenario.start_tf = object()
        scenario.expert_controller = Mock(stop_process=calls.stop_old)
        def setup():
            scenario.expert_controller = Mock(start_process=calls.start_new)
        scenario.setup_expert_controller = setup
        with patch('time.sleep'):
            scenario.configure_drive_from_start()
        names = [call[0] for call in calls.mock_calls]
        self.assertNotIn('grpc.reset_ego_motion', names)
        self.assertLess(names.index('grpc.set_ego_transform'), names.index('stop_old'))
        self.assertLess(names.index('stop_old'), names.index('start_new'))

        expected = RuntimeError('original collection gRPC failure')
        calls.grpc.get_ego_motion_state.side_effect = expected
        with self.assertRaises(RuntimeError) as caught:
            scenario.get_ego_motion_state()
        self.assertIs(caught.exception, expected)

    def test_shared_activation_keeps_control_and_expert_order(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {})
        calls.grpc.set_ego_control_mode_auto.return_value = True

        def setup():
            calls.setup()
            scenario.expert_controller = Mock(start_process=calls.start_new)

        scenario.setup_expert_controller = setup
        result = scenario.activate_expert_drive()
        self.assertTrue(result)
        self.assertTrue(scenario.route_configured)
        self.assertEqual(scenario.route_setup_mode, 'aimdrive_expert')
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ['grpc.set_ego_control_mode_auto', 'grpc.set_ego_gear_drive',
             'setup', 'start_new'],
        )
        calls.grpc.set_ego_transform.assert_not_called()
        calls.grpc.reset_ego_motion.assert_not_called()

    def test_shared_activation_holds_before_setup_and_checks_before_release(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {})
        calls.grpc.set_ego_control_mode_auto.return_value = True
        scenario.route_configured = False
        scenario.route_setup_mode = None

        def setup():
            calls.setup()
            scenario.expert_controller = Mock(start_process=calls.start_new)

        def check():
            self.assertFalse(scenario.route_configured)
            self.assertIsNone(scenario.route_setup_mode)
            calls.ready()

        scenario.setup_expert_controller = setup
        self.assertTrue(scenario.activate_expert_drive(
            before_expert_start=calls.hold,
            after_expert_setup=calls.verify_pose,
            after_expert_start=check,
        ))
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ['grpc.set_ego_control_mode_auto', 'grpc.set_ego_gear_drive',
             'hold', 'setup', 'verify_pose', 'start_new', 'ready'],
        )
        calls.grpc.set_ego_transform.assert_not_called()
        calls.grpc.reset_ego_motion.assert_not_called()

    def test_shared_activation_failed_ready_gate_does_not_mark_route_configured(self):
        calls = Mock()
        scenario = FullLoopDriveBase(calls.grpc, None, {}, {})
        scenario.route_configured = False
        scenario.route_setup_mode = None

        def setup():
            scenario.expert_controller = Mock(start_process=calls.start_new)

        scenario.setup_expert_controller = setup
        rejected = RuntimeError('fresh Expert readiness rejected')
        ready = Mock(side_effect=rejected)
        with self.assertRaises(RuntimeError) as caught:
            scenario.activate_expert_drive(after_expert_start=ready)
        self.assertIs(caught.exception, rejected)
        self.assertFalse(scenario.route_configured)
        self.assertIsNone(scenario.route_setup_mode)
        calls.start_new.assert_called_once_with()

    def test_default_driving_timeout_is_disabled(self):
        from ..main_runner import load_configuration
        _, _, cfg = load_configuration()
        self.assertEqual(cfg['timeout_sec'], 0)
        self.assertFalse(cfg['clean_road_mode'])
        self.assertEqual(cfg['sync_runtime']['startup_timeout_sec'], 30)


if __name__ == "__main__":
    unittest.main()
