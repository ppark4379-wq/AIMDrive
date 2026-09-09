import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..adapters.sync_runtime import SyncRuntimeBridge
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.full_loop_scenario import FullLoopScenario


def ego():
    return SimpleNamespace(position=SimpleNamespace(x=1, y=2, z=3),
                           velocity=SimpleNamespace(x=3, y=4, z=0), heading=90)


class SyncRuntimeTest(unittest.TestCase):
    def test_requires_message_after_active_observed(self):
        bridge = SyncRuntimeBridge({})
        bridge.rospy = Mock()
        bridge.rospy.is_shutdown.return_value = False
        bridge.rospy.get_param.return_value = True
        bridge._on_ego(ego())  # Pre-activation sample must not release the gate.
        sleeps = []
        def advance(_):
            sleeps.append(1)
            if len(sleeps) == 2:
                bridge._on_ego(ego())
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
            scenario.runtime_bridge._on_ego(ego())
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
                    scenario.runtime_bridge._on_ego(ego())
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

    def test_default_driving_timeout_is_disabled(self):
        from ..main_runner import load_configuration
        _, _, cfg = load_configuration()
        self.assertEqual(cfg['timeout_sec'], 0)
        self.assertFalse(cfg['clean_road_mode'])
        self.assertEqual(cfg['sync_runtime']['startup_timeout_sec'], 30)
