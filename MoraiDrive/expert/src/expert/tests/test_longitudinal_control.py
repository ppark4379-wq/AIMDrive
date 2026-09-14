import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.control.pid import Pid
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.adaptive_cruise_control import AdaptiveCruiseControl
from expert.src.planning.highway_lane_planning import parameters


class LongitudinalControlTest(unittest.TestCase):
    def test_recorded_handoff_target_accelerates_gradually(self):
        from expert.src.planning.lattice_planning import parameters as static_p
        controller = self._controller(0.)
        controller.previous_target_velocity = 2.6
        controller.control_dt = .1
        ego = SimpleNamespace(velocity=2.6)
        recorded = SimpleNamespace(mode='static_obstacle', debug_details={'static_avoidance': {
            'planner': 'recorded_route_graph', 'path_validated': True}})
        speed = controller._smooth_static_target_speed(5.55, ego, recorded)
        self.assertAlmostEqual(speed, 2.6+static_p.STATIC_TARGET_ACCEL*.1)
        rejoined = SimpleNamespace(mode='static_obstacle', debug_details={})
        for _ in range(30):
            controller.previous_target_velocity = speed
            previous = speed
            speed = controller._smooth_static_target_speed(16.67, ego, rejoined)
            self.assertLessEqual(speed-previous, static_p.STATIC_TARGET_ACCEL*.1+1e-8)
            self.assertGreater(speed, previous)

    def test_recorded_acceleration_filter_preserves_emergency_and_signal_stops(self):
        controller = self._controller(0.)
        controller._static_speed_ramp = True
        controller.previous_target_velocity = 5.
        result = SimpleNamespace(mode='static_obstacle', debug_details={})
        ego = SimpleNamespace(velocity=5.)
        self.assertEqual(controller._smooth_static_target_speed(0., ego, result), 0.)
        self.assertEqual(controller._smooth_static_target_speed(1., ego, result), 1.)

    def test_static_target_filter_does_not_change_unrelated_driving(self):
        controller = self._controller(0.)
        controller.previous_target_velocity = 2.
        ego = SimpleNamespace(velocity=2.)
        result = SimpleNamespace(mode='inactive', debug_details={})
        self.assertEqual(controller._smooth_static_target_speed(16., ego, result), 16.)
        controller._static_speed_ramp = True
        result.mode = 'roundabout_entry'
        self.assertEqual(controller._smooth_static_target_speed(8., ego, result), 8.)

    def test_recorded_handoff_lookahead_changes_gradually(self):
        controller = self._controller(0.)
        controller.control_dt = .1
        controller.pure_pursuit = SimpleNamespace(lfd_gain=.65, min_lfd=2., max_lfd=30.)
        ego = SimpleNamespace(velocity=4.)
        recorded = SimpleNamespace(active=True, reason='static_recorded_follow', debug_details={'static_avoidance': {
            'planner': 'recorded_route_graph'}})
        self.assertEqual(controller._recorded_lookahead_limits(ego, recorded), (1.5,1.5))
        recorded.active = False
        recorded.reason = 'static_recorded_rejoined'
        recorded.debug_details = {}
        self.assertEqual(controller._recorded_lookahead_limits(ego, recorded), (1.6,1.6))

    def _controller(self, previous_command):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.sampling_rate = parameters.PLANNER_UPDATE_RATE
        controller.control_dt = 1.0 / parameters.PLANNER_UPDATE_RATE
        controller.previous_acc_cmd = previous_command
        controller._merge_curvature_floor_zone = None
        controller._merge_curvature_floor_speed = None
        return controller

    def test_braking_is_applied_without_rate_limit(self):
        controller = self._controller(0.5)

        self.assertEqual(-1.0, controller._smooth_longitudinal_command(-1.0))

    def test_brake_release_is_prompt_but_not_instantaneous(self):
        controller = self._controller(-1.0)

        command = controller._smooth_longitudinal_command(1.0)

        self.assertAlmostEqual(
            -1.0
            + parameters.BRAKE_RELEASE_RATE / parameters.PLANNER_UPDATE_RATE,
            command,
        )

    def test_positive_throttle_rises_smoothly(self):
        controller = self._controller(0.0)

        command = controller._smooth_longitudinal_command(1.0)

        self.assertAlmostEqual(
            parameters.ACCEL_COMMAND_RISE_RATE / parameters.PLANNER_UPDATE_RATE,
            command,
        )

    def test_roundabout_release_skips_brake_ramp_from_full_stop(self):
        controller = self._controller(-1.0)
        controller.previous_target_velocity = 0.0

        self.assertTrue(
            controller._roundabout_entry_fast_release(
                8.33, SimpleNamespace(reason="roundabout_entry_cruise")
            )
        )
        self.assertTrue(
            controller._roundabout_entry_fast_release(
                3.5, SimpleNamespace(reason="roundabout_entry_follow_lead")
            )
        )

    def test_non_roundabout_release_keeps_normal_smoothing(self):
        controller = self._controller(-1.0)
        controller.previous_target_velocity = 0.0

        self.assertFalse(
            controller._roundabout_entry_fast_release(
                8.33, SimpleNamespace(reason="highway_acc_override")
            )
        )

    def test_pid_reset_discards_accumulated_acceleration(self):
        pid = Pid(0.075, 0.0001, 0.025, 1.0 / 30.0)
        for _ in range(100):
            pid.get_output(27.78, 0.0)

        pid.reset()

        self.assertEqual(0.0, pid.integral_error)
        self.assertEqual(0.0, pid.previous_error)

    def test_pid_sampling_time_can_follow_simulation_tick(self):
        pid = Pid(0.075, 0.0001, 0.025, 1.0 / 30.0)

        pid.set_sampling_time(0.1)

        self.assertEqual(0.1, pid.sampling_time)

    def test_runtime_profile_uses_mode_specific_steering_and_dt(self):
        class FakePid:
            def __init__(self):
                self.sampling_time = None
                self.reset_count = 0

            def set_sampling_time(self, value):
                self.sampling_time = value

            def reset(self):
                self.reset_count += 1

        class FakeHighwayPlanner:
            def __init__(self):
                self.control_dt = None

            def set_control_dt(self, value):
                self.control_dt = value

        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.realtime_sampling_rate = 30.0
        controller.sync_fallback_sampling_rate = 10.0
        controller.realtime_steering_sign = -1.0
        controller.sync_steering_sign = 1.0
        controller.sync_mode = False
        controller.pid = FakePid()
        highway_planner = FakeHighwayPlanner()
        controller.lattice_planning_process = SimpleNamespace(
            highway_lane_planner=highway_planner
        )
        controller.previous_acc_cmd = 0.5
        controller.previous_front_steer_rad = 0.2

        controller._configure_runtime(sync_mode=True, control_dt=0.02)

        self.assertEqual(1.0, controller.steering_sign)
        self.assertEqual(0.02, controller.control_dt)
        self.assertEqual(50.0, controller.sampling_rate)
        self.assertEqual(0.02, controller.pid.sampling_time)
        self.assertEqual(0.02, highway_planner.control_dt)
        self.assertEqual(1, controller.pid.reset_count)

        controller._configure_runtime(sync_mode=False, control_dt=1.0 / 30.0)

        self.assertEqual(-1.0, controller.steering_sign)
        self.assertAlmostEqual(1.0 / 30.0, controller.control_dt)
        self.assertEqual(2, controller.pid.reset_count)

    def test_highway_lane_uses_only_lane_planner_for_vehicle_acc(self):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.lattice_planning_process = SimpleNamespace(
            roundabout_entry_should_ignore_legacy_acc=lambda *_: False
        )
        vehicle = {"object_info": SimpleNamespace(type=1)}
        pedestrian = {"object_info": SimpleNamespace(type=0)}
        traffic_light = {"object_info": SimpleNamespace(type=3)}
        result = SimpleNamespace(mode="highway_lane")

        entries = controller._legacy_acc_entries(
            SimpleNamespace(), result, [vehicle, pedestrian, traffic_light]
        )

        self.assertEqual([pedestrian, traffic_light], entries)

    def test_non_highway_keeps_legacy_vehicle_acc(self):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.lattice_planning_process = SimpleNamespace(
            roundabout_entry_should_ignore_legacy_acc=lambda *_: False
        )
        vehicle = {"object_info": SimpleNamespace(type=1)}
        result = SimpleNamespace(mode="inactive")

        entries = controller._legacy_acc_entries(
            SimpleNamespace(), result, [vehicle]
        )

        self.assertEqual([vehicle], entries)

    def test_crossing_pedestrian_is_detected_before_entering_lane(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "src" / "config" / "config.json"
        )
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        acc_config = dict(config["planning"]["adaptive_cruise_control"])
        controller = AdaptiveCruiseControl(
            vehicle_length=config["common"]["vehicle_length"],
            **acc_config
        )
        path = [Point(float(x), 0.0) for x in range(36)]
        pedestrian = ObjectInfo(30.0, 9.0, 0.0, 0)

        controller.check_object(
            path,
            [{"object_info": pedestrian, "local_position": Point(30.0, 9.0)}],
            [],
        )

        self.assertEqual(0, controller.object_type)
        self.assertLess(controller.get_target_velocity(14.7, 16.67), 16.67)

    def test_first_merge_ego_priority_ignores_curvature_speed_drop(self):
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"FIRST_PREDICTIVE": "EGO"}
            }
        )

        planned = AutonomousDriving._merge_ego_curvature_speed_floor(
            12.5, 22.2, result
        )

        self.assertEqual(22.2, planned)

    def test_first_merge_other_priority_keeps_curvature_speed_limit(self):
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"FIRST_PREDICTIVE": "OTHER"}
            }
        )

        planned = AutonomousDriving._merge_ego_curvature_speed_floor(
            12.5, 22.2, result
        )

        self.assertEqual(12.5, planned)

    def test_second_merge_ego_priority_ignores_curvature_speed_drop(self):
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"SECOND_ENTRY": "EGO"}
            }
        )

        planned = AutonomousDriving._merge_ego_curvature_speed_floor(
            13.5, 22.2, result
        )

        self.assertEqual(22.2, planned)

    def test_second_merge_other_priority_keeps_curvature_speed_limit(self):
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"SECOND_ENTRY": "OTHER"}
            }
        )

        planned = AutonomousDriving._merge_ego_curvature_speed_floor(
            13.5, 22.2, result
        )

        self.assertEqual(13.5, planned)

    def test_second_merge_ego_speed_does_not_follow_falling_curvature_profile(self):
        controller = self._controller(0.0)
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"SECOND_ENTRY": "EGO"}
            }
        )

        first = controller._hold_merge_ego_curvature_speed(19.1, result)
        later = controller._hold_merge_ego_curvature_speed(15.2, result)

        self.assertEqual(19.1, first)
        self.assertEqual(19.1, later)

    def test_first_merge_ego_speed_does_not_follow_falling_curvature_profile(self):
        controller = self._controller(0.0)
        result = SimpleNamespace(
            debug_details={
                "merge_zone_priorities": {"FIRST_PREDICTIVE": "EGO"}
            }
        )

        controller._hold_merge_ego_curvature_speed(22.0, result)

        self.assertEqual(
            22.0, controller._hold_merge_ego_curvature_speed(14.0, result)
        )


if __name__ == "__main__":
    unittest.main()
