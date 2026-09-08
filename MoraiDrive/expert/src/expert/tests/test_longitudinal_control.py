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
        vehicle = {"object_info": SimpleNamespace(type=1)}
        pedestrian = {"object_info": SimpleNamespace(type=0)}
        traffic_light = {"object_info": SimpleNamespace(type=3)}
        result = SimpleNamespace(mode="highway_lane")

        entries = AutonomousDriving._legacy_acc_entries(
            result, [vehicle, pedestrian, traffic_light]
        )

        self.assertEqual([pedestrian, traffic_light], entries)

    def test_non_highway_keeps_legacy_vehicle_acc(self):
        vehicle = {"object_info": SimpleNamespace(type=1)}
        result = SimpleNamespace(mode="inactive")

        entries = AutonomousDriving._legacy_acc_entries(result, [vehicle])

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
