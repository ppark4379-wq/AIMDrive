import unittest
from types import SimpleNamespace

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.control.pid import Pid
from expert.src.planning.highway_lane_planning import parameters


class LongitudinalControlTest(unittest.TestCase):
    def _controller(self, previous_command):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.sampling_rate = parameters.PLANNER_UPDATE_RATE
        controller.previous_acc_cmd = previous_command
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


if __name__ == "__main__":
    unittest.main()
