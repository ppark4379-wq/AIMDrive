"""Exercise the entry gain through final speed limits and pedal commands."""

import math
from types import SimpleNamespace
import unittest

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.control.pid import Pid
from expert.src.control.pure_pursuit import PurePursuit
from expert.src.localization.point import Point
from expert.src.planning.roundabout.planner import RoundaboutEntryPlanner
from expert.src.vehicle_state import VehicleState


class RoundaboutEntryGainTest(unittest.TestCase):
    def setUp(self):
        self.target = 26. / 3.6
        self.signal = self.legacy = self.target
        self.path = [Point(float(x), 0.) for x in range(201)]
        self.result = self.make_result()
        self.controller = controller = AutonomousDriving.__new__(AutonomousDriving)
        controller._configure_runtime = lambda **kwargs: None
        controller.path = self.path
        controller.path_manager = SimpleNamespace(
            get_local_path=lambda ego: (self.path[:50], self.target),
            current_waypoint=0, is_closed_path=False)
        controller.forward_object_detector = SimpleNamespace(detect_object=lambda *args, **kwargs: [])
        controller.lattice_planning_process = SimpleNamespace(
            roundabout_planner=SimpleNamespace(active=True),
            run=lambda *args, **kwargs: self.result)
        controller.adaptive_cruise_control = SimpleNamespace(
            check_object=lambda *args: None, pedestrian_released=False,
            get_target_velocity=lambda *args: self.legacy)
        controller.traffic_light_stop_controller = SimpleNamespace(
            get_target_velocity=lambda **kwargs: self.signal)
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        controller.pid = Pid(.075, .0001, .025, 1. / 30.)
        controller.control_dt = 1. / 30.
        controller.previous_target_velocity = self.target
        controller.previous_acc_cmd = 1.
        controller.steering_sign, controller.max_steering = 1., math.radians(40.)

    def make_result(self, phase="cruise", **details):
        state = dict(hold_gap=0., entering=False, committed=False, front_gap=None)
        state.update(details)
        return RoundaboutEntryPlanner._result(self.path[:50], self.target, phase, state)

    def command(self, speed):
        return self.controller.execute(VehicleState(0., 0., 0., speed), [], None)[0]

    def test_permitted_standstill_entry_applies_full_throttle_without_brake(self):
        self.controller.previous_target_velocity = 0.
        self.controller.previous_acc_cmd = -1.
        command = self.command(0.)
        self.assertEqual(command.accel, 1.)
        self.assertEqual(command.brake, 0.)
        self.assertEqual(self.controller.previous_target_velocity, self.target)
        self.assertEqual(self.controller.pid.p_gain, .075)
        self.assertEqual(self.controller.pid.integral_error, 0.)  # saturation anti-windup

    def test_entry_uses_point_three_and_reverts_after_leaving_roundabout(self):
        speed = 18. / 3.6
        command = self.command(speed)
        expected = (.30 + .0001 / 30.) * (self.target - speed)
        self.assertAlmostEqual(command.accel, expected)
        self.result.mode = "inactive"
        command = self.command(speed)
        expected = (.075 + 2. * .0001 / 30.) * (self.target - speed)
        self.assertAlmostEqual(command.accel, expected)
        self.assertEqual(self.controller.pid.p_gain, .075)

    def test_approach_following_yield_and_other_modes_keep_base_gain(self):
        cases = [("roundabout_entry", "cruise", {"hold_gap": 8.}),
                 ("roundabout_entry", "follow", {"front_gap": 12.}),
                 ("roundabout_entry", "yield", {}),
                 ("roundabout_entry", "collision_slowdown", {}),
                 ("highway_lane", "cruise", {}),
                 ("static_obstacle", "cruise", {})]
        for mode, phase, details in cases:
            with self.subTest(mode=mode, phase=phase):
                self.setUp()
                self.result = self.make_result(phase, **details)
                self.result.mode = mode
                speed = 18. / 3.6
                reference = Pid(.075, .0001, .025, 1. / 30.)
                expected = reference.get_output(self.target, speed, derivative_on_measurement=(mode == "roundabout_entry"))
                self.assertAlmostEqual(self.command(speed).accel, expected)

    def test_external_speed_limits_and_stops_take_priority(self):
        for source in ("signal", "legacy"):
            for limit in (0., 3.):
                with self.subTest(source=source, limit=limit):
                    self.setUp()
                    setattr(self, source, limit)
                    command = self.command(1.)
                    self.assertEqual(self.controller.previous_target_velocity, limit)
                    if limit == 0.:
                        self.assertEqual((command.accel, command.brake), (0., 1.))
                    else:
                        expected = (.075 + .0001 / 30.) * (limit - 1.)
                        self.assertAlmostEqual(command.accel, expected)

    def test_collision_stop_and_overspeed_braking_are_unchanged(self):
        for stop in (False, True):
            with self.subTest(stop=stop):
                self.setUp()
                if stop:
                    self.result = RoundaboutEntryPlanner._result(self.path[:50], 0., "emergency", {})
                command = self.command(9.)
                self.assertEqual(command.accel, 0.)
                expected = 1. if stop else -(.075 + .0001 / 30.) * (self.target - 9.)
                self.assertAlmostEqual(command.brake, expected)

    def test_acceleration_override_preserves_derivative_braking(self):
        reference = Pid(.075, .0001, .025, 1. / 30.)
        boosted = Pid(.075, .0001, .025, 1. / 30.)
        for pid in (reference, boosted):
            pid.get_output(3.1, 0., derivative_on_measurement=True)
        expected = reference.get_output(3.1, 3., derivative_on_measurement=True)
        actual = boosted.get_output(3.1, 3., derivative_on_measurement=True, acceleration_p_gain=.30)
        self.assertLess(expected, 0.)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
