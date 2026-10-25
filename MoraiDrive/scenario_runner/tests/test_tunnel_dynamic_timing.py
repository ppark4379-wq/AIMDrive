"""Pure travel-time, trajectory-window and goal-progress calculations."""

import math
import unittest

from ..scenarios.full_loop_parts import tunnel_dynamic_timing as timing


def straight_plan(start=10.0, end=30.0, speed=2.5, tolerance=0.5):
    return dict(waypoints=[dict(x=start, y=2.5, z=0.0),
                           dict(x=end, y=2.5, z=0.0)],
                speed_mps=speed, arrival_tolerance_m=tolerance)


class TunnelTravelTimeTest(unittest.TestCase):
    def test_constant_speed_and_braking_are_not_forecast_as_acceleration(self):
        self.assertEqual(timing.travel_time(12.0, 3.0), 4.0)
        self.assertEqual(timing.travel_time(12.0, 3.0, -2.0), 4.0)

    def test_positive_acceleration_and_speed_bound(self):
        self.assertAlmostEqual(timing.travel_time(8.0, 2.0, 2.0), 2.0)
        # Accelerate from 2 to 6 m/s in 2 s (8 m), then cruise 12 m.
        self.assertAlmostEqual(timing.travel_time(20.0, 2.0, 2.0, 6.0), 4.0)
        self.assertAlmostEqual(timing.travel_time(8.0, 2.0, 2.0, 6.0), 2.0)

    def test_forecast_bound_never_reduces_measured_speed(self):
        self.assertEqual(timing.travel_time(24.0, 8.0, 3.0, 5.0), 3.0)

    def test_past_arrivals_do_not_extrapolate_acceleration_backwards(self):
        self.assertEqual(timing.travel_time(-6.0, 3.0, 5.0), -2.0)
        self.assertEqual(timing.travel_time(-6.0, 0.0, 5.0), 0.0)
        self.assertEqual(timing.travel_time(0.0, 0.0), 0.0)

    def test_stopped_ego_needs_positive_acceleration_to_reach_ahead(self):
        self.assertTrue(math.isinf(timing.travel_time(9.0, 0.0)))
        self.assertAlmostEqual(timing.travel_time(9.0, 0.0, 2.0), 3.0)
        self.assertTrue(math.isinf(timing.travel_time(9.0, 0.0, 2.0, 0.0)))

    def test_nonfinite_values_and_negative_ceiling_are_rejected(self):
        for args in ((math.nan, 3.0), (1.0, math.inf),
                     (1.0, 3.0, True), (1.0, 3.0, 1.0, -1.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                timing.travel_time(*args)


class TunnelMotionWindowTest(unittest.TestCase):
    route = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]

    def test_fixed_speed_samples_end_at_arrival_tolerance(self):
        samples = timing.motion_samples(self.route, straight_plan())
        self.assertAlmostEqual(samples[0]["time_sec"], 0.2)
        self.assertAlmostEqual(samples[0]["route_s"], 10.5)
        self.assertAlmostEqual(samples[-1]["time_sec"], 7.8)
        self.assertAlmostEqual(samples[-1]["route_s"], 29.5)
        for sample in samples:
            self.assertAlmostEqual(sample["lateral_m"], 2.5)
        for first, second in zip(samples, samples[1:]):
            distance = second["route_s"] - first["route_s"]
            self.assertLessEqual(distance, 0.25 + 1e-9)
            self.assertAlmostEqual(distance,
                2.5 * (second["time_sec"] - first["time_sec"]))

    def test_reverse_motion_keeps_route_lateral_sign(self):
        samples = timing.motion_samples(self.route, straight_plan(30.0, 10.0))
        self.assertAlmostEqual(samples[0]["route_s"], 29.5)
        self.assertAlmostEqual(samples[-1]["route_s"], 10.5)
        self.assertAlmostEqual(samples[-1]["time_sec"], 7.8)
        self.assertTrue(all(sample["lateral_m"] == 2.5 for sample in samples))
        self.assertTrue(all(a["route_s"] > b["route_s"]
                            for a, b in zip(samples, samples[1:])))

    def test_turn_uses_full_polyline_and_only_final_tolerance(self):
        plan = dict(waypoints=[dict(x=10.0, y=2.5), dict(x=15.0, y=2.5),
                               dict(x=15.0, y=-2.5)],
                    speed_mps=2.0, arrival_tolerance_m=0.5)
        samples = timing.motion_samples(self.route, plan)
        turn = next(sample for sample in samples
                    if math.isclose(sample["time_sec"], 2.5))
        self.assertEqual(turn["route_s"], 15.0)
        self.assertEqual(turn["lateral_m"], 2.5)
        self.assertEqual(samples[-1]["route_s"], 15.0)
        self.assertEqual(samples[-1]["lateral_m"], -2.0)
        self.assertEqual(samples[-1]["time_sec"], 4.75)

    def test_window_distinguishes_too_early_available_and_missed(self):
        samples = timing.motion_samples(self.route, straight_plan(30.0, 50.0))
        earliest, latest = timing.encounter_start_window(samples, 0.0, 5.0)
        self.assertAlmostEqual(earliest, 1.5)
        self.assertAlmostEqual(latest, 6.5)
        earliest, latest = timing.encounter_start_window(samples, 10.0, 5.0)
        self.assertLessEqual(earliest, 0.0)
        self.assertGreaterEqual(latest, 0.0)
        _, latest = timing.encounter_start_window(samples, 40.0, 5.0)
        self.assertLess(latest, 0.0)
        self.assertEqual(timing.encounter_start_window(samples, 60.0, 5.0),
                         (math.inf, -math.inf))

    def test_unreachable_opportunity_does_not_open_window(self):
        samples = timing.motion_samples(self.route, straight_plan(30.0, 50.0))
        self.assertEqual(timing.encounter_start_window(samples, 0.0, 0.0),
                         (math.inf, -math.inf))

    def test_front_body_reach_opens_only_the_early_bound(self):
        samples = [dict(time_sec=1.0, route_s=10.0, lateral_m=2.5)]
        # After 1 s the reported Ego position is 6.5 m. Its 3.85 m nose
        # reaches this NPC at 10 m; a symmetric 3 m reference would not.
        original = timing.encounter_start_window(samples, 1.5, 5.0)
        front = timing.encounter_start_window(samples, 1.5, 5.0,
                                               front_margin_m=3.85)
        self.assertAlmostEqual(original[0], 0.1)
        self.assertAlmostEqual(front[0], -0.07)
        self.assertAlmostEqual(front[1], 1.3)
        self.assertEqual(front[1], original[1])
        self.assertEqual(timing.encounter_start_window(samples, 1.5, 5.0,
                         front_margin_m=3.0), original)
        self.assertEqual(timing.encounter_start_window(samples, 13.1, 5.0,
                         front_margin_m=3.85), (math.inf, -math.inf))
        for invalid in (-0.1, math.inf, math.nan, True):
            with self.subTest(front_margin_m=invalid), self.assertRaises(ValueError):
                timing.encounter_start_window(samples, 1.5, 5.0,
                                              front_margin_m=invalid)

    def test_goal_gate_matches_tunnel_progress_and_distance_thresholds(self):
        cfg = dict(goal_tolerance_m=2.0, expert_arrival_stop_distance_m=2.0,
                   minimum_completion_progress_ratio=0.95)
        self.assertAlmostEqual(timing.goal_progress_s(137.089, cfg), 135.089)
        self.assertEqual(timing.goal_progress_s(100.0, {}), 94.0)
        self.assertEqual(timing.goal_progress_s(100.0,
            dict(minimum_completion_progress_ratio=0.99)), 99.0)


if __name__ == "__main__":
    unittest.main()
