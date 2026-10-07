"""Instructions contradicted by the ego's next 2.5 s are dropped, others kept."""
import unittest

from MoraiDrive.data_collection.situation_labeling.consistency import drop_contradicted_instructions


def track(speeds_mps, lateral_rate=0.0, dt=0.1, seconds=6.0):
    frames, s, d = [], 0.0, 0.0
    for i in range(int(seconds / dt)):
        frames.append({"index": i, "lidar_source_ns": int(i * dt * 1e9),
                       "ego_path_s_m": s, "ego_path_d_m": d})
        s += speeds_mps(i * dt) * dt
        d += lateral_rate * dt
    return frames


class ConsistencyTests(unittest.TestCase):
    def check(self, frames, text, indices=range(10, 15)):
        rules = [{"index": i, "instruction": text} for i in indices]
        drop_contradicted_instructions(frames, rules)
        return rules[0]

    def test_right_avoidance_while_moving_left_is_dropped(self):
        rule = self.check(track(lambda t: 5.0, lateral_rate=0.5),
                          "Perform an avoidance maneuver to the right.")
        self.assertIsNone(rule["instruction"])
        self.assertEqual(rule["instruction_consistency"], "dropped_direction")
        self.assertIn("to the right", rule["instruction_dropped"])

    def test_obstacle_position_is_not_read_as_the_maneuver_side(self):
        rule = self.check(track(lambda t: 5.0, lateral_rate=-0.5),
                          "There is an obstacle to the left of the ego vehicle. Perform an avoidance maneuver to the right.")
        self.assertEqual(rule["instruction_consistency"], "consistent")

    def test_lane_change_side_against_the_motion_is_dropped(self):
        rule = self.check(track(lambda t: 8.0, lateral_rate=-0.5), "Move into the left lane.")
        self.assertEqual(rule["instruction_consistency"], "dropped_direction")

    def test_right_avoidance_while_moving_right_is_kept(self):
        rule = self.check(track(lambda t: 5.0, lateral_rate=-0.5),
                          "Perform an avoidance maneuver to the right.")
        self.assertEqual(rule["instruction_consistency"], "consistent")
        self.assertIsNotNone(rule["instruction"])

    def test_maneuver_while_stopped_is_dropped(self):
        rule = self.check(track(lambda t: 0.0), "Continue the avoidance maneuver to the right.")
        self.assertEqual(rule["instruction_consistency"], "dropped_stationary")

    def test_continue_while_standing_is_dropped(self):
        rule = self.check(track(lambda t: 0.0), "Continue driving.")
        self.assertEqual(rule["instruction_consistency"], "dropped_keep_going_while_stopped")

    def test_continue_while_braking_to_stop_is_dropped(self):
        rule = self.check(track(lambda t: max(0.0, 6.0 - 3.0 * max(0.0, t - 1.0))), "Continue driving.")
        self.assertEqual(rule["instruction_consistency"], "dropped_keep_going_while_stopping")

    def test_contradiction_shorter_than_three_frames_is_kept(self):
        rule = self.check(track(lambda t: 0.0), "Continue driving.", indices=[10, 11])
        self.assertEqual(rule["instruction_consistency"], "consistent_borderline")
        self.assertEqual(rule["instruction"], "Continue driving.")

    def test_short_kept_island_inside_a_dropped_stop_is_dropped(self):
        # Standing still except a 0.2 s roll at t = 1.5 s: the roll frames are not stopped, but
        # sit between two dropped stretches of the same sentence.
        frames = track(lambda t: 3.0 if 1.5 <= t < 1.7 else 0.0, seconds=8.0)
        rules = [{"index": i, "instruction": "Continue driving."} for i in range(6, 40)]
        counts = drop_contradicted_instructions(frames, rules)
        self.assertEqual([r["instruction"] for r in rules], [None] * len(rules))
        self.assertEqual(counts, {"dropped_keep_going_while_stopped": len(rules)})

    def test_continue_while_cruising_is_kept_and_no_future_is_marked(self):
        frames = track(lambda t: 6.0)
        rules = [{"index": 10, "instruction": "Continue driving."},
                 {"index": len(frames) - 2, "instruction": "Continue driving."}]
        counts = drop_contradicted_instructions(frames, rules)
        self.assertEqual([r["instruction_consistency"] for r in rules], ["consistent", "no_future"])
        self.assertEqual(counts, {"consistent": 1, "no_future": 1})

    def test_static_instructions_survive_external_speed_constraints(self):
        for speed in (lambda t: 0.0, lambda t: max(0.0, 6.0 - 3.0 * max(0.0, t - 1.0))):
            for text in ('Continue driving.', 'Maintain the current driving behavior.',
                         'Continue the avoidance maneuver to the right.'):
                with self.subTest(text=text, speed=speed):
                    rules = [dict(index=i, zone_id='static_obstacle', instruction=text)
                             for i in range(10, 15)]
                    drop_contradicted_instructions(track(speed), rules)
                    self.assertTrue(all(r['instruction'] == text for r in rules))
                    self.assertTrue(all(r['instruction_consistency'] == 'consistent' for r in rules))

    def test_static_opposite_direction_is_still_dropped(self):
        rules = [dict(index=i, zone_id='static_obstacle',
                      instruction='Perform an avoidance maneuver to the right.')
                 for i in range(10, 15)]
        drop_contradicted_instructions(track(lambda t: 5.0, lateral_rate=0.5), rules)
        self.assertTrue(all(r['instruction_consistency'] == 'dropped_direction' for r in rules))



class MotionAlignmentTests(unittest.TestCase):
    def test_static_span_follows_the_swing(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import align_static_avoidance
        # Straight 3 s, swing right 3 s, back left 3 s.
        frames, s, d = [], 0.0, 0.0
        for i in range(90):
            t = i * 0.1
            frames.append({"index": i, "lidar_source_ns": int(t * 1e9), "ego_path_s_m": s, "ego_path_d_m": d})
            s += 0.5
            d += -0.1 if 3.0 <= t < 6.0 else 0.1 if t >= 6.0 else 0.0
        rules = [{"index": i, "zone_id": "static_obstacle", "instruction_state": "avoid_continue",
                  "variables": {"avoid_direction": "right"}, "evidence": {}} for i in range(90)]
        align_static_avoidance(frames, rules)
        states = [r["instruction_state"] for r in rules]
        first_avoid = states.index("avoid")
        self.assertEqual(states[:first_avoid], ["approach_maintain"] * first_avoid)
        self.assertGreaterEqual(first_avoid, 3)          # not 3 s early
        self.assertLess(first_avoid, 31)                 # within 2.5 s of the swing at 3.0 s
        self.assertIn("avoid_continue", states)
        self.assertEqual(states[-1], "return_to_path")

    def test_dynamic_waits_until_ego_moves_then_resumes(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import hold_dynamic_wait_until_moving
        frames = track(lambda t: 0.0 if t < 3.0 else min(6.0, 2.0 * (t - 3.0)), seconds=9.0)
        rules = [{"index": i, "zone_id": "dynamic_obstacle",
                  "instruction_state": "yielding" if i < 5 else "normal", "status": "matched"}
                 for i in range(len(frames))]
        hold_dynamic_wait_until_moving(frames, rules)
        states = [r["instruction_state"] for r in rules]
        self.assertEqual(states[10], "wait_clear")
        self.assertIn("resume", states)
        self.assertEqual(states[60], "normal")

    def test_short_rule_gap_between_identical_sentences_is_filled(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        text = "Continue driving."
        rules = [{"index": i, "zone_id": "dynamic_obstacle", "status": "matched" if i not in (3, 4) else "unknown",
                  "instruction": None if i in (3, 4) else text, "instruction_id": None if i in (3, 4) else "dynamic_normal"}
                 for i in range(8)]
        self.assertEqual(fill_short_instruction_gaps(rules), 2)
        self.assertEqual(rules[3]["instruction"], text)
        self.assertEqual(rules[3]["instruction_gap_filled_from"], 2)


class StaticDirectionLatchTests(unittest.TestCase):
    def rows(self, directions, offsets):
        from MoraiDrive.data_collection.situation_labeling.rules import _latch_static_avoid_direction
        results = [{"index": i, "zone_id": "static_obstacle", "instruction_state": "avoid_continue",
                    "variables": {"avoid_direction": d}, "evidence": {}} for i, d in enumerate(directions)]
        indexed = {i: {"ego_reference_path_offset_m": o} for i, o in enumerate(offsets)}
        _latch_static_avoid_direction(results, indexed)
        return [r["variables"]["avoid_direction"] for r in results], results

    def test_passing_a_far_side_obstacle_keeps_the_swing_direction(self):
        # Right swing (Ego right of the path, negative offset); a right-side obstacle becomes nearest.
        directions, rows = self.rows(["right", "right", "left", "left", "right"], [-0.5, -1.5, -1.8, -1.2, -0.8])
        self.assertEqual(directions, ["right"] * 5)
        self.assertEqual(rows[2]["evidence"]["frame_avoid_direction"], "left")

    def test_real_s_curve_switches_once_ego_crosses_to_the_other_side(self):
        directions, _ = self.rows(["right", "right", "left", "left"], [-1.0, -0.2, 0.5, 1.0])
        self.assertEqual(directions, ["right", "right", "left", "left"])


class RoundaboutCleanupTests(unittest.TestCase):
    def test_one_or_two_frame_island_takes_the_surrounding_sentence(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import smooth_short_instruction_islands
        texts = ["Y", "Y", "R", "Y", "C", "C", "Y", "Y", "A", "B", "C", "Y"]
        rules = [{"index": i, "zone_id": "roundabout", "instruction": t, "instruction_id": t} for i, t in enumerate(texts)]
        self.assertEqual(smooth_short_instruction_islands(rules), 3)
        self.assertEqual([r["instruction"] for r in rules], ["Y"] * 8 + ["A", "B", "C", "Y"])
        self.assertEqual(rules[2]["instruction_smoothed_from"], "R")

    def test_yield_while_standing_becomes_remain_stopped(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import use_stopped_roundabout_sentences
        frames = track(lambda t: 0.0 if 2.0 <= t < 6.0 else 3.0, seconds=9.0)
        rules = [{"index": i, "zone_id": "roundabout", "instruction_id": "yield_moving",
                  "instruction": "There is a vehicle in front of the ego vehicle. Yield while waiting to enter.",
                  "instruction_variables": {"position": "in front of"}} for i in range(len(frames))]
        self.assertGreater(use_stopped_roundabout_sentences(frames, rules), 0)
        self.assertEqual(rules[30]["instruction"], "There is a vehicle in front of the ego vehicle. Remain stopped.")
        self.assertEqual(rules[30]["instruction_moving_variant"], "yield_moving")
        self.assertEqual(rules[5]["instruction_id"], "yield_moving")
