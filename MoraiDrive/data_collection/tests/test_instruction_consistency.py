"""Instructions contradicted by the ego's next 2.5 s are dropped, others kept."""
import copy
import unittest

from MoraiDrive.data_collection.situation_labeling.consistency import drop_contradicted_instructions
from MoraiDrive.data_collection.situation_labeling.renderer import load_catalog, render_instruction


def track(speeds_mps, lateral_rate=0.0, dt=0.1, seconds=6.0):
    frames, s, d = [], 0.0, 0.0
    for i in range(int(seconds / dt)):
        frames.append({"index": i, "lidar_source_ns": int(i * dt * 1e9),
                       "ego_path_s_m": s, "ego_path_d_m": d,
                       "valid": True, "object_inventory_usable": True})
        s += speeds_mps(i * dt) * dt
        d += lateral_rate * dt
    return frames


def instruction(index, state="normal", instruction_id="dynamic_normal", zone="dynamic_obstacle",
                *, label=None, direction=None, actor="1", instance="actor-generation-0", side="left"):
    variables = {"target_kind": "pedestrian", "target_semantic_class": "pedestrian",
                 "target_semantic_valid": True, "target_model_name": "Woman8", "target_side": side}
    evidence = {}
    if state != "normal":
        evidence["target_actor_ref"] = {"actor_id": actor, "instance_key": instance}
        variables["target_actor_id"] = actor
    if direction:
        variables["avoid_direction"] = direction
    text, values, missing = render_instruction(load_catalog(), zone, instruction_id, variables)
    if missing:
        raise AssertionError(missing)
    return {"index": index, "zone_id": zone, "status": "matched", "unknown_reasons": [],
            "instruction_state": state, "instruction_id": instruction_id, "instruction": text,
            "instruction_variables": values, "instruction_source": "rule_state" if label is None else "expert_label_and_rule",
            "instruction_mapping_status": "rendered", "instruction_label": label, "label": label,
            "variables": variables, "evidence": evidence}


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

    def test_short_unknown_rule_gap_is_not_invented_as_normal(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        text = "Continue driving."
        rules = [{"index": i, "zone_id": "dynamic_obstacle", "status": "matched" if i not in (3, 4) else "unknown",
                  "instruction": None if i in (3, 4) else text, "instruction_id": None if i in (3, 4) else "dynamic_normal"}
                 for i in range(8)]
        self.assertEqual(fill_short_instruction_gaps(rules), 0)
        self.assertIsNone(rules[3]["instruction"])
        self.assertEqual(rules[3]["status"], "unknown")


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
    def test_untyped_short_islands_do_not_inherit_neighbor_actions(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import smooth_short_instruction_islands
        texts = ["Y", "Y", "R", "Y", "C", "C", "Y", "Y", "A", "B", "C", "Y"]
        rules = [{"index": i, "zone_id": "roundabout", "instruction": t, "instruction_id": t} for i, t in enumerate(texts)]
        self.assertEqual(smooth_short_instruction_islands(rules), 0)
        self.assertEqual([r["instruction"] for r in rules], texts)

    def test_yield_while_standing_becomes_remain_stopped(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import use_stopped_roundabout_sentences
        frames = track(lambda t: 0.0 if 2.0 <= t < 6.0 else 3.0, seconds=9.0)
        rules = [instruction(i, "decelerate_conflict", "yield_moving", "roundabout", label="yield")
                 for i in range(len(frames))]
        for row in rules:
            row["variables"]["target_side"] = "front"
            row["instruction_variables"] = {"position": "in front of"}
            row["expert_state"] = "YIELD_MOVING"
        self.assertGreater(use_stopped_roundabout_sentences(frames, rules), 0)
        self.assertEqual(rules[30]["instruction"], "There is a vehicle in front of the ego vehicle. Remain stopped.")
        self.assertEqual(rules[30]["instruction_moving_variant"], "yield_moving")
        self.assertEqual(rules[30]["instruction_state"], "wait_conflict")
        self.assertEqual(rules[30]["instruction_label"], "yield")
        self.assertEqual(rules[30]["expert_state"], "YIELD_MOVING")
        self.assertEqual(rules[30]["instruction_postprocessing_history"][0]["before"]["instruction_state"],
                         "decelerate_conflict")
        self.assertEqual(rules[5]["instruction_id"], "yield_moving")


class VerifiedRenderingTests(unittest.TestCase):
    def test_resolved_rendering_gap_can_restore_own_state_with_history(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        rows = [instruction(i) for i in range(3)]
        rows[1]["instruction"] = None
        self.assertEqual(fill_short_instruction_gaps(rows, frames=track(lambda t: 3.0)), 1)
        self.assertEqual(rows[1]["instruction"], "Continue driving.")
        self.assertEqual(rows[1]["instruction_state"], "normal")
        self.assertIsNone(rows[1]["instruction_postprocessing_history"][0]["before"]["instruction"])

    def test_c5_inventory_gap_remains_null(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        rows = [instruction(i) for i in range(220, 223)]
        rows[1].update(instruction=None, instruction_state=None, instruction_id=None,
                       instruction_mapping_status="rule_state_unavailable", status="unknown",
                       unknown_reasons=["object_inventory_unusable"])
        frames = track(lambda t: 8.0, seconds=25)
        frames[221]["object_inventory_usable"] = False
        before = copy.deepcopy(rows[1])
        self.assertEqual(fill_short_instruction_gaps(rows, frames=frames), 0)
        self.assertEqual(rows[1], before)

    def test_frame_evidence_overrides_apparently_matched_gap(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        for flag in ("object_inventory_usable", "valid"):
            rows = [instruction(i) for i in range(3)]
            rows[1]["instruction"] = None
            frames = track(lambda t: 5.0)
            frames[1][flag] = False
            self.assertEqual(fill_short_instruction_gaps(rows, frames=frames), 0)
            self.assertIsNone(rows[1]["instruction"])

    def test_dropped_and_unknown_semantics_are_never_restored(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        for exclusion in ("motion", "semantic", "unknown_actor"):
            rows = [instruction(i) for i in range(3)]
            rows[1]["instruction"] = None
            if exclusion == "motion":
                rows[1].update(instruction_dropped="Continue driving.", instruction_consistency="dropped_direction")
            elif exclusion == "semantic":
                rows[1]["variables"]["target_semantic_valid"] = False
            else:
                rows[1]["unknown_reasons"] = ["relevant_pedestrian_geometry_unverified"]
            self.assertEqual(fill_short_instruction_gaps(rows), 0)
            self.assertIsNone(rows[1]["instruction"])

    def test_text_noise_repaired_without_copying_slot_dictionaries(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import smooth_short_instruction_islands
        rows = [instruction(i, "approach", "dynamic_approach") for i in range(3)]
        rows[1]["instruction"] = "garbled cached rendering"
        self.assertEqual(smooth_short_instruction_islands(rows), 1)
        self.assertEqual(rows[1]["instruction"], rows[0]["instruction"])
        self.assertIsNot(rows[1]["instruction_variables"], rows[0]["instruction_variables"])
        self.assertEqual(rows[1]["instruction_postprocessing_history"][0]["before"]["instruction"],
                         "garbled cached rendering")

    def test_c5_waiting_island_keeps_stop_action_and_state(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import smooth_short_instruction_islands
        rows = [instruction(159, "yielding", "dynamic_yielding"),
                instruction(160, "waiting", "dynamic_waiting"),
                instruction(161, "waiting", "dynamic_waiting"),
                instruction(162, "yielding", "dynamic_yielding")]
        before = copy.deepcopy(rows)
        self.assertEqual(smooth_short_instruction_islands(rows), 0)
        self.assertEqual(rows, before)

    def test_resume_and_direction_and_actor_changes_survive_one_frame(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import smooth_short_instruction_islands
        for variant in ("resume", "direction", "actor", "generation", "position"):
            if variant == "resume":
                rows = [instruction(0), instruction(1, "resume", "dynamic_resume"), instruction(2)]
            elif variant == "direction":
                rows = [instruction(i, "avoid", "static_avoid", "static_obstacle", direction="right")
                        for i in range(3)]
                rows[1] = instruction(1, "avoid", "static_avoid", "static_obstacle", direction="left")
            else:
                rows = [instruction(i, "approach", "dynamic_approach") for i in range(3)]
                key, value = {"actor": ("actor", "2"), "generation": ("instance", "actor-generation-1"),
                              "position": ("side", "right")}[variant]
                rows[1] = instruction(1, "approach", "dynamic_approach", **{key: value})
            before = copy.deepcopy(rows)
            self.assertEqual(smooth_short_instruction_islands(rows), 0, variant)
            self.assertEqual(rows, before, variant)

    def test_mismatched_state_id_tuple_is_not_silently_repaired(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        rows = [instruction(i, "yielding", "dynamic_yielding") for i in range(3)]
        for row in rows:
            row["instruction_state"] = "waiting"
        rows[1]["instruction"] = None
        self.assertEqual(fill_short_instruction_gaps(rows), 0)

    def test_time_and_index_gaps_prevent_rendering_recovery(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import fill_short_instruction_gaps
        for variant in ("time", "index", "zone"):
            rows = [instruction(i) for i in range(3)]
            rows[1]["instruction"] = None
            frames = track(lambda t: 4.0)
            if variant == "time":
                frames[2]["lidar_source_ns"] += 1_000_000_000
            elif variant == "index":
                rows[2]["index"] = 3
            else:
                rows[1]["zone_id"] = "static_obstacle"
            self.assertEqual(fill_short_instruction_gaps(rows, frames=frames), 0, variant)


class ContinuousMotionEvidenceTests(unittest.TestCase):
    def test_contradiction_duration_does_not_merge_across_missing_indices_or_zones(self):
        frames = track(lambda t: 0.0, seconds=12.0)
        for indices, zones in (((10, 11, 80), ("dynamic_obstacle",) * 3),
                               ((10, 11, 12), ("dynamic_obstacle", "dynamic_obstacle", "roundabout"))):
            rows = [dict(index=index, zone_id=zone, instruction="Continue driving.")
                    for index, zone in zip(indices, zones)]
            drop_contradicted_instructions(frames, rows)
            self.assertTrue(all(row["instruction"] is not None for row in rows))
            self.assertTrue(all(row["instruction_consistency"] == "consistent_borderline" for row in rows))

    def test_roundabout_stop_does_not_restore_a_prior_exclusion(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import use_stopped_roundabout_sentences
        rows = [instruction(i, "decelerate_conflict", "yield_moving", "roundabout", label="yield")
                for i in range(50)]
        for row in rows:
            row.update(instruction=None, instruction_dropped="prior rejected instruction",
                       instruction_consistency="dropped_direction")
        before = copy.deepcopy(rows)
        self.assertEqual(use_stopped_roundabout_sentences(track(lambda t: 0.0), rows), 0)
        self.assertEqual(rows, before)

    def test_invalid_or_missing_future_track_samples_do_not_certify_motion(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import _Track
        for variant in ("invalid", "missing", "time_gap", "duplicate_stamp"):
            frames = track(lambda t: 5.0)
            if variant == "invalid":
                frames[20]["valid"] = False
            elif variant == "missing":
                frames.pop(20)
            elif variant == "time_gap":
                for row in frames[20:]:
                    row["lidar_source_ns"] += 400_000_000
            else:
                frames[20]["lidar_source_ns"] = frames[19]["lidar_source_ns"]
            self.assertIsNone(_Track(frames).future(10), variant)

    def test_dynamic_resume_is_not_carried_across_unusable_inventory(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import hold_dynamic_wait_until_moving
        frames = track(lambda t: 0.0, seconds=8)
        rows = [instruction(i, "yielding", "dynamic_yielding") if i < 10 else instruction(i)
                for i in range(len(frames))]
        rows[10].update(status="unknown", instruction_state=None, unknown_reasons=["object_inventory_unusable"])
        frames[10]["object_inventory_usable"] = False
        hold_dynamic_wait_until_moving(frames, rows)
        self.assertIsNone(rows[10]["instruction_state"])
        self.assertEqual(rows[11]["instruction_state"], "normal")

    def test_unknown_highway_row_is_not_promoted_by_observed_lane_change(self):
        from MoraiDrive.data_collection.situation_labeling.consistency import align_highway_lane_changes
        frames = track(lambda t: 5.0, lateral_rate=0.7, seconds=9)
        rows = [{"index": row["index"], "zone_id": "highway", "status": "unknown",
                 "label_candidate": "CHANGE_LEFT", "instruction_state": None,
                 "unknown_reasons": ["object_inventory_unusable"], "variables": {}}
                for row in frames]
        before = copy.deepcopy(rows)
        align_highway_lane_changes(frames, rows, [])
        self.assertEqual(rows, before)
