"""Instruction-state rules require spatially relevant, causal evidence."""
import pytest

from MoraiDrive.data_collection.situation_labeling.rules import (
    dynamic_obstacle_rule, evaluate_frames, roundabout_rule,
    static_obstacle_rule, summarize_rules,
)


def frame(zone, **changes):
    return {"index": 1, "zone_id": zone, "valid": True,
            "object_inventory_usable": True, "ego_speed_mps": 4.0,
            "ego_stopped": False, "ego_decelerating": False,
            "ego_hard_braking": False, **changes}


def actor(kind, identity, distance, **changes):
    return {"index": 1, "kind": kind, "actor_id": identity,
            "instance_key": kind + ":" + str(identity),
            "distance_to_ego_m": distance, "valid": True,
            "rule_geometry_usable": True, **changes}


def test_static_approach_and_return_require_different_current_motion():
    obstacle = actor("obstacle", 1, 10, delta_s_m=10, path_d_m=0.0,
                     actor_width_m=2.0, actor_length_m=2.0)
    row = static_obstacle_rule(frame("static_obstacle", avoidance_active=False), [obstacle])
    assert row["instruction_state"] == "approach_maintain"
    assert row["related_actors"] == [1]
    assert row["variables"]["target_distance_m"] == 10
    behind = dict(obstacle, delta_s_m=-4)
    row = static_obstacle_rule(frame("static_obstacle",
                                     avoidance_active=True,
                                     returning_to_reference_path=True), [behind])
    assert row["instruction_state"] == "return_to_path"


def test_static_follow_keeps_path_for_clear_obstacle_only():
    # lap063 frame 74: the motorcycle is to the right of the nominal route.
    clear = actor("obstacle", 1, 16.85, delta_s_m=16.35, path_d_m=-4.0567,
                  actor_width_m=0.7486, actor_length_m=2.021)
    observed = frame("static_obstacle", expert_label="follow",
                     ego_reference_path_offset_m=-0.187,
                     avoidance_active=False,
                     returning_to_reference_path=False,
                     ego_speed_mps=None)
    row = static_obstacle_rule(observed, [clear])
    assert row["instruction_state"] == "keep_path"
    assert row["label"] == "follow"
    assert row["related_actors"] == [1]
    assert row["evidence"]["path_blocked_candidate"] is False
    assert row["evidence"]["minimum_current_clearance_m"] > 2.0
    assert row["unknown_reasons"] == []

    # Expert approach and ongoing avoidance still need their own decisions.
    assert static_obstacle_rule(dict(observed, expert_label="approach"),
                                [clear])["instruction_state"] == "approach_maintain"
    assert static_obstacle_rule(dict(observed, avoidance_active=True),
                                [clear])["instruction_state"] != "keep_path"
    assert static_obstacle_rule(dict(observed, ego_reference_path_offset_m=-2.5),
                                [clear])["instruction_state"] != "keep_path"
    blocking = dict(clear, path_d_m=-1.0)
    assert static_obstacle_rule(observed, [blocking])["instruction_state"] != "keep_path"


def test_static_obstacle_just_behind_stays_unresolved_on_its_own():
    # The obstacle has gone behind Ego (delta_s_m < 0) while Ego is still
    # laterally displaced (avoidance_active) and hasn't yet been observed
    # recovering (returning_to_reference_path False). static_obstacle_rule
    # alone cannot tell what maneuver was actually in progress (a one-sided
    # avoid, a pass_between, ...), so it stays unresolved here -- it is
    # evaluate_frames' hold-forward step (tested below) that carries the
    # last resolved instruction across this gap, not a guess made per-frame.
    behind = actor("obstacle", 1, 4, delta_s_m=-0.08, path_d_m=0.6,
                   actor_width_m=3.2, actor_length_m=0.6)
    row = static_obstacle_rule(frame("static_obstacle",
                                      avoidance_active=True,
                                      returning_to_reference_path=False,
                                      avoid_direction="right"), [behind])
    assert row["instruction_state"] is None
    assert "return_trajectory_unverified" in row["unknown_reasons"]
    # Confirmed recovery still wins once it starts.
    row = static_obstacle_rule(frame("static_obstacle",
                                      avoidance_active=True,
                                      returning_to_reference_path=True,
                                      avoid_direction="right"), [behind])
    assert row["instruction_state"] == "return_to_path"


def _static_frames_and_objects(specs):
    """specs: list of (index, frame_changes, objects) for zone static_obstacle."""
    frames, objects = [], []
    for index, changes, actors in specs:
        f = {"index": index, "zone_id": "static_obstacle", "valid": True,
             "object_inventory_usable": True, "ego_speed_mps": 4.0,
             "ego_stopped": False, "ego_decelerating": False,
             "ego_hard_braking": False, **changes}
        frames.append(f)
        for a in actors:
            objects.append({"index": index, **a})
    return frames, objects


def test_hold_forward_carries_the_last_resolved_instruction_across_the_gap():
    # Mirrors the real lap064 sequence that prompted this: an obstacle is
    # being avoided (avoid_continue), then two frames where the rule alone
    # can't resolve anything (the obstacle has left range and Ego's offset
    # hasn't been observed shrinking yet), then recovery is confirmed -- the
    # two gap frames should keep saying "avoid_continue", not switch to an
    # unrelated guess or go silent, until recovery is actually observed.
    specs = [
        (0, {"expert_label": "follow", "avoidance_active": True,
             "returning_to_reference_path": False, "avoid_direction": "right",
             "static_avoid_episode_active": True, "static_avoid_initial_hold": False},
         [{"kind": "obstacle", "actor_id": 9, "valid": True,
           "rule_geometry_usable": True, "distance_to_ego_m": 3,
           "delta_s_m": 3, "path_d_m": 1.5,
           "actor_width_m": 0.5, "actor_length_m": 0.5,
           "instance_key": "obstacle:9"}]),
        (1, {"expert_label": "follow", "avoidance_active": True,
             "returning_to_reference_path": False},
         [{"kind": "obstacle", "actor_id": 9, "valid": True,
           "rule_geometry_usable": True, "distance_to_ego_m": 3,
           "delta_s_m": -3, "path_d_m": 1.5,
           "actor_width_m": 0.5, "actor_length_m": 0.5,
           "instance_key": "obstacle:9"}]),
        (2, {"expert_label": "follow", "avoidance_active": True,
             "returning_to_reference_path": False},
         [{"kind": "obstacle", "actor_id": 9, "valid": True,
           "rule_geometry_usable": True, "distance_to_ego_m": 6,
           "delta_s_m": -6, "path_d_m": 1.5,
           "actor_width_m": 0.5, "actor_length_m": 0.5,
           "instance_key": "obstacle:9"}]),
        (3, {"expert_label": "follow", "avoidance_active": True,
             "returning_to_reference_path": True},
         [{"kind": "obstacle", "actor_id": 9, "valid": True,
           "rule_geometry_usable": True, "distance_to_ego_m": 8,
           "delta_s_m": -8, "path_d_m": 1.5,
           "actor_width_m": 0.5, "actor_length_m": 0.5,
           "instance_key": "obstacle:9"}]),
    ]
    frames, objects = _static_frames_and_objects(specs)
    rows = evaluate_frames(frames, objects)
    assert rows[0]["instruction_state"] == "avoid_continue"
    assert rows[1]["instruction_state"] == "avoid_continue"
    assert rows[1]["evidence"]["instruction_held_from_index"] == 0
    assert rows[1]["label"] == "follow"
    assert rows[2]["instruction_state"] == "avoid_continue"
    assert rows[2]["evidence"]["instruction_held_from_index"] == 0
    assert rows[3]["instruction_state"] == "return_to_path"


def test_hold_forward_does_not_cross_a_zone_change_or_index_gap():
    specs = [
        (0, {"expert_label": "follow", "avoidance_active": True,
             "returning_to_reference_path": False, "avoid_direction": "right"},
         [{"kind": "obstacle", "actor_id": 9, "valid": True,
           "rule_geometry_usable": True, "distance_to_ego_m": 3,
           "delta_s_m": 3, "path_d_m": 1.5,
           "actor_width_m": 0.5, "actor_length_m": 0.5,
           "instance_key": "obstacle:9"}]),
    ]
    frames, objects = _static_frames_and_objects(specs)
    # A non-contiguous next frame (index jumps) must not inherit the hold.
    frames.append({"index": 5, "zone_id": "static_obstacle", "valid": True,
                   "object_inventory_usable": True, "ego_speed_mps": 4.0,
                   "ego_stopped": False, "ego_decelerating": False,
                   "ego_hard_braking": False, "expert_label": "follow",
                   "avoidance_active": True, "returning_to_reference_path": False})
    objects.append({"index": 5, "kind": "obstacle", "actor_id": 9, "valid": True,
                    "rule_geometry_usable": True, "distance_to_ego_m": 3,
                    "delta_s_m": -3, "path_d_m": 1.5,
                    "actor_width_m": 0.5, "actor_length_m": 0.5,
                    "instance_key": "obstacle:9"})
    rows = evaluate_frames(frames, objects)
    assert rows[0]["instruction_state"] == "avoid_continue"
    assert rows[1]["instruction_state"] is None
    assert "instruction_held_from_index" not in rows[1]["evidence"]


def test_static_gap_candidate_does_not_claim_safe_passage():
    left = actor("obstacle", 1, 10, delta_s_m=10, path_d_m=1.25,
                 actor_width_m=0.65, actor_length_m=1.21)
    right = actor("obstacle", 2, 12, delta_s_m=12, path_d_m=-4.2,
                  actor_width_m=1.36, actor_length_m=0.6)
    candidate = frame("static_obstacle", corridor_passable_candidate=True,
                      corridor_gap_lower_bound_m=4.0)
    row = static_obstacle_rule(candidate, [left, right])
    assert row["instruction_state"] is None
    assert "corridor_trajectory_unverified" in row["unknown_reasons"]
    row = static_obstacle_rule(dict(candidate, expert_label="approach"),
                               [left, right])
    assert row["instruction_state"] == "approach_maintain"
    row = static_obstacle_rule(dict(candidate, corridor_trajectory_verified=True),
                               [left, right])
    assert row["instruction_state"] == "pass_between"
    row = static_obstacle_rule(dict(candidate, corridor_trajectory_verified=True),
                               [left])
    assert row["instruction_state"] is None


def test_static_follow_between_uses_two_obstacles_and_current_motion():
    left = actor("obstacle", 1, 6.98, delta_s_m=6.98, path_d_m=1.254,
                 actor_width_m=0.647, actor_length_m=1.215)
    right = actor("obstacle", 2, 8.97, delta_s_m=8.97, path_d_m=-4.207,
                  actor_width_m=1.358, actor_length_m=0.6)
    observed = frame("static_obstacle", expert_label="follow",
                     static_avoid_transition_start=True,
                     static_avoid_initial_hold=True,
                     ego_reference_path_offset_m=-0.627,
                     ego_reference_heading_error_deg=-5.2,
                     corridor_passable_candidate=True,
                     corridor_gap_lower_bound_m=4.03)
    row = static_obstacle_rule(observed, [left, right])
    assert row["instruction_state"] == "pass_between"
    assert row["related_actors"] == [1, 2]
    assert row["variables"]["target_position"] == "front"
    assert row["evidence"]["trajectory_safety_verified"] is False
    assert row["evidence"]["corridor_motion_supported"] is True
    row = static_obstacle_rule(dict(observed, ego_reference_heading_error_deg=5.2),
                               [left, right])
    assert row["instruction_state"] == "avoid"
    row = static_obstacle_rule(dict(observed, corridor_passable_candidate=False),
                               [left, right])
    assert row["instruction_state"] == "avoid"


def test_expert_follow_label_enters_avoid_before_gt_offset_threshold():
    # Obstacle on Ego's left, well within the nominal path-conflict band;
    # Ego's GT position has not yet deviated (avoidance_active False).
    left = actor("obstacle", 1, 10, delta_s_m=10, path_d_m=1.0,
                 actor_width_m=0.4, actor_length_m=0.4)
    row = static_obstacle_rule(frame("static_obstacle",
                                      avoidance_active=False,
                                      expert_label="follow",
                                      static_avoid_transition_start=True,
                                      static_avoid_initial_hold=True),
                                [left])
    assert row["instruction_state"] == "avoid"
    assert row["label"] == "follow"
    assert row["variables"]["avoid_direction"] == "right"

    # Once Ego's GT position has actually crossed the threshold, the same
    # (label, state) combination still applies.
    row = static_obstacle_rule(frame("static_obstacle",
                                      avoidance_active=True,
                                      expert_label="follow",
                                      static_avoid_transition_start=True,
                                      static_avoid_initial_hold=True),
                                [left])
    assert row["instruction_state"] == "avoid"
    assert row["label"] == "follow"
    assert row["variables"]["avoid_direction"] == "right"

    # Without an Expert label, the existing GT-threshold gate still applies
    # unchanged (no avoid_direction/candidate evidence -> approach_maintain,
    # no label attached).
    row = static_obstacle_rule(frame("static_obstacle", avoidance_active=False), [left])
    assert row["instruction_state"] == "approach_maintain"
    assert row.get("label") is None


def test_expert_follow_avoid_episode_debounces_to_avoid_continue_and_return():
    # A still-blocking FOLLOW span whose initial-instruction hold window has
    # already elapsed (static_avoid_initial_hold False) says "continue", not
    # "avoid" again -- the bug this debounce exists to fix.
    left = actor("obstacle", 1, 10, delta_s_m=10, path_d_m=1.0,
                 actor_width_m=0.4, actor_length_m=0.4)
    row = static_obstacle_rule(frame("static_obstacle",
                                      expert_label="follow",
                                      static_avoid_transition_start=False,
                                      static_avoid_initial_hold=False,
                                      static_avoid_episode_active=True),
                                [left])
    assert row["instruction_state"] == "avoid_continue"
    assert row["label"] == "follow"
    assert row["variables"]["avoid_direction"] == "right"

    # Once Ego's GT offset starts shrinking back toward the reference path
    # (still mid-episode, obstacle still geometrically in front), the state
    # switches to returning rather than repeating "continue".
    row = static_obstacle_rule(frame("static_obstacle",
                                      expert_label="follow",
                                      static_avoid_transition_start=False,
                                      static_avoid_initial_hold=False,
                                      static_avoid_episode_active=True,
                                      returning_to_reference_path=True),
                                [left])
    assert row["instruction_state"] == "return_to_path"
    assert row["label"] == "follow"


def test_dynamic_yield_wait_and_resume():
    ped = actor("pedestrian", 7, 8, path_d_m=0.2,
                actor_width_m=0.6, actor_length_m=0.6,
                v_d_mps=-1.0, target_side="left")
    row = dynamic_obstacle_rule(frame("dynamic_obstacle",
                                      dynamic_in_yield_family=True,
                                      dynamic_yield_transition_start=True), [ped])
    assert row["instruction_state"] == "yield_start"
    assert row["variables"]["pedestrian_crossing_direction"] == "L2R"
    row = dynamic_obstacle_rule(frame("dynamic_obstacle",
                                      dynamic_in_yield_family=True), [ped])
    assert row["instruction_state"] == "yielding"
    row = dynamic_obstacle_rule(frame("dynamic_obstacle",
                                      ego_stopped=True, dynamic_in_yield_family=True), [ped])
    assert row["instruction_state"] == "waiting"
    row = dynamic_obstacle_rule(frame("dynamic_obstacle",
                                      dynamic_resume_transition_start=True), [])
    assert row["instruction_state"] == "resume"
    assert "resume_front_safety_unverified" in row["unknown_reasons"]
    row = dynamic_obstacle_rule(frame("dynamic_obstacle", ego_speed_mps=None), [ped])
    assert row["instruction_state"] == "behavior_unverified"
    assert "ego_speed_unavailable" in row["unknown_reasons"]


def test_roundabout_approach_ignores_far_invalid_vehicle_but_not_near_invalid():
    far = actor("npc", 1, 200, valid=False, rule_geometry_usable=False)
    approach = frame("roundabout", distance_to_merge_m=20,
                     in_entry_section=True, collision_risk=False)
    row = roundabout_rule(approach, [far])
    assert row["instruction_state"] == "approach_entry"
    near = dict(far, distance_to_ego_m=20)
    row = roundabout_rule(approach, [near])
    assert row["status"] == "unknown"
    assert "relevant_vehicle_geometry_unverified" in row["unknown_reasons"]


@pytest.mark.parametrize("gap,insufficient,decelerating,expected", [
    (20.0, False, False, "follow"),
    (5.0, True, True, "follow_decelerate"),
])
def test_roundabout_following_uses_verified_bumper_gap(gap, insufficient,
                                                         decelerating, expected):
    lead = actor("npc", 17, 12)
    row = roundabout_rule(frame(
        "roundabout", in_circulating_lane=True, collision_risk=False,
        lead_actor_id=17, lead_distance_m=12, lead_bumper_gap_m=gap,
        lead_bumper_gap_verified=True, lead_gap_insufficient=insufficient,
        safe_following_gap_m=7.8, ego_decelerating=decelerating), [lead])
    assert row["instruction_state"] == expected
    assert row["related_actors"] == [17]


def test_roundabout_candidate_is_not_a_collision_risk_claim():
    npc = actor("npc", 8, 15)
    row = roundabout_rule(frame("roundabout", distance_to_merge_m=10,
                                 in_entry_section=True, collision_risk=None,
                                 circulating_vehicle_ids=[8]), [npc])
    assert row["instruction_state"] is None
    assert "future_collision_risk_unverified" in row["unknown_reasons"]
    row = roundabout_rule(frame("roundabout", collision_risk=True,
                                 conflict_actor_id=8, conflict_distance_m=15,
                                 ego_decelerating=True), [npc])
    assert row["instruction_state"] == "decelerate_conflict"


def test_dispatcher_and_summary_use_instruction_states():
    frames = [frame("roundabout", collision_risk=False,
                    distance_to_merge_m=12, in_entry_section=True),
              frame("dynamic_obstacle", index=2),
              frame(None, index=3),
              frame("static_obstacle", index=4, valid=False,
                    invalid_reasons=["not_admitted"])]
    rows = evaluate_frames(frames, [])
    assert [row["status"] for row in rows] == [
        "matched", "matched", "not_applicable", "invalid_frame"]
    assert rows[0]["instruction_state"] == "approach_entry"
    assert rows[1]["instruction_state"] == "normal"
    summary = summarize_rules(rows)
    assert summary["schema"] == "morai_situation_rules_v0_2"
    assert summary["instruction_state_counts"] == {
        "approach_entry": 1, "normal": 1}


def test_dynamic_targets_conflicting_pedestrian_not_merely_nearest():
    clear = actor("pedestrian", 1, 4, path_d_m=4.0,
                  actor_width_m=0.6, actor_length_m=0.6,
                  v_d_mps=0.0)
    conflict = actor("pedestrian", 2, 8, path_d_m=0.2,
                     actor_width_m=0.6, actor_length_m=0.6,
                     v_d_mps=-1.0)
    row = dynamic_obstacle_rule(frame("dynamic_obstacle",
                                      dynamic_in_yield_family=True),
                                [clear, conflict])
    assert row["instruction_state"] == "yielding"
    assert row["related_actors"] == [2]
    assert row["variables"]["pedestrian_count"] == 2
    assert [item["conflict"] for item in row["evidence"]["pedestrians"]] == [False, True]
