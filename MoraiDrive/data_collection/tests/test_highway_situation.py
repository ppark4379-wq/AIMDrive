"""Highway labels use actual lane motion and keep uncertain safety nullable."""
from pathlib import Path

from MoraiDrive.data_collection.mgeo_navigation import MGeoMap
from MoraiDrive.data_collection.situation_labeling.instruction_features import enrich
from MoraiDrive.data_collection.situation_labeling.rules import evaluate_frames
from MoraiDrive.data_collection.situation_labeling.highway_rules import highway_rule

MAP = Path(__file__).resolve().parents[2] / "scenario_runner/data/R_KR_PR_K-city_2025/link_set.json"


def _frame(i, x, y, speed=5.0):
    return {"index": i, "lidar_source_ns": 1_000_000_000+i*100_000_000,
            "zone_id": "highway", "valid": True, "object_inventory_usable": True,
            "ego_world_x_m": x, "ego_world_y_m": y, "ego_speed_mps": speed}


def _route():
    return {"map_id": "R_KR_PR_K-city_2025",
            "nav_waypoints": [[67.0, 85.0], [67.0, 5.0]]}


def test_real_mgeo_lane_change_and_abort_use_current_motion_direction():
    mgeo = MGeoMap(str(MAP))
    y = [mgeo.links["A2256W000420"]["points"][40+i][1] for i in range(9)]
    x = [67.1, 67.1, 67.4, 67.8, 68.4, 68.1, 67.7, 67.3, 67.1]
    frames = [_frame(i, xx, yy) for i, (xx, yy) in enumerate(zip(x, y))]
    enrich(frames, [], _route())
    rules = evaluate_frames(frames, [])
    assert frames[2]["lane_change_active"] is True
    assert frames[2]["source_lane_id"] == "L4"
    assert frames[2]["target_lane_id"] == "L3"
    assert rules[2]["label_candidate"] == "CHANGE_LEFT"
    assert frames[5]["lane_change_aborting"] is True
    assert frames[5]["return_direction"] == "right"
    assert rules[5]["label_candidate"] == "CHANGE_RIGHT"
    assert frames[5]["target_lane_collision_safe"] is None


def test_highway_rule_prioritizes_change_and_requires_causal_response():
    frame = _frame(1, 0, 0)
    frame.update(lane_change_active=True, lane_change_direction="left",
                 ego_decelerating=True, ego_hard_braking=False, ego_stopped=False)
    row = highway_rule(frame, [])
    assert (row["label_candidate"], row["behavior_state"]) == ("CHANGE_LEFT", "decelerating")
    frame.update(lane_change_active=False, lane_change_direction=None)
    row = highway_rule(frame, [])
    assert row["status"] == "unknown"
    assert "ego_response_cause_unverified" in row["unknown_reasons"]


def test_highway_lead_and_target_lane_clearance_from_real_map():
    mgeo = MGeoMap(str(MAP))
    source = mgeo.links["A2256W000420"]["points"]
    target = mgeo.links["A2256W000430"]["points"]
    frames = [_frame(0, source[30][0], source[30][1]),
              _frame(1, source[31][0]+0.25, source[31][1])]
    actor = {"index": 1, "kind": "npc", "actor_id": 17,
             "world_x_m": source[50][0], "world_y_m": source[50][1],
             "speed_mps": 2.0, "actor_heading_deg": -90.0,
             "actor_length_m": 4.5, "actor_width_m": 2.0,
             "invalid_reasons": [], "valid": True}
    front = dict(actor, actor_id=18, world_x_m=target[50][0],
                 world_y_m=target[50][1], speed_mps=3.0)
    enrich(frames, [actor, front], _route())
    current = frames[1]
    assert current["lead_actor_id"] == 17
    assert current["lead_distance_m"] > 0
    assert current["lead_bumper_gap_m"] is not None
    assert current["target_front_actor_id"] == 18
    assert current["target_front_clearance_ok"] is False
    assert current["target_lane_collision_safe"] is None
    # enrich_highway's own ACC-resume debounce runs end to end (as a real
    # boolean, not left at its reset-time None) even when nothing triggers it.
    assert current["highway_acc_prev_responding"] is False


def test_follow_and_merge_need_current_causal_geometry():
    frame = _frame(1, 0, 0)
    frame.update(lane_change_active=False, lane_change_direction=None,
                 ego_decelerating=True, ego_hard_braking=False, ego_stopped=False,
                 lead_actor_id=17, lead_distance_m=9.0, lead_bumper_gap_m=4.0,
                 following_gap_insufficient=True)
    lead = {"index": 1, "kind": "npc", "actor_id": 17,
            "distance_to_ego_m": 9.0, "forward_m": 9.0}
    row = highway_rule(frame, [lead])
    assert (row["label_candidate"], row["behavior_state"]) == ("ACC_FOLLOW", "decelerating")
    assert row["related_actors"] == [{"actor_id": 17, "role": "lead_vehicle"}]
    frame.update(lead_actor_id=None, merging_actor_id=18,
                 is_merging_into_ego_lane=True, merge_conflict=True,
                 merging_actor_side="left", merging_actor_distance_m=7.0)
    merge = {"index": 1, "kind": "npc", "actor_id": 18,
             "distance_to_ego_m": 7.0}
    row = highway_rule(frame, [merge])
    assert (row["label_candidate"], row["behavior_state"]) == ("ACC_FOLLOW", "yielding")
    assert row["variables"]["response_reason"] == "merging_vehicle"


def test_target_lane_front_vehicle_on_successor_link_is_found():
    from MoraiDrive.data_collection.situation_labeling.highway_features import _target_traffic
    mgeo = MGeoMap(str(MAP))
    target = mgeo.links["A2256W000430"]["points"][160]
    next_link = mgeo.links["A2256W000434"]["points"][10]
    actor = {"actor_id": 21, "actor_link_id": "A2256W000434",
             "world_x_m": next_link[0], "world_y_m": next_link[1],
             "actor_heading_deg": -90, "actor_length_m": 4.5,
             "actor_width_m": 2.0}
    front, rear = _target_traffic(mgeo, "A2256W000430", target[:2],
                                  -90, [actor], {}, {})
    assert front is not None and front[2]["actor_id"] == 21
    assert rear is None


def test_sustained_small_lateral_steps_start_lane_change():
    mgeo = MGeoMap(str(MAP))
    points = mgeo.links["A2256W000420"]["points"]
    frames = [_frame(i, points[40+i][0]+offset, points[40+i][1])
              for i, offset in enumerate((0.0, 0.06, 0.12, 0.18))]
    enrich(frames, [], _route())
    assert frames[2]["lane_change_active"] is True
    assert evaluate_frames(frames, [])[2]["label_candidate"] == "CHANGE_LEFT"


def test_lane_change_completion_returns_to_keep_lane():
    mgeo = MGeoMap(str(MAP))
    source = mgeo.links["A2256W000420"]["points"]
    x = (67.1, 67.1, 67.4, 68.0, 69.0, 70.0, 70.4, 70.5, 70.5)
    frames = [_frame(i, xx, source[40+i][1]) for i, xx in enumerate(x)]
    enrich(frames, [], _route())
    rows = evaluate_frames(frames, [])
    assert any(frame["lane_change_completed"] for frame in frames)
    assert rows[-1]["label_candidate"] == "KEEP_LANE"
    # The single frame the maneuver actually finishes on says so, instead of
    # the generic steady-state "normal" every KEEP_LANE frame otherwise says.
    completed_index = next(i for i, f in enumerate(frames) if f["lane_change_completed"])
    assert rows[completed_index]["behavior_state"] == "keep_completed"
    assert rows[-1]["behavior_state"] == "normal"


def test_acc_resume_fires_once_after_a_responding_substate_then_settles_to_follow():
    frame = _frame(1, 0, 0)
    frame.update(lane_change_active=False, lane_change_direction=None,
                 ego_decelerating=False, ego_hard_braking=False, ego_stopped=False,
                 lead_actor_id=17, lead_distance_m=9.0, lead_bumper_gap_m=4.0,
                 following_gap_insufficient=True,
                 highway_acc_prev_responding=True)
    lead = {"index": 1, "kind": "npc", "actor_id": 17,
            "distance_to_ego_m": 9.0, "forward_m": 9.0}
    row = highway_rule(frame, [lead])
    assert (row["label_candidate"], row["behavior_state"]) == ("ACC_FOLLOW", "resume")
    # Without the prior-responding flag the identical current-frame geometry
    # is just the generic, repeatable "follow".
    row = highway_rule(dict(frame, highway_acc_prev_responding=False), [lead])
    assert row["behavior_state"] == "follow"


def test_keep_lane_completed_fires_only_on_the_single_completion_frame():
    frame = _frame(1, 0, 0)
    frame.update(lane_change_active=False, lane_change_direction=None,
                 ego_decelerating=False, ego_hard_braking=False, ego_stopped=False,
                 lane_change_completed=True)
    row = highway_rule(frame, [])
    assert (row["label_candidate"], row["behavior_state"]) == ("KEEP_LANE", "keep_completed")
    row = highway_rule(dict(frame, lane_change_completed=False), [])
    assert row["behavior_state"] == "normal"


def test_adjacent_vehicle_moving_into_ego_lane_supports_yield():
    mgeo = MGeoMap(str(MAP))
    source = mgeo.links["A2256W000420"]["points"]
    target = mgeo.links["A2256W000430"]["points"]
    frames = [_frame(0, source[40][0], source[40][1], 5.2),
              _frame(1, source[41][0], source[41][1], 5.0)]
    actor = {"index": 1, "kind": "npc", "actor_id": 31,
             "world_x_m": target[41][0], "world_y_m": target[41][1],
             "vx_mps": -1.0, "vy_mps": -5.0, "speed_mps": 5.1,
             "actor_heading_deg": -100.0, "actor_length_m": 4.5,
             "actor_width_m": 2.0, "invalid_reasons": [], "valid": True}
    enrich(frames, [actor], _route())
    assert frames[1]["is_merging_into_ego_lane"] is True
    assert frames[1]["merge_conflict"] is True
    result = evaluate_frames(frames, [actor])[1]
    assert result["label_candidate"] == "ACC_FOLLOW"
    assert result["behavior_state"] == "yielding"
