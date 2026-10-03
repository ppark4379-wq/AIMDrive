"""Expert labels require a unique bounded ROS Ego sequence match."""
import json

from MoraiDrive.data_collection.situation_labeling.expert_instructions import (
    apply_expert_instructions, canonical_state, load_config,
)


def _sources(tmp_path, signals, records):
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "active_signals.json").write_text(json.dumps(signals), encoding="utf8")
    log = tmp_path / "expert.log"
    stamps = {row["seq"]:row["source_timestamp_ns"] for row in signals}
    for row in records:
        row["ego_header_seq"] = row["ego_seq"]
        row["ego_source_ns"] = stamps[row["ego_seq"]]
        if "target_speed_mps" in row:
            row.setdefault("planner_target_speed_mps", row["target_speed_mps"])
            row.setdefault("acc_target_speed_mps", row["target_speed_mps"])
    log.write_text("".join("[EXPERT_STATE] " + json.dumps(row) + "\n" for row in records),
                   encoding="utf8")
    return log


def _frame(index, zone, stamp):
    return {"index":index, "zone_id":zone, "lidar_source_ns":stamp}


def _rule(index, zone, variables=None):
    return {"index":index, "zone_id":zone, "variables":variables or {},
            "instruction_state":None, "instruction_id":None}


def test_catalog_and_mapping_keep_distinct_instruction_ids():
    mapping, catalog = load_config()
    assert catalog["highway"]["acc_follow"]["label"] == "ACC_FOLLOW"
    assert catalog["highway"]["acc_decelerate"]["label"] == "ACC_FOLLOW"
    assert catalog["highway"]["acc_follow"]["instruction"] != catalog["highway"]["acc_decelerate"]["instruction"]
    raw = {"mode":"highway_lane", "highway_state":"ACC_FOLLOW",
           "ego_speed_mps":0.0, "planner_target_speed_mps":0.0}
    assert canonical_state("highway", raw, mapping)[0]["state"] == "ACC_STOP"
    raw["ego_speed_mps"] = 5.0
    raw["planner_target_speed_mps"] = 4.0
    assert canonical_state("highway", raw, mapping)[0]["state"] == "ACC_DECEL"


def test_roundabout_committed_cruise_and_follow_are_not_entry_merge():
    mapping, catalog = load_config()
    raw = {"mode": "roundabout_entry", "roundabout_phase": "cruise",
           "roundabout_before_merge": True}
    entry, reason = canonical_state("roundabout", raw, mapping)
    assert reason is None and entry["state"] == "CRUISE_MERGE"
    raw["roundabout_before_merge"] = False
    entry, reason = canonical_state("roundabout", raw, mapping)
    assert reason is None and entry["state"] == "CRUISE_AFTER_MERGE"
    assert entry["instruction_id"] == "cruise_normal"
    assert catalog["roundabout"][entry["instruction_id"]]["instruction"] == catalog["roundabout"]["cruise_normal"]["instruction"]
    raw["roundabout_phase"] = "follow"
    entry, reason = canonical_state("roundabout", raw, mapping)
    assert reason is None and entry["state"] == "FOLLOW"
    assert entry["instruction_id"] == "follow"
    raw["roundabout_before_merge"] = True
    entry, reason = canonical_state("roundabout", raw, mapping)
    assert reason is None and entry["state"] == "FOLLOW_MERGE"
    assert entry["instruction_id"] == "follow_merge"


def test_committed_cruise_renders_continuing_instruction():
    raw = {"mode": "roundabout_entry", "roundabout_phase": "cruise",
           "roundabout_entering": True, "roundabout_committed": True,
           "source": "existing_roundabout_text_log"}
    frame = _frame(144, "roundabout", 52_933_983_000_000)
    row = _rule(144, "roundabout")
    apply_expert_instructions([frame], [row], None, [],
                              index_observations={144: raw})
    assert row["expert_state"] == "CRUISE_AFTER_MERGE"
    assert row["label"] == "cruise"
    assert row["instruction_id"] == "cruise_normal"
    assert row["instruction"] == load_config()[1]["roundabout"]["cruise_normal"]["instruction"]


def test_canonical_state_selects_follow_resume_only_when_flagged():
    mapping, _ = load_config()
    plain = canonical_state("roundabout", {"mode": "roundabout_entry",
                                           "roundabout_phase": "follow"}, mapping)[0]
    assert plain["state"] == "FOLLOW"
    flagged = canonical_state("roundabout", {"mode": "roundabout_entry",
                                             "roundabout_phase": "follow",
                                             "roundabout_resume_detected": True}, mapping)[0]
    assert flagged["state"] == "FOLLOW_RESUME"
    assert flagged["instruction_id"] == "follow_resume"


def test_follow_resume_fires_once_then_settles_back_to_plain_follow():
    # Expert yields, then resolves back to following -- the single frame
    # right after a yield/decel/stop response should say "resume following",
    # not the generic "앞차를 따라 주행해라" that plain FOLLOW repeats
    # indefinitely once already following.
    frames = [dict(_frame(1, "roundabout", 1_000_000_000)),
              dict(_frame(2, "roundabout", 1_100_000_000),
                   lead_actor_id="3", lead_distance_m=12.0),
              dict(_frame(3, "roundabout", 1_200_000_000),
                   lead_actor_id="3", lead_distance_m=11.0)]
    rows = [_rule(1, "roundabout"), _rule(2, "roundabout"), _rule(3, "roundabout")]
    observations = {
        1: {"mode": "roundabout_entry", "roundabout_phase": "yield"},
        2: {"mode": "roundabout_entry", "roundabout_phase": "follow", "roundabout_front_id": "3"},
        3: {"mode": "roundabout_entry", "roundabout_phase": "follow", "roundabout_front_id": "3"},
    }
    apply_expert_instructions(frames, rows, None, [], index_observations=observations)
    assert rows[0]["expert_state"] == "YIELD_MOVING"
    assert rows[1]["expert_state"] == "FOLLOW_RESUME"
    assert rows[1]["label"] == "follow"
    assert rows[1]["instruction"] == load_config()[1]["roundabout"]["follow_resume"]["instruction"].format(position="in front of")
    # The frame after that is just ordinary, continuing follow.
    assert rows[2]["expert_state"] == "FOLLOW"


def test_follow_resume_does_not_fire_across_a_non_contiguous_or_missing_frame():
    # A gap in frame index (or a frame the Expert log says nothing about)
    # breaks the "just resolved" link -- plain FOLLOW, not a guessed resume.
    frames = [_frame(1, "roundabout", 1_000_000_000),
              _frame(5, "roundabout", 1_100_000_000)]
    rows = [_rule(1, "roundabout"), _rule(5, "roundabout")]
    observations = {
        1: {"mode": "roundabout_entry", "roundabout_phase": "yield"},
        5: {"mode": "roundabout_entry", "roundabout_phase": "follow"},
    }
    apply_expert_instructions(frames, rows, None, [], index_observations=observations)
    assert rows[1]["expert_state"] == "FOLLOW"


def test_expert_alignment_and_korean_rendering(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":101}],
        [{"ego_seq":101, "mode":"highway_lane", "highway_state":"ACC_FOLLOW",
          "ego_speed_mps":5.0, "target_speed_mps":4.0}])
    rules = [_rule(7,"highway", {"target_kind":"vehicle", "target_position":"front",
                                 "target_distance_m":13.7})]
    report = apply_expert_instructions([_frame(7,"highway",1_010_000_000)],rules,tmp_path,log)
    assert rules[0]["expert_state"] == "ACC_DECEL"
    assert rules[0]["label"] == "ACC_FOLLOW"
    assert rules[0]["instruction_id"] == "acc_decelerate"
    assert rules[0]["instruction"] == load_config()[1]["highway"]["acc_decelerate"]["instruction"].format(**rules[0]["instruction_variables"] )
    assert rules[0]["expert_alignment"]["gap_ns"] == 10_000_000
    assert report["instruction_rendered"] == 1


def test_missing_or_ambiguous_alignment_never_uses_bundle_index(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":101}],
        [{"ego_seq":101, "mode":"highway_lane", "highway_state":"KEEP_LANE"}])
    rules = [_rule(101,"highway"), _rule(102,"highway")]
    frames = [_frame(101,"highway",1_100_000_000), _frame(102,"highway",1_005_000_000)]
    apply_expert_instructions(frames,rules,tmp_path,log)
    assert rules[0]["label"] is None
    assert rules[0]["expert_alignment"]["reason"] == "ego_source_time_outside_tolerance"
    assert rules[1]["label"] == "KEEP_LANE"
    assert rules[1]["instruction_id"] == "keep_lane_normal"
    assert rules[1]["instruction"] == load_config()[1]["highway"]["keep_lane_normal"]["instruction"]


def test_missing_variables_preserve_label_but_omit_sentence(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":5}],
        [{"ego_seq":5, "mode":"highway_lane", "highway_state":"ACC_FOLLOW",
          "ego_speed_mps":5.0, "target_speed_mps":4.0}])
    rows = [_rule(1,"highway")]
    apply_expert_instructions([_frame(1,"highway",1_000_000_000)],rows,tmp_path,log)
    assert rows[0]["label"] == "ACC_FOLLOW"
    assert rows[0]["instruction"] is None
    assert rows[0]["expert_alignment"]["missing_variables"] == ["position"]


def test_repeated_expert_seq_with_conflicting_state_is_unknown(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":5}],
        [{"ego_seq":5, "mode":"highway_lane", "highway_state":"KEEP_LANE"},
         {"ego_seq":5, "mode":"highway_lane", "highway_state":"ACC_FOLLOW"}])
    rows = [_rule(1,"highway")]
    apply_expert_instructions([_frame(1,"highway",1_000_000_000)],rows,tmp_path,log)
    assert rows[0]["label"] is None
    assert rows[0]["expert_alignment"]["reason"] == "expert_state_missing_or_ambiguous"


def test_internal_callback_count_cannot_act_as_ros_header_seq(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":101}],
        [{"ego_seq":101, "mode":"highway_lane", "highway_state":"KEEP_LANE"}])
    line = log.read_text(encoding="utf8").splitlines()[0]
    record = json.loads(line[len("[EXPERT_STATE] "):])
    record["ego_seq"] = 9999  # Internal callback count deliberately differs.
    log.write_text("[EXPERT_STATE] " + json.dumps(record) + "\n", encoding="utf8")
    rows = [_rule(1,"highway")]
    apply_expert_instructions([_frame(1,"highway",1_000_000_000)],rows,tmp_path,log)
    assert rows[0]["label"] == "KEEP_LANE"
    record["ego_source_ns"] += 1
    log.write_text("[EXPERT_STATE] " + json.dumps(record) + "\n", encoding="utf8")
    rows = [_rule(1,"highway")]
    apply_expert_instructions([_frame(1,"highway",1_000_000_000)],rows,tmp_path,log)
    assert rows[0]["label"] is None


def test_dynamic_expert_mapping_is_excluded_and_roundabout_uses_observed_phase():
    # Dynamic obstacle still has no Expert-state-log contract (no
    # [EXPERT_STATE] tagged observations exist for it), so it's absent from
    # expert_state_mapping.yaml; it does have its own rule-only catalog
    # entries now (label: null, like static obstacle's rule-only states).
    mapping, catalog = load_config()
    assert "dynamic_obstacle" not in mapping
    assert "dynamic_obstacle" in catalog
    assert canonical_state("roundabout", {"mode":"roundabout_entry",
                                          "roundabout_phase":"yield",
                                          "ego_speed_mps":0.0,
                                          "planner_target_speed_mps":0.0}, mapping)[0]["state"] == "YIELD_STOPPED"
    assert canonical_state("roundabout", {"mode":"roundabout_entry",
                                          "roundabout_phase":"rear_pressure"}, mapping)[0] is None



def test_highway_emergency_and_merge_are_selected_from_expert_reason():
    mapping, _ = load_config()
    base = {"mode":"highway_lane", "highway_state":"ACC_FOLLOW",
            "ego_speed_mps":6.0, "planner_target_speed_mps":0.0}
    assert canonical_state("highway", dict(base, reason="highway_lane_emergency_stop"),
                           mapping)[0]["instruction_id"] == "acc_hard_brake"
    assert canonical_state("highway", dict(base, reason="highway_lane_merge_acc"),
                           mapping)[0]["instruction_id"] == "acc_yield"
    assert canonical_state("highway", base, mapping)[0]["instruction_id"] == "acc_decelerate"
    stopped = dict(base, reason="highway_lane_emergency_stop", ego_speed_mps=0.0)
    assert canonical_state("highway", stopped, mapping)[0]["instruction_id"] == "acc_merge_stop"


def test_roundabout_sentence_requires_expert_selected_actor_match(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":3}],
        [{"ego_seq":3, "mode":"roundabout_entry", "roundabout_phase":"follow",
          "roundabout_front_id":"7", "ego_speed_mps":4.0,
          "planner_target_speed_mps":4.0}])
    frame = _frame(1,"roundabout",1_000_000_000)
    frame.update(lead_actor_id=7, lead_distance_m=12.0)
    row = _rule(1,"roundabout", {"target_actor_id":7, "target_kind":"vehicle",
                                "target_distance_m":12.0})
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["instruction_id"] == "follow"
    assert row["instruction"] is not None
    frame["lead_actor_id"] = 8
    row = _rule(1,"roundabout", {"target_actor_id":7, "target_kind":"vehicle",
                                "target_distance_m":12.0})
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["label"] == "follow"
    assert row["instruction"] is None


def test_expert_ego_position_disagreement_blocks_label(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":3}],
        [{"ego_seq":3, "mode":"highway_lane", "highway_state":"KEEP_LANE",
          "ego_xy":[100.0,0.0]}])
    frame = _frame(1,"highway",1_000_000_000)
    frame.update(ego_world_x_m=0.0, ego_world_y_m=0.0)
    row = _rule(1,"highway")
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["label"] is None
    assert row["expert_alignment"]["reason"] == "expert_ego_position_mismatch"


def test_dynamic_zone_remains_unlabeled_even_with_observation(tmp_path):
    log = _sources(tmp_path,
        [{"role":"ego", "source_timestamp_ns":1_000_000_000, "seq":3}],
        [{"ego_seq":3, "mode":"roundabout_entry", "roundabout_phase":"yield"}])
    row = _rule(1,"dynamic_obstacle")
    apply_expert_instructions([_frame(1,"dynamic_obstacle",1_000_000_000)],
                              [row],tmp_path,log)
    assert row["expert_state"] is None and row["label"] is None
    assert row["expert_alignment"]["reason"] == "dynamic_expert_state_unavailable"


def test_existing_roundabout_jsonl_is_processed_without_new_collection_log(tmp_path):
    log = tmp_path / "old_roundabout.jsonl"
    record = {"kind":"plan", "input":{"ego_seq":998},
              "ego":{"xy":[10.0,2.0],"speed_mps":4.0},
              "decision":{"phase":"follow","entering":False,
                          "committed":True,"front_id":7},
              "desired_mps":4.0}
    log.write_text(json.dumps(record)+"\n",encoding="utf8")
    frame = _frame(1,"roundabout",0)
    frame.update(ego_world_x_m=10.0, ego_world_y_m=2.0,
                 lead_actor_id=7, lead_distance_m=12.0)
    row = _rule(1,"roundabout", {"target_actor_id":7,"target_kind":"vehicle",
                                "target_distance_m":12.0})
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["expert_state"] == "FOLLOW"
    assert row["instruction"] is not None
    assert row["expert_alignment"]["method"] == "unique_ego_position"
    assert row["expert_alignment"]["physical_simultaneity_verified"] is False


def test_existing_highway_transition_line_is_sparse_and_position_checked(tmp_path):
    log = tmp_path / "old_driving_debug.log"
    log.write_text("[HighwayDecision] pos=(1.00,2.00) yaw=0.00 speed=6.00 target=0.00 "
                   "reason=highway_lane_emergency_stop lane_conflict=None merge=None\n",
                   encoding="utf8")
    frame = _frame(1,"highway",0)
    frame.update(ego_world_x_m=1.0, ego_world_y_m=2.0)
    row = _rule(1,"highway", {"target_kind":"vehicle", "target_position":"front",
                              "target_distance_m":8.0})
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["expert_state"] == "ACC_EMERGENCY"
    assert row["instruction_id"] == "acc_hard_brake"
    second = _frame(2,"highway",0)
    second.update(ego_world_x_m=20.0, ego_world_y_m=2.0)
    rows = [_rule(1,"highway"),_rule(2,"highway")]
    apply_expert_instructions([frame,second],rows,tmp_path,log)
    assert rows[1]["label"] is None


def test_stationary_position_tie_does_not_guess_sensor_frame(tmp_path):
    log = tmp_path / "old_roundabout.jsonl"
    log.write_text(json.dumps({"kind":"plan", "ego":{"xy":[1.0,2.0],"speed_mps":0.0},
                               "decision":{"phase":"yield"},"desired_mps":0.0})+"\n",
                   encoding="utf8")
    frames = [_frame(i,"roundabout",0) for i in (1,2)]
    for frame in frames:
        frame.update(ego_world_x_m=1.0, ego_world_y_m=2.0)
    rows = [_rule(i,"roundabout") for i in (1,2)]
    apply_expert_instructions(frames,rows,tmp_path,log)
    assert all(row["label"] is None for row in rows)


def test_multiple_existing_expert_logs_can_be_processed_together(tmp_path):
    roundabout = tmp_path / "roundabout.jsonl"
    roundabout.write_text(json.dumps({"kind":"plan", "ego":{"xy":[1.0,2.0],"speed_mps":4.0},
        "decision":{"phase":"cruise", "entering":True}, "desired_mps":4.0})+"\n",encoding="utf8")
    highway = tmp_path / "driving_debug.log"
    highway.write_text("[HighwayDecision] pos=(10.00,20.00) yaw=0.00 speed=6.00 target=0.00 "
        "reason=highway_lane_emergency_stop lane_conflict=None merge=None\n",encoding="utf8")
    a,b = _frame(1,"roundabout",0),_frame(2,"highway",0)
    a.update(ego_world_x_m=1.0,ego_world_y_m=2.0,roundabout_phase="entry")
    b.update(ego_world_x_m=10.0,ego_world_y_m=20.0)
    rows = [_rule(1,"roundabout"),_rule(2,"highway")]
    apply_expert_instructions([a,b],rows,tmp_path,[roundabout,highway])
    assert rows[0]["expert_state"] == "CRUISE_MERGE"
    assert rows[1]["expert_state"] == "ACC_EMERGENCY"


def test_standalone_postprocess_reads_existing_features_without_modifying_them(tmp_path):
    from MoraiDrive.data_collection.situation_labeling.attach_expert_instructions import main
    features = tmp_path / "features"
    features.mkdir()
    frame = _frame(1,"roundabout",0)
    frame.update(ego_world_x_m=1.0,ego_world_y_m=2.0,roundabout_phase="entry")
    rule = _rule(1,"roundabout")
    original_frame = json.dumps(frame)+"\n"
    original_rule = json.dumps(rule)+"\n"
    (features / "frame_features.jsonl").write_text(original_frame,encoding="utf8")
    (features / "situation_rules.jsonl").write_text(original_rule,encoding="utf8")
    log = tmp_path / "old_roundabout.jsonl"
    log.write_text(json.dumps({"kind":"plan", "ego":{"xy":[1.0,2.0],"speed_mps":4.0},
                               "decision":{"phase":"cruise", "entering":True},
                               "desired_mps":4.0})+"\n",encoding="utf8")
    output = tmp_path / "new_instructions"
    main(["--features-dir",str(features),"--expert-log",str(log),"--output",str(output)])
    rows = [json.loads(line) for line in (output / "situation_instructions.jsonl").read_text(encoding="utf8").splitlines()]
    assert rows[0]["expert_state"] == "CRUISE_MERGE"
    assert rows[0]["instruction"] is not None
    assert (features / "frame_features.jsonl").read_text(encoding="utf8") == original_frame
    assert (features / "situation_rules.jsonl").read_text(encoding="utf8") == original_rule


def test_roundabout_uses_planner_output_speed_not_input_desired_speed(tmp_path):
    log = tmp_path / "roundabout.jsonl"
    record = {"kind":"plan", "ego":{"xy":[1.0,2.0],"speed_mps":2.0},
              "decision":{"phase":"yield", "target_velocity":1.0},
              "desired_mps":0.0}
    log.write_text(json.dumps(record)+"\n",encoding="utf8")
    frame = _frame(1,"roundabout",0)
    frame.update(ego_world_x_m=1.0,ego_world_y_m=2.0)
    row = _rule(1,"roundabout")
    apply_expert_instructions([frame],[row],tmp_path,log)
    assert row["expert_state"] == "YIELD_MOVING"
    assert row["expert_raw_state"]["planner_target_speed_mps"] == 1.0
