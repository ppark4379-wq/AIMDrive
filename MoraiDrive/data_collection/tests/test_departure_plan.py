"""Command-intent regressions; synthetic cases do not certify sensor synchrony."""
import copy

import pytest

from MoraiDrive.data_collection.situation_labeling.departure_plan import (
    INSTRUCTION_ID, MARKER, align_current_departure_plans,
    current_instruction_command_reasons, validate_current_departure_plan)
from MoraiDrive.data_collection.situation_labeling.dynamic_normal import align_current_dynamic_normal
from MoraiDrive.data_collection.situation_labeling.dynamic_resume import _corridor, _past_motion
from MoraiDrive.data_collection.situation_labeling.expert_instructions import _roundabout_target_proof
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions
from MoraiDrive.data_collection.situation_labeling.rules import dynamic_obstacle_rule


def case(kind="response"):
    frames = [dict(index=i, lidar_source_ns=1_000_000_000 + i * 100_000_000,
                   zone_id="dynamic_obstacle", valid=True, dataset_build_candidate=True,
                   object_inventory_usable=True, ego_world_x_m=0., ego_world_y_m=0.,
                   ego_path_s_m=0., ego_path_d_m=0., ego_speed_mps=0.,
                   ego_speed_source="backward_gt_ego_xy_difference") for i in range(5)]
    f = frames[-1]
    objects = []
    if kind == "normal":
        rules = [dynamic_obstacle_rule(frame, []) for frame in frames]
        align_current_dynamic_normal(frames, objects, rules)
        attach_rule_instructions(rules)
        r = rules[-1]
    else:
        r = dict(index=f["index"], zone_id="dynamic_obstacle", status="matched",
                 instruction_state="wait_clear", instruction_id="dynamic_wait_clear",
                 instruction="Remain stopped.", instruction_variables={}, instruction_source="rule_state",
                 variables={}, unknown_reasons=[], related_actors=["1"], label=None)
        if kind == "response":
            objects = [dict(index=f["index"], lidar_source_ns=f["lidar_source_ns"], actor_id="1",
                            instance_key="0|OBJECT_TYPE_PEDESTRIAN|1|0", kind="pedestrian", model_name="Man1",
                            model_identity_verified=True, semantic_class="pedestrian", semantic_class_valid=True,
                            rule_geometry_usable=True, world_x_m=5., world_y_m=5., path_d_m=5.,
                            delta_s_m=5., path_s_m=5., ego_path_s_m=0., actor_width_m=.5,
                            actor_length_m=.5, v_d_mps=0., invalid_reasons=[])]
            r["evidence"] = {"current_dynamic_response": dict(
                corridor=_corridor(f, objects), motion=_past_motion(frames),
                physical_front_safety_certified=False, supervision_basis="recorded_ego_motion",
                state_before="normal", state_after="wait_clear")}
        else:
            f.update(zone_id="roundabout", conflict_actor_id="9")
            objects = [dict(index=f["index"], lidar_source_ns=f["lidar_source_ns"], actor_id="9",
                            instance_key="0|OBJECT_TYPE_VEHICLE|9|0", kind="vehicle", semantic_class="vehicle",
                            semantic_class_valid=True, model_identity_verified=True,
                            rule_geometry_usable=True, forward_m=5., left_m=0.)]
            raw = dict(roundabout_phase="yield", roundabout_conflict_id="9")
            r.update(zone_id="roundabout", instruction_state="wait_conflict", instruction_id="yield_stopped",
                     instruction="There is a vehicle in front of the ego vehicle. Remain stopped.",
                     instruction_variables={"position": "in front of"}, instruction_source="expert_observation",
                     expert_state="YIELD_MOVING", expert_alignment={"status": "matched"},
                     expert_raw_state=raw, related_actors=["9"], label="yield",
                     evidence={"roundabout_expert_target": _roundabout_target_proof(f, raw, objects)})
    r.update(action_valid=True, observation_valid=True, instruction_consistency="consistent")
    anchor = 9_000_000_000
    def receipt(record_id, ros_type, **values):
        return dict(record_id=record_id, ros_type=ros_type, payload_sha256="a" * 64,
                    source_timestamp_ns=None, source_timestamp_fabricated=False,
                    receive_monotonic_ns=anchor - 20_000_000, receive_age_ms=20.,
                    preceding_received_clock_ns=f["lidar_source_ns"] - 1_000_000,
                    **values)
    f["expert_publications"] = dict(status="RECEIVE_BOUND_CURRENT_CONTEXT", sensor_ns=f["lidar_source_ns"],
        source_sha256={"quality.json": "b" * 64}, lidar_record_id=100, lidar_receive_monotonic_ns=anchor,
        control=receipt(1, "morai_msgs/CtrlCmd", longlCmdType=1, accel=.4, brake=0.),
        target_velocity=receipt(2, "std_msgs/Float32", value_mps=5.55555534362793))
    return frames, objects, [r]


@pytest.mark.parametrize("kind", ["response", "normal", "roundabout"])
def test_current_departure_intent_does_not_wait_for_observed_motion(kind):
    fs, obs, rs = case(kind)
    original = copy.deepcopy(rs[0])
    counts = align_current_departure_plans(fs, obs, rs)
    r = rs[0]
    assert counts["verified_current_departure_plan"] == 1
    assert r["instruction"] == "Resume driving." and r["instruction_id"] == INSTRUCTION_ID
    assert r["label"] is None and r["related_actors"] == [] and r["instruction_variables"] == {}
    assert r["action_valid"] and r["observation_valid"]
    marker = r["evidence"][MARKER]
    assert marker["original_stop"] == original
    assert marker["uses_future_ego"] is False and marker["observed_departure_claimed"] is False
    assert marker["physical_front_safety_certified"] is False
    assert validate_current_departure_plan(fs, obs, r) == (True, [])


@pytest.mark.parametrize("kind", ["response", "normal", "roundabout"])
def test_current_command_plan_survives_final_consistency_and_quality(kind):
    from MoraiDrive.data_collection.situation_labeling.consistency import drop_contradicted_instructions
    from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality
    fs, obs, rs = case(kind)
    align_current_departure_plans(fs, obs, rs)
    drop_contradicted_instructions(fs, rs, objects=obs)
    r = rs[0]
    others = [dynamic_obstacle_rule(f, []) for f in fs if f["index"] != r["index"]]
    attach_rule_instructions(others)
    finalize_instruction_quality(fs, others + rs, objects=obs)
    assert r["instruction"] == "Resume driving." and r["instruction_consistency"] == "consistent"
    assert r["action_valid"] is True and r["observation_valid"] is True


def test_historical_signed_target_barrier_does_not_replace_or_negate_fresh_command():
    fs, obs, rs = case()
    pub = fs[-1]["expert_publications"]
    pub["reasons"] = ["unsupported_historical_expert_target_velocity"]
    pub["unsupported_publications"] = [{"raw_value_mps": -3., "purpose": "prior_stop_context_search_barrier"}]
    align_current_departure_plans(fs, obs, rs)
    assert rs[0]["instruction_id"] == INSTRUCTION_ID
    assert rs[0]["evidence"][MARKER]["target_velocity"]["value_mps"] > .5


def test_roundabout_source_proof_requires_all_current_fields():
    fs, obs, rs = case("roundabout")
    rs[0]["evidence"]["roundabout_expert_target"].pop("instance_key")
    original = copy.deepcopy(rs[0])
    align_current_departure_plans(fs, obs, rs)
    assert rs[0] == original


def test_roundabout_original_actor_coordinates_cannot_be_replaced_by_marker():
    fs, obs, rs = case("roundabout")
    obs[0]["forward_m"] += .1
    original = copy.deepcopy(rs[0])
    align_current_departure_plans(fs, obs, rs)
    assert rs[0] == original


@pytest.mark.parametrize("mutation", ["action_mask", "observation_mask", "semantic_mask", "unknown",
    "frame_mask", "inventory_missing", "stale_receipt", "future_receipt", "receipt_age_disagrees",
    "stale_clock", "future_clock", "wrong_sensor", "bad_payload_hash", "bad_source_hash",
    "fabricated_stamp", "wrong_mode", "brake", "zero_target", "negative_target", "low_accel",
    "missing_control", "missing_target", "changed_actor", "past_motion_changed"])
def test_excluded_or_conflicting_source_never_creates_departure(mutation):
    fs, obs, rs = case()
    f, r = fs[-1], rs[0]
    pub = f["expert_publications"]
    control = pub["control"]
    if mutation == "action_mask": r["action_valid"] = False
    elif mutation == "observation_mask": r["observation_valid"] = False
    elif mutation == "semantic_mask": r["variables"]["target_semantic_valid"] = False
    elif mutation == "unknown": r["unknown_reasons"] = ["current_scene_unverified"]
    elif mutation == "frame_mask": f["dataset_build_candidate"] = False
    elif mutation == "inventory_missing": obs = None
    elif mutation == "stale_receipt": control.update(receive_age_ms=100.000001, receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]-100_000_001)
    elif mutation == "future_receipt": control.update(receive_age_ms=-.000001, receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]+1)
    elif mutation == "receipt_age_disagrees": control["receive_age_ms"] = 0.
    elif mutation == "stale_clock": control["preceding_received_clock_ns"] = f["lidar_source_ns"]-100_000_001
    elif mutation == "future_clock": control["preceding_received_clock_ns"] = f["lidar_source_ns"]+1
    elif mutation == "wrong_sensor": pub["sensor_ns"] += 1
    elif mutation == "bad_payload_hash": control["payload_sha256"] = "z" * 64
    elif mutation == "bad_source_hash": pub["source_sha256"]["quality.json"] = "0"
    elif mutation == "fabricated_stamp": control["source_timestamp_fabricated"] = True
    elif mutation == "wrong_mode": control["longlCmdType"] = 2
    elif mutation == "brake": control["brake"] = .01
    elif mutation == "zero_target": pub["target_velocity"]["value_mps"] = 0.
    elif mutation == "negative_target": pub["target_velocity"]["value_mps"] = -1.
    elif mutation == "low_accel": control["accel"] = .2
    elif mutation == "missing_control": pub["control"] = None
    elif mutation == "missing_target": pub["target_velocity"] = None
    elif mutation == "changed_actor": obs[0]["path_d_m"] = 0.
    elif mutation == "past_motion_changed": fs[-2]["ego_world_x_m"] = 1.
    original = copy.deepcopy(r)
    align_current_departure_plans(fs, obs, rs)
    assert r == original


@pytest.mark.parametrize("tamper", ["text", "slots", "mask", "marker_hash", "command", "frame_hash",
                                   "original_proof", "claim_safety", "actor_join"])
def test_final_validator_recomputes_sources_and_checks_rendering(tamper):
    fs, obs, rs = case()
    align_current_departure_plans(fs, obs, rs)
    r = rs[0]
    marker = r["evidence"][MARKER]
    if tamper == "text": r["instruction"] = "Remain stopped."
    elif tamper == "slots": r["instruction_variables"] = {"position": "in front of"}
    elif tamper == "mask": r["action_valid"] = False
    elif tamper == "marker_hash": marker["control"]["payload_sha256"] = "c" * 64
    elif tamper == "command": fs[-1]["expert_publications"]["control"]["accel"] = .8
    elif tamper == "frame_hash": fs[-1]["expert_publications"]["source_sha256"]["quality.json"] = "c" * 64
    elif tamper == "original_proof": marker["original_stop"]["evidence"]["current_dynamic_response"]["motion"]["current_speed_mps"] = 1.
    elif tamper == "claim_safety": marker["physical_front_safety_certified"] = True
    elif tamper == "actor_join": obs[0]["lidar_source_ns"] -= 1
    valid, reasons = validate_current_departure_plan(fs, obs, r)
    assert valid is False and reasons


def test_real_command_stop_go_sequence_is_not_smoothed():
    cases = [case() for _ in range(3)]
    cases[0][0][-1]["expert_publications"]["control"].update(accel=0., brake=1.)
    cases[0][0][-1]["expert_publications"]["target_velocity"]["value_mps"] = 0.
    cases[2][0][-1]["expert_publications"]["control"].update(accel=0., brake=1.)
    cases[2][0][-1]["expert_publications"]["target_velocity"]["value_mps"] = 0.
    for fs, obs, rs in cases:
        align_current_departure_plans(fs, obs, rs)
    assert [rs[0]["instruction"] for _, _, rs in cases] == ["Remain stopped.", "Resume driving.", "Remain stopped."]


@pytest.mark.parametrize("target", [None, 0., 5.5])
def test_fresh_hard_hold_excludes_resume_without_filling_target(target):
    fs, _, rs = case()
    f, r = fs[-1], rs[0]
    r.update(instruction_id="dynamic_resume", instruction_state="resume", instruction="Resume driving.")
    pub = f["expert_publications"]
    pub["control"].update(accel=0., brake=1.)
    if target is None: pub["target_velocity"] = None
    else: pub["target_velocity"]["value_mps"] = target
    original = copy.deepcopy((f, r))
    assert current_instruction_command_reasons(f, r) == ["current_dynamic_resume_hard_hold_command_conflict"]
    assert (f, r) == original


def test_hard_hold_conflict_is_quarantined_by_final_quality_with_observation_preserved():
    from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality
    fs, _, rs = case()
    f, r = fs[-1], rs[0]
    r.update(instruction_id="dynamic_resume", instruction_state="resume", instruction="Resume driving.")
    f["expert_publications"]["control"].update(accel=0., brake=1.)
    f["expert_publications"]["target_velocity"] = None
    finalize_instruction_quality([f], [r])
    assert r["instruction"] is None and r["action_valid"] is False
    assert r["observation_valid"] is True
    assert r["instruction_rejected"]["instruction"] == "Resume driving."
    assert "current_dynamic_resume_hard_hold_command_conflict" in r["instruction_quality"]["action_exclusion_reasons"]


@pytest.mark.parametrize("mutation", ["stale", "future", "wrong_clock", "wrong_hash", "wrong_mode",
                                    "wrong_sensor", "light_brake", "active_throttle", "other_sentence"])
def test_incomplete_or_nonhold_receipt_cannot_certify_resume_conflict(mutation):
    fs, _, rs = case()
    f, r = fs[-1], rs[0]
    r.update(instruction_id="dynamic_resume", instruction="Resume driving.")
    pub = f["expert_publications"]
    control = pub["control"]
    control.update(accel=0., brake=1.)
    if mutation == "stale": control.update(receive_age_ms=101., receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]-101_000_000)
    elif mutation == "future": control.update(receive_age_ms=-1., receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]+1_000_000)
    elif mutation == "wrong_clock": control["preceding_received_clock_ns"] = f["lidar_source_ns"]+1
    elif mutation == "wrong_hash": control["payload_sha256"] = "bad"
    elif mutation == "wrong_mode": control["longlCmdType"] = 2
    elif mutation == "wrong_sensor": pub["sensor_ns"] += 1
    elif mutation == "light_brake": control["brake"] = .2
    elif mutation == "active_throttle": control["accel"] = .4
    elif mutation == "other_sentence": r["instruction_id"] = "dynamic_wait_clear"
    assert current_instruction_command_reasons(f, r) == []
