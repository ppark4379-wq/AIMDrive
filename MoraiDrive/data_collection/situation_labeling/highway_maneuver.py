"""Reject highway moving-maneuver wording during a recorded current stop.

Legacy lane-change spans use future trajectory labels. This narrow guard does
not turn that future motion into evidence that Ego is moving now, and does not
certify target-lane safety or replace an excluded sentence with a guessed cause.
"""
from collections import Counter
import math

from .renderer import load_catalog, render_instruction

MARKER = "current_highway_maneuver_motion"
SCHEMA = "current_highway_maneuver_motion_v1"
IDS = frozenset(("change_left", "change_right", "change_left_return", "change_right_return"))
STOP_SPEED_MPS = .2
MAX_INTERVAL_S = .25
MAX_SPEED_DISAGREEMENT_MPS = .02


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_highway_maneuver(rule):
    return rule.get("zone_id") == "highway" and rule.get("instruction_id") in IDS


def _admitted(frame):
    return (frame.get("valid") is True and frame.get("dataset_build_candidate") is True
            and not frame.get("invalid_reasons")
            and all(_finite(frame.get(k)) for k in
                    ("index", "lidar_source_ns", "ego_world_x_m", "ego_world_y_m")))


def current_maneuver_motion(frames, rule):
    """Recompute one backward GT interval; no future pose or text is consulted."""
    if not is_highway_maneuver(rule):
        return None, "not_current_highway_maneuver"
    index = rule.get("index")
    if not _finite(index) or int(index) != index:
        return None, "current_highway_maneuver_index_unverified"
    current = [f for f in frames if f.get("index") == index]
    previous = [f for f in frames if f.get("index") == index - 1]
    if len(current) != 1 or len(previous) != 1:
        return None, "current_highway_maneuver_track_missing_or_ambiguous"
    frame, past = current[0], previous[0]
    if frame.get("zone_id") != "highway" or not _admitted(frame) or not _admitted(past):
        return None, "current_highway_maneuver_frame_unusable"
    dt = (frame["lidar_source_ns"] - past["lidar_source_ns"]) / 1e9
    if not 0 < dt <= MAX_INTERVAL_S:
        return None, "current_highway_maneuver_track_discontinuous"
    distance = math.hypot(frame["ego_world_x_m"] - past["ego_world_x_m"],
                          frame["ego_world_y_m"] - past["ego_world_y_m"])
    speed = distance / dt
    if (frame.get("ego_speed_source") != "backward_gt_ego_xy_difference"
            or not _finite(frame.get("ego_speed_mps"))
            or abs(frame["ego_speed_mps"] - speed) > MAX_SPEED_DISAGREEMENT_MPS):
        return None, "current_highway_maneuver_recorded_speed_disagrees"
    if speed <= STOP_SPEED_MPS or frame.get("ego_stopped") is True:
        return None, "current_highway_maneuver_while_stopped"
    return {"schema": SCHEMA, "valid": True, "frame_index": index,
            "lidar_source_ns": frame["lidar_source_ns"],
            "past_index": past["index"], "past_source_ns": past["lidar_source_ns"],
            "current_xy_m": [frame["ego_world_x_m"], frame["ego_world_y_m"]],
            "past_xy_m": [past["ego_world_x_m"], past["ego_world_y_m"]],
            "backward_interval_seconds": dt, "current_speed_mps": speed,
            "instruction_id": rule["instruction_id"],
            "supervision_basis": "recorded_current_highway_movement",
            "uses_future_ego": False, "uses_neighbor_sentence": False,
            "current_lateral_onset_certified": False,
            "target_lane_safety_certified": False}, None


def align_current_highway_maneuvers(frames, rules):
    """Guard final moving wording; preserve all pre-existing exclusion masks."""
    counts = Counter()
    for rule in rules:
        if not is_highway_maneuver(rule) or not rule.get("instruction"):
            continue
        if (rule.get('evidence',{}).get('current_highway_selected_plan') or {}).get('valid'):
            # The selected lane-change path is valid before execution starts,
            # including while stopped. Its own validator checks captured path
            # geometry; requiring executed motion would destroy that contract.
            counts['captured_selected_plan_uses_own_validator'] += 1
            continue
        if rule.get("action_valid") is False or rule.get("observation_valid") is False:
            counts["upstream_exclusion_preserved"] += 1
            continue
        proof, reason = current_maneuver_motion(frames, rule)
        evidence = rule.setdefault("evidence", {})
        if proof is not None:
            evidence[MARKER] = proof
            counts["current_movement_verified"] += 1
            continue
        evidence[MARKER] = {"schema": SCHEMA, "valid": False, "reason": reason,
                            "frame_index": rule.get("index"), "uses_future_ego": False,
                            "uses_neighbor_sentence": False}
        rule.setdefault("instruction_history", []).append({"pass": MARKER,
            "instruction": rule["instruction"], "instruction_id": rule["instruction_id"],
            "reason": reason})
        rule["instruction"] = None
        rule["action_valid"] = False
        rule["instruction_consistency"] = "excluded_current_highway_maneuver_motion"
        if reason not in rule.setdefault("unknown_reasons", []):
            rule["unknown_reasons"].append(reason)
        counts[reason] += 1
    return dict(sorted(counts.items()))


def validate_current_highway_maneuver(frames, objects, rule):
    """Recompute a final maneuver's motion, receipt and exact catalog text."""
    proof, reason = current_maneuver_motion(frames, rule)
    if proof is None:
        return False, [reason]
    if (rule.get("evidence") or {}).get(MARKER) != proof:
        return False, ["current_highway_maneuver_receipt_disagrees"]
    text, slots, missing = render_instruction(load_catalog(), "highway",
                                             rule["instruction_id"], rule.get("variables") or {})
    expected_label = load_catalog()["highway"][rule["instruction_id"]]["label"]
    effective_label = rule.get("instruction_label") or rule.get("label") or rule.get("label_candidate")
    if (missing or rule.get("instruction") != text
            or rule.get("instruction_variables") != slots or effective_label != expected_label):
        return False, ["current_highway_maneuver_final_rendering_disagrees"]
    return True, []
