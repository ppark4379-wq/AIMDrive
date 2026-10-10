"""Highway primary maneuver and current response, with uncertain cases explicit."""
import math

from .highway_position import body_position
from .highway_target_selection import select_current_highway_target


def lane_dependent_action_reasons(frame, rule):
    """Check final lane semantics after current alignment, before rendering.

    Early raw CHANGE candidates can become a supported non-lane-specific
    current observation. Rejecting those early would discard that evidence.
    This predicate never replaces the final action with another sentence.
    """
    if (frame is None or frame.get("zone_id") != "highway"
            or rule.get("zone_id") != "highway"
            or frame.get("highway_lane_association_ambiguous") is not True):
        return []
    identifier = rule.get("instruction_id")
    if identifier is not None:
        dependent = identifier in {
            "acc_follow", "acc_resume", "change_left", "change_right",
            "change_left_return", "change_right_return"}
    else:
        state = rule.get("instruction_state") or rule.get("behavior_state")
        label = rule.get("label_candidate")
        dependent = ((label == "ACC_FOLLOW" and state in ("follow", "resume"))
                     or (label in ("CHANGE_LEFT", "CHANGE_RIGHT")
                         and state in ("lane_change", "return_to_source_lane")))
    return ["current_highway_lane_association_ambiguous"] if dependent else []


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def highway_rule(frame, objects, config=None):
    result = {"index": frame["index"], "zone_id": "highway",
              "status": "unknown", "instruction_state": None,
              "label_candidate": None, "behavior_state": None,
              "related_actors": [], "variables": {}, "evidence": {},
              "unknown_reasons": [], "instruction_id": None}
    reasons = result["unknown_reasons"]
    if frame.get("valid") is not True:
        result["status"] = "invalid_frame"
        reasons.extend(frame.get("invalid_reasons") or ["frame_invalid"])
        return result
    if frame.get("object_inventory_usable") is not True:
        reasons.append("object_inventory_unusable")
        return result
    if frame.get("highway_nearby_geometry_unverified") is True:
        reasons.append("relevant_vehicle_geometry_unverified")
        return result
    active = frame.get("lane_change_active")
    if active is None:
        reasons.append("lane_change_motion_unverified")
        return result
    direction = frame.get("lane_change_direction")
    if active is True and direction not in ("left", "right"):
        reasons.append("lane_change_direction_unverified")
        return result
    label = ("CHANGE_"+direction.upper() if active else None)
    lead_id, merge_id = frame.get("lead_actor_id"), frame.get("merging_actor_id")
    lead = next((a for a in objects if a.get("actor_id") == lead_id), None)
    merging = next((a for a in objects if a.get("actor_id") == merge_id), None)
    stopped = frame.get("ego_stopped") is True
    braking = frame.get("ego_hard_braking") is True
    decelerating = frame.get("ego_decelerating") is True
    lead_cause = (lead is not None and frame.get("lead_bumper_gap_m") is not None and
                  (frame.get("following_gap_insufficient") is True or
                   frame.get("lead_ttc_s") is not None and frame["lead_ttc_s"] <= 5.0))
    merge_cause = (merging is not None and frame.get("is_merging_into_ego_lane") is True
                   and frame.get("merge_conflict") is True)
    actor, role, reason = None, None, None
    lead_following = (
        lead is not None and _number(frame.get("lead_distance_m")) is not None and
        _number(frame.get("ego_speed_mps")) is not None and
        frame["lead_distance_m"] <= max(20.0, 2.0*frame["ego_speed_mps"]))
    if merge_cause:
        actor, role, reason = merging, "merging_vehicle", "merging_vehicle"
    elif lead_cause or (not active and lead_following and
                        not (braking or stopped or decelerating)):
        actor, role, reason = lead, "lead_vehicle", "lead_vehicle"
    if active:
        if braking and (lead_cause or merge_cause):
            state = "hard_braking"
        elif stopped and (lead_cause or merge_cause):
            state = "stopped"
        elif merge_cause and (decelerating or stopped):
            state = "yielding"
        elif decelerating:
            state = "decelerating"
        elif frame.get("lane_change_aborting") is True:
            state = "return_to_source_lane"
        else:
            state = "lane_change"
    elif lead_cause or merge_cause or (lead_following and not (braking or stopped or decelerating)):
        label = "ACC_FOLLOW"
        if braking and (lead_cause or merge_cause):
            state = "hard_braking"
        elif stopped and (lead_cause or merge_cause):
            state = "stopped"
        elif merge_cause and decelerating:
            state = "yielding"
        elif decelerating and (lead_cause or merge_cause):
            state = "decelerating"
        elif frame.get("highway_acc_prev_responding") is True:
            # The previous contiguous highway frame was itself ACC_FOLLOW in
            # a responding sub-state (hard_braking/stopped/yielding/
            # decelerating) and this frame already resolved to plain,
            # unconstrained following -- say "resume", once, instead of the
            # generic "follow" an already-steady-state follow repeats.
            state = "resume"
        else:
            state = "follow"
    else:
        label = "KEEP_LANE"
        if braking or stopped or decelerating:
            reasons.append("ego_response_cause_unverified")
            return result
        actor, role, selection_reasons, selection_receipt = select_current_highway_target(frame, objects)
        result["evidence"]["current_highway_target_selection"] = selection_receipt
        if selection_reasons:
            reasons.extend(selection_reasons)
            return result
        if actor is not None:
            reason = role
            state = "approach_keep"
        elif frame.get("lane_change_completed") is True:
            # The single frame a lane change (or its abort) actually finishes
            # on, enrich_highway's own state machine -- say the maneuver
            # just ended instead of the generic steady-state "normal".
            state = "keep_completed"
        else:
            state = "normal"
    result["label_candidate"] = label
    result["behavior_state"] = state
    result["instruction_state"] = state
    result["status"] = "matched"
    result["variables"].update(
        source_lane_id=frame.get("source_lane_id"),
        target_lane_id=frame.get("target_lane_id"),
        lane_change_direction=direction,
        target_actor_id=actor.get("actor_id") if actor else None,
        target_kind="vehicle" if actor else None,
        target_position=body_position(actor) if actor else None,
        target_distance_m=(frame.get("lead_distance_m") if role == "lead_vehicle" else
                           frame.get("merging_actor_distance_m") if role == "merging_vehicle" else
                           actor.get("distance_to_ego_m") if actor else None),
        target_position_text=body_position(actor) if actor else None,
        response_reason=reason)
    if actor:
        result["related_actors"].append({"actor_id": actor.get("actor_id"), "role": role})
    result["evidence"].update(
        lane_change_active=active,
        lane_change_aborting=frame.get("lane_change_aborting"),
        lane_change_progress=frame.get("lane_change_progress"),
        ego_current_link_id=frame.get("ego_current_link_id"),
        ego_current_lane_id=frame.get("ego_current_lane_id"),
        ego_decelerating=frame.get("ego_decelerating"),
        ego_hard_braking=frame.get("ego_hard_braking"),
        ego_stopped=frame.get("ego_stopped"),
        lead_vehicle_present=lead is not None,
        lead_bumper_gap_m=frame.get("lead_bumper_gap_m"),
        merge_conflict=frame.get("merge_conflict"),
        target_lane_initially_safe=frame.get("target_lane_initially_safe"),
        target_lane_collision_safe=frame.get("target_lane_collision_safe"))
    return result
