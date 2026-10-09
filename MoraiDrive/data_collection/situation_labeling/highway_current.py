"""Validate observed highway keep/follow without consulting future ego poses.

This adds a current-phase consistency receipt. It does not infer a stopping
cause, repair inventory, rewrite maneuver states, or certify trajectory safety.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from .highway_rules import highway_rule
from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_instruction
from .highway_position import validate_current_highway_position

SCHEMA = "current_highway_response_v1"
DECISION = "verified_current_keep_follow"
SUPPORTED = {("KEEP_LANE", "normal"), ("KEEP_LANE", "approach_keep"),
             ("ACC_FOLLOW", "follow")}


@lru_cache(maxsize=1)
def _render_config():
    catalog = load_catalog()
    return catalog, load_label_instruction_map(catalog=catalog)


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _frame_usable(frame):
    return (frame.get("zone_id") == "highway" and frame.get("valid") is True
            and frame.get("dataset_build_candidate") is True
            and frame.get("object_inventory_usable") is True
            and all(_finite(frame.get(k)) for k in
                    ("index", "lidar_source_ns", "ego_path_s_m", "ego_path_d_m",
                     "ego_world_x_m", "ego_world_y_m", "ego_speed_mps")))


def _connected(a, b):
    if not _frame_usable(a) or not _frame_usable(b):
        return False
    dt = (b["lidar_source_ns"]-a["lidar_source_ns"]) / 1e9
    if b["index"] != a["index"]+1 or not 0 < dt <= .25:
        return False
    distance = math.hypot(b["ego_world_x_m"]-a["ego_world_x_m"],
                          b["ego_world_y_m"]-a["ego_world_y_m"])
    ds = b["ego_path_s_m"]-a["ego_path_s_m"]
    return (distance <= 70*dt+.5 and -.5 <= ds <= 70*dt+.5
            and abs(b["ego_path_d_m"]-a["ego_path_d_m"]) <= distance+.1)


def _actor_usable(actor, frame):
    return (actor.get("index") == frame["index"] and actor.get("kind") == "npc"
            and actor.get("instance_key") and actor.get("model_identity_verified") is True
            and actor.get("rule_geometry_usable") is True
            and actor.get("actor_link_id") is not None
            and actor.get("semantic_class_valid") is not False
            and all(_finite(actor.get(k)) for k in
                    ("world_x_m", "world_y_m", "actor_width_m", "actor_length_m", "distance_to_ego_m"))
            and actor["actor_width_m"] > 0 and actor["actor_length_m"] > 0
            and all(_finite(actor[k]) and actor[k] == frame["lidar_source_ns"]
                    for k in ("lidar_source_ns", "sensor_time_ns") if k in actor))


def _target_id(rule):
    actor = rule.get("variables", {}).get("target_actor_id")
    return str(actor) if actor is not None else None


def _evaluate(frames, objects, rule, *, require_marker):
    reasons, receipt = [], {}
    marker = rule.get("evidence", {}).get("current_highway_response", {})
    if require_marker and (marker.get("schema") != SCHEMA or marker.get("decision") != DECISION
                           or marker.get("supervision_basis") != "recorded_current_motion"):
        reasons.append("current_highway_response_marker_unverified")
    classification = rule.get("label_candidate"), rule.get("instruction_state")
    if (rule.get("zone_id") != "highway" or classification not in SUPPORTED
            or rule.get("status") != "matched" or rule.get("unknown_reasons")):
        return False, sorted(set(reasons + ["unsupported_or_unresolved_current_highway_state"])), receipt
    if any(rule.get(k) is False for k in ("action_valid", "observation_valid")):
        reasons.append("current_highway_preexisting_exclusion")
    current_rows = [f for f in frames if f.get("index") == rule["index"]]
    if len(current_rows) != 1 or not _frame_usable(current_rows[0]):
        return False, sorted(set(reasons + ["current_highway_frame_or_inventory_unusable"])), receipt
    frame = current_rows[0]
    # The raw current rule and the final row must agree with actual body
    # geometry, rather than both repeating a hardcoded "front" default.
    if require_marker:
        position_valid, position_reasons = validate_current_highway_position(frames, objects, rule)
        if not position_valid:
            reasons.extend(position_reasons)
    if (frame.get("lane_change_active") is not False
            or frame.get("highway_nearby_geometry_unverified") is not False):
        reasons.append("current_highway_lane_or_geometry_unverified")
    if (frame.get("ego_stopped") is not False or frame.get("ego_decelerating") is not False
            or frame.get("ego_hard_braking") is not False or frame["ego_speed_mps"] < .5):
        reasons.append("current_highway_keep_follow_motion_not_observed")
    if (not _finite(frame.get("ego_lane_center_offset_m"))
            or abs(frame["ego_lane_center_offset_m"]) > .7
            or not _finite(frame.get("ego_heading_error_deg"))
            or abs(frame["ego_heading_error_deg"]) > 12):
        reasons.append("current_highway_lane_alignment_unverified")
    history = sorted([f for f in frames if _finite(f.get("index"))
                      and _finite(f.get("lidar_source_ns")) and f["index"] <= frame["index"]
                      and 0 <= frame["lidar_source_ns"]-f["lidar_source_ns"] <= 650000000],
                     key=lambda f:f["index"])
    if (len(history) < 5 or history[-1] is not frame
            or not all(_frame_usable(f) for f in history)
            or not all(_connected(a,b) for a,b in zip(history,history[1:]))):
        return False, sorted(set(reasons + ["current_highway_past_pose_not_continuous"])), receipt
    first = min(history, key=lambda f:abs((frame["lidar_source_ns"]-f["lidar_source_ns"])/1e9-.5))
    history = history[history.index(first):]
    duration = (frame["lidar_source_ns"]-first["lidar_source_ns"]) / 1e9
    ds, dd = frame["ego_path_s_m"]-first["ego_path_s_m"], frame["ego_path_d_m"]-first["ego_path_d_m"]
    recent_ds = frame["ego_path_s_m"]-history[-3]["ego_path_s_m"] if len(history) >= 3 else 0
    receipt.update(past_start_index=first["index"], past_window_seconds=duration,
                   observed_progress_m=ds, observed_lateral_change_m=dd, recent_progress_m=recent_ds)
    if not .4 <= duration <= .65 or ds < .25 or recent_ds < .1 or abs(dd) > .25:
        reasons.append("current_highway_past_keep_follow_motion_unverified")
    indexed = defaultdict(list)
    for actor in objects:
        indexed[actor.get("index")].append(actor)
    current_objects = indexed[frame["index"]]
    # Independent reproduction of the original current traffic/cause rule.
    expected = highway_rule(frame, current_objects)
    if ((expected.get("label_candidate"), expected.get("instruction_state")) != classification
            or expected.get("status") != "matched" or expected.get("unknown_reasons")
            or _target_id(expected) != _target_id(rule)):
        reasons.append("current_highway_rule_recomputation_disagrees")
    if require_marker:
        catalog, mapping = _render_config()
        identifier = mapping.get(("highway",expected.get("label_candidate"),expected.get("instruction_state")))
        if identifier is None:
            reasons.append("current_highway_expected_rendering_unavailable")
        else:
            text, slots, missing = render_instruction(catalog,"highway",identifier,expected.get("variables",{}))
            if (missing or rule.get("instruction_id") != identifier or rule.get("instruction") != text
                    or rule.get("instruction_variables") != slots):
                reasons.append("current_highway_rendered_state_or_slots_disagree")
    for actor in current_objects:
        if actor.get("kind") != "npc":
            continue
        distance = actor.get("distance_to_ego_m")
        if not _finite(distance) or (distance <= 85 and not _actor_usable(actor,frame)):
            reasons.append("current_highway_nearby_actor_unverified")
    target = _target_id(rule)
    if target is not None:
        current = [a for a in current_objects if str(a.get("actor_id")) == target]
        if len(current) != 1 or not _actor_usable(current[0],frame):
            reasons.append("current_highway_target_instance_unverified")
        else:
            actor = current[0]
            expected_variables = expected.get("variables", {})
            variables = rule.get("variables", {})
            for key in ("target_kind", "target_position", "response_reason"):
                if variables.get(key) != expected_variables.get(key):
                    reasons.append("current_highway_target_semantics_disagree")
            ref = {"actor_id": actor["actor_id"], "instance_key": actor["instance_key"]}
            receipt["current_actor_ref"] = ref
            old_ref = rule.get("evidence", {}).get("target_actor_ref")
            if old_ref is not None and (str(old_ref.get("actor_id")),old_ref.get("instance_key")) != (target,actor["instance_key"]):
                reasons.append("current_highway_target_reference_disagrees")
            previous_actor, previous_frame = None, None
            for old_frame in history:
                matches = [a for a in indexed[old_frame["index"]]
                           if str(a.get("actor_id")) == target and a.get("instance_key") == actor["instance_key"]]
                if len(matches) != 1 or not _actor_usable(matches[0],old_frame):
                    reasons.append("current_highway_target_past_inventory_unverified")
                    continue
                if previous_actor is not None:
                    dt = (old_frame["lidar_source_ns"]-previous_frame["lidar_source_ns"])/1e9
                    travel = math.hypot(matches[0]["world_x_m"]-previous_actor["world_x_m"],
                                        matches[0]["world_y_m"]-previous_actor["world_y_m"])
                    if not 0 < dt <= .25 or travel > 70*dt+.5:
                        reasons.append("current_highway_target_pose_discontinuity")
                previous_actor, previous_frame = matches[0], old_frame
    elif classification != ("KEEP_LANE","normal"):
        reasons.append("current_highway_target_missing")
    return not reasons, sorted(set(reasons)), receipt


def align_current_highway_response(frames, objects, rules):
    """Attach independently supported receipts; never change state or masks."""
    counts = Counter()
    for rule in rules:
        if rule.get("zone_id") != "highway":
            continue
        if (rule.get('evidence',{}).get('current_highway_selected_plan') or {}).get('valid'):
            counts['captured_selected_plan_uses_own_validator'] += 1
            continue
        valid, reasons, receipt = _evaluate(frames,objects,rule,require_marker=False)
        if not valid:
            counts["not_certified"] += 1
            continue
        receipt.update(schema=SCHEMA, decision=DECISION, uses_future_ego=False,
                       supervision_basis="recorded_current_motion")
        rule.setdefault("evidence", {})["current_highway_response"] = receipt
        if receipt.get("current_actor_ref") is not None:
            rule["evidence"]["target_actor_ref"] = dict(receipt["current_actor_ref"])
        counts["verified_current_keep_follow"] += 1
    return dict(sorted(counts.items()))


def validate_current_highway_response(frames, objects, rule):
    """Recompute current motion, actor identity and causal rule after rendering."""
    if objects is None:
        return False, ["current_highway_inventory_not_supplied"]
    valid, reasons, _ = _evaluate(frames,objects,rule,require_marker=True)
    return valid, reasons
