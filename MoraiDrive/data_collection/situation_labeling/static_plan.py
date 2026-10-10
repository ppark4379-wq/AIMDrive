"""Compose current longitudinal commands with the selected static avoidance path.

A selected avoidance path can coexist with a red-light stop. Conversely a
positive departure target is not cancelled merely because actual speed is still
below 0.2 m/s. Expert publications without timestamps are receive-clock bound;
that limitation is retained in evidence rather than promoted to sensor synchrony.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

POLICY = "current_static_selected_plan_v1"
ORIGINAL_STATES = frozenset(("avoid", "avoid_continue"))
COMPOSED_STATES = frozenset(("static_signal_stop_plan", "static_stop_plan",
                            "static_prepare_resume_plan", "static_resume_plan"))
MAX_COMMAND_AGE_NS = 50_000_000


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _current_receipt(receipt, sensor_ns, lidar_receive_ns):
    return (isinstance(receipt, dict)
            and isinstance(receipt.get("preceding_received_clock_ns"), int)
            and 0 <= sensor_ns - receipt["preceding_received_clock_ns"] <= MAX_COMMAND_AGE_NS
            and isinstance(receipt.get("receive_monotonic_ns"), int)
            and 0 <= lidar_receive_ns - receipt["receive_monotonic_ns"] <= 100_000_000
            and isinstance(receipt.get("record_id"), int) and bool(receipt.get("payload_sha256")))


def _route_red(route, index, sensor_ns):
    bracket = route.get("light_source_bracket_ns") or []
    return (route.get("index") == index and route.get("sensor_source_ns") == sensor_ns
            and route.get("valid") is True and route.get("color") == 1
            and route.get("route_rpc_current_link_agreement") is True
            and bool(route.get("light_id")) and bool(route.get("light_source_record_ids"))
            and len(bracket) == 2 and all(isinstance(t, int) for t in bracket)
            and bracket[0] <= sensor_ns <= bracket[1]
            and _finite(route.get("distance_to_stopline_m"))
            and 0 <= route["distance_to_stopline_m"] <= 15)


def compose_current_static_plan(frame):
    """Return a supported composite state, or (None, explicit reason).

    This function is independently rerun by the final consistency validator.
    It does not consult instruction text, future poses or adjacent labels.
    """
    index, sensor_ns = frame.get("index"), frame.get("lidar_source_ns")
    plan = frame.get("expert_static_selected_plan") or {}
    pub = frame.get("expert_publications") or {}
    if (not isinstance(index, int) or not isinstance(sensor_ns, int)
            or frame.get("valid") is not True or frame.get("dataset_build_candidate") is not True
            or frame.get("object_inventory_usable") is not True):
        return None, "current_frame_or_inventory_unusable"
    if (plan.get("anchor_interval_all_plans_agree") is not True
            or plan.get("incoming_plan_agrees") is not True
            or plan.get("path_validated") is not True or plan.get("phase") != "follow"
            or plan.get("direction") not in ("left", "right")
            or not plan.get("selected_routes") or not plan.get("source_sha256")):
        return None, "selected_path_unverified"
    lidar_receive_ns = pub.get("lidar_receive_monotonic_ns")
    if (pub.get("status") != "RECEIVE_BOUND_CURRENT_CONTEXT"
            or pub.get("sensor_ns") != sensor_ns or not isinstance(lidar_receive_ns, int)):
        return None, "current_expert_publications_unbound"
    control, target = pub.get("control"), pub.get("target_velocity")
    if not all(_current_receipt(receipt, sensor_ns, lidar_receive_ns) for receipt in (control, target)):
        return None, "current_expert_command_stale_or_missing"
    if control.get("longlCmdType", control.get("command_type")) != 1:
        return None, "throttle_brake_command_mode_unverified"
    speed, target_speed = frame.get("ego_speed_mps"), target.get("value_mps")
    brake, accel = control.get("brake"), control.get("accel")
    if (not all(_finite(value) and value >= 0 for value in (speed, target_speed, brake, accel))
            or frame.get("ego_speed_source") != "backward_gt_ego_xy_difference"):
        return None, "current_motion_or_commands_invalid"
    route = frame.get("static_plan_route_signal") or {}
    # "At the red traffic light" describes the current route-bound light and
    # stop location. It does not assert the cause. Causal attribution below is
    # separately stricter: an unchanged zero signal limiter in the log interval.
    red = _route_red(route, index, sensor_ns)
    if target_speed <= .01 and brake >= .5 and accel <= .001 and speed <= .2:
        state = "static_signal_stop_plan" if red else "static_stop_plan"
    elif target_speed > .01 and speed <= .2:
        prior = target.get("prior_zero_target") or {}
        prior_stamp, prior_receive = prior.get("preceding_received_clock_ns"), prior.get("receive_monotonic_ns")
        if (not isinstance(prior_stamp, int) or not 0 < sensor_ns-prior_stamp <= 1_000_000_000
                or prior.get("value_mps") != 0.0 or not prior.get("payload_sha256")
                or not isinstance(prior_receive, int) or prior_receive >= target["receive_monotonic_ns"]):
            return None, "departure_requires_recent_zero_target"
        state = "static_prepare_resume_plan" if brake > .01 else "static_resume_plan"
    else:
        return None, "no_stopped_or_departure_composition_needed"
    return {"state": state, "direction": plan["direction"],
            "current_red_signal_observed": red,
            "red_signal_cause_verified": (red and state == "static_signal_stop_plan"
                                           and plan.get("signal_zero_limit_all") is True),
            "supervision_basis": "selected_static_path_and_current_expert_commands",
            "uses_future_ego": False, "physical_sensor_synchrony_certified": False,
            "trajectory_safety_verified": False, "observation_mask_promotion": False,
            "timing_basis": "preceding received sim-clock plus causal receive monotonic; no Expert source timestamp",
            "plan": copy.deepcopy(plan), "control": copy.deepcopy(control),
            "target_velocity": copy.deepcopy(target), "route_signal": copy.deepcopy(route)}, None


def _fresh_target(frame, objects, rule):
    ref = (rule.get("evidence") or {}).get("target_actor_ref") or {}
    if ref.get("actor_id") is None or not ref.get("instance_key"):
        return False
    matches = [actor for actor in objects
               if actor.get("index") == frame["index"]
               and str(actor.get("actor_id")) == str(ref["actor_id"])
               and actor.get("instance_key") == ref["instance_key"]]
    if len(matches) != 1:
        return False
    actor = matches[0]
    return (actor.get("kind") == "obstacle" and actor.get("rule_geometry_usable") is True
            and actor.get("model_identity_verified") is True
            and actor.get("semantic_class_valid") is not False
            and all(actor.get(key) == frame["lidar_source_ns"]
                    for key in ("lidar_source_ns", "sensor_time_ns") if key in actor))


def attach_static_route_signals(frames, episode):
    """Load only the route-bound per-frame light audit, preserving its source pin."""
    path = Path(episode) / "route_signal_audit_v1" / "frames.json"
    if not path.is_file():
        return {"bound_frames": 0, "reason": "route_signal_audit_missing"}
    data = path.read_bytes()
    rows = json.loads(data)
    if not isinstance(rows, list) or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Invalid or duplicate route signal rows")
    indexed, bound = {row["index"]: row for row in rows}, 0
    for frame in frames:
        row = indexed.get(frame["index"])
        if (frame.get("zone_id") == "static_obstacle" and row is not None
                and row.get("sensor_source_ns") == frame.get("lidar_source_ns")):
            frame["static_plan_route_signal"] = copy.deepcopy(row)
            bound += 1
    return {"bound_frames": bound, "source_path": str(path),
            "source_sha256": hashlib.sha256(data).hexdigest()}


def align_current_static_plan(frames, objects, rules):
    """Compose only already supported avoidance rows; never upgrade a mask."""
    indexed = {frame["index"]: frame for frame in frames}
    counts = Counter()
    for rule in rules:
        if rule.get("zone_id") != "static_obstacle" or rule.get("instruction_state") not in ORIGINAL_STATES:
            continue
        frame = indexed.get(rule["index"])
        if (frame is None or rule.get("status") != "matched" or rule.get("unknown_reasons")
                or rule.get("label") not in (None, "follow")
                or any(rule.get(key) is False for key in ("action_valid", "observation_valid"))):
            counts["upstream_evidence_preserved"] += 1
            continue
        if not _fresh_target(frame, objects, rule):
            counts["fresh_target_unverified"] += 1
            continue
        result, reason = compose_current_static_plan(frame)
        if result is None:
            counts[reason] += 1
            continue
        evidence = rule.setdefault("evidence", {})
        evidence["current_static_selected_plan"] = dict(result, schema=POLICY,
            previous_instruction_state=rule["instruction_state"])
        rule.update(instruction_state=result["state"], instruction_id=None,
                    instruction=None, instruction_variables=None)
        rule.setdefault("variables", {})["avoid_direction"] = result["direction"]
        counts["to_" + result["state"]] += 1
    return dict(sorted(counts.items()))


def validate_current_static_plan(frames, objects, rule):
    """Recompute every supported field; an audit marker alone is not proof."""
    reasons = []
    matches = [frame for frame in frames if frame.get("index") == rule.get("index")]
    marker = (rule.get("evidence") or {}).get("current_static_selected_plan") or {}
    if (rule.get("zone_id") != "static_obstacle" or rule.get("instruction_state") not in COMPOSED_STATES
            or marker.get("schema") != POLICY or len(matches) != 1):
        return False, ["not_a_current_static_selected_plan"]
    frame = matches[0]
    result, reason = compose_current_static_plan(frame)
    if result is None:
        reasons.append(reason)
    elif (result["state"] != rule["instruction_state"]
          or result["direction"] != (rule.get("instruction_variables") or {}).get("direction")):
        reasons.append("current_static_composed_state_or_direction_disagrees")
    if rule.get("status") != "matched" or rule.get("unknown_reasons"):
        reasons.append("current_rule_unresolved")
    if any(rule.get(key) is False for key in ("action_valid", "observation_valid")):
        reasons.append("upstream_instruction_mask")
    if not _fresh_target(frame, objects, rule):
        reasons.append("fresh_target_unverified")
    return not reasons, sorted(set(reasons))
