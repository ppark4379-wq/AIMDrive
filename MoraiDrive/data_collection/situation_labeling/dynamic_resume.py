"""Current-response dynamic labels, supported by causal recorded Ego motion.

This is a GT supervision pass, not a physical safety certificate. It never
reads future Ego positions, copies a prior target, or overrides an explicit
upstream eligibility mask. Full current inventory is checked across API kinds.
"""
from collections import Counter, defaultdict
import math

from .renderer import actor_semantic_class

YIELD_STATES = frozenset(("yield_start", "yielding", "waiting"))
CLEAR_STATES = frozenset(("normal", "approach", "resume", "wait_clear"))
FRONT_REASON = "resume_front_safety_unverified"
MAX_INTERVAL_S = 0.25
PAST_MIN_S, PAST_MAX_S = 0.3, 0.6
STOP_SPEED_MPS, RESUME_SPEED_MPS = 0.2, 3.0
NEAR_M, HALF_CORRIDOR_M = 35.0, 1.5


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _frame_usable(frame):
    return bool(frame and frame.get("valid") is True
                and frame.get("dataset_build_candidate") is True
                and frame.get("object_inventory_usable") is True
                and all(_number(frame.get(key)) is not None for key in
                        ("index", "lidar_source_ns", "ego_world_x_m",
                         "ego_world_y_m", "ego_path_s_m", "ego_path_d_m")))


def _connected(previous, current):
    if not _frame_usable(previous) or not _frame_usable(current):
        return False
    dt = (current["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9
    return (current["index"] == previous["index"] + 1
            and 0 < dt <= MAX_INTERVAL_S)


def _masked(rule):
    return any(scope.get(key) is False
               for scope in (rule, rule.get("variables") or {}, rule.get("evidence") or {})
               for key in ("valid", "frame_valid", "dataset_build_candidate",
                           "object_inventory_usable", "observation_valid",
                           "action_valid", "target_semantic_valid", "semantic_valid"))


def _actor_keys(objects):
    return {actor.get("actor_id"): actor.get("instance_key") for actor in objects
            if actor.get("actor_id") is not None}


def _corridor(frame, actors):
    """Conservatively audit every located nearby actor, never just the target."""
    checks, reasons = [], []
    for actor in actors:
        check = {key: actor.get(key) for key in
                 ("actor_id", "instance_key", "kind", "model_name", "delta_s_m",
                  "path_d_m", "actor_width_m", "actor_length_m")}
        x, y = _number(actor.get("world_x_m")), _number(actor.get("world_y_m"))
        if x is None or y is None:
            reasons.append("unlocated_current_inventory_actor")
            check["verified"] = False
            checks.append(check)
            continue
        distance = math.hypot(x - frame["ego_world_x_m"], y - frame["ego_world_y_m"])
        check["distance_m"] = distance
        if distance > NEAR_M:
            check["outside_nearby_radius"] = True
            checks.append(check)
            continue
        semantic, semantic_reason = actor_semantic_class(
            actor.get("kind"), actor.get("model_name"),
            identity_verified=actor.get("model_identity_verified") is True)
        semantic_valid = (semantic_reason is None and actor.get("instance_key") is not None
                          and actor.get("semantic_class_valid") is not False
                          and actor.get("semantic_class") in (None, semantic))
        check.update(semantic_class=semantic, semantic_verified=semantic_valid)
        if not semantic_valid:
            reasons.append("current_inventory_semantics_unverified")
        stamp_ok = actor.get("lidar_source_ns") == frame["lidar_source_ns"]
        delta, lateral, width, length = (_number(actor.get(key)) for key in
                                        ("delta_s_m", "path_d_m", "actor_width_m",
                                         "actor_length_m"))
        geometry_valid = (stamp_ok and actor.get("rule_geometry_usable") is True
                          and None not in (delta, lateral, width, length)
                          and width > 0 and length > 0
                          and not (set(actor.get("invalid_reasons") or ()) - {
                              "outside_training_bev_or_unknown", "object_geometry_not_candidate"}))
        actor_s, actor_ego_s = _number(actor.get("path_s_m")), _number(actor.get("ego_path_s_m"))
        if actor_ego_s is not None and abs(actor_ego_s - frame["ego_path_s_m"]) > 1e-3:
            geometry_valid = False
        if actor_s is not None and delta is not None and abs(
                actor_s - frame["ego_path_s_m"] - delta) > 1e-3:
            geometry_valid = False
        check["geometry_verified"] = geometry_valid
        if not geometry_valid:
            reasons.append("current_inventory_geometry_unverified")
        check["verified"] = semantic_valid and geometry_valid
        if geometry_valid:
            radius = math.hypot(width, length) / 2
            front_overlap = delta + radius >= -2.0 and delta - radius <= NEAR_M
            blocks = front_overlap and abs(lateral) - radius <= HALF_CORRIDOR_M
            vd = _number(actor.get("v_d_mps"))
            approaching = (actor.get("kind") == "pedestrian" and front_overlap
                           and vd is not None and lateral * vd < -0.1)
            check.update(corridor_blocker=blocks, crossing_toward_corridor=approaching)
            if blocks or approaching:
                reasons.append("current_route_corridor_not_clear")
        checks.append(check)
    return {"all_current_actors_evaluated": True, "nearby_radius_m": NEAR_M,
            "corridor_half_width_m": HALF_CORRIDOR_M, "actors": checks,
            "clear": not reasons, "reasons": sorted(set(reasons))}


def _past_motion(history):
    current = history[-1]
    # Use only a short, contiguous past interval; a future departure cannot
    # turn a currently stationary frame into a Resume instruction.
    prior = next((frame for frame in reversed(history[:-1])
                  if PAST_MIN_S <= (current["lidar_source_ns"] - frame["lidar_source_ns"]) / 1e9
                  <= PAST_MAX_S), None)
    if prior is None:
        return None
    dt = (current["lidar_source_ns"] - prior["lidar_source_ns"]) / 1e9
    distance = math.hypot(current["ego_world_x_m"] - prior["ego_world_x_m"],
                          current["ego_world_y_m"] - prior["ego_world_y_m"])
    speed = distance / dt
    progress = current["ego_path_s_m"] - prior["ego_path_s_m"]
    previous = history[-2] if len(history) >= 2 else None
    if not _connected(previous, current):
        return None
    instant_dt = (current["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9
    instant = math.hypot(current["ego_world_x_m"] - previous["ego_world_x_m"],
                         current["ego_world_y_m"] - previous["ego_world_y_m"]) / instant_dt
    recorded = _number(current.get("ego_speed_mps"))
    if (speed > 50 or instant > 50 or recorded is None
            or not math.isclose(recorded, instant, rel_tol=1e-4, abs_tol=1e-4)):
        return None
    return {"source": "backward_gt_ego_xy_difference", "start_index": prior["index"],
            "end_index": current["index"], "interval_s": dt,
            "start_source_ns": prior["lidar_source_ns"],
            "end_source_ns": current["lidar_source_ns"],
            "speed_mps": speed, "current_speed_mps": instant,
            "route_progress_m": progress, "uses_future_ego": False}


def _state_from_motion(motion):
    speed, instant = motion["speed_mps"], motion["current_speed_mps"]
    if speed <= STOP_SPEED_MPS and instant <= STOP_SPEED_MPS:
        return "wait_clear"
    if (speed > STOP_SPEED_MPS and instant > STOP_SPEED_MPS
            and motion["route_progress_m"] > 0.02):
        return "resume" if max(speed, instant) < RESUME_SPEED_MPS else "normal"
    return None


def validate_current_dynamic_response(frames, objects, rule):
    """Independently recheck the current state and original GT, not a marker.

    An inventory array is mandatory, even when empty. No objects argument is
    not evidence of an empty inventory. No future Ego sample is consulted.
    """
    if objects is None:
        return False, ["current_dynamic_inventory_not_supplied"]
    indexed = {frame["index"]: frame for frame in frames}
    if len(indexed) != len(frames):
        return False, ["duplicate_current_dynamic_frame"]
    frame = indexed.get(rule.get("index"))
    if (not _frame_usable(frame) or frame.get("zone_id") != "dynamic_obstacle"
            or rule.get("zone_id") != "dynamic_obstacle" or rule.get("status") != "matched"
            or _masked(rule) or rule.get("unknown_reasons")):
        return False, ["current_dynamic_frame_or_rule_unverified"]
    marker = (rule.get("evidence") or {}).get("current_dynamic_response")
    if (not isinstance(marker, dict) or marker.get("supervision_basis") != "recorded_ego_motion"
            or marker.get("physical_front_safety_certified") is not False
            or not isinstance(marker.get("motion"), dict)):
        return False, ["current_dynamic_response_marker_invalid"]
    # Rebuild a contiguous past window from original frames. Gaps, invalid
    # frames or changed current timestamps cannot be skipped to find evidence.
    history = [frame]
    while len(history) <= 20:
        previous = indexed.get(history[0]["index"] - 1)
        if (not _connected(previous, history[0]) or
                (frame["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9 > PAST_MAX_S):
            break
        history.insert(0, previous)
    motion = _past_motion(history)
    if motion is None:
        return False, ["current_dynamic_past_motion_unverified"]
    reasons = []
    recorded = marker["motion"]
    for key in ("source", "start_index", "end_index", "start_source_ns", "end_source_ns", "uses_future_ego"):
        if recorded.get(key) != motion[key]:
            reasons.append("current_dynamic_motion_evidence_disagrees")
            break
    for key in ("interval_s", "speed_mps", "current_speed_mps", "route_progress_m"):
        value = _number(recorded.get(key))
        if value is None or not math.isclose(value, motion[key], rel_tol=1e-5, abs_tol=1e-5):
            reasons.append("current_dynamic_motion_evidence_disagrees")
            break
    current_actors = [actor for actor in objects if actor.get("index") == frame["index"]]
    corridor = _corridor(frame, current_actors)
    reasons.extend(corridor["reasons"])
    expected = _state_from_motion(motion)
    if expected is None or rule.get("instruction_state") != expected or marker.get("state_after") != expected:
        reasons.append("current_dynamic_state_disagrees_with_gt_motion")
    return not reasons, sorted(set(reasons))


def align_current_dynamic_response(frames, objects, rules):
    """Mutate supported post-yield states; return auditable operation counters."""
    indexed = {frame["index"]: frame for frame in frames}
    if len(indexed) != len(frames) or len({row["index"] for row in rules}) != len(rules):
        raise ValueError("Duplicate dynamic response frame index")
    grouped = defaultdict(list)
    for actor in objects:
        if actor["index"] not in indexed:
            raise ValueError("Current inventory contains unknown frame index")
        grouped[actor["index"]].append(actor)
    counts, history = Counter(), []
    active, previous, previous_keys = False, None, {}
    for rule in sorted(rules, key=lambda row: row["index"]):
        frame = indexed.get(rule["index"])
        actors = grouped[rule["index"]]
        keys = _actor_keys(actors)
        respawn = any(key in previous_keys and previous_keys[key] != value
                      for key, value in keys.items())
        if not _connected(previous, frame) or respawn:
            active, history = False, []
        previous, previous_keys = frame, keys
        if (not _frame_usable(frame) or rule.get("zone_id") != "dynamic_obstacle"
                or rule.get("status") != "matched" or _masked(rule)
                or set(rule.get("unknown_reasons") or ()) - {FRONT_REASON}):
            active, history = False, []
            counts["ineligible_preserved"] += 1
            continue
        history.append(frame)
        history = [entry for entry in history if
                   (frame["lidar_source_ns"] - entry["lidar_source_ns"]) / 1e9 <= PAST_MAX_S]
        state = rule.get("instruction_state")
        if state in YIELD_STATES:
            active = True
            continue
        if not active or state not in CLEAR_STATES:
            active = False
            continue
        corridor = _corridor(frame, actors)
        motion = _past_motion(history)
        evidence = rule.setdefault("evidence", {})
        evidence["current_dynamic_response"] = {
            "corridor": corridor, "motion": motion,
            "physical_front_safety_certified": False}
        if not corridor["clear"] or motion is None:
            # Do not leave an unsupported normal/resume action trainable.
            reasons = corridor["reasons"] or ["current_ego_motion_unverified"]
            rule["unknown_reasons"] = sorted(set(rule.get("unknown_reasons") or ()) | set(reasons))
            rule["action_valid"] = False
            counts["current_response_excluded"] += 1
            # A fully verified blocker is current evidence that the same
            # yielding episode has not cleared yet. It excludes this frame's
            # Resume action, but must not erase the episode before the next
            # clear frame. Unknown geometry, semantics or motion still reset.
            active = (motion is not None and corridor["reasons"] == [
                "current_route_corridor_not_clear"])
            if active:
                counts["verified_blocker_context_retained"] += 1
            continue
        new = _state_from_motion(motion)
        if new is None:
            rule["unknown_reasons"] = sorted(set(rule.get("unknown_reasons") or ()) |
                                             {"current_ego_motion_unverified"})
            rule["action_valid"] = False
            counts["current_response_excluded"] += 1
            # The causal mean intentionally lags an instantaneous departure
            # or stop. Keep this transition frame excluded, while preserving
            # the already verified episode for the next continuous frame.
            # Both speeds moving but no forward route progress is a different
            # unknown behavior and does not receive this exception.
            active = ((motion["speed_mps"] <= STOP_SPEED_MPS)
                      != (motion["current_speed_mps"] <= STOP_SPEED_MPS))
            if active:
                counts["motion_phase_transition_context_retained"] += 1
            continue
        evidence["current_dynamic_response"].update(
            supervision_basis="recorded_ego_motion", state_before=state, state_after=new)
        if FRONT_REASON in (rule.get("unknown_reasons") or ()):
            rule["unknown_reasons"] = [reason for reason in rule["unknown_reasons"] if reason != FRONT_REASON]
            counts["front_reason_revalidated"] += 1
        if new != state:
            rule["instruction_state"] = new
            counts["to_" + new] += 1
        if new == "normal":
            active = False
    return dict(sorted(counts.items()))
