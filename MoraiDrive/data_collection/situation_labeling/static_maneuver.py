"""Current static maneuver phase from fresh inventory and recorded past motion.

This pass labels an observed maneuver, not a safe candidate trajectory. It never
reads future ego poses and never carries an actor across an evidence gap.
"""
from collections import Counter, defaultdict
import copy
import math

POLICY = "current_static_maneuver_v1"
MAX_DT_S = 0.25
PAST_WINDOW_S = 0.5
ONSET_SHIFT_M = 0.15
RETURN_SHIFT_M = 0.08
MIN_FORWARD_PROGRESS_M = 0.4
ALLOWED_UNKNOWN = {"return_trajectory_unverified", "avoidance_corridor_safety_unverified"}
MANEUVER_STATES = {"approach_maintain", "avoid", "avoid_continue", "return_to_path"}


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value))


def _frame_usable(frame):
    return (frame is not None and frame.get("valid") is True
            and frame.get("dataset_build_candidate") is True
            and frame.get("object_inventory_usable") is True
            and frame.get("zone_id") == "static_obstacle"
            and all(_finite(frame.get(k)) for k in
                    ("index", "lidar_source_ns", "ego_path_s_m", "ego_path_d_m",
                     "ego_world_x_m", "ego_world_y_m")))


def _actor_usable(actor):
    return (actor.get("kind") == "obstacle" and actor.get("instance_key")
            and actor.get("rule_geometry_usable") is True
            and actor.get("model_identity_verified") is True
            and actor.get("semantic_class_valid") is not False
            and all(_finite(actor.get(k)) for k in
                    ("path_d_m", "delta_s_m", "actor_width_m", "actor_length_m",
                     "world_x_m", "world_y_m"))
            and actor["actor_width_m"] > 0 and actor["actor_length_m"] > 0)


def _identity(actor):
    return str(actor.get("actor_id")), actor.get("instance_key")


def _connected(first, last):
    if not _frame_usable(first) or not _frame_usable(last):
        return False
    dt = (last["lidar_source_ns"] - first["lidar_source_ns"]) / 1e9
    if last["index"] != first["index"] + 1 or not 0 < dt <= MAX_DT_S:
        return False
    distance = math.hypot(last["ego_world_x_m"]-first["ego_world_x_m"],
                          last["ego_world_y_m"]-first["ego_world_y_m"])
    progress = last["ego_path_s_m"] - first["ego_path_s_m"]
    return (distance <= 50 * dt + .5 and -.5 <= progress <= 50 * dt + .5
            and abs(last["ego_path_d_m"]-first["ego_path_d_m"]) <= distance + .1)


def _actor_stamp_matches(actor, frame):
    # Object tables normally inherit the admitted frame index. If a caller
    # supplies its source stamp as well, contradictory stamps cannot be ignored.
    return (actor.get("index") == frame["index"] and
            all(_finite(actor[key]) and actor[key] == frame["lidar_source_ns"]
                for key in ("lidar_source_ns", "sensor_time_ns") if key in actor))


def _motion(history):
    """Only a continuous 0.4-0.65 s past window can support a phase change."""
    if len(history) < 5:
        return None
    current = history[-1]
    window = [f for f in history
              if (current["lidar_source_ns"] - f["lidar_source_ns"]) / 1e9 <= 0.65]
    if len(window) < 5:
        return None
    first = min(window, key=lambda f: abs(
        (current["lidar_source_ns"] - f["lidar_source_ns"]) / 1e9 - PAST_WINDOW_S))
    window = window[window.index(first):]
    duration = (current["lidar_source_ns"] - first["lidar_source_ns"]) / 1e9
    if len(window) < 5 or not 0.4 <= duration <= 0.65:
        return None
    ds = current["ego_path_s_m"] - first["ego_path_s_m"]
    dd = current["ego_path_d_m"] - first["ego_path_d_m"]
    changes = [b["ego_path_d_m"] - a["ego_path_d_m"] for a, b in zip(window, window[1:])]
    abs_changes = [abs(b["ego_path_d_m"]) - abs(a["ego_path_d_m"])
                   for a, b in zip(window, window[1:])]
    recent = window[-3]
    tail = window[-4]
    return {"past_window_seconds": duration, "past_start_index": first["index"],
            "delta_s_m": ds, "delta_d_m": dd,
            "delta_abs_d_m": abs(current["ego_path_d_m"]) - abs(first["ego_path_d_m"]),
            "recent_delta_d_m": current["ego_path_d_m"] - recent["ego_path_d_m"],
            "recent_delta_abs_d_m": abs(current["ego_path_d_m"]) - abs(recent["ego_path_d_m"]),
            "recent_window_seconds": (current["lidar_source_ns"] - recent["lidar_source_ns"]) / 1e9,
            "recent_delta_s_m": current["ego_path_s_m"] - recent["ego_path_s_m"],
            "tail_window_seconds": (current["lidar_source_ns"] - tail["lidar_source_ns"]) / 1e9,
            "tail_delta_s_m": current["ego_path_s_m"] - tail["ego_path_s_m"],
            "step_delta_d_m": changes, "step_delta_abs_d_m": abs_changes,
            "moving_forward": ds >= MIN_FORWARD_PROGRESS_M}


def _outward(motion, direction):
    sign = 1 if direction == "left" else -1
    return (motion is not None and motion["moving_forward"]
            and sign * motion["delta_d_m"] >= ONSET_SHIFT_M
            and motion["delta_abs_d_m"] >= ONSET_SHIFT_M * 0.7
            and sign * motion["recent_delta_d_m"] >= 0.03
            and all(sign * x >= -0.02 for x in motion["step_delta_d_m"]))


def _return_support(motion, d):
    if (motion is None or not motion["moving_forward"] or abs(d) <= 0.3
            or motion["recent_delta_s_m"] < 0.1):
        return None
    if (motion["delta_abs_d_m"] <= -RETURN_SHIFT_M
            and motion["recent_delta_abs_d_m"] <= -0.02
            and all(x <= 0.02 for x in motion["step_delta_abs_d_m"])):
        return "past_0_5s_sustained_convergence"
    # At a genuine turn-back boundary the preceding 0.5 s still contains
    # outward movement. Inspect three consecutive current steps, requiring
    # two meaningful shrinking steps and no outward step beyond 1 mm jitter.
    # A single shrinking sample cannot trigger this branch.
    steps = motion["step_delta_abs_d_m"][-3:]
    if (0.29 <= motion["tail_window_seconds"] <= 0.41
            and motion["tail_delta_s_m"] >= 0.3
            and len(steps) == 3 and all(x <= 0.001 for x in steps)
            and sum(x <= -0.005 for x in steps) >= 2
            and steps[-1] <= -0.01 and sum(steps) <= -0.015):
        return "recent_0_3s_confirmed_turn_back"
    return None


def _returning(motion, d):
    return _return_support(motion, d) is not None


def _set_state(row, state, actor, direction, reason, counters):
    before = row.get("instruction_state")
    if before != state:
        row.setdefault("evidence", {})["current_static_maneuver_before"] = {
            "instruction_state": before, "status": row.get("status"),
            "unknown_reasons": copy.deepcopy(row.get("unknown_reasons", [])),
            "target_actor_ref": copy.deepcopy(row.get("evidence", {}).get("target_actor_ref"))}
        counters["to_" + state] += 1
    ref = {"actor_id": actor["actor_id"], "instance_key": actor["instance_key"]}
    row.update(instruction_state=state, status="matched", unknown_reasons=[], instruction_id=None)
    row["related_actors"] = [actor["actor_id"]]
    row.setdefault("evidence", {}).update(target_actor_ref=ref)
    variables = row.setdefault("variables", {})
    variables.update(target_actor_id=actor["actor_id"], target_kind="obstacle",
                     target_distance_m=abs(actor["delta_s_m"]),
                     target_side="left" if actor["path_d_m"] > .3 else
                                 "right" if actor["path_d_m"] < -.3 else "center")
    if state in ("avoid", "avoid_continue"):
        variables["avoid_direction"] = direction
    else:
        variables.pop("avoid_direction", None)
    row["evidence"]["current_static_maneuver_policy"]["decision"] = reason


def align_current_static_maneuvers(frames, objects, rules):
    """Mutate raw rule states before rendering, return auditable counters.

    Only a verified one-sided Expert FOLLOW maneuver may begin/continue.
    Return requires current sustained recorded convergence. Protected actions
    (normal/keep-path/pass-between/etc.) and invalid evidence are preserved.
    """
    indexed = {f["index"]: f for f in frames}
    by_index = defaultdict(list)
    for obj in objects:
        by_index[obj["index"]].append(obj)
    counts, history, context = Counter(), [], None
    previous = None
    for row in rules:
        if row.get("zone_id") != "static_obstacle":
            history, context, previous = [], None, None
            continue
        frame = indexed.get(row["index"])
        evidence = row.setdefault("evidence", {})
        audit = {"schema": POLICY, "uses_future_ego": False, "decision": "preserved"}
        evidence["current_static_maneuver_policy"] = audit
        connected = _connected(previous, frame)
        if not connected:
            history, context = [], None
        previous = frame
        if not _frame_usable(frame) or any(row.get(k) is False for k in ("action_valid", "observation_valid")):
            history, context = [], None
            audit["decision"] = "unusable_current_evidence"
            counts["unusable_current_evidence"] += 1
            continue
        history.append(frame)
        history = [f for f in history if frame["lidar_source_ns"] - f["lidar_source_ns"] <= 1000000000]
        motion = _motion(history)
        audit["motion"] = motion
        state = row.get("instruction_state")
        reasons = set(row.get("unknown_reasons") or ())
        if reasons - (ALLOWED_UNKNOWN | {"future_avoidance_trajectory_unverified"}):
            context = None
            audit["decision"] = "unresolved_rule_evidence"
            continue
        if (state not in MANEUVER_STATES and not (state is None and reasons and reasons <= ALLOWED_UNKNOWN)):
            context = None
            audit["decision"] = "protected_non_maneuver_state"
            continue
        if state is None and not reasons <= ALLOWED_UNKNOWN:
            context = None
            continue
        actors = [a for a in by_index[row["index"]]
                  if _actor_usable(a) and _actor_stamp_matches(a, frame)]
        ref = evidence.get("target_actor_ref") or {}
        expected = ((str(ref.get("actor_id")), ref.get("instance_key")) if ref.get("instance_key")
                    else context["identity"] if context is not None else None)
        matches = [a for a in actors if _identity(a) == expected]
        # No fallback to an actor ID alone, nearest actor, or previous pose.
        if len(matches) != 1:
            context = None
            audit["decision"] = "fresh_same_instance_missing"
            continue
        actor = matches[0]
        identity = _identity(actor)
        if motion is not None:
            actor_history_valid = True
            for old_frame in history:
                if old_frame["index"] < motion["past_start_index"]:
                    continue
                old = [a for a in by_index[old_frame["index"]]
                       if _actor_usable(a) and _identity(a) == identity
                       and _actor_stamp_matches(a, old_frame)]
                if (len(old) != 1 or
                        math.hypot(actor["world_x_m"] - old[0]["world_x_m"],
                                   actor["world_y_m"] - old[0]["world_y_m"]) > 0.35):
                    actor_history_valid = False
                    break
            if not actor_history_valid:
                context = None
                motion = None
                audit["motion"] = None
                audit["actor_history_unverified"] = True
        if context is not None:
            prior_actor = context["actor"]
            if (identity != context["identity"] or
                    math.hypot(actor["world_x_m"] - prior_actor["world_x_m"],
                               actor["world_y_m"] - prior_actor["world_y_m"]) > 0.35):
                context = None
                audit["decision"] = "actor_lifecycle_or_pose_discontinuity"
                continue
        audit["current_actor_ref"] = {"actor_id": actor["actor_id"], "instance_key": actor["instance_key"]}
        front = 0 < actor["delta_s_m"] <= 35
        behind = -15 <= actor["delta_s_m"] <= 0
        nominal = abs(actor["path_d_m"]) <= 1.0 + actor["actor_width_m"] / 2 + 0.5
        conflicts = [a for a in actors if 0 < a["delta_s_m"] <= 35
                     and abs(a["path_d_m"]) <= 1.0 + a["actor_width_m"] / 2 + .5]
        one_sided = frame.get("corridor_passable_candidate") is not True and evidence.get("obstacle_layout") != "corridor"
        if not nominal or not (front or behind) or not one_sided:
            context = None
            audit["decision"] = "not_single_sided_conflict_context"
            continue
        if context is not None and any(_identity(a) != identity for a in conflicts):
            context = None
            audit["decision"] = "different_current_conflict_actor"
            continue
        # Raw return itself can be verified without reconstructing a past
        # avoid label; a new return is only inferred within a verified episode.
        if _returning(motion, frame["ego_path_d_m"]) and (context is not None or state == "return_to_path"):
            _set_state(row, "return_to_path", actor, None, "current_past_window_converging", counts)
            audit["return_support"] = _return_support(motion, frame["ego_path_d_m"])
            context = {"identity": identity, "actor": actor, "direction":
                       context["direction"] if context else ("left" if frame["ego_path_d_m"] > 0 else "right"),
                       "onset_index": context["onset_index"] if context else row["index"]}
            continue
        direction = "left" if motion is not None and motion["delta_d_m"] > 0 else "right"
        same_episode = context is not None and context["direction"] == direction
        can_start = front and frame.get("expert_label") == "follow" and len(conflicts) == 1
        if (_outward(motion, direction) and frame.get("expert_label") == "follow"
                and (can_start or (behind and same_episode))):
            if context is None:
                # An already observed Continue is not a new maneuver start.
                onset = row["index"] - 5 if state == "avoid_continue" else row["index"]
                context = {"identity": identity, "actor": actor, "direction": direction, "onset_index": onset}
            state = "avoid" if row["index"] < context["onset_index"] + 5 else "avoid_continue"
            _set_state(row, state, actor, direction, "fresh_current_outward_motion", counts)
            context["actor"] = actor
            continue
        if context is not None:
            context["actor"] = actor
        if state == "return_to_path":
            # A raw one-frame/noise proxy cannot assert current return when
            # the sustained current past window does not corroborate it.
            row.update(instruction_state=None, status="unknown", instruction_id=None,
                       unknown_reasons=sorted(reasons | {"current_return_motion_unverified"}))
            audit["decision"] = "return_not_supported_by_current_motion"
            counts["return_not_supported_by_current_motion"] += 1
    return dict(sorted(counts.items()))


def validate_current_static_maneuver(frames, objects, rule):
    """Independently recheck current phase; never trust the saved audit numbers.

    Returns (bool, reasons). No mutation and no future-pose access. This is for
    current-policy successful maneuver rows, not a blanket static exemption.
    """
    reasons = []
    state = rule.get("instruction_state")
    marker = rule.get("evidence", {}).get("current_static_maneuver_policy", {})
    if (rule.get("zone_id") != "static_obstacle" or
            state not in ("avoid", "avoid_continue", "return_to_path") or
            marker.get("schema") != POLICY or marker.get("decision") not in
            ("fresh_current_outward_motion", "current_past_window_converging")):
        return False, ["not_a_successful_current_static_maneuver"]
    if rule.get("status") != "matched" or rule.get("unknown_reasons"):
        reasons.append("current_rule_unresolved")
    if any(rule.get(k) is False for k in ("action_valid", "observation_valid")):
        reasons.append("current_rule_excluded")
    matching = [f for f in frames if f.get("index") == rule.get("index")]
    if len(matching) != 1 or not _frame_usable(matching[0]):
        return False, sorted(set(reasons + ["current_ego_or_inventory_unusable"]))
    current = matching[0]
    history = sorted([f for f in frames if _finite(f.get("lidar_source_ns"))
                      and _finite(f.get("index")) and f["index"] <= current["index"]
                      and 0 <= current["lidar_source_ns"]-f["lidar_source_ns"] <= 650000000],
                     key=lambda f:f["index"])
    if (not history or history[-1] is not current or not all(_frame_usable(f) for f in history)
            or not all(_connected(a,b) for a,b in zip(history,history[1:]))):
        return False, sorted(set(reasons + ["past_ego_track_not_continuous"]))
    motion = _motion(history)
    if motion is None:
        return False, sorted(set(reasons + ["past_motion_window_unavailable"]))
    ref = rule.get("evidence", {}).get("target_actor_ref") or {}
    if ref.get("actor_id") is None or not ref.get("instance_key"):
        return False, sorted(set(reasons + ["target_instance_unverified"]))
    identity = str(ref["actor_id"]), ref["instance_key"]
    indexed = defaultdict(list)
    for actor in objects:
        indexed[actor.get("index")].append(actor)
    observed = []
    for frame in history:
        if frame["index"] < motion["past_start_index"]:
            continue
        matched = [a for a in indexed[frame["index"]] if a.get("actor_id") is not None
                   and _identity(a) == identity]
        if len(matched) != 1 or not _actor_usable(matched[0]):
            reasons.append("fresh_same_instance_geometry_unverified")
            continue
        if not _actor_stamp_matches(matched[0], frame):
            reasons.append("object_source_stamp_mismatch")
        observed.append(matched[0])
    if observed:
        actor = observed[-1]
        if any(math.hypot(a["world_x_m"]-actor["world_x_m"],
                          a["world_y_m"]-actor["world_y_m"]) > .35 for a in observed):
            reasons.append("static_actor_pose_discontinuity")
        if not -15 <= actor["delta_s_m"] <= 35:
            reasons.append("actor_outside_current_maneuver_context")
        if abs(actor["path_d_m"]) > 1+actor["actor_width_m"]/2+.5:
            reasons.append("actor_not_nominal_path_conflict")
    else:
        reasons.append("fresh_same_instance_geometry_unverified")
    if state == "return_to_path":
        if not _returning(motion,current["ego_path_d_m"]):
            reasons.append("current_return_motion_unverified")
    else:
        direction = rule.get("variables", {}).get("avoid_direction")
        if direction not in ("left","right") or not _outward(motion,direction):
            reasons.append("current_avoid_direction_motion_mismatch")
        rendered_direction = rule.get("instruction_variables", {}).get("direction")
        if rendered_direction is not None and rendered_direction != direction:
            reasons.append("rendered_avoid_direction_mismatch")
        if current.get("expert_label") != "follow":
            reasons.append("current_avoid_expert_phase_unverified")
    return not reasons, sorted(set(reasons))
