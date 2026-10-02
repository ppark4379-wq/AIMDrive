"""Evidence-preserving instruction situation rules, schema v0.2.

A state describes a supported current-frame instruction. Missing proof stays
unknown; candidate geometry is never promoted to a safety certificate.
"""
from collections import Counter, defaultdict
from .highway_rules import highway_rule
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RuleConfig:
    static_lookahead_m: float = 35.0
    # Include obstacles flanking the avoidance corridor, even when their
    # centers are outside the nominal path conflict band.
    static_near_path_m: float = 4.0
    ego_width_m: float = 2.0
    # Expert's static-obstacle collision safety margin (lattice_planning
    # parameters.py COLLISION_MARGIN), not the smaller tracking-only or
    # road-boundary margins defined alongside it.
    static_collision_margin_m: float = 0.5
    static_candidate_gap_m: float = 3.0
    dynamic_nearby_m: float = 35.0
    roundabout_nearby_m: float = 80.0
    lead_tracking_m: float = 60.0
    stop_speed_mps: float = 0.2
    deceleration_mps2: float = -0.5
    hard_braking_mps2: float = -3.0
    lead_min_gap_m: float = 3.0
    lead_headway_s: float = 1.2


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _ref(row):
    return {"actor_id": row.get("actor_id"), "instance_key": row.get("instance_key")}


def _result(frame):
    return {"index": frame["index"], "zone_id": frame.get("zone_id"),
            "status": "no_match", "instruction_state": None,
            "related_actors": [], "variables": {}, "evidence": {},
            "unknown_reasons": [], "instruction_id": None}


def _finish(row, state=None, reasons=()):
    row["instruction_state"] = state
    row["unknown_reasons"] = sorted(set(reasons))
    row["status"] = "matched" if state is not None else (
        "unknown" if row["unknown_reasons"] else "no_match")
    return row


def _distance(frame, actor):
    distance = _number(actor.get("distance_to_ego_m"))
    if distance is not None:
        return distance
    x, y = _number(actor.get("world_x_m")), _number(actor.get("world_y_m"))
    ex, ey = _number(frame.get("ego_world_x_m")), _number(frame.get("ego_world_y_m"))
    return math.hypot(x - ex, y - ey) if None not in (x, y, ex, ey) else None


def _relevant(frame, objects, kinds, radius):
    relevant, unlocated = [], []
    for row in objects:
        if row.get("kind") not in kinds:
            continue
        distance = _distance(frame, row)
        if distance is None:
            unlocated.append(_ref(row))
        elif distance <= radius:
            relevant.append(row)
    return relevant, unlocated


def _usable(row):
    if row.get("rule_geometry_usable") is True:
        return True
    # The rule functions also accept constructed evidence rows in unit tests.
    if "rule_geometry_usable" not in row and row.get("valid") is True:
        return True
    return False


def _target(result, actor, distance, side, kind):
    result["related_actors"] = [actor.get("actor_id")]
    result["evidence"]["target_actor_ref"] = _ref(actor)
    result["variables"].update(target_actor_id=actor.get("actor_id"),
                               target_kind=kind, target_distance_m=distance,
                               target_side=side)


def static_obstacle_rule(frame, objects, config=RuleConfig()):
    result = _result(frame)
    # Expert's raw recorded phase, already translated to an instruction label
    # (expert_static_phase.py RAW_STATE_TO_LABEL) -- rule logic below branches
    # on this translated label, never on Expert's raw state string, so a
    # renamed/merged raw state only needs an edit at the translation layer.
    expert_label = frame.get("expert_label")
    if expert_label is not None:
        result["label"] = expert_label
    nearby, unlocated = _relevant(frame, objects, ("obstacle",),
                                 config.static_lookahead_m + 15)
    if unlocated:
        result["evidence"]["unlocated_actors"] = unlocated
        return _finish(result, reasons=["unlocated_obstacle"])
    unresolved = [_ref(row) for row in nearby if not _usable(row)]
    if unresolved:
        result["evidence"]["unresolved_actors"] = unresolved
        return _finish(result, reasons=["relevant_obstacle_geometry_unverified"])
    front = []
    behind = []
    for row in nearby:
        delta, d = _number(row.get("delta_s_m")), _number(row.get("path_d_m"))
        width, length = _number(row.get("actor_width_m")), _number(row.get("actor_length_m"))
        if None in (delta, d, width, length):
            return _finish(result, reasons=["missing_static_path_geometry"])
        radius = math.hypot(width, length) / 2
        if abs(d) > config.static_near_path_m + radius:
            continue
        if 0 < delta <= config.static_lookahead_m:
            front.append(row)
        elif -15 <= delta <= 0:
            behind.append(row)
    front.sort(key=lambda row: row["delta_s_m"])
    result["evidence"].update(
        relevant_obstacle_ids=[row.get("actor_id") for row in front],
        obstacle_layout=frame.get("obstacle_layout"),
        corridor_gap_m=frame.get("corridor_gap_m"),
        corridor_gap_is_lower_bound=True,
        corridor_passable_candidate=frame.get("corridor_passable_candidate"),
        avoidance_active=frame.get("avoidance_active"),
        returning_to_reference_path=frame.get("returning_to_reference_path"),
        trajectory_safety_verified=False)
    if not front:
        if behind and frame.get("returning_to_reference_path") is True:
            _target(result, behind[0], abs(behind[0]["delta_s_m"]),
                    behind[0].get("target_side"), "obstacle")
            return _finish(result, "return_to_path")
        if behind and frame.get("avoidance_active") is True:
            return _finish(result, reasons=["return_trajectory_unverified"])
        return _finish(result, "normal")
    first = front[0]

    def _nominal_conflict(row):
        # A fixed geometric test against the official route, independent of
        # Ego's current (possibly already-deviated) position: does this
        # obstacle's static lateral offset intrude on Ego's nominal body
        # envelope plus Expert's real static-obstacle collision margin
        # (lattice_planning parameters.py COLLISION_MARGIN=0.5m)?
        return abs(row["path_d_m"]) <= (
            config.ego_width_m / 2 + row["actor_width_m"] / 2
            + config.static_collision_margin_m)

    conflicting_front = [row for row in front if _nominal_conflict(row)]
    # Report the nearest obstacle that actually conflicts with the nominal
    # path, not just the nearest obstacle by distance; a closer but
    # laterally-clear obstacle should not mask a real conflict farther along.
    relevant = conflicting_front[0] if conflicting_front else first
    blocking = bool(conflicting_front)

    side = "left" if relevant["path_d_m"] > .3 else "right" if relevant["path_d_m"] < -.3 else "center"
    _target(result, relevant, relevant["delta_s_m"], side, "obstacle")

    ego_d = _number(frame.get("ego_reference_path_offset_m"))
    if ego_d is not None:
        # Separate from nominal_path_conflict: how much room Ego's *current*
        # (possibly already-avoiding) position actually leaves next to this
        # obstacle, independent of whether the nominal path conflicts.
        result["evidence"]["current_clearance_m"] = (
            abs(relevant["path_d_m"] - ego_d)
            - config.ego_width_m / 2 - relevant["actor_width_m"] / 2)
    result["evidence"]["path_blocked_candidate"] = blocking
    result["evidence"]["nominal_path_conflict_ids"] = [
        row.get("actor_id") for row in conflicting_front]
    result["evidence"]["avoid_direction"] = frame.get("avoid_direction")
    if frame.get("corridor_passable_candidate") is True:
        left = next((row for row in front if row["path_d_m"] > 0), None)
        right = next((row for row in front if row["path_d_m"] < 0), None)
        gap = _number(frame.get("corridor_gap_lower_bound_m"))
        if gap is None:
            gap = _number(frame.get("corridor_gap_m"))
        if (left is not None and right is not None and gap is not None
                and gap > config.static_candidate_gap_m
                and abs(left["delta_s_m"] - right["delta_s_m"]) <= 5):
            result["evidence"]["corridor_pair_actor_refs"] = [_ref(left), _ref(right)]
            result["evidence"]["corridor_gap_lower_bound_m"] = gap
            verified = frame.get("corridor_trajectory_verified") is True
            ego_d = _number(frame.get("ego_reference_path_offset_m"))
            heading_error = _number(frame.get("ego_reference_heading_error_deg"))
            left_inner = left["path_d_m"] - math.hypot(
                left["actor_width_m"], left["actor_length_m"]) / 2
            right_inner = right["path_d_m"] + math.hypot(
                right["actor_width_m"], right["actor_length_m"]) / 2
            margin = config.ego_width_m / 2 + config.static_collision_margin_m
            safe_min, safe_max = right_inner + margin, left_inner - margin
            in_corridor = ego_d is not None and safe_min <= ego_d <= safe_max
            heading_into_corridor = (
                ego_d is not None and heading_error is not None
                and ((ego_d < safe_min and heading_error > 0.5)
                     or (ego_d > safe_max and heading_error < -0.5)))
            expert_passage = (
                expert_label == "follow"
                and (_number(frame.get("ego_speed_mps")) or 0) > config.stop_speed_mps
                and (in_corridor or heading_into_corridor))
            if verified or expert_passage:
                result["related_actors"] = [left.get("actor_id"), right.get("actor_id")]
                result["variables"]["target_position"] = "front"
                result["evidence"]["trajectory_safety_verified"] = verified
                result["evidence"]["corridor_motion_supported"] = expert_passage
                return _finish(result, "pass_between")
    if expert_label == "follow" and blocking:
        # Expert's own recorded planner state already confirms an avoidance
        # maneuver is underway for this (geometrically conflicting) obstacle
        # -- unlike avoidance_active below, this does not require Ego's GT
        # position to have already crossed a lateral-offset threshold, nor
        # the motion-derived avoid-candidate flag (which can lag Ego's GT
        # motion, or be suppressed once Ego is already rejoining). Direction
        # comes from the obstacle's own side so it is available even before
        # Ego's GT position has started to move.
        obstacle_direction = ("right" if side == "left" else
                              "left" if side == "right" else None)
        direction = obstacle_direction or frame.get("avoid_direction")
        incoherent = side in ("left", "right") and direction == side
        if direction in ("left", "right") and not incoherent:
            result["variables"]["avoid_direction"] = direction
            # A held FOLLOW+blocking span previously returned "avoid" every
            # frame, repeating "회피해라" for the whole maneuver instead of
            # saying it for the first few frames and then "계속 진행해라" --
            # static_avoid_initial_hold/_episode_active (attach_static_avoid_
            # episode) debounce this the same way dynamic_yield_transition_
            # start does for dynamic_obstacle_rule, just over a short window
            # (default 5 frames from the start) instead of a single frame.
            if frame.get("static_avoid_initial_hold") is True:
                return _finish(result, "avoid")
            if frame.get("returning_to_reference_path") is True:
                return _finish(result, "return_to_path")
            return _finish(result, "avoid_continue")
    if frame.get("avoidance_active") is True:
        if frame.get("avoidance_corridor_verified") is True and not blocking:
            return _finish(result, "pass_avoidance_corridor")
        direction = frame.get("avoid_direction")
        # Incoherent only when Ego avoids toward the side the obstacle is
        # already on (actively wrong); a centered obstacle is compatible
        # with either avoid direction.
        incoherent = side in ("left", "right") and direction == side
        if (direction in ("left", "right") and blocking and not incoherent
                and frame.get(direction + "_avoid_candidate") is True):
            result["variables"]["avoid_direction"] = direction
            return _finish(result, "avoid")
        return _finish(result, reasons=["avoidance_corridor_safety_unverified"])
    if frame.get("corridor_passable_candidate") is True:
        if expert_label in ("approach", "drive_approach"):
            return _finish(result, "approach_maintain",
                           ["future_avoidance_trajectory_unverified"])
        return _finish(result, reasons=["corridor_trajectory_unverified"])
    return _finish(result, "approach_maintain",
                   ["future_avoidance_trajectory_unverified"])


def dynamic_obstacle_rule(frame, objects, config=RuleConfig()):
    result = _result(frame)
    nearby, unlocated = _relevant(frame, objects, ("pedestrian",),
                                 config.dynamic_nearby_m)
    if unlocated:
        result["evidence"]["unlocated_actors"] = unlocated
        return _finish(result, reasons=["unlocated_pedestrian"])
    unresolved = [_ref(row) for row in nearby if not _usable(row)]
    if unresolved:
        result["evidence"]["unresolved_actors"] = unresolved
        return _finish(result, reasons=["relevant_pedestrian_geometry_unverified"])
    nearby.sort(key=lambda row: _distance(frame, row))
    evidence = []
    for row in nearby:
        d = _number(row.get("path_d_m"))
        width, length = _number(row.get("actor_width_m")), _number(row.get("actor_length_m"))
        if None in (d, width, length):
            return _finish(result, reasons=["pedestrian_path_conflict_unverified"])
        clearance = abs(d) - math.hypot(width, length) / 2 - config.ego_width_m / 2
        vd = _number(row.get("v_d_mps"))
        direction = ("R2L" if vd is not None and vd > .2 else
                     "L2R" if vd is not None and vd < -.2 else
                     "stationary" if vd is not None else "unknown")
        evidence.append({"actor_id": row.get("actor_id"),
                         "distance_m": _distance(frame, row),
                         "path_d_m": d, "v_d_mps": vd,
                         "crossing_direction": direction,
                         "clearance_m": clearance,
                         "conflict": clearance <= 0})
    result["evidence"]["pedestrians"] = evidence
    result["variables"].update(
        pedestrian_count=len(nearby),
        crossing_directions=[item["crossing_direction"] for item in evidence])
    if not nearby:
        if frame.get("dynamic_resume_transition_start") is True:
            result["evidence"]["front_safety_verified"] = False
            return _finish(result, "resume", ["resume_front_safety_unverified"])
        return _finish(result, "normal")
    chosen_index = next((i for i, item in enumerate(evidence)
                         if item["conflict"]), 0)
    first, first_evidence = nearby[chosen_index], evidence[chosen_index]
    _target(result, first, first_evidence["distance_m"],
            first.get("target_side"), "pedestrian")
    result["variables"].update(
        pedestrian_path_d_m=first_evidence["path_d_m"],
        pedestrian_v_d_mps=first_evidence["v_d_mps"],
        pedestrian_crossing_direction=first_evidence["crossing_direction"],
        pedestrian_clearance_m=first_evidence["clearance_m"])
    conflict = first_evidence["conflict"]
    result["evidence"]["pedestrian_on_conflict_path"] = conflict
    result["evidence"]["all_relevant_pedestrians_clear"] = not any(
        item["conflict"] for item in evidence)
    if not conflict:
        if frame.get("dynamic_resume_transition_start") is True:
            result["evidence"]["front_safety_verified"] = False
            return _finish(result, "resume", ["resume_front_safety_unverified"])
        if any(item["v_d_mps"] is not None
               and item["path_d_m"] * item["v_d_mps"] < -0.1
               for item in evidence):
            return _finish(result, "approach")
        return _finish(result, "normal")
    # The situation itself is established (a relevant pedestrian is on a
    # conflict path) -- but without Ego's own speed, whether its current
    # behavior is yielding for this pedestrian or something else can't be
    # told apart, so the behavior dimension stays unverified rather than
    # guessed.
    speed = _number(frame.get("ego_speed_mps"))
    if speed is None:
        return _finish(result, "behavior_unverified", ["ego_speed_unavailable"])
    stopped = frame.get("ego_stopped") is True
    # dynamic_in_yield_family (instruction_features.py) already debounces a
    # single noisy ego_decelerating frame against an otherwise continuous,
    # monotonic slow-down; using the raw per-frame flag here would flip this
    # state back to "approach" for one frame mid-yield.
    yielding_now = frame.get("dynamic_in_yield_family") is True
    if yielding_now:
        if frame.get("dynamic_yield_transition_start") is True:
            return _finish(result, "yield_start")
        return _finish(result, "waiting" if stopped else "yielding")
    return _finish(result, "approach", ["future_pedestrian_motion_unverified"])

def roundabout_rule(frame, objects, config=RuleConfig()):
    result = _result(frame)
    nearby, unlocated = _relevant(frame, objects, ("npc", "vehicle"),
                                 config.roundabout_nearby_m)
    if unlocated:
        result["evidence"]["unlocated_actors"] = unlocated
        return _finish(result, reasons=["unlocated_vehicle"])
    # Validity is checked only after spatial filtering; a distant NPC does not
    # veto the whole frame. BEV exclusion alone does not invalidate sealed GT.
    unresolved = [_ref(row) for row in nearby if not _usable(row)]
    if unresolved:
        result["evidence"]["unresolved_actors"] = unresolved
        return _finish(result, reasons=["relevant_vehicle_geometry_unverified"])
    result["evidence"].update(
        nearby_vehicle_ids=[row.get("actor_id") for row in nearby],
        circulating_vehicle_ids=frame.get("circulating_vehicle_ids") or [],
        collision_risk=frame.get("collision_risk"),
        future_collision_predicted=frame.get("future_collision_predicted"),
        lead_bumper_gap_verified=frame.get("lead_bumper_gap_verified") is True)
    for name in ("roundabout_phase", "distance_to_entry_m", "distance_to_merge_m"):
        result["variables"][name] = frame.get(name)
    risk = frame.get("collision_risk")
    lead_id = frame.get("lead_actor_id")
    lead = next((row for row in nearby if row.get("actor_id") == lead_id), None)
    if lead is not None:
        result["related_actors"] = [lead_id]
        result["evidence"]["target_actor_ref"] = _ref(lead)
        result["variables"].update(
            target_actor_id=lead_id, target_kind="vehicle",
            target_distance_m=frame.get("lead_distance_m"),
            # A lead vehicle is always ahead along the ring path, not a
            # literal lateral-relative position like static/dynamic targets.
            target_side="front",
            lead_distance_m=frame.get("lead_distance_m"),
            lead_bumper_gap_m=frame.get("lead_bumper_gap_m"),
            safe_following_gap_m=frame.get("safe_following_gap_m"),
            relative_speed_mps=frame.get("relative_speed_mps"))
    if risk is True:
        conflict_id = frame.get("conflict_actor_id")
        conflict = next((row for row in nearby if row.get("actor_id") == conflict_id), None)
        if conflict is None:
            return _finish(result, reasons=["conflict_actor_unverified"])
        result["related_actors"] = [conflict_id]
        result["evidence"]["target_actor_ref"] = _ref(conflict)
        result["variables"].update(target_actor_id=conflict_id,
                                   target_kind="vehicle",
                                   target_distance_m=frame.get("conflict_distance_m"),
                                   # A circulating conflict vehicle's ETA is a
                                   # ring-position/time measure, not a literal
                                   # lateral-relative position.
                                   target_side="front")
        if frame.get("ego_hard_braking") is True:
            return _finish(result, "hard_brake_conflict")
        if frame.get("ego_stopped") is True:
            return _finish(result, "wait_conflict")
        if frame.get("ego_decelerating") is True:
            return _finish(result, "decelerate_conflict")
        return _finish(result, reasons=["conflict_response_unverified"])
    if lead is not None:
        if frame.get("lead_bumper_gap_verified") is not True:
            return _finish(result, reasons=["lead_bumper_gap_unverified"])
        insufficient = frame.get("lead_gap_insufficient")
        result["evidence"]["lead_gap_insufficient"] = insufficient
        if risk is False and frame.get("ego_hard_braking") is True and insufficient is True:
            return _finish(result, "hard_brake_lead")
        if risk is False and frame.get("ego_stopped") is True and insufficient is True:
            return _finish(result, "wait_lead")
        if (risk is False and frame.get("ego_stopped") is True
                and insufficient is False and
                _number(frame.get("lead_speed_mps")) is not None
                and frame["lead_speed_mps"] > config.stop_speed_mps):
            result["evidence"]["resume_candidate"] = True
            return _finish(result, "resume_candidate",
                           ["future_following_safety_unverified"])
        if risk is False and frame.get("ego_decelerating") is True and insufficient is True:
            return _finish(result, "follow_decelerate")
        if risk is False and insufficient is False:
            # A verified, sufficient lead gap is ordinary following regardless
            # of which exact link Ego matched; entry/merge traffic queues the
            # same way a fully circulating vehicle does.
            return _finish(result, "follow")
        return _finish(result, reasons=["lead_response_or_conflict_unverified"])
    if risk is None:
        return _finish(result, reasons=["future_collision_risk_unverified"])
    if frame.get("ego_stopped") is True and risk is False:
        result["evidence"]["resume_candidate"] = True
        return _finish(result, "resume_candidate",
                       ["future_entry_safety_unverified"])
    if frame.get("in_merge_section") is True:
        if frame.get("ego_merging_candidate") is True:
            return _finish(result, "merge")
        return _finish(result, reasons=["merge_motion_unverified"])
    distance = _number(frame.get("distance_to_merge_m"))
    if distance is not None and distance > 0 and frame.get("in_entry_section") is True:
        return _finish(result, "approach_entry")
    if frame.get("in_circulating_lane") is True or frame.get("roundabout_phase") == "exit":
        # Exit covers a short first-exit maneuver that never matches a
        # declared ring link; past the merge with no lead or conflict is
        # ordinary driving again.
        return _finish(result, "normal")
    return _finish(result, reasons=["roundabout_phase_unverified"])


def evaluate_frames(frames, objects, config=RuleConfig()):
    indexed = {}
    for frame in frames:
        if frame["index"] in indexed:
            raise ValueError("duplicate frame feature index")
        indexed[frame["index"]] = frame
    grouped = defaultdict(list)
    for row in objects:
        if row["index"] not in indexed:
            raise ValueError("object feature has unknown frame index")
        grouped[row["index"]].append(row)
    results = []
    for index in sorted(indexed):
        frame = indexed[index]
        if frame.get("valid") is not True:
            result = _finish(_result(frame), reasons=frame.get("invalid_reasons") or ["frame_invalid"])
            result["status"] = "invalid_frame"
        elif frame.get("zone_id") in ("static_obstacle", "dynamic_obstacle", "roundabout", "highway") and (
                frame.get("object_inventory_usable") is not True):
            result = _finish(_result(frame), reasons=["object_inventory_unusable"])
        elif frame.get("zone_id") == "static_obstacle":
            result = static_obstacle_rule(frame, grouped[index], config)
        elif frame.get("zone_id") == "dynamic_obstacle":
            result = dynamic_obstacle_rule(frame, grouped[index], config)
        elif frame.get("zone_id") == "roundabout":
            result = roundabout_rule(frame, grouped[index], config)
        elif frame.get("zone_id") == "highway":
            result = highway_rule(frame, grouped[index], config)
        else:
            result = _result(frame)
            result["status"] = "not_applicable"
        if frame.get("zone_id") == "highway":
            result.setdefault("label_candidate", None)
            result.setdefault("behavior_state", None)
        results.append(result)
    return results


def summarize_rules(results, config=RuleConfig()):
    return {"schema": "morai_situation_rules_v0_2",
            "total_frames": len(results),
            "status_counts": dict(sorted(Counter(row["status"] for row in results).items())),
            "instruction_state_counts": dict(sorted(Counter(
                row["instruction_state"] for row in results
                if row["instruction_state"] is not None).items())),
            "label_candidate_counts": dict(sorted(Counter(
                row["label_candidate"] for row in results
                if row.get("label_candidate") is not None).items())),
            "behavior_state_counts": dict(sorted(Counter(
                row["behavior_state"] for row in results
                if row.get("behavior_state") is not None).items())),
            "unknown_reason_counts": dict(sorted(Counter(
                reason for row in results for reason in row["unknown_reasons"]).items())),
            "thresholds": asdict(config),
            "instruction_templates_applied": False,
            "future_collision_prediction_reproduced": False}