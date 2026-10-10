"""Validate moving roundabout claims against current and contiguous past GT.

Expert's desired speed and interpolated phase are observations, not proof that
the ego has departed. This pass never invents a stop cause, a clear entry, or a
replacement stop sentence. A verified entry follow uses the existing generic
Follow sentence; ordinary FOLLOW/YIELD and their departure policy are untouched.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

from .renderer import load_catalog, render_template

SCHEMA = "roundabout_current_cruise_v1"
BASIS = "recorded_current_roundabout_motion"
MARKER = "current_roundabout_cruise"
STATES = {"CRUISE": "cruise_normal", "CRUISE_AFTER_MERGE": "cruise_normal",
          "CRUISE_APPROACH": "cruise_approach_plain", "CRUISE_MERGE": "cruise_merge_plain",
          "FOLLOW_MERGE": "follow_merge"}
MIN_WINDOW_S = .3
MAX_WINDOW_S = .6
MAX_INTERVAL_S = .25
STOP_SPEED_MPS = .2
MAX_SPEED_DISAGREEMENT_MPS = .02
MAX_WINDOW_DECEL_MPS = .3
ENTRY_FOLLOW_DECISION = "verified_current_entry_follow"
EXPERT_FOLLOW_PLAN_DECISION = "verified_current_expert_follow_plan"
PUBLICATION_DRIVE_DECISION = "verified_receive_bound_current_path_plan"
MAX_ENTRY_TARGET_DISTANCE_M = 35.
MAX_ENTRY_ACTOR_SPEED_MPS = 30.


def _publication_drive_decision(rule, tables):
    """Recover only the published path plan, without inventing a follow cause.

    The controller topics have no source timestamp. This is explicitly a
    causal receive-bound context, never an exact source-time observation or a
    safe-entry claim. A legacy phase is retained as provenance, not promoted
    to a freshly selected actor or a merge command.
    """
    if (rule.get("zone_id") != "roundabout"
            or (rule.get("expert_raw_state") or {}).get("source") != "existing_roundabout_text_log"
            or (rule.get("expert_raw_state") or {}).get("roundabout_phase") != "follow"
            or rule.get("observation_valid") is False or rule.get("action_valid") is False
            or set(rule.get("unknown_reasons") or ()) - {"conflict_response_unverified"}):
        return None, "current_roundabout_path_plan_upstream_unverified"
    motion, reason = _past_motion(rule.get("index"), tables)
    if reason:
        return None, reason
    if min(motion["backward_speeds_mps"]) <= STOP_SPEED_MPS:
        return None, "current_roundabout_path_plan_departure_not_observed"
    frame = tables[0][rule["index"]][0]
    context = frame.get("expert_publications") or {}
    if (context.get("status") != "RECEIVE_BOUND_CURRENT_CONTEXT"
            or context.get("sensor_ns") != frame["lidar_source_ns"]
            or not isinstance(context.get("lidar_record_id"), int)
            or not isinstance(context.get("lidar_receive_monotonic_ns"), int)
            or not isinstance(context.get("source_sha256"), dict)
            or not context["source_sha256"]):
        return None, "current_roundabout_path_plan_context_missing"
    receipts = {}
    for role in ("ego", "control", "target_velocity", "local_path"):
        item = context.get(role)
        if not isinstance(item, dict):
            return None, "current_roundabout_path_plan_publication_missing"
        age, clock = item.get("receive_age_ms"), item.get("preceding_received_clock_ns")
        payload_hash = item.get("payload_sha256")
        exact_gt_ego = (role == "ego" and item.get("source_timestamp_ns") == frame["lidar_source_ns"]
                        and item.get("ego_pose_source_age_ms") == 0
                        and item.get("source_association_basis") ==
                        "exact_sealed_gt_scene_source_hypothesis_and_prior_receipt")
        if (not _finite(age) or age < 0 or (not exact_gt_ego and age > 100)
                or not isinstance(item.get("receive_monotonic_ns"), int)
                or context["lidar_receive_monotonic_ns"] < item["receive_monotonic_ns"]
                or abs(age - (context["lidar_receive_monotonic_ns"] - item["receive_monotonic_ns"])/1e6) > 1e-6
                or (not exact_gt_ego and (not isinstance(clock, int)
                    or not 0 <= frame["lidar_source_ns"] - clock <= 100000000))
                or not isinstance(item.get("record_id"), int)
                or not isinstance(payload_hash, str) or len(payload_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in payload_hash)):
            return None, "current_roundabout_path_plan_receipt_unverified"
        if role == "ego" and (not isinstance(item.get("source_timestamp_ns"), int)
                or not 0 <= frame["lidar_source_ns"] - item["source_timestamp_ns"] <= 100000000
                or item.get("ego_pose_source_age_ms") != (
                    frame["lidar_source_ns"] - item["source_timestamp_ns"])/1e6):
            return None, "current_roundabout_path_plan_ego_source_unverified"
        receipts[role] = {key: item.get(key) for key in (
            "record_id", "payload_sha256", "receive_age_ms", "receive_monotonic_ns",
            "preceding_received_clock_ns", "source_timestamp_ns", "raw_chunk")}
    control, target, ego, path = (context[k] for k in (
        "control", "target_velocity", "ego", "local_path"))
    target_speed, ego_speed = target.get("value_mps"), ego.get("speed_mps")
    if (control.get("longlCmdType") != 1
            or not all(_finite(control.get(k)) for k in ("accel", "brake"))
            or not 0 <= control["brake"] <= 1 or not 0 <= control["accel"] <= 1
            or not _finite(target_speed) or target_speed < 0
            or not _finite(ego_speed) or ego_speed <= STOP_SPEED_MPS):
        return None, "current_roundabout_path_plan_control_unverified"
    if (control["brake"] <= 1e-6 and target_speed > STOP_SPEED_MPS
            and (control["accel"] > 0 or target_speed >= ego_speed)):
        identifier = "current_plan_drive"
    elif control["brake"] > 1e-6 and control["accel"] <= 1e-6:
        if target_speed == 0:
            identifier = "current_plan_stop"
        elif target_speed < ego_speed:
            identifier = "current_plan_slowdown"
        else:
            return None, "current_roundabout_path_plan_brake_target_disagrees"
    else:
        return None, "current_roundabout_path_plan_control_target_disagrees"
    xyz, yaw, points = ego.get("world_xyz"), ego.get("yaw_rad"), path.get("world_xyz")
    if (not isinstance(xyz, list) or len(xyz) < 2 or not all(_finite(v) for v in xyz)
            or not _finite(yaw) or not isinstance(points, list) or len(points) < 2
            or any(not isinstance(p, list) or len(p) < 2 or not all(_finite(v) for v in p)
                   for p in points)):
        return None, "current_roundabout_path_plan_geometry_missing"
    distance = math.hypot(points[0][0] - xyz[0], points[0][1] - xyz[1])
    forward = [(p[0]-xyz[0])*math.cos(yaw)+(p[1]-xyz[1])*math.sin(yaw) for p in points]
    if distance > 3. or max(forward) < 2.:
        return None, "current_roundabout_path_plan_current_path_unverified"
    return dict(motion, schema=SCHEMA, supervision_basis=BASIS, valid=True,
                decision=PUBLICATION_DRIVE_DECISION, expert_state=None,
                instruction_id=identifier, current_expert_target=None,
                current_plan={"status": context["status"], "sensor_ns": context["sensor_ns"],
                              "lidar_record_id": context["lidar_record_id"],
                              "source_sha256": dict(context["source_sha256"]), "receipts": receipts,
                              "target_speed_mps": target_speed, "ego_speed_mps": ego_speed,
                              "accel": control["accel"], "brake": control["brake"],
                              "selected_plan_action": identifier,
                              "path_points": len(points), "path_start_distance_m": distance,
                              "path_forward_extent_m": max(forward),
                              "path_geometry_sha256": hashlib.sha256(json.dumps(
                                  points, separators=(",", ":"), allow_nan=False).encode()).hexdigest()},
                uses_future_ego=False, uses_future_actor=False,
                physical_simultaneity_verified=False, physical_front_safety_certified=False,
                instruction_semantics="published_current_path_plan_without_actor_cause"), None


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _admitted(frame):
    return (frame.get("zone_id") == "roundabout" and frame.get("valid") is True
            and frame.get("dataset_build_candidate") is True
            and frame.get("object_inventory_usable") is True
            and frame.get("observation_valid") is not False
            and frame.get("action_valid") is not False
            and not frame.get("invalid_reasons")
            and all(_finite(frame.get(k)) for k in
                    ("index", "lidar_source_ns", "ego_world_x_m", "ego_world_y_m")))


def _tables(frames, objects):
    positions = defaultdict(list)
    stamps = Counter(f.get("lidar_source_ns") for f in frames)
    actors = defaultdict(list)
    for frame in frames:
        positions[frame.get("index")].append(frame)
    for actor in objects:
        actors[actor.get("index")].append(actor)
    return positions, stamps, actors


def _current_inventory(frame, actors):
    """Check freshness/identity; do not turn complete inventory into safety."""
    ids = [str(a.get("actor_id")) for a in actors]
    if len(ids) != len(set(ids)):
        return None, "current_roundabout_actor_identity_ambiguous"
    refs = []
    for actor in actors:
        if (actor.get("actor_id") is None or not actor.get("instance_key")
                or actor.get("index") != frame["index"]
                or actor.get("lidar_source_ns") != frame["lidar_source_ns"]):
            return None, "current_roundabout_actor_stale_or_unidentified"
        refs.append({"actor_id": str(actor["actor_id"]), "instance_key": actor["instance_key"]})
    return sorted(refs, key=lambda a: (a["actor_id"], a["instance_key"])), None


def _past_motion(index, tables):
    positions, stamps, actors = tables
    current = positions.get(index, ())
    if len(current) != 1:
        return None, "current_roundabout_frame_ambiguous"
    frame = current[0]
    if not _admitted(frame) or stamps[frame["lidar_source_ns"]] != 1:
        return None, "current_roundabout_frame_unusable"
    track = [frame]
    for earlier_index in range(index - 1, -1, -1):
        candidates = positions.get(earlier_index, ())
        if len(candidates) != 1:
            return None, "current_roundabout_past_track_missing_or_ambiguous"
        earlier = candidates[0]
        if not _admitted(earlier) or stamps[earlier["lidar_source_ns"]] != 1:
            return None, "current_roundabout_past_track_unusable"
        dt = (track[0]["lidar_source_ns"] - earlier["lidar_source_ns"]) / 1e9
        elapsed = (frame["lidar_source_ns"] - earlier["lidar_source_ns"]) / 1e9
        if not 0 < dt <= MAX_INTERVAL_S or elapsed > MAX_WINDOW_S + 1e-8:
            return None, "current_roundabout_past_track_discontinuous"
        track.insert(0, earlier)
        if elapsed >= MIN_WINDOW_S - 1e-8:
            break
    window = (frame["lidar_source_ns"] - track[0]["lidar_source_ns"]) / 1e9
    if window < MIN_WINDOW_S - 1e-8:
        return None, "current_roundabout_past_window_too_short"
    speeds, progress = [], 0.
    for first, last in zip(track, track[1:]):
        dt = (last["lidar_source_ns"] - first["lidar_source_ns"]) / 1e9
        distance = math.hypot(last["ego_world_x_m"] - first["ego_world_x_m"],
                              last["ego_world_y_m"] - first["ego_world_y_m"])
        speed = distance / dt
        if (last.get("ego_speed_source") != "backward_gt_ego_xy_difference"
                or not _finite(last.get("ego_speed_mps"))
                or abs(last["ego_speed_mps"] - speed) > MAX_SPEED_DISAGREEMENT_MPS):
            return None, "current_roundabout_recorded_speed_disagrees"
        speeds.append(speed)
        progress += distance
    refs, reason = _current_inventory(frame, actors.get(index, ()))
    if reason:
        return None, reason
    return {"frame_index": index, "lidar_source_ns": frame["lidar_source_ns"],
            "past_start_index": track[0]["index"], "past_start_ns": track[0]["lidar_source_ns"],
            "past_window_seconds": window, "backward_speeds_mps": speeds,
            "current_speed_mps": speeds[-1], "recorded_progress_m": progress,
            "current_actor_refs": refs}, None


def _legacy_phase_source_proof(frame, raw, past_start_ns):
    """Verify a stable qualitative plan between log clock anchors.

    The phase log does not timestamp each transition. Its bracketing anchors
    bound uncertainty; interpolated time alone never certifies a transition.
    A historical numerical target is deliberately not used as a current one.
    """
    from .expert_roundabout_phase import _parse_log
    source = raw.get("transition_source") or {}
    receipt = raw.get("legacy_transition_observation") or {}
    if (raw.get("source") != "existing_roundabout_text_log"
            or receipt.get("transition_source") != source
            or receipt.get("frame_index") != frame["index"]
            or receipt.get("lidar_source_ns") != frame["lidar_source_ns"]
            or not source.get("path") or not isinstance(source.get("line_no"), int)):
        return None, "current_roundabout_follow_plan_source_missing"
    try:
        anchors, _, transitions = _parse_log(source["path"])
    except (OSError, ValueError):
        return None, "current_roundabout_follow_plan_source_unreadable"
    selected = [(i, line, item) for i, (line, item) in enumerate(transitions)
                if line == source["line_no"]]
    if len(selected) != 1:
        return None, "current_roundabout_follow_plan_source_line_ambiguous"
    pos, line, recorded = selected[0]
    if (recorded["transition_source"] != source
            or any(raw.get(k) != value for k, value in recorded.items())):
        return None, "current_roundabout_follow_plan_source_tuple_disagrees"
    before = max((a for a in anchors if a[0] <= line), default=None)
    after = min((a for a in anchors if a[0] >= line), default=None)
    next_line = transitions[pos+1][0] if pos+1 < len(transitions) else None
    next_before = (max((a for a in anchors if a[0] <= next_line), default=None)
                   if next_line is not None else None)
    latest = max((a[1] for a in anchors), default=None)
    t, past_t = frame["lidar_source_ns"]/1e9, past_start_ns/1e9
    if (before is None or after is None or before == after or latest is None
            or after[1] <= before[1]
            or past_t < after[1] or t > latest
            or (next_line is not None and (next_before is None or t >= next_before[1]))):
        return None, "current_roundabout_follow_plan_transition_time_unverified"
    if (receipt.get("transition_anchor_lower_s") != before[1]
            or receipt.get("transition_anchor_upper_s") != after[1]
            or receipt.get("next_transition_anchor_lower_s") != (next_before[1] if next_before else None)
            or receipt.get("numeric_target_current_verified") is not False):
        return None, "current_roundabout_follow_plan_anchor_receipt_disagrees"
    return {"transition_source": source, "phase": recorded["roundabout_phase"],
            "selected_actor_id": recorded["roundabout_front_id"],
            "transition_anchor_lower_s": before[1], "transition_anchor_upper_s": after[1],
            "next_transition_anchor_lower_s": next_before[1] if next_before else None,
            "last_log_anchor_s": latest, "entire_past_window_within_phase_bounds": True,
            "numeric_target_current_verified": False,
            "physical_simultaneity_verified": False}, None


def _expert_follow_plan_decision(rule, tables):
    """Describe a source-bounded Expert follow plan, not a safe-entry claim."""
    from .expert_instructions import _roundabout_actor, _roundabout_target_proof
    raw = rule.get("expert_raw_state") or {}
    if (rule.get("expert_state") not in ("FOLLOW_MERGE", "FOLLOW")
            or raw.get("mode") != "roundabout_entry" or raw.get("roundabout_phase") != "follow"):
        return None, "current_roundabout_follow_plan_state_unverified"
    # A GT-rule conflict uncertainty cannot veto a separately recorded plan,
    # but masks or unrelated source/semantic exclusions are never cleared.
    if (rule.get("action_valid") is False or rule.get("observation_valid") is False
            or (rule.get("expert_alignment") or {}).get("status") != "matched"
            or set(rule.get("unknown_reasons") or ()) - {"conflict_response_unverified"}
            or rule.get("status") not in (None, "matched", "unknown")):
        return None, "current_roundabout_follow_plan_upstream_unverified"
    motion, reason = _past_motion(rule.get("index"), tables)
    if reason:
        return None, reason
    if min(motion["backward_speeds_mps"]) <= STOP_SPEED_MPS:
        return None, "current_roundabout_follow_plan_current_motion_unverified"
    frame = tables[0][rule["index"]][0]
    source, reason = _legacy_phase_source_proof(frame, raw, motion["past_start_ns"])
    if reason:
        return None, reason
    target = _roundabout_target_proof(frame, raw, tables[2].get(frame["index"], ()))
    if (not target or not target.get("valid") or target.get("forward_m", -1) <= 0
            or target != (rule.get("evidence") or {}).get("roundabout_expert_target")):
        return None, "current_roundabout_follow_plan_current_target_unverified"
    observations = []
    for index in range(motion["past_start_index"], frame["index"]+1):
        past_frame, actors = tables[0][index][0], tables[2].get(index, ())
        _, reason = _current_inventory(past_frame, actors)
        actor = _roundabout_actor(past_frame, actors, target["selected_actor_id"])
        if (reason or actor is None or actor.get("instance_key") != target["instance_key"]
                or not all(_finite(actor.get(k)) for k in ("world_x_m", "world_y_m"))):
            return None, "current_roundabout_follow_plan_past_actor_unverified"
        world_distance = math.hypot(actor["world_x_m"]-past_frame["ego_world_x_m"],
                                    actor["world_y_m"]-past_frame["ego_world_y_m"])
        if index == frame["index"] and not math.isclose(
                world_distance, target["distance_m"], abs_tol=1e-5, rel_tol=1e-6):
            return None, "current_roundabout_follow_plan_local_world_distance_disagrees"
        observations.append({"index": index, "lidar_source_ns": past_frame["lidar_source_ns"],
                             "actor_id": str(actor["actor_id"]), "instance_key": actor["instance_key"],
                             "world_x_m": actor["world_x_m"], "world_y_m": actor["world_y_m"]})
    for first, last in zip(observations, observations[1:]):
        dt = (last["lidar_source_ns"]-first["lidar_source_ns"])/1e9
        if math.hypot(last["world_x_m"]-first["world_x_m"], last["world_y_m"]-first["world_y_m"])/dt > MAX_ENTRY_ACTOR_SPEED_MPS:
            return None, "current_roundabout_follow_plan_actor_discontinuous"
    return dict(motion, schema=SCHEMA, supervision_basis=BASIS, valid=True,
                decision=EXPERT_FOLLOW_PLAN_DECISION, expert_state="FOLLOW", instruction_id="follow",
                geometry_phase=frame.get("roundabout_phase"), current_expert_target=target,
                target_past_observations=observations, qualitative_plan_source=source,
                collision_risk=frame.get("collision_risk"), uses_future_ego=False,
                uses_future_actor=False, physical_front_safety_certified=False,
                instruction_semantics="expert_selected_current_follow_plan"), None


def _entry_follow_decision(rule, tables):
    """Recompute an entry FOLLOW tuple without certifying a safe merge.

    The old before-merge boolean maps entry to FOLLOW_MERGE. Only a fresh
    geometric lead matching the Expert-selected identity permits a generic
    Follow description. The current actor must also have a contiguous admitted
    past track; no adjacent instruction or future actor observation is used.
    """
    from .expert_instructions import (_roundabout_actor, _roundabout_target_proof,
                                      canonical_state, enrich_roundabout_current_observation,
                                      load_config)
    if (rule.get("zone_id") != "roundabout"
            or rule.get("expert_state") not in ("FOLLOW_MERGE", "FOLLOW")
            or (rule.get("expert_raw_state") or {}).get("roundabout_phase") != "follow"
            or (rule.get("expert_raw_state") or {}).get("mode") != "roundabout_entry"):
        return None, "current_roundabout_entry_follow_state_unverified"
    if (rule.get("action_valid") is False or rule.get("observation_valid") is False
            or rule.get("status") not in (None, "matched")
            or rule.get("unknown_reasons")
            or (rule.get("expert_alignment") or {}).get("status") != "matched"):
        return None, "current_roundabout_entry_follow_upstream_unverified"
    motion, reason = _past_motion(rule.get("index"), tables)
    if reason:
        return None, reason
    if (motion["current_speed_mps"] <= STOP_SPEED_MPS
            or min(motion["backward_speeds_mps"]) <= STOP_SPEED_MPS):
        return None, "current_roundabout_entry_follow_motion_unverified"
    if motion["backward_speeds_mps"][0] - motion["current_speed_mps"] >= MAX_WINDOW_DECEL_MPS:
        return None, "current_roundabout_entry_follow_deceleration_unverified"
    frame = tables[0][rule["index"]][0]
    if frame.get("roundabout_phase") != "entry":
        return None, "current_roundabout_entry_follow_phase_unverified"
    if frame.get("collision_risk") is not False or frame.get("conflict_actor_id") is not None:
        return None, "current_roundabout_entry_follow_conflict_unverified"
    # Independently recover the original ordered mapping. Stop/deceleration and
    # other raw Expert semantics must not be promoted to generic Follow.
    raw = rule["expert_raw_state"]
    match, _ = enrich_roundabout_current_observation(frame, raw)
    match["roundabout_before_merge"] = True
    match["roundabout_resume_detected"] = False
    entry, _ = canonical_state("roundabout", match, load_config()[0])
    if entry is None or entry.get("state") != "FOLLOW_MERGE":
        return None, "current_roundabout_entry_follow_raw_mapping_disagrees"
    current_actors = tables[2].get(frame["index"], ())
    target = _roundabout_target_proof(frame, raw, current_actors)
    if (not target or target.get("valid") is not True
            or target.get("geometric_target_matches") is not True
            or target != (rule.get("evidence") or {}).get("roundabout_expert_target")
            or target.get("forward_m", -1) <= 0
            or not _finite(target.get("distance_m"))
            or target["distance_m"] > MAX_ENTRY_TARGET_DISTANCE_M):
        return None, "current_roundabout_entry_follow_current_lead_unverified"
    for actor in current_actors:
        distance = actor.get("distance_to_ego_m")
        if not _finite(distance):
            return None, "current_roundabout_entry_follow_nearby_position_unverified"
        if distance <= MAX_ENTRY_TARGET_DISTANCE_M and (
                actor.get("rule_geometry_usable") is not True
                or actor.get("model_identity_verified") is not True
                or actor.get("semantic_class_valid") is not True):
            return None, "current_roundabout_entry_follow_nearby_geometry_unverified"
    selected_id, observations = target["selected_actor_id"], []
    for index in range(motion["past_start_index"], frame["index"] + 1):
        past_frame = tables[0][index][0]
        past_actors = tables[2].get(index, ())
        _, reason = _current_inventory(past_frame, past_actors)
        if reason:
            return None, "current_roundabout_entry_follow_past_inventory_unverified"
        actor = _roundabout_actor(past_frame, past_actors, selected_id)
        if (actor is None or actor.get("instance_key") != target["instance_key"]
                or not all(_finite(actor.get(k)) for k in ("world_x_m", "world_y_m"))):
            return None, "current_roundabout_entry_follow_past_actor_unverified"
        observations.append({"index": index, "lidar_source_ns": past_frame["lidar_source_ns"],
                             "actor_id": str(selected_id), "instance_key": actor["instance_key"],
                             "world_x_m": actor["world_x_m"], "world_y_m": actor["world_y_m"]})
    for first, last in zip(observations, observations[1:]):
        dt = (last["lidar_source_ns"] - first["lidar_source_ns"]) / 1e9
        speed = math.hypot(last["world_x_m"] - first["world_x_m"],
                           last["world_y_m"] - first["world_y_m"]) / dt
        if speed > MAX_ENTRY_ACTOR_SPEED_MPS:
            return None, "current_roundabout_entry_follow_actor_discontinuous"
    actor = _roundabout_actor(frame, current_actors, selected_id)
    dx, dy = actor["world_x_m"] - frame["ego_world_x_m"], actor["world_y_m"] - frame["ego_world_y_m"]
    # Audited body-local geometry uses captured pose. The feature heading is
    # backward XY motion, which need not have exactly the same yaw. Compare
    # rotation-invariant distance; do not falsely equate those two headings.
    world_distance = math.hypot(dx, dy)
    if not math.isclose(world_distance, target["distance_m"], abs_tol=1e-5, rel_tol=1e-6):
        return None, "current_roundabout_entry_follow_local_world_distance_disagrees"
    return dict(motion, schema=SCHEMA, supervision_basis=BASIS, valid=True,
                decision=ENTRY_FOLLOW_DECISION, entry_follow_policy="current_entry_follow_v1",
                original_expert_state="FOLLOW_MERGE", expert_state="FOLLOW", instruction_id="follow",
                geometry_phase="entry", current_expert_target=target,
                target_past_observations=observations, target_world_distance_m=world_distance,
                local_coordinate_source="sealed_current_actor_geometry",
                world_track_source="sealed_current_actor_positions", collision_risk=False,
                uses_future_ego=False, uses_future_actor=False,
                physical_front_safety_certified=False), None


def _decision(rule, tables):
    state = rule.get("expert_state")
    if ((rule.get("evidence") or {}).get(MARKER) or {}).get("decision") == PUBLICATION_DRIVE_DECISION:
        return _publication_drive_decision(rule, tables)
    if (state == "FOLLOW" and ((rule.get("evidence") or {}).get(MARKER) or {}).get("decision") == EXPERT_FOLLOW_PLAN_DECISION):
        return _expert_follow_plan_decision(rule, tables)
    if (state == "FOLLOW"
            and ((rule.get("evidence") or {}).get(MARKER) or {}).get("decision") == ENTRY_FOLLOW_DECISION):
        return _entry_follow_decision(rule, tables)
    raw_phase = "follow" if state == "FOLLOW_MERGE" else "cruise"
    if (rule.get("zone_id") != "roundabout" or rule.get("expert_state") not in STATES
            or (rule.get("expert_raw_state") or {}).get("roundabout_phase") != raw_phase
            or (rule.get("expert_raw_state") or {}).get("mode") != "roundabout_entry"):
        return None, "not_roundabout_expert_cruise"
    if (rule.get("observation_valid") is False or rule.get("action_valid") is False
            or (rule.get("expert_alignment") or {}).get("status") != "matched"):
        return None, "current_roundabout_upstream_mask_or_source_unverified"
    motion, reason = _past_motion(rule.get("index"), tables)
    if reason:
        return None, reason
    if motion["current_speed_mps"] <= STOP_SPEED_MPS:
        return None, ("current_roundabout_merge_while_stopped" if state == "FOLLOW_MERGE"
                      else "current_roundabout_cruise_while_stopped")
    if min(motion["backward_speeds_mps"]) <= STOP_SPEED_MPS:
        return None, "current_roundabout_departure_not_sustained"
    if motion["backward_speeds_mps"][0] - motion["current_speed_mps"] >= MAX_WINDOW_DECEL_MPS:
        return None, "current_roundabout_cruise_deceleration_unverified"
    frame = tables[0][rule["index"]][0]
    phase = frame.get("roundabout_phase")
    phase_state = {"approach": "CRUISE_APPROACH", "entry": "CRUISE_APPROACH",
                   "merge": "CRUISE_MERGE", "circulating": "CRUISE_AFTER_MERGE",
                   "exit": "CRUISE_AFTER_MERGE"}.get(phase)
    if state != "FOLLOW_MERGE" and phase_state is not None and state != phase_state:
        return None, "current_roundabout_cruise_state_phase_disagrees"
    if state == "CRUISE_APPROACH" and phase not in ("approach", "entry"):
        return None, "current_roundabout_entry_phase_unverified"
    if state in ("CRUISE_MERGE", "FOLLOW_MERGE") and phase != "merge":
        return None, "current_roundabout_merge_phase_unverified"
    if state == "CRUISE_AFTER_MERGE" and phase not in ("circulating", "exit"):
        return None, "current_roundabout_after_merge_phase_unverified"
    entry_states = ("CRUISE_APPROACH", "CRUISE_MERGE", "FOLLOW_MERGE")
    if state in entry_states and frame.get("collision_risk") is not False:
        return None, "current_roundabout_entry_conflict_unverified"
    if state in entry_states:
        if frame.get("conflict_actor_id") is not None:
            return None, "current_roundabout_entry_conflict_present"
        for actor in tables[2].get(frame["index"], ()):
            distance = actor.get("distance_to_ego_m")
            if not _finite(distance):
                forward, left = actor.get("forward_m"), actor.get("left_m")
                if all(_finite(v) for v in (forward, left)):
                    distance = math.hypot(forward, left)
            if not _finite(distance):
                return None, "current_roundabout_entry_actor_position_unverified"
            if distance <= 35. and (actor.get("rule_geometry_usable") is not True
                    or actor.get("model_identity_verified") is not True
                    or actor.get("semantic_class_valid") is not True):
                return None, "current_roundabout_entry_actor_geometry_unverified"
    target_proof = None
    if state == "FOLLOW_MERGE":
        from .expert_instructions import _roundabout_target_proof
        target_proof = _roundabout_target_proof(frame, rule["expert_raw_state"],
                                              tables[2].get(frame["index"], ()))
        recorded = (rule.get("evidence") or {}).get("roundabout_expert_target")
        if not target_proof or target_proof.get("valid") is not True or recorded != target_proof:
            return None, "current_roundabout_merge_target_unverified"
    return dict(motion, schema=SCHEMA, supervision_basis=BASIS, valid=True,
                expert_state=state, instruction_id=STATES[state], geometry_phase=phase,
                current_expert_target=target_proof,
                collision_risk=frame.get("collision_risk"), uses_future_ego=False,
                uses_future_actor=False, physical_front_safety_certified=False), None


def align_current_roundabout_cruise(frames, objects, rules):
    """Fail closed unsupported moving CRUISE claims; never replace with Wait."""
    tables = _tables(frames, objects)
    counts = Counter()
    for rule in rules:
        if rule.get("zone_id") != "roundabout" or rule.get("expert_state") not in STATES:
            continue
        previous = (rule.get("evidence") or {}).get(MARKER)
        if previous is not None:
            counts["already_guarded"] += 1
            continue
        frame = tables[0].get(rule.get("index"), ())
        legacy = ((rule.get("expert_raw_state") or {}).get("legacy_transition_observation") or {})
        if rule.get("expert_state") == "FOLLOW_MERGE" and legacy:
            proof, reason = _expert_follow_plan_decision(rule, tables)
            if proof is None:
                rule.setdefault("evidence", {})["current_expert_follow_plan_rejection"] = {"reason": reason}
                # Independently published path/control can support a weaker
                # current path-plan sentence. They cannot recover the missing
                # phase timestamp or certify the selected follow actor.
                proof, reason = _publication_drive_decision(rule, tables)
        elif (rule.get("expert_state") == "FOLLOW_MERGE" and len(frame) == 1
                and frame[0].get("roundabout_phase") == "entry"):
            proof, reason = _entry_follow_decision(rule, tables)
            if proof is None:
                rule.setdefault("evidence", {})["current_entry_follow_rejection"] = {
                    "policy": "current_entry_follow_v1", "reason": reason,
                    "uses_future_ego": False, "uses_future_actor": False}
                # The new optional mapping must not weaken or change the
                # established stopped/phase exclusion when it cannot prove
                # a generic Follow tuple.
                proof, reason = _decision(rule, tables)
        else:
            proof, reason = _decision(rule, tables)
        if proof is not None:
            if proof.get("decision") == PUBLICATION_DRIVE_DECISION:
                template = load_catalog()["roundabout"][proof["instruction_id"]]
                rule.setdefault("instruction_history", []).append({
                    "pass": MARKER, "decision": PUBLICATION_DRIVE_DECISION,
                    "expert_state": rule.get("expert_state"), "instruction": rule.get("instruction"),
                    "instruction_id": rule.get("instruction_id")})
                rule.update(expert_state=None, instruction_state=proof["instruction_id"], label=None,
                            instruction_id=proof["instruction_id"], instruction=template["instruction"],
                            instruction_variables={}, instruction_source="rule_state", status="matched",
                            instruction_mapping_status="rendered")
                rule["unknown_reasons"] = [r for r in rule.get("unknown_reasons", [])
                                           if r != "conflict_response_unverified"]
                counts[PUBLICATION_DRIVE_DECISION] += 1
            elif proof.get("decision") in (ENTRY_FOLLOW_DECISION, EXPERT_FOLLOW_PLAN_DECISION):
                template = load_catalog()["roundabout"]["follow"]
                text, values, missing = render_template(template, {"position": proof["current_expert_target"]["position"]})
                assert not missing
                rule.setdefault("instruction_history", []).append({
                    "pass": MARKER, "decision": proof["decision"],
                    "expert_state": rule.get("expert_state"),
                    "instruction": rule.get("instruction"), "instruction_id": rule.get("instruction_id")})
                rule.update(expert_state="FOLLOW", instruction_id="follow", label=template["label"],
                            instruction=text, instruction_variables=values,
                            instruction_mapping_status="rendered")
                counts[proof["decision"]] += 1
            rule.setdefault("evidence", {})[MARKER] = proof
            counts["current_cruise_verified"] += 1
            continue
        rule.setdefault("evidence", {})[MARKER] = {
            "schema": SCHEMA, "supervision_basis": BASIS, "valid": False,
            "frame_index": rule.get("index"), "reason": reason,
            "uses_future_ego": False, "uses_future_actor": False,
            "physical_front_safety_certified": False}
        rule.setdefault("instruction_history", []).append({
            "pass": MARKER, "instruction": rule.get("instruction"),
            "instruction_id": rule.get("instruction_id"), "reason": reason})
        rule["instruction"] = None
        rule["action_valid"] = False
        rule["instruction_consistency"] = "excluded_current_roundabout_cruise"
        if reason not in rule.setdefault("unknown_reasons", []):
            rule["unknown_reasons"].append(reason)
        counts[reason] += 1
    return dict(sorted(counts.items()))


def validate_current_roundabout_cruise(frames, objects, rule):
    """Recompute the current proof and final rendered tuple independently."""
    proof, reason = _decision(rule, _tables(frames, objects))
    if proof is None:
        return False, [reason]
    marker = (rule.get("evidence") or {}).get(MARKER)
    if marker != proof:
        return False, ["current_roundabout_cruise_receipt_disagrees"]
    expected = load_catalog()["roundabout"][proof["instruction_id"]]
    values = ({"position": proof["current_expert_target"]["position"]}
              if proof["current_expert_target"] is not None else {})
    text, slots, missing = render_template(expected, values)
    if (rule.get("instruction_id") != proof["instruction_id"]
            or rule.get("label") != expected.get("label")
            or missing or rule.get("instruction") != text
            or rule.get("instruction_variables") != slots
            or rule.get("instruction_source") != (
                "rule_state" if proof.get("decision") == PUBLICATION_DRIVE_DECISION
                else "expert_observation")):
        return False, ["current_roundabout_cruise_final_rendering_disagrees"]
    return True, []
