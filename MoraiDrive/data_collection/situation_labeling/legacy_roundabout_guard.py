"""Reject specifically unsupported actor actions from interpolated legacy logs.

This read-only gate neither repairs a sentence nor infers an action from nearby
frames. Unknown historical numeric targets alone do not invalidate an action.
"""

LEGACY_ACTOR_ACTIONS = frozenset((
    "yield_moving", "yield_stopped", "collision_slowdown", "collision_stopped",
))
PHASE_REASON = "legacy_roundabout_phase_time_unverified"
ACTOR_REASON = "legacy_roundabout_actor_response_unverified"
CONFLICT_REASON = "legacy_roundabout_conflict_response_unverified"


def _validated_current_response(evidence):
    marker = evidence.get("current_roundabout_cruise") or {}
    consistency = evidence.get("current_phase_consistency") or {}
    return (marker.get("schema") == "roundabout_current_cruise_v1"
            and marker.get("supervision_basis") == "recorded_current_roundabout_motion"
            and marker.get("valid") is True
            and marker.get("uses_future_ego") is False
            and consistency.get("valid") is True
            and consistency.get("recomputed_from_current_evidence") is True
            and consistency.get("uses_future_ego") is False)


def legacy_roundabout_action_reasons(rule):
    """Return exclusions for an unbounded legacy phase with unsupported cause.

The current response pass and its independent validator take precedence. This
gate only covers final actor-action wording still sourced from legacy Expert
logs, with an explicitly unverified transition time and an independently
recorded actor mismatch or unknown conflict response.
    """
    if (rule.get("zone_id") != "roundabout"
            or rule.get("instruction_source") != "expert_observation"
            or rule.get("instruction_id") not in LEGACY_ACTOR_ACTIONS):
        return []
    evidence = rule.get("evidence") or {}
    if _validated_current_response(evidence):
        return []
    raw = rule.get("expert_raw_state") or {}
    transition = raw.get("legacy_transition_observation") or {}
    if (raw.get("source") != "existing_roundabout_text_log"
            or transition.get("schema") != "roundabout_transition_observation_v1"
            or transition.get("frame_index") != rule.get("index")
            or transition.get("phase_within_anchor_bounds") is not False):
        return []
    reasons = []
    actor = evidence.get("roundabout_expert_target") or {}
    selected, current = actor.get("selected_actor_id"), actor.get("current_actor_id")
    if (actor.get("valid") is True
            and actor.get("geometric_target_matches") is False
            and selected is not None and current is not None
            and str(selected) != str(current)):
        reasons.append(ACTOR_REASON)
    if (rule.get("status") == "unknown"
            and "conflict_response_unverified" in (rule.get("unknown_reasons") or [])):
        reasons.append(CONFLICT_REASON)
    return sorted([PHASE_REASON] + reasons) if reasons else []
