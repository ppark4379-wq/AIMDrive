"""Resolve a stopped sentence against a fresh captured departure command.

The imperative describes current Expert intent. It does not claim that Ego has
already moved, certify a safe entry, attribute a hazard, or infer a transition
through an unsupported historical target. Original stop evidence stays in the
receipt and is independently rechecked before accepting the replacement.
"""
from collections import Counter
import copy
import math

POLICY = "current_departure_plan_v1"
MARKER = "current_departure_plan"
INSTRUCTION_ID = "current_departure_plan"
INSTRUCTION = "Resume driving."
ORIGINAL_IDS = {"dynamic_obstacle": "dynamic_wait_clear", "roundabout": "yield_stopped"}
MAX_COMMAND_AGE_NS = 100_000_000
MASKS = ("valid", "frame_valid", "dataset_build_candidate", "object_inventory_usable",
         "observation_valid", "action_valid", "target_semantic_valid", "semantic_valid")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _masked(rule):
    return any(scope.get(key) is False
               for scope in (rule, rule.get("variables") or {}, rule.get("evidence") or {})
               for key in MASKS)


def _receipt(receipt, sensor_ns, anchor_ns, ros_type):
    if not isinstance(receipt, dict):
        return False
    clock, received, age = (receipt.get(k) for k in
                            ("preceding_received_clock_ns", "receive_monotonic_ns", "receive_age_ms"))
    return (type(clock) is int and type(received) is int
            and 0 <= sensor_ns - clock <= MAX_COMMAND_AGE_NS
            and 0 <= anchor_ns - received <= MAX_COMMAND_AGE_NS
            and _finite(age) and 0 <= age <= MAX_COMMAND_AGE_NS / 1e6
            and abs(age - (anchor_ns - received) / 1e6) <= 1e-6
            and type(receipt.get("record_id")) is int and receipt["record_id"] >= 0
            and _hash(receipt.get("payload_sha256"))
            and receipt.get("ros_type") == ros_type
            and receipt.get("source_timestamp_fabricated") is False)


def _publication_bound(frame):
    pub = frame.get("expert_publications") or {}
    sensor_ns, anchor_ns, source_pins = (frame.get("lidar_source_ns"),
                                       pub.get("lidar_receive_monotonic_ns"), pub.get("source_sha256"))
    return (type(sensor_ns) is int and pub.get("status") == "RECEIVE_BOUND_CURRENT_CONTEXT"
            and pub.get("sensor_ns") == sensor_ns and type(anchor_ns) is int
            and type(pub.get("lidar_record_id")) is int
            and isinstance(source_pins, dict) and bool(source_pins)
            and all(isinstance(k, str) and k and _hash(v) for k, v in source_pins.items()))


def current_instruction_command_reasons(frame, rule):
    """Reject explicit progress wording contradicted by current commands.

    Resume retains its independent hard-brake guard. For Continue, require both
    a fresh hard brake and a fresh zero target: braking while still requesting
    forward travel does not by itself disprove Continue. This closes the motion
    check's 1--2 m/s boundary without guessing a stop cause or replacing text.
    Lateral avoidance and generic Maintain wording are deliberately separate.
    """
    if not isinstance(rule, dict):
        return []
    zone, text = rule.get("zone_id"), rule.get("instruction")
    resume = zone == "dynamic_obstacle" and rule.get("instruction_id") == "dynamic_resume" and bool(text)
    continuing = zone in ("dynamic_obstacle", "static_obstacle", "roundabout") and text in (
        "Continue driving.", "Continue driving along the current path.")
    if (not isinstance(frame, dict) or frame.get("index") != rule.get("index")
            or frame.get("zone_id") != zone or not (resume or continuing)
            or any(frame.get(k) is not True for k in
                   ("valid", "dataset_build_candidate", "object_inventory_usable"))
            or not _publication_bound(frame)):
        return []
    pub = frame["expert_publications"]
    control = pub.get("control")
    if not _receipt(control, frame["lidar_source_ns"], pub["lidar_receive_monotonic_ns"], "morai_msgs/CtrlCmd"):
        return []
    accel, brake = control.get("accel"), control.get("brake")
    if (type(control.get("longlCmdType")) is int and control["longlCmdType"] == 1
            and _finite(accel) and _finite(brake) and 0 <= accel <= .001 and .7 <= brake <= 1):
        if resume:
            return ["current_dynamic_resume_hard_hold_command_conflict"]
        target = pub.get("target_velocity")
        if (_receipt(target, frame["lidar_source_ns"], pub["lidar_receive_monotonic_ns"], "std_msgs/Float32")
                and _finite(target.get("value_mps")) and 0 <= target["value_mps"] <= .01):
            return ["current_{}_continue_hard_hold_command_conflict".format(zone)]
    return []


def _original_stop_valid(frames, objects, rule, frame):
    if frame["zone_id"] == "dynamic_obstacle":
        evidence = rule.get("evidence") or {}
        if evidence.get("current_dynamic_response"):
            from .dynamic_resume import validate_current_dynamic_response
            return validate_current_dynamic_response(frames, objects, rule)
        if evidence.get("current_dynamic_normal"):
            from .dynamic_normal import validate_current_dynamic_normal
            return validate_current_dynamic_normal(frames, objects, rule)
        return False, ["current_departure_original_dynamic_stop_unverified"]
    # The new command sentence removes the actor/cause claim, but never uses
    # that removal to rescue an ineligible or stale original target join.
    from .expert_instructions import _roundabout_target_proof
    raw, alignment = rule.get("expert_raw_state") or {}, rule.get("expert_alignment") or {}
    if (rule.get("instruction_source") != "expert_observation"
            or alignment.get("status") != "matched" or raw.get("roundabout_phase") != "yield"
            or rule.get("label") != "yield"):
        return False, ["current_departure_original_roundabout_source_unverified"]
    actors = [a for a in objects if a.get("index") == frame["index"]]
    target = _roundabout_target_proof(frame, raw, actors)
    recorded = (rule.get("evidence") or {}).get("roundabout_expert_target")
    if not target or target.get("valid") is not True or target != recorded:
        return False, ["current_departure_original_roundabout_target_unverified"]
    return True, []


def _decision(frames, objects, rule):
    matches = [f for f in frames if f.get("index") == rule.get("index")]
    if len(matches) != 1 or objects is None:
        return None, ["current_departure_frame_or_inventory_missing"]
    frame = matches[0]
    zone, sensor_ns = frame.get("zone_id"), frame.get("lidar_source_ns")
    if (zone not in ORIGINAL_IDS or rule.get("zone_id") != zone
            or rule.get("instruction_id") != ORIGINAL_IDS[zone]
            or rule.get("status") != "matched" or not rule.get("instruction")
            or rule.get("unknown_reasons") or _masked(rule) or _masked(frame)
            or any(frame.get(k) is not True for k in
                   ("valid", "dataset_build_candidate", "object_inventory_usable"))
            or type(sensor_ns) is not int or type(frame.get("index")) is not int):
        return None, ["current_departure_original_frame_or_rule_ineligible"]
    original_valid, reasons = _original_stop_valid(frames, objects, rule, frame)
    if not original_valid:
        return None, reasons
    pub = frame.get("expert_publications") or {}
    anchor_ns, source_pins = pub.get("lidar_receive_monotonic_ns"), pub.get("source_sha256")
    if not _publication_bound(frame):
        return None, ["current_departure_publication_anchor_unverified"]
    control, target = pub.get("control"), pub.get("target_velocity")
    if (not _receipt(control, sensor_ns, anchor_ns, "morai_msgs/CtrlCmd")
            or not _receipt(target, sensor_ns, anchor_ns, "std_msgs/Float32")):
        return None, ["current_departure_command_receipt_unverified"]
    accel, brake, target_speed = control.get("accel"), control.get("brake"), target.get("value_mps")
    if (type(control.get("longlCmdType")) is not int or control["longlCmdType"] != 1
            or not all(_finite(v) for v in (accel, brake, target_speed))
            or not .2 < accel <= 1 or not 0 <= brake <= 1e-6 or not target_speed > .5):
        return None, ["current_departure_command_not_unambiguous"]
    # Full receipts and input hashes are compared again by the final validator.
    # A historical signed target is provenance, never a substituted command.
    return dict(schema=POLICY, instruction_id=INSTRUCTION_ID,
                supervision_basis="current_expert_departure_command",
                source_sha256=copy.deepcopy(source_pins), sensor_ns=sensor_ns,
                lidar_record_id=pub["lidar_record_id"], lidar_receive_monotonic_ns=anchor_ns,
                control=copy.deepcopy(control), target_velocity=copy.deepcopy(target),
                original_stop=copy.deepcopy(rule), uses_future_ego=False,
                observed_departure_claimed=False, physical_front_safety_certified=False,
                physical_sensor_synchrony_certified=False, hazard_cause_verified=False,
                upstream_mask_promotion=False,
                timing_basis="causal receipt and preceding received simulation-clock bounds; no fabricated Expert source time"), []


def align_current_departure_plans(frames, objects, rules):
    """Replace only eligible stopped wording; return operation counters.

    Call after the stopped-variant pass and before the final consistency pass.
    Every upstream false mask and unknown reason is preserved without changes.
    """
    if len({f["index"] for f in frames}) != len(frames) or len({r["index"] for r in rules}) != len(rules):
        raise ValueError("Duplicate current departure frame or rule index")
    counts = Counter()
    for rule in rules:
        if rule.get("instruction_id") != ORIGINAL_IDS.get(rule.get("zone_id")):
            continue
        decision, reasons = _decision(frames, objects, rule)
        if decision is None:
            counts.update(reasons)
            continue
        rule.setdefault("evidence", {})[MARKER] = decision
        rule.setdefault("instruction_postprocessing_history", []).append({
            "operation": POLICY, "before_instruction_id": rule["instruction_id"],
            "before_instruction_state": rule.get("instruction_state"),
            "after_instruction_id": INSTRUCTION_ID, "uses_future_ego": False})
        rule.update(instruction_state=INSTRUCTION_ID, instruction_id=INSTRUCTION_ID,
                    instruction=INSTRUCTION, instruction_variables={}, instruction_source="rule_state",
                    instruction_mapping_status="rendered", expert_state=None, label=None,
                    label_candidate=None, related_actors=[])
        counts["verified_current_departure_plan"] += 1
    return dict(sorted(counts.items()))


def validate_current_departure_plan(frames, objects, rule):
    """Recheck original eligibility, current commands and final rendering."""
    marker = (rule.get("evidence") or {}).get(MARKER) or {}
    original = marker.get("original_stop")
    if (marker.get("schema") != POLICY or not isinstance(original, dict)
            or rule.get("instruction_id") != INSTRUCTION_ID or rule.get("instruction_state") != INSTRUCTION_ID
            or rule.get("instruction") != INSTRUCTION or rule.get("instruction_variables") != {}
            or rule.get("instruction_source") != "rule_state" or rule.get("status") != "matched"
            or rule.get("expert_state") is not None or rule.get("label") is not None
            or rule.get("label_candidate") is not None or rule.get("related_actors") != []
            or rule.get("index") != original.get("index") or rule.get("zone_id") != original.get("zone_id")
            or rule.get("unknown_reasons") or _masked(rule)):
        return False, ["current_departure_final_rule_or_marker_unverified"]
    expected, reasons = _decision(frames, objects, original)
    if expected is None:
        return False, reasons
    if expected != marker:
        return False, ["current_departure_recorded_command_evidence_disagrees"]
    return True, []
