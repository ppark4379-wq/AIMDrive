"""Conservative offline checks for the GPS/VE driving journal."""

import math


def _check(identifier, status, finding, rows=(), missing=(), module=None, code=None):
    return dict(check_id=identifier, status=status, finding=finding,
                rule="same run_id, host clock and explicit cycle lineage",
                data_quality_requirement="valid stage records and clock identity",
                evidence=[dict(file=stage + "_trace.jsonl", line=line) for stage, line in rows],
                time_range=None, next_check=code, suspected_module=module,
                missing_data=list(missing), observed_facts=[finding] if status == "OBSERVED" else [])


def _same_cycle(rows, cycle):
    return next(((i, row) for i, row in reversed(list(enumerate(rows or [], 1)))
                 if row.get("cycle_id") == cycle), None)


def _path_hits_box(path, box):
    """Exact segment vs current OBB footprint; no forecast of moving actors."""
    cx, cy = box["position_xy"]
    half_x, half_y = box["length_m"] / 2, box["width_m"] / 2
    angle = math.radians(box["heading_deg"])
    cosine, sine = math.cos(angle), math.sin(angle)
    local = [((x-cx)*cosine+(y-cy)*sine, -(x-cx)*sine+(y-cy)*cosine)
             for x, y in path]
    for (x0, y0), (x1, y1) in zip(local, local[1:]):
        lower, upper = 0., 1.
        for origin, delta, half in ((x0, x1-x0, half_x), (y0, y1-y0, half_y)):
            if delta == 0:
                if abs(origin) > half:
                    break
                continue
            a, b = sorted(((-half-origin)/delta, (half-origin)/delta))
            lower, upper = max(lower, a), min(upper, b)
            if lower > upper:
                break
        else:
            return True
    return False


def _diagnose_pipeline(run_id, failure, perception, planning, control, errors):
    insufficient = "INSUFFICIENT_EVIDENCE"
    observed = "OBSERVED"
    suspected = "SUSPECTED"
    result = []
    identities_valid = True
    for stage, rows in (("perception", perception), ("planning", planning), ("control", control)):
        if errors.get(stage) or not rows:
            result.append(_check("pipeline." + stage + ".quality", insufficient,
                                 stage + " log unavailable or damaged", missing=[stage + "_trace.jsonl"],
                                 code="Check driver pipeline logging"))
            continue
        valid = all(row.get("run_id") == run_id and row.get("clock_domain") == "host_monotonic_ns"
                    and type(row.get("timestamp_ns")) is int and type(row.get("sequence")) is int
                    for row in rows)
        valid = valid and all(rows[i]["timestamp_ns"] >= rows[i-1]["timestamp_ns"] and
                              rows[i]["sequence"] > rows[i-1]["sequence"]
                              for i in range(1, len(rows)))
        identities_valid = identities_valid and valid
        result.append(_check("pipeline." + stage + ".quality", observed if valid else insufficient,
                             stage + (" identity and clock verified" if valid else " identity or clock mismatch"),
                             rows=[(stage, 1)], missing=[] if valid else ["common run_id and clock"]))
    if not identities_valid or any(errors.get(stage) or not rows for stage, rows in
           (("perception", perception), ("planning", planning), ("control", control))):
        result.append(_check("perception.delivery", insufficient, "Stage logs cannot be linked",
                             missing=["complete stage logs"]))
        result.append(_check("planning.footprint", insufficient, "No verified path and object geometry",
                             missing=["same-cycle path and boxes"]))
        result.append(_check("control.handoff", insufficient, "Planning and Control cannot be linked",
                             missing=["same-cycle stage logs"]))
        result.append(_check("control.ego_response", insufficient, "No command-to-Ego causal acknowledgement",
                             missing=["actuator command id or response latency contract"]))
        return result
    hosts = {row.get("clock_host") for rows in (perception, planning, control) for row in rows}
    if len(hosts) != 1:
        result.append(_check("pipeline.time_link", insufficient, "Pipeline stages have different clock hosts",
                             missing=["one shared producer host"])); return result
    cross_host = str(failure.get("pipeline_window_basis") or "").startswith("cross-host wall window")
    anchor = failure.get("pipeline_anchor_ns")
    end_ns = (anchor if cross_host and type(anchor) is int else
              int(failure.get("captured_monotonic_sec", 0) * 1e9))
    if end_ns <= 0 or any(row["timestamp_ns"] > end_ns for rows in (perception, planning, control) for row in rows):
        result.append(_check("pipeline.time_link", insufficient, "Host time cannot be linked to failure",
                             missing=["verified capture-to-pipeline time anchor"])); return result
    result.append(_check("pipeline.time_link", observed,
                         "Cross-host wall window verified; pipeline stages share producer monotonic clock"
                         if cross_host else "Same-host monotonic failure window verified",
                         rows=[("perception", len(perception)), ("planning", len(planning)), ("control", len(control))]))
    p_line, p = len(perception), perception[-1]
    matched_plan = _same_cycle(planning, p.get("cycle_id"))
    matched_control = _same_cycle(control, p.get("cycle_id"))
    if not matched_plan or not matched_control:
        result.append(_check("perception.delivery", insufficient, "Final cycle has no linked plan/control",
                             missing=["matching cycle_id"])); return result
    l_line, plan = matched_plan
    c_line, ctrl = matched_control
    refs = [("perception", p_line), ("planning", l_line), ("control", c_line)]
    if p.get("ve_valid") is False:
        result.append(_check("perception.freshness", observed, "VE input explicitly invalid or expired", refs,
                             code="Inspect VE input gate and sensor receipts"))
    elif p.get("ve_valid") is True and isinstance(p.get("ve_received_ns"), int) and p.get("ve_sequence") == plan.get("ve_sequence"):
        result.append(_check("perception.freshness", observed, "Planner used the logged valid VE frame", refs))
    else:
        result.append(_check("perception.freshness", insufficient, "VE freshness/lineage unverified", refs,
                             missing=["VE receipt time and frame sequence"]))
    source = p.get("objects")
    used = plan.get("input_objects")
    if p.get("ve_valid") is True and isinstance(source, list) and isinstance(used, list) and p.get("ve_sequence") == plan.get("ve_sequence"):
        missing_ids = sorted({box.get("track_id") for box in source if box.get("track_id") is not None} -
                             {box.get("track_id") for box in used if box.get("track_id") is not None})
        result.append(_check("perception.delivery", suspected if missing_ids else observed,
                             "VE tracks absent from Planner input: " + str(missing_ids) if missing_ids else
                             "All logged VE tracks reached Planner input", refs,
                             module="VE-to-Planner adapter" if missing_ids else None,
                             code="Inspect planner_inputs()" if missing_ids else None))
    else:
        result.append(_check("perception.delivery", insufficient, "VE-to-Planner object comparison unavailable", refs,
                             missing=["same VE sequence and track IDs"]))
    path = plan.get("planned_trajectory")
    valid_path = (isinstance(path, list) and len(path) >= 2 and
                  all(isinstance(pt, list) and len(pt) >= 2 and
                      all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt[:2]) for pt in path))
    if not valid_path:
        result.append(_check("planning.path_generated", suspected if plan.get("started_ns") else insufficient,
                             "Planner returned no valid path", refs,
                             module="Planning" if plan.get("started_ns") else None,
                             code="Inspect AutonomousDriving.execute()"))
    else:
        result.append(_check("planning.path_generated", observed, "Planner produced a finite path", refs))
    static = [obj for obj in used or [] if isinstance(obj, dict) and obj.get("motion_state") == "stationary"]
    geometry = all(obj.get("coordinate_frame") == plan.get("coordinate_frame") and
                   isinstance(obj.get("position_xy"), list) and len(obj["position_xy"]) == 2 and
                   all(isinstance(obj.get(key), (int, float)) and math.isfinite(obj[key]) and obj[key] > 0
                       for key in ("length_m", "width_m")) and
                   isinstance(obj.get("heading_deg"), (int, float)) and math.isfinite(obj["heading_deg"])
                   for obj in static)
    if valid_path and static and geometry and plan.get("coordinate_frame") == "projected_grid":
        hits = [obj.get("track_id") for obj in static if _path_hits_box(path, obj)]
        result.append(_check("planning.footprint", suspected if hits else observed,
                             "Path overlaps current stationary VE footprint: " + str(hits) if hits else
                             "No overlap with logged stationary VE footprints", refs,
                             module="Planning" if hits else None,
                             code="Inspect lattice obstacle handling" if hits else None))
    else:
        result.append(_check("planning.footprint", insufficient, "Moving-object prediction or matched static geometry unavailable", refs,
                             missing=["validated stationary object geometry and same-frame path"] ))
    linked = ctrl.get("planning_cycle_id") == plan.get("cycle_id") and ctrl.get("planning_cycle_id") is not None
    if linked and ctrl.get("input_path") == path and ctrl.get("target_velocity_mps") == plan.get("target_velocity_mps"):
        result.append(_check("control.handoff", observed, "Planner path and target reached Control unchanged", refs))
    elif linked:
        result.append(_check("control.handoff", suspected, "Control input differs from linked Planner output", refs,
                             module="Planning-to-Control adapter", code="Inspect driver cycle handoff"))
    else:
        result.append(_check("control.handoff", insufficient, "Planning-to-Control cycle link unavailable", refs,
                             missing=["planning_cycle_id"]))
    calc = ctrl.get("calculated_command")
    wire = ctrl.get("final_wire_command")
    if isinstance(wire, dict) and all(isinstance(wire.get(key), (int, float)) and math.isfinite(wire[key])
                                      for key in ("accel", "brake", "steer_normalized")):
        invalid = not (0 <= wire["accel"] <= 1 and 0 <= wire["brake"] <= 1 and
                       -1 <= wire["steer_normalized"] <= 1)
        result.append(_check("control.command", suspected if invalid else observed,
                             "Invalid UDP command fields" if invalid else
                             ("UDP command sent" if ctrl.get("sent") and ctrl.get("sent_ns") else "UDP send unverified"), refs,
                             module="Control transport" if invalid else None,
                             code="Inspect encode_planner_control()" if invalid else None))
    else:
        result.append(_check("control.command", insufficient, "Wire command fields unavailable", refs,
                             missing=["final_wire_command"]))
    max_steer = ctrl.get("max_steering_rad")
    if isinstance(calc, dict) and isinstance(wire, dict) and isinstance(max_steer, (int, float)) and max_steer > 0:
        try:
            expected = max(-1., min(1., calc["front_steer_rad"] / max_steer))
            changed = any(not math.isclose(wire[key], calc[key], abs_tol=1e-6) for key in ("accel", "brake")) or not math.isclose(
                wire["steer_normalized"], expected, abs_tol=1e-6)
        except (KeyError, TypeError, ValueError):
            changed = None
        result.append(_check("control.wire_conversion", insufficient if changed is None else
                             suspected if changed else observed,
                             "UDP wire values differ from calculated command" if changed else
                             "UDP wire values match documented steering normalization" if changed is False else
                             "Command conversion cannot be verified", refs,
                             module="Control transport" if changed else None,
                             missing=["calculated and wire command"] if changed is None else [],
                             code="Inspect encode_planner_control()" if changed else None))
    else:
        result.append(_check("control.wire_conversion", insufficient, "Command conversion cannot be verified", refs,
                             missing=["calculated command, wire command, maximum steering angle"]))
    target = plan.get("target_velocity_mps")
    if linked and isinstance(target, (int, float)) and target <= .05 and isinstance(calc, dict):
        mismatch = calc.get("accel") != 0 or calc.get("brake") != 1
        result.append(_check("control.stop_target", suspected if mismatch else observed,
                             "Stop target contradicted by calculated command" if mismatch else
                             "Stop target produced full braking", refs,
                             module="Control" if mismatch else None,
                             code="Inspect PathController._calculate_control()" if mismatch else None))
    else:
        result.append(_check("control.stop_target", insufficient, "No applicable stop-target comparison", refs,
                             missing=["linked stop target and calculated command"]))
    ego = ctrl.get("observed_ego") or {}
    contract = ctrl.get("response_contract") or {}
    measured = ego.get("longitudinal_acceleration_mps2")
    acknowledged = (ego.get("applied_command_cycle_id") == ctrl.get("cycle_id") and
                    ctrl.get("sent") is True and isinstance(ctrl.get("sent_ns"), int) and
                    isinstance(ctrl.get("observed_ego_received_ns"), int) and
                    ctrl["observed_ego_received_ns"] > ctrl["sent_ns"])
    if (acknowledged and isinstance(measured, (int, float)) and math.isfinite(measured) and
            isinstance(contract.get("min_acceleration_mps2"), (int, float)) and
            isinstance(contract.get("max_acceleration_mps2"), (int, float))):
        mismatch = not contract["min_acceleration_mps2"] <= measured <= contract["max_acceleration_mps2"]
        result.append(_check("control.ego_response", suspected if mismatch else observed,
                             "Acknowledged Ego acceleration outside supplied response contract" if mismatch else
                             "Acknowledged Ego acceleration within supplied response contract", refs,
                             module="Actuation or vehicle response" if mismatch else None,
                             code="Inspect MORAI actuator application and vehicle dynamics" if mismatch else None))
    else:
        result.append(_check("control.ego_response", insufficient,
                             "Observed Ego speed has no verified command acknowledgement or latency contract", refs,
                             missing=["actuator command id or response latency contract"],
                             code="Add MORAI applied-command acknowledgement before attributing response"))
    if failure.get("event_type") == "collision_started":
        result.append(_check("perception.collision_object", insufficient,
                             "NPC actor ID cannot be equated with VE track ID", refs,
                             missing=["validated GT/VE coordinate frame, timestamp and geometry match"]))
    else:
        result.append(_check("perception.collision_object", "NOT_APPLICABLE",
                             "No collision event in this failure", refs))
    return result


def diagnose_pipeline(run_id, failure, perception, planning, control, errors):
    checks = _diagnose_pipeline(run_id, failure, perception, planning, control, errors)
    sources = {"perception_trace.jsonl": perception or [],
               "planning_trace.jsonl": planning or [], "control_trace.jsonl": control or []}
    for entry in checks:
        stamps = []
        for reference in entry["evidence"]:
            rows = sources.get(reference["file"], [])
            line = reference.get("line")
            if isinstance(line, int) and 1 <= line <= len(rows):
                stamp = rows[line-1].get("timestamp_ns")
                if isinstance(stamp, int):
                    stamps.append(stamp)
        entry["time_range"] = (dict(start_ns=min(stamps), end_ns=max(stamps),
                                    clock_domain="host_monotonic_ns") if stamps else None)
        entry["problem_timestamp_ns"] = (max(stamps) if stamps and
                                           entry["status"] == "SUSPECTED" else None)
    return checks
