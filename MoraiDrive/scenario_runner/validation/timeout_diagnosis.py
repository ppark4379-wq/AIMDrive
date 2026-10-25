"""Evidence-based checks for the existing tunnel-vehicle runtime timeout."""

import math


def _ref(name, line=None):
    return {"file": name, "line": line}


def _check(identifier, status, finding, rule, quality, refs=(), *,
           time_range=None, next_check=None, module=None):
    return {"check_id": identifier, "status": status, "finding": finding,
            "rule": rule, "data_quality_requirement": quality,
            "evidence": list(refs), "time_range": time_range,
            "next_check": next_check, "suspected_module": module,
            "missing_data": [quality] if status == "INSUFFICIENT_EVIDENCE" else [],
            "observed_facts": [finding] if status == "OBSERVED" else []}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def diagnose_timeout(manifest, scenario, summary, ego, planning, control,
                     *, trace_error=None, pipeline_errors=None):
    """Join only same-run, same-host observations; never assign module blame by timeout alone."""
    if (manifest.get("failure") or {}).get("event_type") != "scenario_timeout":
        return []
    failure = manifest["failure"]
    run_id = manifest["source_run_id"]
    pipeline_errors = pipeline_errors or {}
    valid_scenario = (isinstance(scenario, dict) and
                      scenario.get("run_seed") == manifest.get("seed") and
                      scenario.get("failure_reason") == failure.get("reason") == "Tunnel vehicle timed out")
    valid_exit = (isinstance(summary, dict) and summary.get("run_id") == run_id and
                  summary.get("status") == "exception" and
                  summary.get("error_message") == "Tunnel vehicle timed out")
    checks = []
    checks.append(_check("timeout.event", "OBSERVED" if valid_scenario and valid_exit else
                         "INSUFFICIENT_EVIDENCE",
                         "터널 차량 시나리오가 Timeout으로 종료됨" if valid_scenario and valid_exit else
                         "Timeout 종료와 시나리오 결과의 동일 실행 증거 부족",
                         "manifest 사유, 같은 seed의 NPC 결과, 같은 run_id의 Runner 예외가 모두 일치",
                         "manifest.json, scenario_result.json, summary.json의 일치",
                         [_ref("manifest.json"), _ref("scenario_result.json"), _ref("summary.json")],
                         time_range={"start_sec": failure.get("captured_monotonic_sec"),
                                     "end_sec": failure.get("captured_monotonic_sec"),
                                     "time_basis": "host_monotonic_sec"}
                         if _number(failure.get("captured_monotonic_sec")) else None))

    valid_ego = not trace_error and ego and all(row.get("run_id") == run_id and
                    _number(row.get("captured_monotonic_sec")) for row in ego)
    progress = [(i, row) for i, row in enumerate(ego or [], 1)
                if _number(row.get("route_progress_s_m"))]
    if valid_ego and len(progress) >= 2:
        first_i, first = progress[0]
        last_i, last = progress[-1]
        delta = last["route_progress_s_m"] - first["route_progress_s_m"]
        duration = last["captured_monotonic_sec"] - first["captured_monotonic_sec"]
        finding = ("Ego 경로 진행 %.2fm; 직전 %.1f초 동안 %.2fm 진행; 마지막 속도 %.3fm/s" %
                   (last["route_progress_s_m"], duration, delta, last.get("speed_mps", float("nan"))))
        checks.append(_check("timeout.ego_progress", "OBSERVED", finding,
                             "저장된 route_progress_s_m의 첫·마지막 값만 비교; 정체 임계값을 새로 만들지 않음",
                             "같은 run_id의 순서·단조시각·경로 진행거리",
                             [_ref("ego_trace.jsonl", first_i), _ref("ego_trace.jsonl", last_i)],
                             time_range={"start_sec": first["captured_monotonic_sec"],
                                         "end_sec": last["captured_monotonic_sec"],
                                         "time_basis": "host_monotonic_sec"}))
    else:
        checks.append(_check("timeout.ego_progress", "INSUFFICIENT_EVIDENCE",
                             "Timeout 직전 Ego 진행거리 비교 불가",
                             "같은 실행의 두 개 이상 경로 진행 관측 필요",
                             "완전한 ego_trace.jsonl 및 route_progress_s_m",
                             [_ref("ego_trace.jsonl")]))

    end_ns = failure.get("captured_monotonic_sec")
    end_ns = int(end_ns * 1e9) if _number(end_ns) else None
    valid_pipeline = (end_ns is not None and
                      not pipeline_errors.get("planning") and not pipeline_errors.get("control") and
                      all(isinstance(rows, list) and rows for rows in (planning, control)) and
                      all(row.get("run_id") == run_id and row.get("clock_domain") == "host_monotonic_ns" and
                          type(row.get("timestamp_ns")) is int and row["timestamp_ns"] <= end_ns
                          for rows in (planning, control) for row in rows))
    linked = []
    if valid_pipeline:
        plans = {row["cycle_id"]: (i, row) for i, row in enumerate(planning, 1)
                 if type(row.get("cycle_id")) is int}
        for i, row in enumerate(control, 1):
            pair = plans.get(row.get("planning_cycle_id")) if type(row.get("planning_cycle_id")) is int else None
            if pair and row.get("target_velocity_mps") == pair[1].get("target_velocity_mps") and row.get("sent") is True:
                linked.append((pair[0], pair[1], i, row))
    if linked:
        plan_i, plan, ctrl_i, ctrl = linked[-1]
        wire = ctrl.get("final_wire_command")
        wire = wire if isinstance(wire, dict) else {}
        target = plan.get("target_velocity_mps")
        brake = wire.get("brake")
        if _number(target) and _number(brake) and type(ctrl.get("sent_ns")) is int:
            finding = "Planner 목표속도 %.3fm/s, 송신된 제동 %.3f (cycle_id=%s)" % (
                target, brake, ctrl.get("planning_cycle_id"))
            checks.append(_check("timeout.plan_control", "OBSERVED", finding,
                                 "같은 cycle_id의 Planning 출력과 Control 입력·실제 송신값 확인",
                                 "같은 run_id·host_monotonic_ns·cycle_id 및 sent_ns",
                                 [_ref("planning_trace.jsonl", plan_i), _ref("control_trace.jsonl", ctrl_i)],
                                 time_range={"start_ns": plan["timestamp_ns"],
                                             "end_ns": ctrl["sent_ns"],
                                             "clock_domain": "host_monotonic_ns"}))
        else:
            linked = []
    if not linked:
        checks.append(_check("timeout.plan_control", "INSUFFICIENT_EVIDENCE",
                             "Timeout 직전 Planning 목표속도와 송신 제어값의 연결 불가",
                             "같은 cycle_id와 host monotonic 시각의 Planning·Control 기록만 연결",
                             "유효한 planning_trace.jsonl, control_trace.jsonl, sent_ns",
                             [_ref("planning_trace.jsonl"), _ref("control_trace.jsonl")]))

    cfg = (manifest.get("effective_scenario_config") or {}).get("tunnel_vehicle_dynamic") or {}
    last_npc = (scenario or {}).get("last_observation")
    last_npc = last_npc if isinstance(last_npc, dict) else {}
    gap = last_npc.get("route_gap_m")
    encounter = cfg.get("encounter_gap_m")
    planned_speed = ((scenario or {}).get("plan") or {}).get("speed_mps")
    commanded_speed = (scenario or {}).get("approach_speed_mps")
    if valid_scenario and all(_number(v) for v in (gap, encounter, commanded_speed, planned_speed)):
        checks.append(_check("timeout.npc_state", "OBSERVED",
                             "NPC 마지막 간격 %.1fm, 조우 기준 %.1fm, 명령속도 %.3fm/s, 계획속도 %.3fm/s, 조우확인=%s" %
                             (gap, encounter, commanded_speed, planned_speed,
                              scenario.get("encounter_verified")),
                             "저장된 NPC 결과와 실행 설정의 수치를 그대로 보고",
                             "동일 seed의 scenario_result.json 및 effective_scenario_config",
                             [_ref("manifest.json"), _ref("scenario_result.json")]))
    else:
        checks.append(_check("timeout.npc_state", "INSUFFICIENT_EVIDENCE",
                             "NPC 간격·속도·조우 기준 비교 불가",
                             "같은 실행의 NPC 결과 및 조우 기준 확인",
                             "scenario_result.json의 마지막 간격·속도와 manifest 설정",
                             [_ref("manifest.json"), _ref("scenario_result.json")]))
    yield_rows = [(i, row) for i, row in enumerate(planning or [], 1)
                  if row.get("planning_reason") in ("jamming_dynamic_yield", "jamming_whole_body_collision")]
    candidate = (valid_scenario and valid_exit and valid_pipeline and bool(linked) and
                 linked[-1][3].get("planning_cycle_id") == planning[-1].get("cycle_id") and
                 (scenario or {}).get("encounter_verified") is False and
                 ((scenario or {}).get("plan") or {}).get("behavior") == "vehicle_slow" and
                 _number(gap) and _number(encounter) and gap > encounter and
                 _number(commanded_speed) and _number(planned_speed) and
                 commanded_speed < planned_speed and len(yield_rows) >= 2 and
                 planning[-1].get("planning_reason") in
                 ("jamming_dynamic_yield", "jamming_whole_body_collision"))
    if candidate:
        checks.append(_check("timeout.yield_npc_coupling", "SUSPECTED",
                             "교착 가능성: Planner 양보/충돌회피 판단 반복, NPC 간격 %.1fm > 조우 기준 %.1fm; NPC 명령속도 %.3fm/s < 계획 %.3fm/s" %
                             (gap, encounter, commanded_speed, planned_speed),
                             "기존 조우 기준·NPC 속도 연동 설정과 반복 Planning 사유의 동시 관측",
                             "같은 실행의 NPC 결과·설정 및 유효한 Planning 시각/순서",
                             [_ref("manifest.json"), _ref("scenario_result.json"),
                              _ref("planning_trace.jsonl", yield_rows[0][0]),
                              _ref("planning_trace.jsonl", yield_rows[-1][0])],
                             time_range={"start_ns": yield_rows[0][1]["timestamp_ns"],
                                         "end_ns": yield_rows[-1][1]["timestamp_ns"],
                                         "clock_domain": "host_monotonic_ns"},
                             module="Scenario NPC 속도 연동 / Planning 양보 상호작용",
                             next_check="vehicle_slow 접근 속도 정책과 jamming_dynamic_yield 조건을 함께 검토"))
        checks[-1]["observed_facts"] = [
            "NPC 간격 %.1fm > 조우 기준 %.1fm" % (gap, encounter),
            "NPC 명령속도 %.3fm/s < 계획속도 %.3fm/s" % (commanded_speed, planned_speed),
            "Planning 양보 또는 whole-body collision 판단이 직전 기록에서 반복됨",
        ]
    else:
        checks.append(_check("timeout.yield_npc_coupling", "INSUFFICIENT_EVIDENCE",
                             "Timeout만으로 NPC·Planner 교착 원인을 확인할 수 없음",
                             "NPC 간격이 설정된 조우 기준 밖이고 속도 연동 및 반복 양보 판단이 함께 기록돼야 함",
                             "NPC 조우·속도 결과와 동일 실행 Planning 사유",
                             [_ref("scenario_result.json"), _ref("planning_trace.jsonl")]))

    if manifest.get("controller") == "ve_gps_udp":
        uncertain = [(i, obj) for i, row in enumerate(planning or [], 1)
                     for obj in (row.get("input_objects") if isinstance(row.get("input_objects"), list) else [])
                     if isinstance(obj, dict) and obj.get("motion_state") == "uncertain"]
        checks.append(_check("timeout.ve_uncertainty", "INSUFFICIENT_EVIDENCE",
                             "VE 객체 운동 상태가 uncertain으로 기록됨; 불필요한 정지의 직접 원인인지는 미확인" if uncertain else
                             "VE 객체 속도 불확실성이 정지에 미친 영향 미확인",
                             "인지 추적 품질과 Planning 선택의 인과관계는 별도 검증 필요",
                             "VE 객체 속도 신뢰도, 당시 NPC 상태, Planning 선택 근거",
                             [_ref("planning_trace.jsonl", uncertain[-1][0])] if uncertain else
                             [_ref("planning_trace.jsonl")],
                             time_range={"start_ns": planning[uncertain[-1][0]-1]["timestamp_ns"],
                                         "end_ns": planning[uncertain[-1][0]-1]["timestamp_ns"],
                                         "clock_domain": "host_monotonic_ns"}
                             if uncertain and valid_pipeline else None,
                             next_check="VE 추적 속도와 같은 시각의 NPC 속도·좌표계 대조"))
    else:
        checks.append(_check("timeout.ve_uncertainty", "NOT_APPLICABLE",
                             "VE 제어 경로가 아님", "VE 입력 사용 실행에만 적용",
                             "controller 식별", [_ref("manifest.json")]))
    return checks
