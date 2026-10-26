"""Offline, evidence-limited diagnosis of one Scenario Runner incident.

No MORAI, ROS, Expert, or VE imports. A finding never implies a controller
fault merely because a collision was observed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .incident import _write_json


OBSERVED = "OBSERVED"
SUSPECTED = "SUSPECTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
NOT_APPLICABLE = "NOT_APPLICABLE"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _read_json(path):
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        return None, "invalid: " + str(error)
    return (value, None) if isinstance(value, dict) else (None, "not an object")


def _read_jsonl(path):
    if not path.is_file():
        return None, "missing"
    try:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError("truncated final line")
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("record is not an object")
    except (OSError, UnicodeError, ValueError) as error:
        return None, "invalid: " + str(error)
    return rows, None


def _ref(file, line=None):
    return {"file": file, "line": line}


def _result(check_id, status, finding, rule, quality, evidence=(), time_range=None,
            next_check=None):
    return {"check_id": check_id, "status": status, "finding": finding,
            "rule": rule, "data_quality_requirement": quality,
            "evidence": list(evidence), "time_range": time_range,
            "next_check": next_check, "suspected_module": None,
            "missing_data": [quality] if status == INSUFFICIENT else [],
            "observed_facts": [finding] if status == OBSERVED else []}


def _range(rows):
    values = [row.get("timestamp") for row in rows if _finite(row.get("timestamp"))]
    return {"start_sec": min(values), "end_sec": max(values),
            "time_basis": rows[0].get("time_basis", "lap_elapsed_sec")} if values else None


def _trace_quality(rows, run_id):
    if rows is None or not rows:
        return "Ego 기록이 없거나 손상됨"
    previous_sequence = previous_clock = None
    previous_lap = None
    domain = rows[0].get("clock_domain")
    for index, row in enumerate(rows, 1):
        sequence, clock = row.get("sequence"), row.get("captured_monotonic_sec")
        lap = row.get("lap_index")
        position, speed = row.get("position_xyz"), row.get("speed_mps")
        if (row.get("run_id") != run_id or type(sequence) is not int or
                (previous_sequence is not None and sequence != previous_sequence + 1) or
                not _finite(clock) or (previous_clock is not None and clock < previous_clock) or
                type(lap) is not int or (previous_lap is not None and lap < previous_lap) or
                row.get("clock_domain") != domain or row.get("time_basis") != "lap_elapsed_sec" or
                not _finite(row.get("timestamp")) or
                not isinstance(position, (list, tuple)) or len(position) != 3 or
                not all(_finite(value) for value in position) or
                not _finite(speed) or speed < 0):
            return "Ego 기록 %d행의 run_id/순서/시각/위치/속도 이상" % index
        previous_sequence, previous_clock, previous_lap = sequence, clock, lap
    return None


def diagnose_incident(folder):
    """Write diagnosis.json and diagnosis.txt; return the structured report."""
    folder = Path(folder).resolve()
    manifest, error = _read_json(folder / "manifest.json")
    if error or manifest.get("schema_version") != 1 or folder.name != "incident_" + str(manifest.get("incident_id")):
        raise ValueError("Incident manifest is missing, invalid, or belongs to another folder")
    run_id = manifest.get("source_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Incident source_run_id is missing")
    ego, ego_error = _read_jsonl(folder / "ego_trace.jsonl")
    actors, actors_error = _read_jsonl(folder / "actors_trace.jsonl")
    events, events_error = _read_jsonl(folder / "validation_events.jsonl")
    planner, planner_error = _read_jsonl(folder / "planner_trace.jsonl")
    scenario, scenario_error = _read_json(folder / "scenario_result.json")
    scenario_digest = manifest.get("scenario_result_sha256")
    if scenario_error is None and scenario_digest is not None:
        if hashlib.sha256((folder / "scenario_result.json").read_bytes()).hexdigest() != scenario_digest:
            scenario, scenario_error = None, "checksum mismatch"
    summary, summary_error = _read_json(folder / "summary.json")
    if summary_error == "missing" and re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        source_root = manifest.get("runner_package_root")
        if isinstance(source_root, str):
            original = Path(source_root) / ".runtime" / "validation_runs" / run_id / "summary.json"
            candidate, candidate_error = _read_json(original)
            if candidate_error is None and candidate.get("run_id") == run_id and candidate.get("scenario_name") == manifest.get("scenario_name"):
                _write_json(folder / "summary.json", candidate)
                summary, summary_error = candidate, None
    rerun, rerun_error = _read_json(folder / "rerun_result.json")
    failure = manifest.get("failure") or {}
    checks = []
    add = checks.append

    quality_error = _trace_quality(ego, run_id)
    add(_result("ego.trace_quality", INSUFFICIENT if ego_error or not ego else OBSERVED,
                quality_error or "Ego 위치·속도 및 기록 순서가 유효함",
                "각 기록의 run_id, 순서, 단조시각, 위치, 비음수 속도 검증",
                "Ego JSONL 전체가 읽혀야 함", [_ref("ego_trace.jsonl")],
                _range(ego or []), "손상된 행 원본 확인" if quality_error else None))
    # Data anomalies are directly observed facts, never a presumed control fault.
    checks[-1]["anomaly"] = bool(quality_error)

    if summary_error:
        add(_result("runtime.exit", INSUFFICIENT, "종료 요약이 Incident에 없음",
                    "동일 run_id의 summary.status와 error를 읽음", "summary.json 필요",
                    [_ref("summary.json")], next_check="Runner 종료 후 진단 재실행"))
    elif summary.get("run_id") != run_id:
        add(_result("runtime.exit", INSUFFICIENT, "종료 요약 run_id 불일치",
                    "Incident source_run_id와 summary.run_id 동일", "동일 실행 요약 필요",
                    [_ref("summary.json")], next_check="요약 파일 출처 확인"))
    else:
        status = summary.get("status")
        add(_result("runtime.exit", OBSERVED,
                    "Runner 종료 상태=%s; %s" % (status, summary.get("error_message") or "오류 메시지 없음"),
                    "저장된 Runner 종료 상태를 그대로 보고", "동일 run_id의 summary.json",
                    [_ref("summary.json")], next_check="예외 스택 추적은 별도 터미널 로그 확인" if status == "exception" else None))

    if scenario_error:
        add(_result("scenario.npc", INSUFFICIENT, "시나리오 결과가 없거나 손상됨",
                    "저장된 NPC 생성·이동·완료 상태 확인", "scenario_result.json 필요",
                    [_ref("scenario_result.json")], next_check="시나리오 결과 저장 여부 확인"))
    elif scenario.get("run_seed") != manifest.get("seed"):
        add(_result("scenario.npc", INSUFFICIENT, "NPC 결과 seed와 Incident seed 불일치",
                    "동일 seed의 결과만 연결", "동일 실행 NPC 결과 필요",
                    [_ref("manifest.json"), _ref("scenario_result.json")], next_check="결과 파일 출처 확인"))
    else:
        phase = scenario.get("phase")
        failed = phase == "failed" or bool(scenario.get("failure_reason"))
        moving = bool(scenario.get("observed_moving") or scenario.get("startup_motion_verified"))
        add(_result("scenario.npc", OBSERVED,
                    "NPC phase=%s, spawned=%s, spawn_verified=%s, moving=%s, completed=%s%s" % (
                        phase, scenario.get("spawned"), scenario.get("spawn_verified"), moving,
                        scenario.get("completed"),
                        ", reason=" + str(scenario.get("failure_reason")) if failed else ""),
                    "시나리오가 이미 기록한 상태를 재계산 없이 보고",
                    "동일 seed의 scenario_result.json; 충돌 시점 복사본은 최종 정리 상태가 아닐 수 있음",
                    [_ref("scenario_result.json")], next_check="NPC 실패 사유 확인" if failed else None))
        checks[-1]["anomaly"] = failed

    if rerun_error:
        add(_result("runtime.setup_match", NOT_APPLICABLE, "재실행 비교 결과 없음",
                    "rerun_result.matches에 false가 있는지 확인", "재실행 후 비교 파일 필요",
                    [_ref("rerun_result.json")]))
    else:
        matches = rerun.get("matches")
        if not isinstance(matches, dict) or rerun.get("source_run_id") != run_id:
            add(_result("runtime.setup_match", INSUFFICIENT, "재실행 비교 결과의 출처 불일치",
                        "동일 원본 run_id와 matches 필요", "유효한 rerun_result.json",
                        [_ref("rerun_result.json")]))
        else:
            mismatches = [key for key, value in matches.items() if value is False]
            placement = matches.get("npc_placement")
            if isinstance(placement, dict):
                mismatches.extend("npc_placement." + key for key, value in placement.items()
                                  if value is False)
            add(_result("runtime.setup_match", OBSERVED,
                        "불일치: " + ", ".join(mismatches) if mismatches else "기록된 설정 불일치 없음; 미검증 항목은 별도 확인",
                        "matches가 명시적으로 false인 항목만 불일치로 보고",
                        "동일 원본 run_id의 비교 결과", [_ref("rerun_result.json")],
                        next_check="MORAI 버전·맵 내용은 비교 범위 밖"))
            checks[-1]["anomaly"] = bool(mismatches)

    event_matches = [] if events is None else [(line, row) for line, row in enumerate(events, 1)
        if row.get("event_type") == "collision_started" and row.get("run_id") == run_id]
    failure_ids = failure.get("actor_ids") or []
    matched = [(line, row) for line, row in event_matches
               if row.get("actor_id") in failure_ids and row.get("timestamp") == failure.get("timestamp")]
    if events_error or (failure.get("event_type") == "collision_started" and not matched):
        add(_result("collision.event", INSUFFICIENT, "충돌 이벤트 증거 누락·손상·불일치",
                    "manifest의 actor ID·timestamp와 동일 run_id의 collision_started 일치",
                    "manifest 및 validation_events.jsonl", [_ref("manifest.json"), _ref("validation_events.jsonl")],
                    next_check="원본 Event Monitor 기록 대조"))
    elif matched:
        line, event = matched[0]
        sequence = failure.get("observation_sequence")
        ego_match = next(((i, row) for i, row in enumerate(ego or [], 1)
                          if row.get("sequence") == sequence and event["actor_id"] in (row.get("collision_objects") or [])), None)
        actor_match = next(((i, actor) for i, sample in reversed(list(enumerate(actors or [], 1)))
                            if type(sample.get("sequence")) is int and sample["sequence"] <= sequence
                            for actor in sample.get("actors", []) if actor.get("actor_id") == event["actor_id"]), None)
        references = [_ref("manifest.json"), _ref("validation_events.jsonl", line)]
        if ego_match:
            references.append(_ref("ego_trace.jsonl", ego_match[0]))
        if actor_match:
            references.append(_ref("actors_trace.jsonl", actor_match[0]))
        add(_result("collision.event", OBSERVED,
                    "충돌 시각 %.3fs, actor=%s, Ego 위치=%s%s" % (
                        event["timestamp"], event["actor_id"], failure.get("position_xyz"),
                        ", 직전 NPC 위치=" + str(actor_match[1].get("position_xyz")) if actor_match else ", NPC 상태 없음"),
                    "일치하는 Event Monitor 이벤트와 Ego 충돌 객체를 확인",
                    "동일 run_id·timestamp·actor ID; Ego/NPC는 기록 순서로 연결",
                    references, {"start_sec": event["timestamp"], "end_sec": event["timestamp"],
                                 "time_basis": failure.get("time_basis")},
                    next_check="Ego 충돌 관측 누락 확인" if not ego_match else None))
    else:
        add(_result("collision.event", NOT_APPLICABLE, "충돌 이벤트 없음",
                    "collision_started가 없으면 충돌 원인 검사 제외", "완전한 이벤트 파일",
                    [_ref("validation_events.jsonl")]))

    watchdog = (manifest.get("effective_scenario_config") or {}).get("progress_watchdog") or {}
    if not watchdog.get("enabled", True):
        add(_result("ego.stall", NOT_APPLICABLE, "이 실행의 progress watchdog 비활성",
                    "Runner 설정의 enabled 플래그 사용", "effective_scenario_config",
                    [_ref("manifest.json")]))
    elif quality_error:
        add(_result("ego.stall", INSUFFICIENT, "Ego 기록 품질 부족",
                    "기존 watchdog의 시간·진행거리 기준만 사용", "연속된 유효 Ego 진행거리 필요",
                    [_ref("ego_trace.jsonl")]))
    else:
        timeout = max(1.0, float(watchdog.get("timeout_sec", 120.0)))
        minimum = max(0.1, float(watchdog.get("minimum_progress_m", 2.0)))
        grace = max(0.0, float(watchdog.get("startup_grace_sec", 30.0)))
        eligible = [row for row in ego if row.get("lap_index") == ego[-1].get("lap_index")
                    and row.get("timestamp", -1) >= grace and _finite(row.get("route_progress_s_m"))]
        if not eligible or eligible[-1]["timestamp"] - eligible[0]["timestamp"] < timeout:
            add(_result("ego.stall", INSUFFICIENT, "기존 watchdog 판정에 필요한 연속 시간 부족",
                        "startup_grace_sec=%s, timeout_sec=%s, minimum_progress_m=%s" % (grace, timeout, minimum),
                        "같은 lap에서 grace 이후 연속 %ss 이상 진행거리" % timeout,
                        [_ref("manifest.json"), _ref("ego_trace.jsonl")], _range(ego)))
        else:
            recent = [row for row in eligible if row["timestamp"] >= eligible[-1]["timestamp"] - timeout]
            progress = max(row["route_progress_s_m"] for row in recent) - recent[0]["route_progress_s_m"]
            stalled = progress < minimum
            add(_result("ego.stall", SUSPECTED if stalled else OBSERVED,
                        "정체 후보: %ss간 진행거리 %.2fm < %.2fm" % (timeout, progress, minimum)
                        if stalled else "관측 구간의 진행거리 %.2fm" % progress,
                        "Runner progress_watchdog: timeout_sec=%s, minimum_progress_m=%s" % (timeout, minimum),
                        "같은 lap의 유효 route_progress_s_m 및 단조 관측",
                        [_ref("manifest.json"), _ref("ego_trace.jsonl")], _range(recent),
                        next_check="신호 대기·정상 정차 여부 확인" if stalled else None))

    manual = manifest.get("controller") == "morai_keyboard"
    planner_rows = planner if isinstance(planner, list) else None
    if manual:
        add(_result("control.response", NOT_APPLICABLE, "수동 키보드 제어 실행",
                    "자동 제어 명령과 차량 반응 비교 제외", "controller 식별",
                    [_ref("manifest.json")]))
        add(_result("planning.path", NOT_APPLICABLE, "수동 키보드 제어 실행",
                    "Planning 경로 검사 제외", "controller 식별", [_ref("manifest.json")]))
        add(_result("perception.ve", NOT_APPLICABLE, "VE 입력을 사용하지 않은 수동 실행",
                    "VE 입력 검사 제외", "controller 식별", [_ref("manifest.json")]))
    else:
        command_rows = [] if planner_rows is None else [(i, row) for i, row in enumerate(planner_rows, 1)
                        if isinstance(row.get("control_command"), dict)]
        add(_result("control.response", INSUFFICIENT,
                    "제어 명령/목표속도와 Ego 반응의 시간 동기화 근거 부족" if command_rows else "제어 명령·목표속도 구조화 로그 없음",
                    "동일 run_id·frame/sequence 및 clock domain의 명령과 Ego 반응 비교 필요",
                    "실제 발행 명령, 목표속도, Ego 상태, 검증된 동기화 및 기존 응답 기준",
                    [_ref("planner_trace.jsonl") if command_rows else _ref("manifest.json")],
                    next_check="Expert/Planner 제어 출력에 run_id·sequence·시각 연결"))
        path_rows = [] if planner_rows is None else [(i, row) for i, row in enumerate(planner_rows, 1)
                     if "planned_trajectory" in row]
        linked = [(i, row) for i, row in path_rows if type(failure.get("observation_sequence")) is int and
                  row.get("run_id") == run_id and
                  row.get("observation_sequence") == failure.get("observation_sequence") and
                  row.get("clock_domain") == (ego[-1].get("clock_domain") if ego else None)]
        if not linked:
            add(_result("planning.path", INSUFFICIENT,
                        "충돌 관측과 동일 sequence·clock domain의 계획 경로 없음",
                        "일치하는 frame의 구조화 planned_trajectory만 판정",
                        "동일 run_id·observation_sequence·clock_domain의 경로",
                        [_ref("planner_trace.jsonl")], next_check="Planner 구조화 경로 로그 연결"))
        else:
            line, row = linked[-1]
            path = row["planned_trajectory"]
            invalid = not isinstance(path, list) or len(path) < 2 or any(
                not isinstance(point, (list, tuple)) or len(point) < 2 or
                not all(_finite(value) for value in point[:2]) for point in path)
            add(_result("planning.path", SUSPECTED if invalid else OBSERVED,
                        "충돌 frame의 계획 경로가 없거나 유효하지 않음" if invalid else "충돌 frame의 계획 경로 형식 유효; 점유영역 충돌은 미검증",
                        "동일 frame 경로의 2개 이상 유한 XY 점 확인",
                        "연결된 구조화 경로; 점유영역 판단에는 객체 형상·프레임 필요",
                        [_ref("planner_trace.jsonl", line)],
                        next_check="Planner가 의도적으로 정지 경로를 반환했는지 확인" if invalid else "객체 형상·좌표계 연결"))
        ve_enabled = "--ve-objects" in manifest.get("run_options", []) or (manifest.get("data_sources") or {}).get("ve_input") == "connected"
        if not ve_enabled:
            add(_result("perception.ve", NOT_APPLICABLE, "이 실행에서 VE 입력 사용 근거 없음",
                        "VE 모드 또는 연결 로그가 있는 경우만 검사", "VE 사용 증거",
                        [_ref("manifest.json")]))
        else:
            ve_rows = [] if planner_rows is None else [(i, row.get("ve_input")) for i, row in enumerate(planner_rows, 1)
                        if row.get("run_id") == run_id and
                        type(failure.get("observation_sequence")) is int and
                        row.get("observation_sequence") == failure.get("observation_sequence") and
                        row.get("clock_domain") == (ego[-1].get("clock_domain") if ego else None) and
                        isinstance(row.get("ve_input"), dict)]
            bad = [(i, item) for i, item in ve_rows if item.get("ready") is False or item.get("valid") is False or item.get("stale") is True]
            add(_result("perception.ve", OBSERVED if bad else INSUFFICIENT,
                        "VE 입력이 무효/미준비/지연으로 명시됨" if bad else "VE 유효성·수신 시각·순서 로그 부족",
                        "입력 로그의 명시적 ready/valid/stale 플래그만 사용",
                        "run_id, 수신·소스 시각, 순서, 프레임, 유효성 및 기존 freshness 기준",
                        [_ref("planner_trace.jsonl", bad[0][0]) if bad else _ref("planner_trace.jsonl")],
                        next_check="VE 입력 audit를 Incident run_id와 연결"))

    add(_result("planning.occupancy", INSUFFICIENT if not manual else NOT_APPLICABLE,
                "경로·객체 점유영역·좌표계의 동기화 자료 부족" if not manual else "수동 실행",
                "동일 frame의 계획 경로와 객체 점유영역 교차 검증",
                "경로·객체 형상·좌표계·동일 frame/시각 필요",
                [_ref("planner_trace.jsonl"), _ref("actors_trace.jsonl")],
                next_check="동기화된 객체 형상 및 경로 기록" if not manual else None))
    pipeline_logs = {}
    pipeline_errors = {}
    for stage in ("perception", "planning", "control"):
        pipeline_logs[stage], pipeline_errors[stage] = _read_jsonl(folder / (stage + "_trace.jsonl"))
    if manifest.get("controller") == "ve_gps_udp" or any(value is not None for value in pipeline_logs.values()):
        from .pipeline_diagnosis import diagnose_pipeline
        checks.extend(diagnose_pipeline(run_id, failure, pipeline_logs["perception"],
                                        pipeline_logs["planning"], pipeline_logs["control"],
                                        pipeline_errors))
    if failure.get("event_type") == "scenario_timeout":
        from .timeout_diagnosis import diagnose_timeout
        checks.extend(diagnose_timeout(manifest, scenario, summary, ego,
                                      pipeline_logs["planning"], pipeline_logs["control"],
                                      trace_error=quality_error, pipeline_errors=pipeline_errors))
    report = {"schema_version": 1, "incident_id": manifest["incident_id"],
              "source_run_id": run_id, "scenario_name": manifest.get("scenario_name"),
              "controller": manifest.get("controller"), "checks": checks,
              "cause_candidates": [entry["check_id"] for entry in checks
                                   if (entry["check_id"] in ("scenario.npc", "ego.stall", "planning.path",
                                                              "timeout.yield_npc_coupling")
                                       or entry.get("suspected_module"))
                                   and (entry["status"] == SUSPECTED or entry.get("anomaly"))
                                   and (failure.get("event_type") != "scenario_timeout" or
                                        entry["check_id"] != "scenario.npc")],
              "reproduction_issues": [entry["check_id"] for entry in checks
                                      if entry["check_id"] == "runtime.setup_match" and entry.get("anomaly")],
              "data_quality_issues": [entry["check_id"] for entry in checks
                                      if entry["check_id"] == "ego.trace_quality" and entry.get("anomaly")],
              "log_quality": {"ego": ego_error, "actors": actors_error,
                              "events": events_error, "planner": planner_error,
                              "scenario": scenario_error, "summary": summary_error,
                              "rerun": rerun_error, "pipeline": pipeline_errors},
              "limits": ["충돌 자체만으로 Planning/Control 원인을 판정하지 않음",
                         "시간 기준 또는 frame 연결이 없는 Expert 문자열 로그는 사용하지 않음"]}
    _write_json(folder / "diagnosis.json", report)
    lines = ["Incident 자동 진단: " + manifest["incident_id"],
             "시나리오: %s | 제어기: %s | 원본 실행: %s" % (
                 report["scenario_name"], report["controller"], run_id), ""]
    for entry in checks:
        lines.append("[%s] %s: %s" % (entry["status"], entry["check_id"], entry["finding"]))
        if entry.get("suspected_module"):
            lines.append("  의심 모듈: " + entry["suspected_module"])
        if entry.get("missing_data"):
            lines.append("  부족한 근거: " + ", ".join(entry["missing_data"]))
        if entry.get("evidence"):
            lines.append("  근거: " + ", ".join(
                reference["file"] + (":" + str(reference["line"]) if reference.get("line") else "")
                for reference in entry["evidence"]))
        if entry["next_check"]:
            lines.append("  추가 확인: " + entry["next_check"])
    lines.extend(["", "원인 후보: " + (", ".join(report["cause_candidates"]) or "확정 가능한 후보 없음"),
                  "재현 조건 문제: " + (", ".join(report["reproduction_issues"]) or "없음"),
                  "데이터 품질 문제: " + (", ".join(report["data_quality_issues"]) or "없음"),
                  "충돌만으로 Planning/Control 책임을 추정하지 않습니다."])
    (folder / "diagnosis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose an Incident offline; no MORAI required")
    parser.add_argument("incident_dir", type=Path)
    parser.add_argument("--html", action="store_true", help="also write failure_report/report.html")
    args = parser.parse_args(argv)
    try:
        report = diagnose_incident(args.incident_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, "Diagnosis failed: %s\n" % error)
    print((args.incident_dir / "diagnosis.txt").read_text(encoding="utf-8"))
    if args.html:
        from .failure_report import build_report
        try:
            print("report: %s" % build_report(args.incident_dir)[0])
        except (OSError, ValueError) as error:
            print("Report skipped: %s" % error)
    return 0


if __name__ == "__main__":
    sys.exit(main())
