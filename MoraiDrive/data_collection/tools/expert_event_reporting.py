"""Describe observed driving events without training acceptance or exclusion rules."""

from collections import Counter
import json
import math

from expert_frame_matching import match_collision, valid_ns, valid_xy


REPORT_COMMENTS = {'읽는_순서': ['summary로 전체 집계를 확인한 뒤 laps의 events에서 개별 이벤트와 프레임 후보를 확인하세요.',
           '이 보고서는 관측 사실을 설명합니다. 학습 사용 여부나 주행의 좋고 나쁨을 판정하지 않습니다.',
           'sensor_bundle_index는 0-based입니다. 1667은 사람이 세는 1668번째 bundle입니다. bundle은 함께 묶인 센서 데이터 단위입니다.',
           'null은 해당 값을 알 수 없거나 선택하지 않았다는 뜻입니다. 0과 다릅니다.',
           'HIGH도 실제 충돌 순간이나 카메라 촬영 순간을 확정한 값이 아닙니다.',
           'collisions는 events에 포함된 충돌을 다시 보여주는 호환용 목록입니다. 두 목록의 개수를 합산하지 마세요.'],
 '최상위': {'_comments': '보고서를 읽기 위한 한국어 필드 설명입니다. 이벤트 데이터에 포함해 집계하지 않습니다.',
         'schema': '기존 평가 보고서 구조의 버전.',
         'frame_matching_schema': '프레임 후보 매칭 규칙과 필드의 버전.',
         'event_report_schema': '통합 이벤트 보고서의 버전.',
         'campaign': '평가한 수집 campaign 이름.',
         'laps': 'Lap별 결과 목록. Lap 번호순으로 정렬됩니다.',
         'summary': '전체 Lap의 완주·충돌 및 입력 오류 집계.'},
 'Lap_필드': {'lap': 'Lap 번호. 1부터 시작하며 센서 bundle index와 다릅니다.',
            'run_name': '이 Lap의 수집 폴더 이름.',
            'episode': '평가에 사용한 episode 폴더의 경로.',
            'collection_status': 'runs.tsv에 기록된 해당 Lap의 마지막 수집 상태. captured가 곧 완주 성공을 뜻하지는 않습니다.',
            'completed': 'outcome의 명시적 완주 여부. true=완주, false=미완주, null=확인 불가.',
            'scenario_status': 'outcome에 저장된 runner의 종료 상태 문자열.',
            'scenario_reason': 'outcome에 저장된 종료 또는 실패 이유.',
            'collision_summary': 'runner가 남긴 충돌 요약 원문. 이벤트 목록과 별도로 참고합니다.',
            'collision_count': '읽어낸 collision 이벤트 수. 로그가 누락·손상되면 null로 둡니다.',
            'collisions': '기존 형식의 충돌 상세 목록. 같은 충돌이 events에도 포함됩니다.',
            'events': '지원하는 이벤트의 통합 목록. 원본 로그의 줄 순서로 정렬하며, 줄이 없는 outcome 기반 이벤트는 마지막에 둡니다.',
            'event_counts': 'events의 type별 개수. 신호 변경 명령과 변경 확인은 서로 다른 종류로 셉니다.',
            'events_complete': 'events.jsonl 읽기 및 충돌 필드 처리에 오류가 없었는지. 실제 모든 상황이 기록됐다는 보장은 아닙니다.',
            'event_detection': '파생 이벤트 탐지에 사용한 설정과 출처.',
            'event_report_warnings': '기본 설정 사용, 잘못된 progress, 탐지 불가 등 이벤트 해석의 한계.',
            'errors': '수집 결과 입력의 누락·손상·불일치. 충돌 자체를 오류로 표시하는 항목은 아닙니다.'},
 '이벤트_필드': {'type': '이벤트 종류. 아래 이벤트_종류 설명을 참고하세요.',
            'report_event_id': '이 Lap의 통합 events 목록에서 1부터 부여한 번호.',
            'event_id': '충돌 목록 안에서 1부터 부여한 충돌 번호. report_event_id와 다를 수 있습니다.',
            'source_event': '이벤트 근거가 된 원본 event 이름. progress이면 관측 표본에서 구간을 계산한 이벤트일 수 있습니다.',
            'source_line': 'events.jsonl의 원본 줄 번호, 1-based. 구간 이벤트는 시작 줄.',
            'event_sequence': 'runner가 남긴 원본 sequence 번호. 센서 index나 보고서 번호와 다릅니다.',
            'receive_monotonic_ns': '이벤트를 기록한 host의 monotonic 수신 시각, 나노초. 날짜·시각이나 시뮬레이션 source time과 직접 비교하면 '
                                    '안 됩니다.',
            'elapsed_seconds': '원본 이벤트의 경과 시간, 초. 이번 데이터에서는 시나리오 경과 시간이며 host 수신 시간과 속도가 다를 수 있습니다.',
            'route_s': '기준 경로를 따라 진행한 거리, m. 차량의 총 이동거리와 동일하지 않을 수 있습니다.',
            'ego_xy': '이벤트에 기록된 Ego의 평면 위치 [x,y], m. 위치 후보 탐지에 사용합니다.',
            'current_link': '이벤트 당시 Ego가 위치한 지도 link의 ID.',
            'collision_objects': 'runner가 충돌로 관측한 객체 ID 목록. ID 변경·재접촉은 별도 collision 이벤트로 기록될 수 있습니다.',
            'sensor_frame': '기존 확정 frame 필드. 현재는 확정하지 않으므로 null을 유지합니다.',
            'details': '신호 ID, 명령·확인 결과, 원본 메시지 등 종류별 부가 정보.',
            'reason': '미완주 이벤트의 outcome 원문 이유.',
            'scenario_status': '미완주 이벤트의 outcome 원문 종료 상태.',
            'evidence_source': '이벤트 판정의 입력 파일. 예: outcome.json.'},
 '이벤트_종류': {'collision': '원본 로그의 충돌 이벤트.',
            'traffic_light_override': 'runner가 신호 강제 변경을 시도한 기록. 명령 성공 여부는 details.result에서 확인합니다.',
            'traffic_light_override_verified': '신호 변경 후 확인 기록. 별도 충돌이나 별도 변경 명령이 아닙니다.',
            'traffic_light_override_verification_timeout': '신호 변경 확인이 제한 시간 내 완료되지 않은 기록.',
            'off_route': '원본의 명시적 경로 이탈 이벤트 또는 설정 기준을 넘은 progress 관측 구간. source_event로 구분합니다.',
            'stuck': '원본에 명시적으로 기록된 stuck 이벤트. 정차한 차량을 새로 추정한 결과가 아닙니다.',
            'expert_input_warning': '원본 Expert 입력 경고.',
            'scenario_intervention': '명시적 개입 이벤트 또는 intervention=true인 다른 원본 이벤트. source_event에 원래 이름을 '
                                     '보존합니다.',
            'lap_incomplete': 'outcome에 명시된 미완주·실패 종료 사실. outcome 파일이 없다는 이유만으로 생성하지 않습니다.'},
 '매칭_필드': {'sensor_bundle_index': '선택된 근처 센서 bundle의 0-based index. HIGH 또는 TIME_ONLY에서만 채웁니다.',
           'sensor_bundle_number': '선택된 index+1. 사람이 세는 1-based 순서 번호.',
           'sensor_bundle_index_base': 'index의 시작 번호. 항상 0.',
           'nearest_sensor_bundle_index': '호환용 시간 기준 최근접 후보. 최종 선택이 없더라도 남을 수 있습니다.',
           'nearest_sensor_frame_number': '호환용 시간 후보 index+1. 확정 프레임 번호가 아닙니다.',
           'receive_time_candidate_index': '유일한 시간 기준 최근접 후보 index.',
           'position_candidate_index': '유일한 위치 기준 최근접 후보 index.',
           'receive_time_candidate_indices': '시간 기준 최근접 후보 전체. 동률이면 여러 index가 들어갑니다.',
           'position_candidate_indices': '위치 기준 최근접 후보 전체. 동률이면 여러 index가 들어갑니다.',
           'receive_time_gap_ms': '이벤트 수신 시각과 시간 후보의 GPS 수신 시각 사이 절대 차이, ms. 실제 센서 노출 시간 오차가 아닙니다.',
           'position_error_m': '시간 후보의 GT Ego 위치와 이벤트 Ego 위치 사이 평면 거리, m.',
           'nearest_position_error_m': '위치 후보에서의 최소 위치 오차, m. 두 후보가 다르면 position_error_m와 다를 수 있습니다.',
           'candidate_index_distance': '시간 후보와 위치 후보의 index 차이의 절댓값.',
           'candidate_source_time_gap_ms': '후보 양 끝의 GT source time 차이, ms. 1 index를 무조건 100ms로 환산하지 않습니다.',
           'match_status': '후보의 관계 또는 근거 상태. 아래 매칭_상태를 참고하세요.',
           'match_confidence': '후보 근거의 수준. 주행 품질 점수나 학습 적합성 점수가 아닙니다.',
           'match_reasons': '매칭 결과의 근거 및 입력 한계.',
           'match_interpretation': '매칭 결과를 문장으로 설명한 해석.',
           'matching_method': '사용한 매칭 방법 이름.',
           'neighbor_position_errors': '시간 후보의 직전·현재·직후 index별 위치 오차. 각 항목의 index도 0-based입니다.',
           'possible_collision_between_bundle_indices': '인접 후보가 다를 때 가능한 충돌 구간의 양 끝 index. 가능성 표시이며 확정 구간이나 '
                                                        '보간 결과가 아닙니다.',
           'collision_between_frames': 'HIGH 후보 주변 좌표를 투영한 보간 구간의 양 끝 index, 0-based.',
           'estimated_fraction_after_frame': '보간 구간 첫 index 이후 진행 비율. 예: 0.19=첫 bundle 위치에서 다음 위치로 약 19%.',
           'estimated_collision_source_time_ns': '위치 투영 비율과 실제 GT source 간격으로 계산한 추정 source 시각, ns.',
           'estimated_collision_source_time_seconds': '위 추정 source 시각의 초 표현. elapsed_seconds 또는 host 수신 시각과 '
                                                      '다른 시간 기준입니다.',
           'interpolation_position_error_m': '보간 선분에 투영한 점과 이벤트 Ego 위치 사이 잔여 거리, m.',
           'interpolation_method': '위치 투영·보간 방법. 보간하지 못하면 null.',
           'physical_collision_time_verified': '물리적 충돌 시각을 검증했는지. 현재 항상 false이며 HIGH에서도 동일합니다.'},
 '매칭_상태': {'CANDIDATES_AGREE': '유일한 시간·위치 후보가 일치하고 시간 근거 검사를 통과함. confidence=HIGH.',
           'BETWEEN_FRAMES': '두 후보가 인접 index. 그 사이 충돌 가능성을 표시하며 선택 index는 null. confidence=LOW.',
           'CANDIDATE_MISMATCH': '두 후보가 2개 이상 index 간격으로 떨어짐. 선택 index는 null. confidence=LOW.',
           'INCONSISTENT_EVIDENCE': '센서 수신 범위 밖이거나 후보의 source time 근거가 누락·불일치함. confidence=LOW.',
           'TIME_ONLY': 'Ego 좌표 없이 수신 시각 후보만 선택됨. 위치 교차 검증을 하지 못했으므로 HIGH와 구분합니다.',
           'AMBIGUOUS': '최근접 동률 때문에 후보 하나를 선택할 수 없음.',
           'UNAVAILABLE': '필수 시각·좌표·후처리 파일 등의 근거 부족.'},
 '신뢰도': {'HIGH': '시간·위치 최근접 후보가 유일하게 일치함. 임의의 ms/m 허용 한계는 적용하지 않으므로 오차 수치도 같이 확인하세요.',
         'LOW': '후보 불일치 또는 시간 근거 문제. 반드시 매칭 실패를 뜻하지 않으며 BETWEEN_FRAMES일 수 있습니다.',
         'TIME_ONLY': '시간 후보만 존재하고 위치 확인은 하지 못함.',
         'AMBIGUOUS': '동률 후보로 판단이 모호함.',
         'UNAVAILABLE': '후보를 평가할 근거 부족.'},
 '신호_상세_details': {'light_id': '대상 신호등 ID.',
                   'target_color': '변경하려는 신호 상태 코드 또는 비트값.',
                   'observed_color': '변경 후 관측한 신호 상태 코드 또는 비트값.',
                   'exact_target_color_match': '목표 상태 코드와 관측 코드의 정확한 일치 여부.',
                   'verification_semantics': 'runner가 신호 변경 확인에 사용한 의미·조건.',
                   'result': '원본 변경 명령의 반환값. accepted, verified, before, after, error 등을 포함할 수 있습니다.',
                   'command_result': '확인 이벤트에 첨부된 앞선 변경 명령 결과.',
                   'wait_wall_sec': '변경 명령 전 host 시간 기준 대기 시간, 초.',
                   'wait_scenario_sec': '변경 명령 전 시나리오 시간 기준 대기 시간, 초.',
                   'command_to_readback_wall_sec': '명령부터 확인까지 host 시간, 초.',
                   'intervention': 'runner가 개입으로 명시했는지.',
                   'mission_distribution_changed': '원본이 시나리오 조건 변경으로 명시했는지.',
                   'reason / message / warning': '해당 이벤트가 남긴 원문 이유·메시지·경고.'},
 '경로_이탈_구간': {'end_source_line': '구간 마지막 progress의 1-based 로그 줄 번호.',
              'start_event_sequence / end_event_sequence': '시작·끝 progress의 원본 sequence 번호.',
              'start_receive_monotonic_ns / end_receive_monotonic_ns': '시작·끝 progress의 host monotonic 수신 시각, '
                                                                       'ns.',
              'start_elapsed_seconds / end_elapsed_seconds': '시작·끝 progress의 원본 경과 시간, 초.',
              'duration_seconds': '첫 이탈 관측부터 마지막 이탈 관측까지 시간. duration_clock 기준으로 읽으세요.',
              'duration_clock': '구간 길이 계산 기준. 기본 receive_monotonic_ns, 전체 수신 시각 부재 시 elapsed_seconds 대체.',
              'cross_track_limit_m': '경로 이탈 관측에 사용한 절대 cross-track 오차 한계, m. 이 값보다 클 때만 관측 구간에 포함합니다.',
              'timeout_seconds': '구간 이벤트를 남기기 위한 최소 관측 시간, 초.',
              'max_cross_track_error_m': '구간에서 관측한 cross-track 오차 절댓값의 최댓값, m.',
              'progress_samples': '이탈 구간에 포함된 progress 표본 수.',
              'max_sample_gap_seconds': '연속 progress 관측 사이 최대 시간 간격. 관측 공백을 해석할 때 참고합니다.',
              'observation_basis': '구간이 연속된 유효 progress 표본에서 만들어졌음을 표시.',
              'continuous_off_route_verified': '표본 사이에도 계속 경로 이탈이었는지 검증했는가. 현재 false.',
              'start_sensor_bundle_index / end_sensor_bundle_index': '구간 시작·끝의 선택된 0-based bundle 후보. 양쪽 중 '
                                                                     '하나가 null이면 해당 끝을 확정하지 못한 것입니다.',
              'start_match / end_match': '구간 시작·끝의 전체 매칭 근거. 내부 필드는 매칭_필드 설명과 같습니다.'},
 '경로_이탈_설정_event_detection_off_route': {'enabled': '파생 off-route 관측 구간 탐지를 실행할 수 있는지. 주행 허용 여부가 아닙니다.',
                                        'source': '탐지 설정 출처. episode.json/scenario_config 또는 runner 기본값.',
                                        'defaulted_fields': '수집 metadata에 없어서 기본값을 사용한 설정 이름들.',
                                        'expert_max_cross_track_error_m': '수집 당시 경로 오차 한계. 없으면 runner 기본 '
                                                                          '10m.',
                                        'off_route_timeout_sec': '수집 당시 이탈 timeout. 없으면 기본 3초. 이번 165549 '
                                                                 'campaign에는 1e12초로 저장되어 있어 그 값을 유지합니다.',
                                        'off_route_grace_sec': '초기 경과 시간 중 이탈 관측을 시작하지 않는 유예 구간. 없으면 runner '
                                                               '기본 2초.',
                                        'duration_clock': '실제 구간 계산에 선택한 시간 기준.'},
 '충돌_원본_요약_collision_summary': {'contact_events': 'runner가 집계한 접촉 이벤트 수. evaluator는 events.jsonl의 collision '
                                                  '개수와 비교합니다.',
                                'nonempty_contact_observations': '충돌 객체 ID가 비어 있지 않았던 관측 횟수. 충돌 이벤트 수와 다릅니다.',
                                'first_collision_receive_monotonic_ns': 'runner가 처음 접촉을 관측한 host monotonic '
                                                                        '시각, ns. collision 이벤트 기록 시각과 약간 다를 '
                                                                        '수 있습니다.',
                                'last_observed_contact_ids': '마지막 관측에서 접촉 중이던 객체 ID 목록.'},
 '전체_요약_summary': {'total_laps': 'ledger에서 읽은 서로 다른 Lap 수.',
                   'completed_laps': 'completed=true인 Lap 수.',
                   'collision_free_laps': '완주했고 collision_count=0이며 입력 오류가 없는 Lap 수. 학습 사용 승인 수가 아닙니다.',
                   'collision_events': '읽어낸 충돌 이벤트 총수. 입력 불완전 시 확인된 건수만 포함합니다.',
                   'laps_with_errors': 'errors가 하나 이상 있는 Lap 수.',
                   'evaluation_complete': '기존 충돌·완주 입력 처리에 오류가 없는지. 모든 이벤트의 프레임 매칭 또는 파생 탐지가 성공했다는 뜻은 아닙니다. '
                                          'match_status와 event_report_warnings를 함께 확인하세요.'}}


TRAFFIC_EVENTS = {
    "traffic_light_override", "traffic_light_override_verified",
    "traffic_light_override_verification_timeout",
}
RECORDED_EVENTS = {"stuck", "expert_input_warning", "scenario_intervention", "off_route"}
DETAIL_FIELDS = (
    "light_id", "target_color", "observed_color", "exact_target_color_match",
    "verification_semantics", "result", "command_result", "wait_wall_sec",
    "wait_scenario_sec", "command_to_readback_wall_sec", "reason", "message",
    "warning", "intervention", "mission_distribution_changed",
)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_off_route_config(episode, warnings):
    defaults = {"expert_max_cross_track_error_m": 10.0,
                "off_route_timeout_sec": 3.0, "off_route_grace_sec": 2.0}
    path = episode / "episode.json"
    config = {}
    source = "runner_defaults"
    try:
        with path.open(encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("scenario_config"), dict):
            raise ValueError("scenario_config must be an object")
        config = metadata["scenario_config"]
        source = "episode.json/scenario_config"
    except FileNotFoundError:
        warnings.append("episode.json missing; off-route detection uses documented runner defaults")
    except (OSError, ValueError) as exc:
        warnings.append("off-route detection unavailable: {}".format(exc))
        return {"enabled": False, "source": str(path)}
    values = {key: config.get(key, default) for key, default in defaults.items()}
    if any(not finite_number(value) or value < 0 for value in values.values()):
        warnings.append("off-route detection unavailable: invalid scenario thresholds")
        return {"enabled": False, "source": source}
    return dict(values, enabled=True, source=source,
                defaulted_fields=[key for key in defaults if key not in config])


def match_event(event, sources):
    ego = event.get("ego")
    ego = ego if isinstance(ego, dict) else {}
    evidence = {"receive_monotonic_ns": event.get("receive_monotonic_ns"),
                "ego_xy": [ego.get("x"), ego.get("y")]}
    match = match_collision(evidence, sources)
    # Intervention messages often have no ego snapshot. Expose the same receipt
    # candidate explicitly as TIME_ONLY rather than pretending spatial agreement.
    index = match["receive_time_candidate_index"]
    if not valid_xy(evidence["ego_xy"]) and index is not None:
        receipt = evidence["receive_monotonic_ns"]
        times = [row["receive_ns"] for row in sources["receipt"].values()]
        if times and min(times) <= receipt <= max(times):
            match.update(sensor_bundle_index=index, sensor_bundle_number=index + 1,
                         match_status="TIME_ONLY", match_confidence="TIME_ONLY",
                         match_interpretation="Receipt-time candidate only; event ego position unavailable.")
            match["match_reasons"] = list(sources["warnings"]) + [
                "event ego position unavailable; no spatial cross-check"]
    return match


def point_event(event_type, event, line, sources):
    ego = event.get("ego")
    ego = ego if isinstance(ego, dict) else {}
    item = {"type": event_type, "source_event": event.get("event"),
            "source_line": line, "event_sequence": event.get("sequence"),
            "receive_monotonic_ns": event.get("receive_monotonic_ns"),
            "elapsed_seconds": event.get("elapsed_seconds"),
            "route_s": event.get("route_s"), "current_link": ego.get("current_link"),
            "ego_xy": [ego.get("x"), ego.get("y")],
            "details": {key: event[key] for key in DETAIL_FIELDS if key in event}}
    item.update(match_event(event, sources))
    return item


def detect_off_route_events(records, config, sources, warnings):
    if not config["enabled"]:
        return []
    progress = [(line, event) for line, event in records
                if event is not None and event.get("event") == "progress"]
    # The runner's timeout measures host duration, not elapsed scenario time.
    clock = "receive_monotonic_ns" if any(valid_ns(e.get("receive_monotonic_ns"))
                                         for _, e in progress) else "elapsed_seconds"
    config["duration_clock"] = clock
    if progress and clock == "elapsed_seconds":
        warnings.append("progress receipt timestamps absent; off-route duration uses elapsed_seconds fallback")
    scale = 1e9 if clock == "receive_monotonic_ns" else 1
    results, segment = [], []
    previous_stamp = None

    def finish():
        if not segment:
            return
        start_line, start, start_stamp = segment[0]
        end_line, end, end_stamp = segment[-1]
        duration = (end_stamp - start_stamp) / scale
        if duration >= config["off_route_timeout_sec"]:
            first_match, last_match = match_event(start, sources), match_event(end, sources)
            results.append({
                "type": "off_route", "source_event": "progress",
                "source_line": start_line, "end_source_line": end_line,
                "start_event_sequence": start.get("sequence"), "end_event_sequence": end.get("sequence"),
                "start_receive_monotonic_ns": start.get("receive_monotonic_ns"),
                "end_receive_monotonic_ns": end.get("receive_monotonic_ns"),
                "start_elapsed_seconds": start.get("elapsed_seconds"),
                "end_elapsed_seconds": end.get("elapsed_seconds"),
                "duration_seconds": duration, "duration_clock": clock,
                "cross_track_limit_m": config["expert_max_cross_track_error_m"],
                "timeout_seconds": config["off_route_timeout_sec"],
                "max_cross_track_error_m": max(abs(e["cross_track_error_m"]) for _, e, _ in segment),
                "progress_samples": len(segment),
                "max_sample_gap_seconds": max(
                    ((b[2] - a[2]) / scale for a, b in zip(segment, segment[1:])), default=0),
                "observation_basis": "consecutive_valid_progress_samples",
                "continuous_off_route_verified": False,
                "sensor_bundle_index_base": 0,
                "start_sensor_bundle_index": first_match["sensor_bundle_index"],
                "end_sensor_bundle_index": last_match["sensor_bundle_index"],
                "start_match": first_match, "end_match": last_match,
            })
        segment.clear()

    for line, event in records:
        if event is None:  # A malformed JSONL record breaks an observed interval.
            finish()
            previous_stamp = None
            continue
        if event.get("event") != "progress":
            continue
        cte, stamp, elapsed = event.get("cross_track_error_m"), event.get(clock), event.get("elapsed_seconds")
        valid_stamp = valid_ns(stamp) if clock == "receive_monotonic_ns" else finite_number(stamp)
        if not finite_number(cte) or not valid_stamp or not finite_number(elapsed):
            finish()
            previous_stamp = None
            warnings.append("progress line {} has invalid timing or cross-track evidence".format(line))
            continue
        if previous_stamp is not None and stamp <= previous_stamp:
            finish()
            warnings.append("progress clock did not increase at line {}; interval split".format(line))
        previous_stamp = stamp
        if elapsed >= config["off_route_grace_sec"] and abs(cte) > config["expert_max_cross_track_error_m"]:
            segment.append((line, event, stamp))
        else:
            finish()
    finish()
    return results


def build_event_report(episode, records, collisions, outcome, sources):
    warnings = []
    config = load_off_route_config(episode, warnings)
    items = [dict(collision, type="collision", source_event="collision") for collision in collisions]
    for line, event in records:
        if event is None or event.get("event") == "collision":
            continue
        name = event.get("event")
        if name in TRAFFIC_EVENTS or name in RECORDED_EVENTS:
            items.append(point_event(name, event, line, sources))
        elif event.get("intervention") is True:
            items.append(point_event("scenario_intervention", event, line, sources))
    items.extend(detect_off_route_events(records, config, sources, warnings))
    # A missing outcome is unknown, not proof of an interrupted lap.
    if isinstance(outcome, dict) and (
        outcome.get("full_mission_completed") is False or
        ("full_mission_completed" not in outcome and outcome.get("status") in {
            "failed", "diagnostic_lap_incomplete", "diagnostic_window_completed"})
    ):
        finished = [(line, e) for line, e in records if e and e.get("event") == "episode_finished"]
        line, event = finished[-1] if finished else (None, {})
        item = point_event("lap_incomplete", event, line, sources)
        item.update(reason=outcome.get("reason"), scenario_status=outcome.get("status"),
                    evidence_source="outcome.json")
        items.append(item)
    items.sort(key=lambda item: item["source_line"] if item.get("source_line") is not None else float("inf"))
    for number, item in enumerate(items, 1):
        item["report_event_id"] = number
    return {"events": items, "event_counts": dict(sorted(Counter(e["type"] for e in items).items())),
            "event_detection": {"off_route": config}, "event_report_warnings": warnings}
