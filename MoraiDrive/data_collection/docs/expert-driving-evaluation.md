# Expert Event Report

수집된 campaign의 충돌·신호 개입·경로 이탈·미완주 이벤트를 기록합니다. Python 표준
라이브러리만 사용하며 로그 읽기는 ROS·시뮬레이터 없이 동작합니다. 프레임 매칭에는
03번이 생성한 association·GT 후처리 파일이 필요합니다.
시나리오와 수집 연결을 변경하지 않습니다.

## 실행

`MoraiDrive/`에서 실행합니다. Windows에서는 `python3` 대신 `python`을 사용할 수 있습니다.

```bash
python3 data_collection/tools/04_evaluate_expert_driving.py campaign_01
python3 data_collection/tools/04_evaluate_expert_driving.py test_campaign --base /tmp/expert_test
```

기본 입력은 `dataset/sensor_record_auto_v1`이며 다른 경로는 `--base`로 지정합니다.

E: 드라이브 데이터를 WSL에서 평가하는 예시:

```bash
python3 data_collection/tools/04_evaluate_expert_driving.py campaign_name \
  --base /mnt/e/MoraiDriveData/sensor_record_auto_v1
```

Windows에서는 `--base E:/MoraiDriveData/sensor_record_auto_v1`로 지정합니다.
04번은 `MORAI_STORAGE_ROOT`를 직접 읽거나 수집 경로를 변경하지 않습니다.
출력 보고서도 지정한 base 아래 campaign 폴더에 저장됩니다.

`full_loop_static` 같은 구간 주행도 동일한 `full_mission_completed=true` 필드로
완주를 인식합니다. `route_s`는 원본 이벤트 값을 그대로 보존하므로, 구간의 거리인지
전체 경로의 거리인지는 수집 시나리오 기준을 따릅니다.

```text
sensor_record_auto_v1/
  campaigns/campaign_01/runs.tsv
  campaign_01_lap_001/episode_000/events.jsonl
  campaign_01_lap_001/episode_000/outcome.json
  campaign_01_lap_002/episode_000/...
```

`runs.tsv`의 필수 열은 `lap`, `run_name`, `status`입니다. 각 Lap의 마지막 행을
선택하고 Lap 번호순으로 `run_name/episode_000/`을 읽습니다. `started`·`failed` Lap도
포함하며 수집 상태와 완주 여부는 별도로 표시합니다.

콘솔 출력과 함께 `campaigns/<campaign>/expert_driving_evaluation_v1/report.json`을
저장합니다. 재실행하면 이 보고서를 갱신하며 수집 원본은 수정하지 않습니다.
보고서 맨 위 `_comments`에는 필드·단위·상태의 한국어 설명이 있습니다.
`summary.evaluation_complete`는 충돌·완주 입력 오류 여부이며 모든 프레임 매칭의 완료를 뜻하지 않습니다.

## 판정과 보고서

- 완주는 `outcome.json`의 `full_mission_completed`가 JSON `true`인 경우만 인정합니다.
  `false`는 미완주이며, 누락 또는 잘못된 타입은 미확인입니다.
- 충돌은 `events.jsonl`의 `event == "collision"`만 셉니다.
  `collision_cleared`와 `progress`는 제외합니다. 접촉 객체 변경이나 재접촉으로
  기록된 이벤트도 각각 셉니다. 물리적 사고의 중복 제거는 수행하지 않습니다.
- 각 충돌의 객체 ID, `route_s`, 경과 시간, 원본 sequence, `receive_monotonic_ns`,
  JSONL 줄 번호를 보존합니다. outcome의 `collision_summary`는 참고용으로 보존하고,
  `contact_events`가 있으면 이벤트 수와 불일치하는 경우 오류를 표시합니다.
- `collision_free_laps`는 완주했고 충돌이 0회이며 입력 오류가 없는 Lap 수입니다.
  파일 누락·손상은 오류와 `UNKNOWN`(JSON `null`)으로 표시합니다. 읽을 수 있는
  충돌 기록은 남기되 불완전한 이벤트 파일의 전체 충돌 횟수는 확정하지 않습니다.
- `summary.collision_events`는 읽어낸 충돌 이벤트 합계입니다. 입력 오류가 있으면
  `evaluation_complete: false`와 `laps_with_errors`를 확인해야 합니다.

종료 코드는 정상 평가 `0`, Lap 입력 오류가 있는 보고서 `1`, campaign/TSV/저장 오류 `2`입니다.
충돌이나 미완주 자체는 실행 오류가 아닙니다. 빈 ledger는 유효한 평가로 처리하지 않습니다.

## 충돌 근처 센서 프레임 자동 매칭

충돌이 있으면 아래 후처리 파일로 두 후보를 독립적으로 구합니다.

- `native_associations/associations.json`: 충돌의 `receive_monotonic_ns`와
  `gps.receive_monotonic_ns` 차이가 최소인 bundle.
- `gt_timeline_v1/frames.json`: 충돌 Ego x,y와 GT `ego_xy`의 2차원 거리가 최소인 bundle.
  두 파일은 배열 순서가 아닌 `index`로 연결합니다.

| 필드 | 의미 |
| --- | --- |
| `receive_time_candidate_index` | 유일한 시간 기준 후보 |
| `position_candidate_index` | 유일한 위치 기준 후보 |
| `nearest_sensor_bundle_index` | 시간 기준 후보, 0부터 시작 |
| `nearest_sensor_frame_number` | 위 index + 1, 사람이 세는 번호 |
| `receive_time_gap_ms` | 충돌과 GPS 수신 시각의 절대 차이 |
| `position_error_m` | 시간 후보 프레임과 충돌 위치의 거리 |
| `nearest_position_error_m` | 위치 후보의 최소 거리 |
| `neighbor_position_errors` | 시간 후보의 앞·현재·뒤 위치 오차 |
| `current_link` | 충돌 이벤트의 link ID |

두 후보가 유일하게 일치하고, 충돌 수신 시각이 센서 수신 범위 안에 있으며,
동일 index의 GPS source time과 GT LiDAR source time도 일치하면 `HIGH`입니다.
후보 불일치·수신 범위 밖·source time 불일치는 `LOW`, 최근접 동률은
`AMBIGUOUS`, 입력 부재는 `UNAVAILABLE`입니다. 동률 후보 전체는
`receive_time_candidate_indices`, `position_candidate_indices`에 남깁니다.

`HIGH`는 두 최근접 후보의 일치 신뢰도이며 실제 충돌·카메라 노출 시각을 확정하지
않습니다. 임의의 ms/m 허용 오차를 적용하지 않습니다. 기존 `sensor_frame`은
`null`, `physical_collision_time_verified`는 `false`로 유지합니다.
후처리 파일 누락·손상은 `match_reasons`에 표시하며 충돌·완주 집계나 종료 코드는
바꾸지 않습니다. 후처리 결과가 준비된 뒤 같은 명령으로 재평가할 수 있습니다.

HIGH 후보 앞/뒤의 연속된 index 구간에 충돌 위치를 투영할 수 있으면 보간합니다.

- `collision_between_frames`: 구간 양 끝의 **0-based index**.
- `estimated_fraction_after_frame`: 구간 첫 index 이후 진행 비율.
- `estimated_collision_source_time_ns`, `estimated_collision_source_time_seconds`:
  GT source time의 실제 간격을 같은 비율로 보간한 추정 시각. 100ms 고정값을 쓰지 않습니다.
- `interpolation_position_error_m`: 투영 구간과 충돌 위치 사이의 잔여 오차.

인접 index 누락, 정지 상태, 증가하지 않는 source time, 구간 밖 투영은 보간하지
않습니다. 양쪽 구간의 투영 오차가 같아 모호한 경우도 비워 둡니다.
보간은 선형 위치 투영에 따른 추정이며 물리적 충돌 시각을 복원한 값은 아닙니다.

실제 `scenario_campaign_20260915_000958` Lap 001에서 두 후보 모두 `1416`,
시간차 `69.67ms`, 위치 오차 `0.082m`, 신뢰도 `HIGH`를 재현했습니다.
보간 구간은 `[1416, 1417]`, 비율 약 `0.190`, 추정 시각 `172.360029초`입니다.
같은 campaign의 Lap 002 완주 필드 누락은 별도 입력 오류로 남습니다.

보고서 경로와 기존 schema는 유지하고 추가 기능 버전은
`frame_matching_schema: expert_collision_frame_proximity_v2`로 기록합니다.
속도·신호·차선 위반은 범위에 포함하지 않습니다.

## 실제 데이터 없이 검증

테스트가 임시 폴더에 가짜 campaign을 만들고 정리하므로 수집 데이터가 필요하지 않습니다.
정상 완주, 충돌, 중복 ledger, 미완주, 누락·손상 입력, CLI 보고서 저장을 확인합니다.

```bash
python3 -m unittest discover -s data_collection/tests -p "test_expert*.py" -v
```

기본 가짜 campaign은 두 Lap이 모두 완주하고 두 번째 Lap에서 `NPC_17`과 충돌한 예입니다.
기대 집계는 완주 `2/2`, 무충돌 완주 `1/2`, 충돌 이벤트 `1`입니다.

## 후보 차이 분류 (proximity v2)

`sensor_bundle_index`는 선택된 **0-based 근처 bundle index**입니다.
HIGH 또는 TIME_ONLY일 때 채우고, `sensor_bundle_number`에는 index + 1인 순서 번호를 넣습니다.
TIME_ONLY는 Ego 좌표 없는 이벤트의 시간 후보로, 위치 검증을 통과한 값이 아닙니다.
물리적 충돌 frame 확정을 뜻하지 않습니다. LOW·동률·입력 부재에서는 두 값을 비워 둡니다.
기존 `nearest_sensor_bundle_index`는 호환성을 위해 시간 후보로 보존합니다.

| match_status | confidence | 의미 |
| --- | --- | --- |
| CANDIDATES_AGREE | HIGH | 유일한 시간·위치 후보 일치 |
| TIME_ONLY | TIME_ONLY | Ego 위치 없이 유일한 수신 시각 후보만 선택 |
| BETWEEN_FRAMES | LOW | 두 후보가 인접 index, 사이에서 충돌했을 가능성 |
| CANDIDATE_MISMATCH | LOW | 후보가 2개 이상의 index 간격으로 떨어짐 |
| INCONSISTENT_EVIDENCE | LOW | 수신 범위 밖 또는 후보 source time 연결 불일치 |
| AMBIGUOUS | AMBIGUOUS | 최근접 동률 |
| UNAVAILABLE | UNAVAILABLE | 필수 근거 부재 |

`candidate_index_distance`는 후보 index의 절대 차이입니다.
`candidate_source_time_gap_ms`에는 실제 GT source 시각 차이를 저장하므로
1 index 차이를 항상 100ms로 간주하지 않습니다.
BETWEEN_FRAMES는 양쪽 index의 association/GT source 시각 일치와 증가 순서를
확인한 경우에만 표시합니다.

예를 들어 Lap 001 / 객체 5는 시간 후보 979, 위치 후보 980이며,
`possible_collision_between_bundle_indices: [979, 980]`,
`sensor_bundle_index: null`, `match_status: BETWEEN_FRAMES`,
`match_confidence: LOW`입니다. 이는 충돌 구간의 확정이 아닌 가능한 해석입니다.
기존 HIGH의 위치 보간 결과인 `collision_between_frames`와 구분하며,
이 LOW 상태에서는 보간 비율·추정 충돌 시각을 채우지 않습니다.

## Expert Event Report

최신 출력은 학습 적합성 판정이 아닌 관찰 이벤트 보고서입니다.
기존 충돌 요약과 `collisions`는 호환성을 위해 유지하고, 각 Lap의 `events` 배열에
모든 지원 이벤트를 원본 로그 순서로 모읍니다. `report_event_id`는 이 배열의 번호이며
충돌별 `event_id`와 별개입니다. 추가 버전 필드는 `event_report_schema: expert_event_report_v1`입니다.
출력 파일은 기존 campaign의 `expert_driving_evaluation_v1/report.json`입니다.

- `collision`: 기존 충돌 기록과 매칭 결과를 그대로 재사용.
- `traffic_light_override`, `traffic_light_override_verified`,
  `traffic_light_override_verification_timeout`: 원본 종류를 구분해 기록.
  신호 ID, 명령/확인 결과는 `details`에 남깁니다.
- `off_route`: 저장된 `episode.json/scenario_config`의
  `expert_max_cross_track_error_m`, `off_route_timeout_sec`, `off_route_grace_sec`를 사용.
  절대 cross-track 오차가 한계를 넘는 연속 progress 관측의 시간 범위를 기록합니다.
- `lap_incomplete`: outcome의 명시적 미완주 또는 완주 필드가 없는 실패 종료 상태를
  이유와 함께 기록. outcome 누락·손상만으로 미완주를 만들어내지 않습니다.
- `stuck`, `expert_input_warning`, `scenario_intervention`, 명시적 `off_route`:
  해당 이름의 원본 이벤트가 있을 때 기록합니다. 임의의 정지·제어 이상 탐지는 하지 않습니다.
  다른 이름이라도 `intervention: true`이면 `scenario_intervention`으로 기록하고
  `source_event`에 원래 이름을 보존합니다. 신호 개입을 일반 개입으로 중복 집계하지 않습니다.

Ego 좌표 없는 이벤트도 기존 matcher의 수신 시각 후보를 재사용합니다.
유일한 후보이며 수신 범위 안인 경우 `sensor_bundle_index`를 채우되
`match_confidence`와 `match_status`를 **TIME_ONLY**로 표시합니다.
위치 검증이 된 HIGH와 구분하며, 동률·범위 밖에서는 선택 index를 비워 둡니다.
모든 bundle index는 계속 0-based입니다.

경로 이탈 구간은 `start_sensor_bundle_index`, `end_sensor_bundle_index`,
`start_match`, `end_match`에 양 끝 후보와 근거를 저장합니다.
`duration_seconds`, `max_cross_track_error_m`, 표본 수·최대 표본 간격도 보존합니다.
runner의 timeout은 host 시간 기준이므로 수신 monotonic 시간으로 계산하며,
전체 progress에 수신 시각이 없을 때만 `elapsed_seconds`로 대체하고 경고합니다.
잘못된 progress, 손상된 JSONL, 시간 역행은 구간을 끊습니다.
이는 연속된 **관측 표본**의 범위이며, 표본 사이의 지속적 이탈까지 입증하지 않습니다.

episode metadata가 없으면 확인된 runner 기본값(10m, 3초, grace 2초)을 사용하고
`event_report_warnings`와 `event_detection.off_route`에 출처를 기록합니다.
설정 파일이 손상됐거나 값이 유효하지 않으면 파생 off-route 탐지를 중단하고 경고합니다.
이번 165549 campaign에는 timeout이 `1e12초`로 저장되어 있어 그 값을 유지합니다.
실제 기록된 최대 cross-track 오차도 세 Lap 모두 약 3.7m로 10m 미만입니다.

`events`에는 EXCLUDE, REJECT, BAD, usable_for_training 같은 학습 판정 필드를
추가하지 않습니다. 원본 데이터 삭제, clip 필터링, 학습 공개 상태 변경도 수행하지 않습니다.
`event_counts`는 관찰된 종류별 개수이며 `event_report_warnings`는 탐지 입력의 한계입니다.

실제 165549 campaign 재실행 결과:

| Lap | collision | traffic_light_override | traffic_light_override_verified |
| --- | ---: | ---: | ---: |
| 001 | 2 | 3 | 3 |
| 002 | 1 | 2 | 2 |
| 003 | 1 | 4 | 4 |

미완주·파생 off-route 이벤트는 이 실행에서 검출되지 않았습니다.
