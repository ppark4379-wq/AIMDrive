# Expert Event Report push 전 점검 — 2026-09-16

현재 Windows 작업 공간 코드로 실제 데이터와 테스트를 확인했다.
WSL Ubuntu-20.04의 실행 파일 3개도 SHA-256이 동일하다. 이번 점검에서 push는 수행하지 않았다.

## 체크리스트

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| report.json 생성 | 통과 | 실제 3개 campaign 모두 JSON 재생성·파싱 성공, 한국어 _comments 포함 |
| 최신 3 Lap 실행 | 통과 | 165549 campaign, 3/3 완주, 충돌 4건, 입력 오류 0, 종료 코드 0 |
| 여러 campaign 모두 오류 없음 | 조건부 | 3개 모두 traceback 없이 보고서 생성. 이전 2개는 원본 outcome 누락·필드 누락으로 종료 코드 1 |
| collision spot-check | 통과 | 서로 다른 campaign의 충돌 5건을 원본 이벤트·GPS 수신 시각·GT 좌표로 독립 계산 |
| signal override spot-check | 통과 | Lap 001, sequence 242, 신호 C1256W000077, bundle index 365, TIME_ONLY |
| README와 로직 일치 | 수정 후 통과 | README에 04번 추가. 상세 문서의 HIGH 전용 index 설명을 HIGH/TIME_ONLY로 수정 |
| 노션 설명 | 미확인 | 확인할 노션 페이지가 제공되지 않아 저장소 문서만 검증 |
| 0-based 명시 | 통과 | README, 상세 문서, JSON _comments, 콘솔 출력 |
| 세 가지 매칭 상태 설명 | 통과 | CANDIDATES_AGREE / BETWEEN_FRAMES / TIME_ONLY 표 및 설명 |
| 회귀 테스트 | 통과 | test_expert*.py: 46개 통과 |

## 실제 campaign 실행

| campaign | Lap 수 | 종료 코드 | 입력 상태 |
| --- | ---: | ---: | --- |
| scenario_campaign_20260914_204500 | 1 | 1 | Lap 001 outcome.json 없음 |
| scenario_campaign_20260915_000958 | 2 | 1 | Lap 002 full_mission_completed 필드 없음. 원본 실패 상태로 lap_incomplete 이벤트는 기록 |
| scenario_campaign_20260915_165549 | 3 | 0 | 입력 오류 없음 |

종료 코드 1은 기존 입력 문제를 보고한 것으로, 코드 예외로 실행이 중단된 경우와 다르다.
그러나 “여러 campaign 모두 오류 없이 실행”이라는 조건을 충족했다고 표시하지는 않는다.
원본 기록을 수정하거나 성공으로 덮어쓰지 않았다.

## 독립 spot-check

평가기 함수를 호출하지 않고 원본 JSONL의 source_line에서 이벤트를 읽어,
전체 association의 GPS receive_monotonic_ns 차이 최소값과 전체 GT ego_xy 거리 최소값을 구했다.
객체 ID·route_s·수신 시각 오차·0-based 번호 변환도 보고서와 대조했다.

| campaign / Lap / collision | 시간 index | 위치 index | 시간차 ms | 상태 |
| --- | ---: | ---: | ---: | --- |
| 000958 / 1 / 1 | 1416 | 1416 | 69.67 | CANDIDATES_AGREE |
| 165549 / 1 / 1 | 979 | 980 | 72.99 | BETWEEN_FRAMES |
| 165549 / 1 / 2 | 1667 | 1667 | 68.25 | CANDIDATES_AGREE |
| 165549 / 2 / 1 | 1356 | 1356 | 201.47 | CANDIDATES_AGREE |
| 165549 / 3 / 1 | 1424 | 1424 | 37.18 | CANDIDATES_AGREE |

신호 표본은 165549 Lap 001의 첫 traffic_light_override다.
sequence 242, light_id C1256W000077, 시간 후보 index 365와 원본 result가 일치했다.
원본에 Ego 좌표가 없으므로 TIME_ONLY가 맞다.

HIGH는 최근접 시간·위치 후보의 일치를 의미한다. Lap 002의 201.47ms도 이 정의로
HIGH이며, 임의의 시간차 한계를 통과했다거나 물리적 노출 시각이 맞는다는 의미가 아니다.

## 재현

```bash
python3 -m unittest discover -s data_collection/tests -p "test_expert*.py" -v
python3 data_collection/tools/04_evaluate_expert_driving.py scenario_campaign_20260915_165549 --base /mnt/c/MoraiDriveData/sensor_record_auto_v1
```

이번 점검에서 실제 결과 파일은 각 campaign의 expert_driving_evaluation_v1/report.json에 갱신했다.
구현 파일·테스트·문서는 아직 Git에 추적되지 않은 파일을 포함하므로 commit 대상 확인이 필요하다.
관련 없는 .runtime, diagnostics 및 별도 decode 스크립트는 이 기능의 검증 대상에 포함하지 않았다.

## 추가 실제 데이터 확인

`scenario_campaign_20260914_194237`의 04 평가는 exit 0, 완주 3/3, 충돌 4건이었다.
시간·위치 후보는 Lap 1의 1368, Lap 2의 1514, Lap 3의 595·1285에서 각각 일치했다.
모두 0-based CANDIDATES_AGREE/HIGH이며, 신호 변경 10건도 원본과 일치했다.
Lap 1/1368 차량과 Lap 3/595 보행자는 원본 카메라 이미지로 근접 상황을 확인했다.
이 기록은 04 평가 결과이며, 원본 campaign 완료 파일이 없는 아카이브의
전체 03 검증 통과를 의미하지 않는다.

`static_010101`에서는 실제 `full_loop_static` 데이터 1·25·50번을 표본으로 사용했다.
기존 03 후처리 13단계의 산출물 해시와 입력 연결을 확인한 뒤 04를 다시 실행했다.
전체 센서 바이트 감사와 02 수집을 새로 실행한 검증은 아니다.

| 표본 | bundles | completed | 신호 변경/확인 | 신호 bundle index |
| --- | ---: | --- | --- | --- |
| 001 | 218 | true | 1/1 | 116/116 |
| 025 | 197 | true | 0/0 | 해당 없음 |
| 050 | 232 | true | 1/1 | 129/130 |

04는 exit 0, 입력 오류 0이었다. 구간 길이는 약 132.35m이며 progress의 route_s는
약 0m에서 시작한다. 신호 이벤트의 route_s와 TIME_ONLY 후보도 원본과 일치했다.
세 표본에 충돌이 없어 구간 모드의 실제 충돌 매칭은 이 검증에서 확인하지 못했다.
표본 선택용 runs.tsv는 로컬에 별도로 만들었으며 원본 수집 ledger로 간주하지 않는다.
