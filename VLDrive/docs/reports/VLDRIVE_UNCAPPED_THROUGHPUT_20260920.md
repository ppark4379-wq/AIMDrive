# 시각화 및 추론 빈도 제한 제거 실측 — 2026-09-20

현재 best checkpoint와 실제 MORAI UDP/navigation으로 60초 실행했다. 유효 waypoint 122개, 평균 **2.03Hz**였다. 이는 현재 환경의 무제한 실행 관측 처리율이며 하드웨어의 절대 최대치나 안정적인 고정 주기를 의미하지 않는다.

## 조건

- GPU: NVIDIA GeForce RTX 5060 Ti. MORAI는 계속 실행했다. 실험 전 GPU 사용률은 순간 측정 53–56%였다.
- 기존 best checkpoint, 전처리, FP32 encoder/기존 Stage-2 autocast, 11프레임 이력, hard Gumbel, reconstruction off를 유지했다.
- Dashboard 객체/HTTP 서버/브라우저 시각화를 만들지 않았다. 기존 결과 JSONL 및 실행 보고서 저장은 유지했다.
- 임시 스크립트에서 `live_pipeline.py`를 메모리로 읽고 `now-last_prediction_at < 1/runtime['prediction_hz']` 조건 한 곳만 `False`로 바꿔 실행했다. 모델과 repository 실행 코드를 수정하지 않았다.
- 입력 이력의 5Hz 목표, 0.2±0.06초 간격 검사, 큐 3개, 센서 freshness 및 출력 유효성 검사는 유지했다. 같은 anchor를 중복 추론하지 않는다.
- 모델 로딩 이후 최초 이력 수집을 포함한 60초를 측정했다. 로딩 시간은 처리율에 포함하지 않았다.
- 별도 모델 계산 실험은 live 수신/encoder 종료 후 같은 실제 입력 window로 Stage-2를 warm-up 3회, 측정 20회 실행했다. MORAI는 이때도 실행 중이었다.

## 결과

| 측정 항목 | 결과 |
|---|---:|
| 유효 waypoint 출력 | 122 / 60초 = **2.03Hz** |
| 최초~마지막 출력 구간 처리율 | 2.12Hz |
| Stage-2 forward 호출 | 132 / 60초 = 2.20Hz |
| Live Stage-2 계산 시간 중앙값 | 292.2ms |
| 센서 anchor 수신 → 유효 출력 중앙값 | 835.7ms |
| 입력 이력 초기화 | 15회 |
| 입력 큐 drop | 15개 |
| forward 이후 유효성 검사에서 폐기 | 10개 |
| 준비된 window의 Stage-2만 반복 계산 | 20회 / 3.476초 = **5.75Hz** |
| Stage-2 단독 계산 중앙값 | 169.2ms |

전체 파이프라인의 처리율과 한 출력의 지연은 다르다. 병렬 수집/encoder/Stage-2가 겹치므로 출력 2.03Hz를 지연 약 492ms라고 해석하면 안 된다. 실제 anchor→출력 지연 중앙값은 약 836ms다. 이력 초기화와 재수집이 반복되어 출력 간격이 일정하지 않았다.

Stage-2 단독 5.75Hz는 준비된 동일 token window를 재사용한 계산 성능이며, 실제 센서에서 새 visual token을 만들고 과거 이력을 갱신하는 비용을 포함하지 않는다. 전날 관측한 470ms의 역수 2.13Hz는 당시의 추정치다. 이번 수치는 다른 장면/시점에서 직접 측정했으므로 전부 시각화 제거에 따른 개선이라고 단정할 수 없다.

기본 실행 설정의 `prediction_hz=0.5`와 시각화 옵션은 변경하지 않았다. 임시 벤치마크 스크립트는 측정 후 삭제했고 결과는 남겼다. 차량 제어기는 연결하지 않았다.

## 측정 원본

- `outputs/inference/uncapped_20260920_004222/benchmark.json`: 호출별 시간 및 집계
- `outputs/inference/uncapped_20260920_004222/inference_report.json`: checkpoint 및 수신/이력 상태
- `outputs/inference/uncapped_20260920_004222/waypoints.jsonl`: 유효 출력별 waypoint, 입력 문장 및 anchor 이력
