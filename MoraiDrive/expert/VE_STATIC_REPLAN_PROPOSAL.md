# VE 입력 정적 장애물 2개 이상 — Expert 쪽 개선 제안

작성: 2026-10-17 · 대상: Expert(정적 장애물 planner) 담당자 · 이 문서는 코드 변경이 아니라 분석 결과와 제안입니다.

## 요약
VE(P1) 박스로 정적 장애물 2개 이상을 만날 때 경로가 안 나오는 실주행 실패 3건을 로그로 재현했더니,
**원인이 bbox 정확도만이 아니었고 planner 쪽 정책 두 가지가 결정적**이었습니다.
같은 정지 위치에서 VE 박스로 **새로 계획하면 경로가 나옵니다**(pair081, pair096). 막히는 이유는 "한 번 blocked가 되면 다시 계획하지 않는 것"과 "준비해 둔 계획을 현재 장애물과 대조하지 않는 것"입니다.

## 근거 (실주행 로그 재현)
| 실주행 | GT 박스 +δ 여유 | VE 박스로 새 계획 | 실제 원인 |
|---|---|---|---|
| attempt_020 (rot90) | +5 cm에서 이미 경로 없음 | raw 3/9 지점만 가능 | 배치 자체의 여유 부족. Expert/인지 모두 해결 어려움 |
| pair081 | +30 cm까지 가능 | 9/9 지점 가능 | 신호 대기 중 2번째 장애물이 VE 범위(~30 m) 밖이었고, blocked 시점 로그가 `selected_routes=[right_01]`, `search={}`(탐색 없음)라서 **장애물 1개 기준 preplan이 채택된 것으로 추정**(코드 경로 확인 필요) → 2개가 보이자 `remaining_path_collision` → 영구 blocked |
| pair096 | +5 cm 가능, +10 cm 불가 | 9/9 지점 가능 | 계획은 정상 확정 → 프레임마다 박스가 흔들려 `_remaining_clear`가 통과/실패를 오가며 blocked↔follow 반복 → 6~7 m 앞에서 정지 후 영구 blocked |
재현 스크립트·결과: `MoraiDrive/.runtime/ve_static_bench_20261017/` (README 참고), `ve_static_pair081_review_20261016/`, `ve_static_pair096_review_20261016/`.

## 추가 근거 (2026-10-17 캠페인 10회 × 2, 막힌 9건 전수)
`static_stuck`으로 끝난 9건 모두, 정지한 자세에서 VE 박스 전부(중복 박스 포함)로 새로 계획하면 **경로가 나옵니다**. 같은 자세에서 GT 박스, GT +5 cm도 모두 경로가 나옵니다(`classify_stuck.py`: 9/9). 즉 막힌 원인은 최종 박스의 정확도나 배치 여유가 아니라, **먼저 확정한 계획이 이후 박스 변화로 무효화된 뒤 다시 계획하지 않는 것**입니다. 막힌 회차의 `STATIC_FAILURE`는 1~7회뿐이고 이후 영구 blocked입니다. 아래 제안 1(재계획)이 가장 직접적인 해결입니다.
(오프라인 근사: 새 planner를 정지 자세에서 한 번 호출한 결과이며, 실제 Expert 상태 전이 전체를 재현한 것은 아닙니다.)

## 제안
1. **blocked 상태에서 재계획** — `정적장애물/recorded_planner.py` `plan()`의 `failure`가 `remaining_path_collision`/`tracking_error`로 `blocked`가 된 뒤, 자차가 거의 정지(`speed < ~0.3 m/s`)하고 일정 시간(예: 1 s) 지속되면 `reset()` 후 **측정된 현재 자세와 현재 장애물**로 `_prepare()`를 다시 호출. (`reset_after_teleport`처럼 이미 reset 경로가 있음.)
2. **preplan 채택 전 장애물 대조** — `plan()`에서 `self.pending.result()`를 채택할 때, 워커가 쓴 `pending_key`(= `_obstacle_key`)와 현재 `static_objects`를 비교(개수 + 위치 허용오차). 다르면 폐기하고 측정 자세에서 재계획. 현재 key는 float 정확 일치라 VE 입력과 맞지 않으므로 `_preview_obstacle_key`처럼 반올림(예: 위치 0.1 m, 각도 5°) 비교 권장.
3. **박스 흔들림 내성(선택)** — 같은 경로가 `remaining_clear` 실패를 낼 때, 연속 k프레임(예: 3) 실패해야 blocked로 전이. 또는 실패해도 기존 확정 경로를 최대 N 프레임 유지.
4. **참고: `uncertain` 트랙** — `planning_process.run()`의 `static_objects`는 `is_static`(= VE `motion_state == 'stationary'`)만 포함. 정지 물체가 `uncertain`으로 몇 프레임 깜빡이면 그 프레임엔 장애물 집합에서 빠집니다(실측: 확정 후 약 2~3% 프레임). 이전에 stationary였던 트랙은 `uncertain`이어도 정적으로 유지하는 방안도 검토 가치가 있습니다.

## 이미 인지 쪽에서 한 것 (참고)
- `VLDrive/inference/ve_object_tracking.py` `static_geometry_fusion=True`: stationary 박스를 world 좌표로 융합(시점별 샘플, 8 m 이내 갱신 중지). 로그 리플레이에서 중심 오차 −25%, 확정 경로 무효화 −65%(pair096). 기본값 off, 러너 플래그 `--static-geometry-fusion`.
- bbox 모델 재학습(파일럿 3종)은 정적 회전 박스 5~15% 개선에 그쳤고 동적·일반 정적 박스가 악화되어 채택하지 않음.

## 검증 방법
- 위 벤치마크의 `commit_test.py`(확정 후 무효화율)와 `budget.py`(GT 여유폭)로 변경 전후 비교.
- 실주행: `run_ve_full_route_campaign.sh --seed S --runs N [--static-geometry-fusion]`를 같은 `--seed`로 A/B 반복.
