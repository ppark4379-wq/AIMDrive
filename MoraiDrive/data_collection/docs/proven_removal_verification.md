# Proven 모드 제거 검증

검증일: 2026-09-16. 기준은 HEAD `03e36fef3714d7c814391ba81922722b45e5e77d`만이
아니라 **작업 시작 시점의 기존 미커밋 변경을 포함한 작업 트리**다.

## 범위와 판정

- 제거: `internal_sensor_record/proven_three_lap_runtime`의 추적 파일 130개
  (동결 Expert 94개, 동결 Scenario Runner 31개, 전용 관리 파일 5개),
  `proven` 선택 분기와 고정 참조 해시 검사.
- 유지: 최상위 `MoraiDrive/expert` 및 `MoraiDrive/scenario_runner`를 사용하는
  기존 `current` 실행 경로. 6개 시나리오와 공개 01/02/03 명령을 유지한다.
- 실제 소스 변경은 `data_collection` 안에만 있다. 최상위 `expert`와
  `scenario_runner`에는 이번 작업으로 추가한 변경이 없다. 작업 전에 있던 변경도 보존했다.
- 최상위 Expert의 파일 내용, 권한, 디렉터리, 심볼릭 링크를 포함한
  13,349개 항목을 시작 시점과 비교하여 추가·삭제·변경이 없음을 확인했다.

`run_proven_three_lap_scenario.py`와 `collect_async_raw_proven.py`라는 파일명은
기존 호출·프로세스 잠금 판별·수집 증거의 호환성을 위해 유지한다.
현재 수집에 사용되는 구현이므로 이름만 보고 삭제하지 않았다.
`collect_async_raw_proven.py` 자체의 내용은 변경하지 않았다.
별도 legacy/probe 구현의 통합은 이번 범위에 포함하지 않았다.

## 기능 보존 검증

| 확인 항목 | 결과 |
| --- | --- |
| Runner·Supervisor의 `current` 분기 전후 AST 비교 | 동일 |
| 캠페인 실행·반복·잠금·중단 처리 전후 비교 | 동일; current 정책 문자열은 동일 값으로 단순화 |
| 6개 시나리오에 전달되는 전체 설정 전후 비교 | 모두 동일 |
| 6개 시나리오의 `--check` 및 최상위 Expert 경로 확인 | 모두 통과 |
| 기본 선택과 명시적 `current` 선택의 출력 비교 | 동일 |
| 제거된 `proven`을 Python·캠페인 진입점에서 지정 | 명시적 거부; 캠페인은 PC 준비 전에 종료 |
| 임시 Expert 설정 변경 후 현재 경로 확인 | 고정된 옛 해시 없이 수용 |
| 과거 스냅샷·실행 계획 변조 감지 회귀 테스트 | 통과 |
| Data Collection unittest | 99개 통과 |
| Scenario Runner unittest | 80개 통과 |
| 두 폴더 전체 pytest | 226개 통과 |
| 변경 Python 3개 및 관련 셸 진입점 6개의 문법 검사 | 통과 |

AST 비교에서는 제거 대상인 모드 선택 인자와 `proven` 분기만 정규화했다.
센서 기록, 원본 저장, cadence 판정, GT, 실패 시 원본 보존, 잠금, 종료,
오프라인 검증, 회차별 코드·실행 계획 해시 검증은 변경하지 않았다.
런별 `software_snapshot`은 증거용 보관이며, 동결 Expert를 실행하는 기능이 아니다.

## 실제 환경 및 과거 자료 확인

- 실제 Simulator에 대한 Supervisor `--check`가 종료 코드 0으로 완료됐다.
  센서 준비 `ready=true`, `failures=[]`, 불변 설정 검사 통과,
  Simulation Time 및 External Sync OFF를 확인했다.
  분석 Python 의존성·저장소·Scenario Runner·최상위 Expert 연결도 통과했다.
- 기존 수집 자료 116건의 `software_snapshot` 및 `capture_contract`를
  현재 검증 코드로 다시 읽었으며 **116/116 모두 유효**했다.
  과거 자료나 과거 해시는 변경하지 않았다.
- 이 검사는 새 주행·Sensor Record 녹화를 시작하지 않는 읽기 전용 검사다.
  이번 작업에서 6개 시나리오 실주행, 50/100회 재수집, 모든 원본의 03 전체
  byte/GT 재처리를 수행한 것은 아니다. 코드 동등성과 위 검사 범위를 넘어
  모든 PC·모든 실주행 상황의 무결함을 100% 보증한다는 의미로 해석하면 안 된다.
  기존 Expert 행동 로직은 수정하거나 재평가하지 않았다.

## 복구 및 로컬 증거

삭제 전 자료와 상세 결과는 Git에서 제외되는 다음 경로에 보관했다.

`data_collection/.runtime/proven_removal_audit/`

- `before.json`, `before_status.txt`: 작업 전 상태 및 Expert 내용·권한·링크 기준
- `before_sources.tar.gz`: 기존 미커밋 내용을 포함한 변경 전 소스
- `frozen_full_backup.tar.gz`, `frozen_residual/`: 제거한 동결 트리와 남아 있던 보고서
- `scope_result.json`: 허용 폴더 외 변경 없음 및 Expert 무변경 확인
- `current_equivalence.json`: current 분기·6개 설정 동등성 결과
- `data_after.log`, `scenario_after.log`, `pytest_all.log`: 회귀 검사 결과
- `live_supervisor_check.log`, `historical_evidence.json`: 읽기 전용 통합·과거 자료 검사

백업은 실행 경로나 배포 대상이 아니다. 삭제된 파일은 이 백업으로 복구할 수 있다.
이 문서 및 테스트 변경과 별개로, 기존 사용자 변경을 되돌리거나 커밋·푸시하지 않았다.
