# 수집 실행 스택의 컴포넌트 경계

대규모 수집은 메인 시나리오 러너, 데이터 수집기, 검증된 Expert를 역할별로 분리한다.

| 역할 | 소스 위치 | 책임 |
|---|---|---|
| 시나리오 러너 | ../../../scenario_runner/ | 공식 경로, 날씨·시간, 장애물·NPC·신호, lap lifecycle |
| 데이터 수집기 | ../async_capture/, ../native_capture/, ../tools/ | Sensor Record, clock/readiness, ROS·GT 원본, 저장·품질 검증 |
| Expert | expert/ | /ctrl_cmd를 만드는 검증된 로컬 Expert fork |

실행 경로는 다음과 같다.

    02_collect_internal_sensor_record_laps.sh
      -> tools/collect_scenario_laps.sh
         -> tools/run_sensor_record_scenario.py
            -> tools/run_proven_three_lap_scenario.py  수집 어댑터
               -> 저장소 메인 scenario_runner/
               -> proven_three_lap_runtime/expert/

proven_three_lap_runtime/에는 Scenario Runner 복사본을 두지 않는다. 시나리오 동작의 원본은 저장소 메인 scenario_runner/ 하나이며, 이 폴더에는 별도 안전·주행 변경이 있는 Expert와 그 무결성 증빙만 유지한다.

## Expert 증빙

- PROVEN_RUNTIME_MANIFEST.json: 실행 허용 Expert 파일의 SHA256
- UPSTREAM_PROVENANCE.json: 최초 복사 시점 Expert 파일의 SHA256
- LOCAL_FORK_CHANGES.json: 허용된 Expert 변경과 현재 SHA256
- audit_local_forks.py: 누락·미기록 변경·해시 불일치를 실패 처리

    python3 data_collection/internal_sensor_record/proven_three_lap_runtime/audit_local_forks.py

수집 기능은 data_collection에, 환경·actor·traffic 동작은 메인 scenario_runner에, 주행 판단은 동결 Expert에 둔다. .runtime/, 수집 결과, validation 산출물은 이 구조 정리의 대상이 아니다.
