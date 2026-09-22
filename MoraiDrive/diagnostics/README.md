# diagnostics

MORAI 주행 및 센서 동기화 문제를 조사하는 진단 도구와 분석 보고서를 보관합니다.

- `sync_frame_probe.py`: 차량을 제동 상태로 유지하며 SyncMode를 진행하고, 센서 메시지의 프레임·타임스탬프와 캡처 요청에 따른 발행 동작을 확인합니다.
- `incident_20260908_062258.md`: 2026년 9월 8일 주행 중단 사건의 로그·궤적 분석과 원인 추정, 개선 방향을 정리한 보고서입니다.
- `benchmark_highway_projection.py`: 기존 선분별 계산과 배열 계산의 고속도로 플래너 출력을 비교하고 처리 시간을 측정합니다. MORAI 연결 없이 `python3 MoraiDrive/diagnostics/benchmark_highway_projection.py --iterations 30`으로 실행합니다. 지도·교통 입력 생성과 ROS 통신·시각화 시간은 측정에서 제외합니다.
- 회전교차로 진입 기록: `expert/`에서 평소처럼 `python main.py`를 실행하면 `expert/.runtime/debug_output/driving_debug_*_roundabout.jsonl`이 함께 생성됩니다. 시작 시 `[ROUNDABOUT_DEBUG] file:`로 경로를 출력합니다. 재현 후 같은 실행의 `.log`와 `_roundabout.jsonl`을 함께 확인합니다.

회전교차로 JSONL의 `plan`에는 적용 파라미터, 입력 프레임/동기화 정보, 자차 상태, 차량별 위치·속도·지도 매칭·추적 포함/제외 사유, 입력/추적 ID의 추가·제거, 0/2/4/6/8초의 상대 차량 예측 분기, 검사한 가속·지연·감속 후보별 충돌 결과와 자차 예측 위치가 들어 있습니다. `control`은 같은 `frame`의 최종 목표 속도·페달·조향과 속도 제한 출처를 기록합니다. `closest_reference_distance`는 검사 중 자차 기준점과 상대 차량 중심의 최소 거리 제곱이며, 범퍼 간격이 아닙니다. 검사가 첫 충돌이나 경로 끝에서 종료될 수 있으므로 `checked_until_s`와 `collision`을 같이 확인합니다. JSON의 `null`은 비유한 수치 또는 해당 값이 없음을 뜻합니다.

일반 기록은 0.2초 간격이고, 진입 상태·판단·선행차/충돌 차량·입력/추적 차량 목록이 바뀌면 즉시 기록합니다. 예측 계산을 재실행하지 않고 이미 계산 중인 결과를 수집하며, JSON 변환·파일 쓰기는 별도 스레드에서 처리합니다. 기록 큐가 차면 제어를 기다리게 하지 않고 `dropped_records`를 증가시킵니다. `ROUNDABOUT_DEBUG_ENABLED = False`로 끌 수 있으며 진입·속도 판단에는 사용하지 않습니다.

진입 전 후방 회전 차량에는 `ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN = 4.0`으로 차량 앞쪽의 충돌 예측 영역을 4m 확장합니다. 실제 차체를 포함한 영역이므로 기존 충돌을 누락시키지 않으며, 앞차 추종과 진입 후 판단은 기존 차체 크기를 사용합니다. 값이 0이면 추가 여유가 없습니다. 적용 여부는 `checks[].rear_front_margin_m_by_track` 및 충돌 결과의 `rear_front_margin_m`에 기록됩니다.

실행 중 생성되는 원본 기록은 `../.runtime/sync_runtime/`에 저장되며, 이 폴더는 자동 생성 캐시가 아닙니다.

`sync_frame_probe.py`는 별도 SyncMode 세션을 사용하므로 다른 프로세스가 SyncMode를 사용하지 않을 때 실행해야 합니다.
