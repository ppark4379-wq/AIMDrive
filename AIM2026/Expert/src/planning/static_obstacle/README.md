# 정적 장애물 구간

`recorded_planner.py`는 기록 경로 연결·전역 경로 복귀·추종 중 재검사를 수행한다.
`corridor_graph.py`는 기록 경로 연결만으로 해결되지 않을 때 통로 전체를 탐색한다.
`avoidance.py`는 기존 래티스 정적 회피 상태를 관리한다.
`speed_policy.py`는 기존 상위 계획기의 정적 구간 속도 처리를 보관한다.

기록 데이터는 Expert 루트의 `data/recorded_paths/static_obstacle/`에 있다.

```text
static_obstacle/
├── routes.json
└── recordings/
    ├── right_01.csv
    ├── right_02.csv
    ├── right_03.csv
    ├── right_04.csv
    ├── right_05.csv
    ├── right_06.csv
    ├── right_06_retry.csv
    └── right_4.csv
```

`routes.json`의 사용 순서는 `right_01`, `right_02`, `right_03`, `right_04`,
`right_05`, `right_06_retry`다. 목록 밖의 두 파일도 원본으로 보관한다.
리팩토링에서는 CSV 좌표·행 순서·파일 내용을 변경하지 않았다.
경로 로딩 시 수행하던 초반 후진 제거·샘플 선택·연결 검사는 그대로 유지한다.

좌표는 기존 Expert의 projected-grid 지도 좌표 [m]다. 기존 파일/메타데이터의
`ENU` 표기는 과거 명칭이며, 고정 기준점 ENU로 다시 변환하거나 이동하지 않는다.

설정은 `src/config/parameters/static_obstacle.py`와
`lattice.py`의 정적 장애물 항목에서 관리한다. 구간 활성 영역은 `sections.py`다.
각 설정의 기존 값·단위·초기 계산 시점을 유지한다.

경로 기록은 `tools/paths/record_route.py`를 사용한다.
기록기는 개발용 ROS `/Ego_topic`을 사용하며 차량 제어 명령을 보내지 않는다.
대회용 UDP 주행 실행과 별도 도구이므로 기록 도구를 켜지 않아도 Expert가 실행된다.

Expert 루트에서 기존 ROS 기록 환경을 준비한 뒤 실행한다.

```bash
python tools/paths/record_route.py --section static --check-only
python tools/paths/record_route.py --section static new_route_name
```
