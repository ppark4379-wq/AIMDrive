# 재밍 구간

기존 재밍 구간의 기록 경로 계획·동적 장애물 회피·보행자 검사·도로 경계 검사를
보관하는 폴더다. 구간 진입 조건과 탐색·재계획·속도 판단은 변경하지 않았다.

| 파일 | 역할 |
|---|---|
| `section.py` | 시작/종료 영역과 구간 진행축 변환, 기록 유효성 검사 |
| `recorded_planner.py` | 기록 경로 기반 계획과 정적 회피 구현 연결; 이동체가 없으면 `corridor.py`로 넘김 |
| `corridor.py` | 누적 점유 지도(`Expert/tunnel_map.py`) 위 매 주기 재최적화 경로, 통로는 지도 도로 경계 |
| `dynamic_planner.py` | 이동 객체에 대한 추월·복귀 판단 |
| `dynamic_recovery.py` | 기존 계획 경로 재연결 및 복구 |
| `motion.py` | 정지/이동 객체 판별과 객체 식별 |
| `pedestrian_safety.py` | 선택 경로의 보행자 충돌 검사 |
| `pedestrian_speed.py` | 기존 상위 계획기의 보행자 속도 처리 |
| `road_boundary.py` | 지도 차로와 차체 형상 기반 도로 경계 검사 |
| `prepare_routes.py` | 공통 종료선 가공 함수와 기존 주행용 경로 검사 |

정지 물체만 있을 때는 누적 점유 지도와 `corridor.py`가 경로를 맡는다. VE 트랙이 2 m 이상 이동했거나
5프레임 연속 moving이면 이동체로 보고, 그동안(이후 10틱까지)은 기존 기록 경로 계획과 이동체 처리가 그대로 맡는다.

기록 데이터는 Expert 루트의 `data/recorded_paths/jamming/`에 있다.

```text
jamming/
├── routes.json                     # 기존 01~09 순서와 가공 감사 정보
├── recordings/
│   ├── jamming_01.csv ... jamming_09.csv
│   └── jamming_01.status.json ... jamming_09.status.json
└── prepared/
    └── jamming_01.csv ... jamming_09.csv
```

원본 9개와 공통 종료선 가공본 9개의 CSV 내용은 모두 보존한다.
`routes.json`의 `directory`는 `prepared`, `source_directory`는 `recordings`다.
이번 파일 이동 때 가공 함수를 다시 실행하지 않았다. 원본 해시와 선택 순서도 유지한다.

시작/종료 설정은 `src/config/jamming_section.json`,
튜닝값은 `config/parameters/jamming.py`에서 관리한다.
기록 경로 연결·차체 검사에서 공유하는 기존 정적 설정은 `lattice.py`와
`static_obstacle.py`를 참조한다.
외부/테스트 경로 라이브러리를 명시하면 기존처럼 그 라이브러리의 `section.json`을 읽는다.

CSV와 `*_enu`라는 기존 설정 키의 값은 Expert의 projected-grid 지도 좌표 [m]다.
키 이름과 수치를 보존했으며 새 고정 기준점 ENU로 재계산하지 않았다.
`section.py`는 이 지도 좌표를 구간 진행축으로 회전하는 기존 수식만 사용한다.

기록 도구는 개발용 ROS `/Ego_topic` 입력을 사용하는 별도 도구다.
대회용 UDP 주행 중에는 실행하지 않는다. Expert 루트에서 기존 ROS 기록 환경을
준비한 뒤 아래 명령을 사용한다. 기록기는 제어 명령을 발행하지 않는다.

```bash
python tools/paths/record_route.py --section jamming --check
```

신규 원본 기록을 의도적으로 공통 종료선에 맞출 때 사용하는 가공 명령은 다음과 같다.
이 명령은 `prepared`와 등록 메타데이터를 갱신하므로 단순 확인용 명령이 아니다.

```bash
python tools/paths/prepare_jamming_routes.py --end-enu -80.10 -546.66
```
