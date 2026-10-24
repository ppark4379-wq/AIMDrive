# Expert 구조와 호출 흐름

센서·모델·좌표변환·UDP는 AIM2026의 책임이다. Expert의 입력은 이미 기존 지도 좌표계로 맞춰진 자차·객체 상태이며 출력은 제어 계산 결과다.

```text
AIM2026/Expert/
├── main.py                         # on_control_tick: 계획 → 제어
├── runtime.py                      # 통합 실행·인지 프로세스·제어 송신
├── adapter.py                      # 기존 경로 준비·판제 자료형 연결
├── data/recorded_paths/
│   ├── static_obstacle/             # recordings, routes.json
│   └── jamming/                     # recordings, prepared, routes.json
├── tools/paths/                     # 경로 기록·가공
├── ros/                            # ROS 메시지 소스·RViz 설정
├── tools/run/run_ros_expert.sh      # 보관된 ROS 실행
└── src/
    ├── autonomous_driving.py        # 상태 소유, execute() 호환
    ├── interfaces/                 # 입출력 자료형
    ├── planning/
    │   ├── planning_process.py      # 계획 처리 순서
    │   ├── section_planning.py      # 기존 구간 선택 순서
    │   ├── common/                  # 기준 경로·객체 선택·속도·ACC
    │   ├── highway/
    │   ├── static_obstacle/
    │   ├── jamming/
    │   ├── roundabout/
    │   ├── lattice/
    │   ├── traffic/
    │   └── ve_planning.py           # 기존 인지 기반 계획 분기
    ├── control/                    # ControlProcess, PID, Pure Pursuit
    ├── config/parameters/          # 구간별 설명·단위·파라미터
    ├── localization/               # 기존 Point/PathManager 호환 import
    ├── network/ros/                # 보관된 ROS 통신
    ├── runtime/ros_runtime.py
    └── utils/
```

`on_control_tick`은 `driving.planning.run()` 다음에 `driving.control.run()`을 호출한다. `AutonomousDriving.execute()`도 같은 순서다. PID 누적값, 이전 조향·속도, 합류 유지값, 기록 경로 상태는 같은 인스턴스에 유지한다.

| 경계 | 값 |
|---|---|
| 자차 위치 | 기존 Expert 지도 projected-grid 좌표, m |
| 자차 yaw / 속도 | rad / m/s |
| 객체 위치 / heading | 같은 지도 좌표, m / degree |
| 제어 출력 | accel/brake 0~1, front_steer rad |

UDP 조향 정규화·부호·기어·패킷 생성은 Expert 밖에서 담당한다. GPS/ENU 객체를 그대로 Expert 지도 좌표로 간주하지 않는다. 인지 시점에 변환한 객체를 최신 자차 pose로 중복 변환하지 않는다.

`planning/common`은 구간 공통 입력 준비와 속도 종합을 담당한다. 구간 회피 알고리즘을 합친 것은 아니다. 구간 선택 순서·조기 반환·상태 초기화·수식은 변경하지 않았다.

기록 CSV 26개(정적 8, 재밍 원본 9, 재밍 가공 9)와 사용 경로 순서를 유지했다. 파라미터는 `config/parameters`, 공통 PID·Pure Pursuit·ACC는 `config/config.json`에서 관리한다.
