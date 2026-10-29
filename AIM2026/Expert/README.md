# Expert 계획·제어

`src`를 열면 계획·제어·파라미터 코드를 바로 볼 수 있다. Python import는 `AIM2026.Expert.src`를 사용한다.

```text
AIM2026/Expert/
├── main.py                    # 실행 진입점: 계획 → 제어
├── runtime.py                 # 인지 실행, 입력 수명 관리, 제어 송신
├── adapter.py                 # 지도 경로 준비, 판제 입출력 자료형 연결
├── src/
│   ├── autonomous_driving.py  # 주행 상태와 프로세서 보관
│   ├── planning/              # 공통 처리·고속도로·정적·재밍·회전교차로·신호
│   ├── control/               # PID·Pure Pursuit·ControlProcess
│   ├── config/                # config.json, 구간별 parameters/
│   ├── interfaces/            # 자차·객체·경로·제어 자료형
│   ├── path/                  # 지도 로딩·경로 생성
│   ├── obstacle/              # 기존 객체 자료형 호환
│   ├── visualization/         # RViz 마커 생성
│   └── utils/
├── data/recorded_paths/       # 직접 기록한 정적·재밍 경로
├── map_data/                  # Expert 지도
├── tools/paths/               # 경로 기록·재밍 경로 가공
├── ros/                       # ROS 메시지 소스·RViz 설정
└── docs/                      # 구조·주행·통신 설명
```

## 통합 실행

### 고밀도 고속도로 반복 실행

저장소 루트에서 WSL/Linux Bash로 실행한다. 스크립트는 자신의 위치를 기준으로
경로를 찾으므로 사용자 이름이나 체크아웃 위치가 달라도 된다.

```sh
sh highway_dense_repeat.sh               # 혼합: 버스 8 + 트럭 2 + 승용차 12
sh highway_dense_repeat.sh --bus-only    # Universe 버스 20대
sh highway_dense_repeat.sh --truck-only  # Xcient P520 트럭 20대
```

혼합 교통은 합류 인접 버스 2대, 북쪽 접근 차량 5대의 남향 본선 재배치,
20–100 km/h 목표 속도를 사용한다. 세 모드 모두 차선 기반 대형차 추적을 켜며,
`config_local.yaml`의 LiDAR 보정·버스/트럭 보완 검출·VE 선행차 기억 설정을 사용한다.
완료 후 1초 간격으로 반복하며 Ctrl+C로 종료한다. 현재 고속도로 충돌 검증은
접촉을 기록만 하고 주행을 중단하지 않는다. 실행 오류는 반복하지 않고 종료한다.

다른 컴퓨터에서는 같은 커밋과 MORAI K-City 지도·차량 모델 목록, 센서 설정을
사용해야 한다. [시나리오 실행 환경 안내](../utils/scenario_runner/README.md)에 따라
Python/ROS/gRPC 환경을 준비하고 `AIM2026/config_local.yaml`의 IP·포트를 맞춘다.
모델 가중치는 Git에 포함되지 않으므로 `AIM2026/Weights/VE/`에 별도로 준비한다:
`ve_p1_20261014.pth.tar`, `signal_head_p1_20261014.pt`,
`object_head_truck_20261024.pt`. 선택 파일 `signal_phase_head_p1_20261027.pt`의
존재 여부도 원래 컴퓨터와 맞춰야 한다. 기본 모델과 객체 헤드의 SHA256은
`Inference/vision/checkpoint.py`에 고정되어 있다.

위 반복 명령은 매회 새로운 무작위 시드를 사용한다. 동일한 초기 배치·목표 속도를
재현하려면 기록된 시드를 지정해 아래처럼 실행한다(예시 시드: 123):

```sh
bash AIM2026/Expert/run_highway.sh --highway-density dense --lane-motion-tracking --highway-bus-heavy --seed 123
bash AIM2026/Expert/run_highway.sh --highway-density dense --lane-motion-tracking --highway-npc-model 2016_Hyundai_Universe --seed 123
bash AIM2026/Expert/run_highway.sh --highway-density dense --lane-motion-tracking --highway-npc-model '2013_Hyundai_Xcient(P520)' --seed 123
```

같은 시드는 초기 시나리오 재현용이며, 비동기 센서 처리·시뮬레이터 타이밍까지
동일한 주행 결과를 보장하지는 않는다.

`AIM2026/main.py`는 자율주행과 시각화를 기본으로 켜고, 사용자 폴더의 Windows `lmdrive` 환경을 자동으로 선택한다. 모델 추론도 자동으로 시작된다. WSL Bash에서 실행한다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/AIM2026
python3 main.py
```

별도 WSL 터미널에서 RViz를 실행한다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive
bash AIM2026/Visualization/run_rviz.sh
```

기본 화면은 전체 주행용 지도 + 자차를 따라 회전하는 VE ROI다. `--map`도 동일하다.
예전 자차 중심 네모 화면만 보려면 `--roi`를 붙인다. RViz Views의 `Whole Course`는
전체 코스, `Follow Ego`는 자차 주변을 보여준다. 지도는 저장된 MGeo 중심선이며
실제 도로 경계선이나 카메라 차선 검출 결과로 표시하는 것은 아니다.
전역경로 밖 일반 도로 중심선은 숨기고 고속도로 차선은 유지한다.
청록색은 고속도로 차선과 ROI, 옅은 보라색은 정적 구간 기록
참고 경로다. 현재 판제가 생성한 후보와 선택 경로는 기존 후보/선택 색으로 별도 표시한다.
회전교차로는 접근 영역·진입 판단 영역·정지선·회전 차로 감시 영역을 표시한다.
신호등 구체는 지도상의 정지선 노드 위치이며 실제 등기구 위치가 아니다. 현재 타겟의
VE 기반 판제 신호만 색으로 표시하고, 미확인·다른 타겟·오래된 상태는 회색이다.
신호 표시를 위해 Expert를 재시작해야 한다. 시각화는 센서/제어 UDP를 사용하지 않는다.
시각화 스냅샷은 최대 10Hz로 기록하고, RViz는 새 스냅샷의 변경 마커만 갱신한다.
변경 없는 마커는 만료시키지 않고, 실제 없어진 객체는 삭제한다. RViz가 다시 연결되면
전체 마커를 재전송한다. 지도 화면에서는 현재 인지 객체만 표시하고 미래 예측 박스는
숨긴다(판제 충돌 예측에는 영향 없음).

PowerShell에서는 다음 명령을 사용한다.

```powershell
cd C:\Users\autonav009\Desktop\AIMDrive\AIM2026
python main.py
```

통합 실행기의 `--check`는 차량을 움직이거나 소켓을 열지 않고 설정·가중치·경로를 확인한다 (`python3 main.py --check`). 관찰 모드는 기존 `AIM2026/Expert/main.py`를 `--drive` 없이 실행한다.

GPS 재밍 시험에서는 정상 GPS로 초기 위치를 잡은 뒤 재밍 구간에 진입한다. 기본 로컬 설정의 `Localization.enabled: true`이면 GPS가 끊길 때 IMU·차속 기반 위치 추정으로 자동 전환한다. 재밍 구간에서 실행을 새로 시작하면 초기 GPS 위치가 없어 주행할 수 없다.

위치 추정은 `AIM2026/Localization/`에서 관리하며 설정은 `AIM2026/Localization/config.yaml`에 있다. 기존 UDP 수신기의 GPS·IMU·차속 이력을 공유하므로 별도 실행이나 센서 포트가 필요 없다. GPS가 정상일 때는 GPS 위치를 사용하고, 유효 GPS가 0.15초 이상 갱신되지 않으면 마지막 GPS 기준 위치에서 IMU 방향과 부호 있는 차속을 적분한다. GPS가 복구되면 GPS 위치로 돌아간다. LiDAR 벽 검출 결과는 위치 추정에 사용하지 않는다.

콘솔의 `position_source`는 정상 수신 시 `GPS`, 추정 주행 시 `GPS_dead_reckoning`, 사용 가능한 위치가 없으면 `unavailable`이다. 사용한 위치와 센서 시각은 실행 폴더의 `control.jsonl`, 원시 입력과 추정 기록은 `localization/`에 저장한다. 초기 GPS 없음, IMU·차속 중단 또는 추정 한도(120초/250m) 초과로 위치가 무효화되면 통합 제어는 정지한다. GPS 복구 시 위치를 즉시 GPS로 되돌리므로 추정 오차만큼 위치가 바뀔 수 있다. 잘못된 좌표가 유효한 GPS처럼 계속 수신되는 상황의 탐지는 지원하지 않는다.

기본 실행은 터널의 미지원 신호 `LCS01`을 무시한다. `--competition` 실행은 기본 무시 신호가 없으며, 필요하면 `--ignore-signals LCS01`로 지정한다. 여러 신호는 쉼표로 구분한다. 이 옵션은 해당 실행의 계획·신호 정책에만 적용되며 원본 Expert 설정은 변경하지 않는다.

## 실행 흐름

```text
main.py → runtime.py
  ├── adapter.py → 기존 지도·공식 경로 준비
  ├── AIM2026/Inference/main.py 자동 실행
  │   → UDP/Manager 센서 수신
  │   → Localization: GPS 또는 IMU·차속 위치 추정 + P1/신호등 추론
  │   → 인지 결과·최신 자차 상태 전달
  └── adapter → main.on_control_tick
      → PlanningProcess → ControlProcess
      → AIM2026/UDP/Cmd.py → MORAI
```

통신 설정은 `AIM2026/config_local.yaml`, 가중치는 `AIM2026/Weights/VE/`의 `ve_p1_20261014.pth.tar`와 `signal_head_p1_20261014.pt`를 사용한다. 실행 기록은 `AIM2026/outputs/expert/`에 저장한다.

구조 정리로 계산식·구간 선택·파라미터·DR·센서 설정은 변경하지 않았다. 기존 시나리오 실행기에서 참조하는 VE·GT 도구는 `MoraiDrive/scenario_runner/legacy_expert/`로 옮겼다. 현재 통합 실행은 이 도구를 호출하지 않는다. Expert 내부 테스트·검사 전용 파일과 사용하지 않는 `rosauth`·`rosbridge_suite`, 이전 catkin 빌드 산출물은 제거했다.

자세한 입출력 단위와 처리 순서는 [구조 설명](docs/architecture.md), 센서·포트는 [통합 실행 안내](docs/udp_runtime.md)를 참고한다.

센서 수신·명령 송신은 `AIM2026/UDP`, 위치 추정·좌표변환은 `AIM2026/GPSJamming`이 담당한다. Expert 안의 `network`, `localization` 폴더는 제거했다. `Point`는 `src/interfaces/geometry.py`, `PathManager`는 `src/planning/common/reference_path.py`에서 직접 가져온다. `interfaces`는 통신 구현이 아니라 판제에 필요한 자료형이다.

기존 시나리오 도구가 사용하는 ROS 구현·실행기는 [legacy_expert/ros](../../MoraiDrive/scenario_runner/legacy_expert/ros/README.md)로 옮겼다. 현재 UDP 실행에서는 불러오지 않는다. `ros/`와 `src/visualization/`은 RViz 시각화에 필요하므로 유지한다.
# 고속도로 시나리오 테스트

저장소 루트에서 WSL로 실행한다.

```bash
bash AIM2026/Expert/run_highway.sh --check  # 센서 연결·주행 없이 구성 검사
bash AIM2026/Expert/run_highway.sh          # 시나리오 배치와 통합 Expert 실행
```

기존 `MoraiDrive/scenario_runner`의 `full_loop_highway` 구간(약 825m),
`full_loop.yaml`의 3개 NPC SpawnPoint·날씨 설정을 재사용한다.
이 검증 실행에서는 `highway_merge_static_vehicle.enabled=False`로 덮어써 합류 정차 차량을 생성하지 않는다.
원본 시나리오 YAML과 일반 주행 설정은 변경하지 않는다.
예전 `run_gt_scenario.sh highway`, `run_ve_scenario.sh highway`의 ROS/GT 주행 프로세스는
이 명령으로 시작하지 않는다. `highway_tunnel`은 터널까지 포함한 별도 시험이다.

실행 흐름은 `run_highway.sh → run_highway.py → AIM2026/main.py → Expert/main.py`
이며, 센서 수신·인지 추론·위치 추정·UDP 제어는 일반 주행과 같은 런타임을 사용한다.
시나리오는 gRPC로 차량/NPC를 배치하고 경로를 전달하며 진행을 관찰한다.
완주 또는 Ctrl+C 시 종료 파일을 통해 Expert와 인지 프로세스를 함께 정리한다.
초기부터 지속되는 MORAI `Ego` 단독 충돌 표시는 별도 기록하며,
이를 포함한 엄격한 충돌 검사는 `--strict-collisions`로 실행한다.

일반 주행은 저장소 루트에서 `python3 AIM2026/main.py`(WSL) 또는
`python AIM2026/main.py`(Windows)로 실행한다. 설치된 `lmdrive`, `vldrive` 순서로
Windows Python을 선택한다. 모델·포트 설정은 같은 `AIM2026/config_local.yaml`을 쓴다.
검증이 끝나 테스트 런처를 제거해도 일반 `main.py` 주행 경로는 그대로 유지된다.
