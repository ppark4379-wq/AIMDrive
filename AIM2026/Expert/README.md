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

WSL Bash에서 실행한다. 모델 추론은 자동으로 시작된다.

```bash
cd /mnt/c/Users/dydwn/AIMDrive
/mnt/c/Users/dydwn/anaconda3/envs/vldrive/python.exe -u -B AIM2026/Expert/main.py --visualize --drive
```

별도 WSL 터미널에서 RViz를 실행한다.

```bash
cd /mnt/c/Users/dydwn/AIMDrive
bash AIM2026/Visualization/run_rviz.sh
```

PowerShell에서는 다음 명령을 사용한다.

```powershell
cd C:\Users\dydwn\AIMDrive
& "C:\Users\dydwn\anaconda3\envs\vldrive\python.exe" -u -B AIM2026/Expert/main.py --visualize --drive
```

`--check`는 소켓을 열지 않고 설정·가중치·경로를 확인한다. `--drive`를 생략하면 제어 명령을 송신하지 않는다.

## 실행 흐름

```text
main.py → runtime.py
  ├── adapter.py → 기존 지도·공식 경로 준비
  ├── AIM2026/Inference/main.py 자동 실행
  │   → UDP/Manager 센서 수신
  │   → GPSJamming 위치 추정·좌표변환 + P1/신호등 추론
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
