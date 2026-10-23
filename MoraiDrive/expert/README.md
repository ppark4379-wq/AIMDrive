# Expert 계획·제어

Expert는 지도·인지·자차 상태를 받아 계획과 제어를 계산하는 라이브러리다. 센서 수신, 비전 인코더 추론, 좌표변환, UDP 송수신은 [AIM2026](../../AIM2026)에서 관리한다. 이 폴더의 별도 UDP 실행기·추론 실행기·릴레이·통신 설정·UDP 로그 RViz 도구는 제거했다.

```text
AIM2026: 센서 수신 / 추론 / 좌표변환 / 통신
                ↓ Expert 인터페이스 객체로 변환된 입력
main.on_control_tick(driving, vehicle, objects, traffic)
                ↓
PlanningProcess.run → ControlProcess.run
                ↓ ControlInput (accel, brake, front_steer)
AIM2026: 단위·부호 처리 / 제어 송신
```

통합 실행은 `AIM2026/Expert/runtime.py`가 담당한다. Expert `main.py`가 이 실행기를 호출하면 AIM2026 인지 송신기가 자동으로 시작된다. 기존 `AIM2026/main.py`의 센서 수신 전용 예제 대신 아래 명령을 사용한다. 이전 `--external-perception --drive` 명령과 `tools/run/run_expert.*`는 더 이상 사용하지 않는다.

PowerShell에서 기존 인지 실행기를 종료한 후 실행한다. 별도 모델 실행 명령은 필요 없다.

```powershell
cd C:\Users\dydwn\AIMDrive
& "C:\Users\dydwn\anaconda3\envs\vldrive\python.exe" -u -B MoraiDrive/expert/main.py --visualize --drive
```

`--drive` 생략 시 센서·추론·계획만 확인하고 차량 명령은 송신하지 않는다. `--check`는 소켓 없이 가중치 SHA256과 경로 설정을 검사한다. 센서·포트는 [AIM2026 로컬 설정](../../AIM2026/config_local.yaml)을 사용한다. RViz는 WSL에서 `bash AIM2026/Visualization/run_rviz.sh`로 별도 실행한다.

## 남긴 구조

- `main.py`: 통신 없는 `on_control_tick` 호출 인터페이스.
- `src/expert/src/autonomous_driving.py`: 기존 상태 보존 및 `execute()` 호환.
- `src/expert/src/planning/`: 공통 처리와 고속도로·정적 장애물·재밍·회전교차로·신호 계획.
- `src/expert/src/control/`: 기존 PID·Pure Pursuit 및 `ControlProcess`.
- `src/expert/src/interfaces/`: 자차·객체·계획 결과·제어 명령 자료형.
- `src/expert/src/config/parameters/`: 구간별 파라미터와 주석.
- `data/recorded_paths/`: 정적 장애물·재밍 구간 기록 경로.
- `tools/paths/`: 경로 기록·검사·가공 도구.
- `src/expert/src/network/ros/`, `runtime/ros_runtime.py`: 보관 요청한 기존 ROS 경로.

입출력 단위와 호출 흐름은 [architecture.md](docs/architecture.md), 검증 범위는 [verification.md](docs/verification.md)를 참고한다.

## 호출

한 번 생성한 `AutonomousDriving` 인스턴스를 매 tick 재사용한다. 실행 환경의 지도·경로 설정을 먼저 적용해야 한다.

```python
from MoraiDrive.expert.main import on_control_tick
from expert.src.autonomous_driving import AutonomousDriving

driving = AutonomousDriving()
# AIM2026에서 수신·변환한 Expert 인터페이스 객체를 전달한다.
command = on_control_tick(driving, vehicle, objects, traffic)
```

`python MoraiDrive/expert/main.py`는 AIM2026 통합 실행기를 관찰 모드로 실행한다. 보관된 ROS 실행은 ROS 환경에서 `bash MoraiDrive/expert/tools/run/run_ros_expert.sh`를 사용한다.

기존 scenario_runner의 VE/GT UDP 전용 도구와 VLDrive의 Expert 고정 ENU 실행 분기는 제거한 실행기에 의존하므로 더 이상 지원하지 않는다. AIM2026로 통신 경로를 통일하며 해당 옛 실행 명령을 대체 명령으로 안내하지 않는다.
