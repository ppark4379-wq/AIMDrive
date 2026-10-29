# WSL2에서 실행할 때의 UDP 설정 (`wsl_network.py`)

MORAI는 Windows에서, Expert는 WSL2에서 돌리면 두 쪽의 주소가 PC마다 다르고 WSL을 다시 시작할 때마다 바뀝니다. 저장소의
기본 설정(같은 PC용 `127.0.0.1`, 대회용 `192.168.0.1 → 192.168.0.10`)은 이 구성에 맞지 않아서, 센서와 차량 상태가 하나도
들어오지 않고 Expert가 `VE not ready: missing_gps`로 멈춥니다.

`wsl_network.py`는 실행할 때마다 그 PC의 Windows 게이트웨이와 WSL 주소를 읽어 **PC별 UDP 설정 사본**을 만듭니다.
저장소 파일은 고치지 않습니다. WSL에서 `--config`를 따로 주지 않았을 때만 동작하고, Windows나 일반 Linux, 또는 `--config`를
직접 준 실행은 지금까지와 같습니다.

## 쓰는 곳

`run_highway.sh`, `run_roundabout.sh`, `run_static.sh`, `run_tunnel.sh`와 전체 코스 대회 실행기
`utils/scenario_runner/tools/run_competition_aim2026.sh`에서 같은 옵션을 씁니다.

```bash
bash AIM2026/Expert/run_tunnel.sh --variant pedestrian --seed 2000
bash AIM2026/utils/scenario_runner/tools/run_competition_aim2026.sh --tunnel-scene pedestrian --seed 2004 --validation-events --competition-morai
```

생성된 설정은 `AIM2026/outputs/network/`(git 제외)에 남고, 실행 로그에 `[Network] ...` 줄로 경로가 나옵니다.

## MORAI의 센서 파일을 어떻게 맞출지

| 방식 | 설정 | 하는 일 |
|---|---|---|
| 이미 MORAI에 불러온 센서 파일을 그대로 쓴다 (권장) | `--morai-sensor FILE`, 환경변수 `MORAI_SENSOR_FILE`, 또는 `AIM2026/outputs/network/morai_sensor_path.txt`에 경로 한 줄 | 파일의 센서별 목적지 포트를 읽어 수신 설정을 맞추고, 파일의 주소(`hostIP`, `destinationIP`)가 현재 Windows/WSL 주소와 같은지 확인한다. 다르면 무엇을 고칠지 알려 주고 멈춘다. |
| 지정하지 않는다 | (기본) | 저장소의 대회 센서 세트(`aim_sensor.json`, 센서마다 포트 하나)에서 주소만 바꾼 `outputs/network/sensor_set_wsl.json`을 만든다. 이 파일을 MORAI Sensor Settings에서 불러와야 한다. |
| 대회 주소를 그대로 쓴다 | `--competition-addresses` | 대회 세트를 한 글자도 안 바꾸고 쓴다. 이 PC에 `192.168.0.1`(Windows)과 `192.168.0.10`(WSL) 주소가 있어야 한다. 실행 때 붙이는 명령이 출력된다. |

센서 파일 경로는 보통 `<MORAI 런처>/SaveFile/Sensor/<세트>/aim_sensor.json`입니다. `C:\...` 형식으로 적어도 됩니다.
센서 파일의 주소가 바뀌었을 때(WSL 재시작 등)는 파일의 `hostIP`(Windows), `destinationIP`(WSL)를 고치고 MORAI에서 다시
불러옵니다. 현재 주소는 `ip -4 route show default`(게이트웨이=Windows)와 `ip -4 addr show eth0`(WSL)로 볼 수 있습니다.

## 그 밖의 옵션

- `--status-bytes 181|229|245`(또는 환경변수 `MORAI_STATUS_BYTES`): MORAI가 보내는 Ego 상태 크기. 기본 229(전체 상태),
  181은 대회 빌드의 Competition Vehicle Status, 245는 예전 로컬 빌드. 틀리면 차량 상태를 못 읽습니다(`control_pose_frames: 0`).
  대회 실행기의 `--competition-morai`는 이 크기를 181로 고정하지 않고 위 값을 그대로 씁니다.
- `--morai-build competition|local`: 대회용 조향 설정(기본)이나 예전 로컬 빌드 설정.
- `--prepare-network`: 설정만 만들고 MORAI 연결 없이 끝냅니다.

## 안 될 때

- `VE not ready: missing_gps`, `VE_frames: 0`, `control_pose_frames: 0`: 센서나 차량 상태가 도착하지 않는 것입니다. 센서
  파일의 주소와 포트, Windows 방화벽(UDP 포트 허용), `--status-bytes`를 확인합니다.
- MORAI gRPC는 별도입니다. 위 `run_*.sh`는 Windows 게이트웨이를 자동으로 찾습니다. 직접 줄 때는
  `MORAI_GRPC_TARGET=<Windows 주소>:7789`를 쓰며, `tunnel_repeat.sh`는 기본값이 `127.0.0.1:7789`라 WSL에서는 이 변수를 줘야 합니다.
