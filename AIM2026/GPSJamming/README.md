# GPSJamming

LiDAR 터널 벽 경계 검출·RViz 시각화·플래닝용 UDP 전송 모듈입니다.

## 구조

```text
GPSJamming/
├── main.py
├── src/
│   ├── __init__.py
│   ├── Process.py
│   ├── UDP.py
│   ├── LidarDecoder.py
│   ├── ROIDownsampling.py
│   ├── Ransac.py
│   └── Boundary.py
├── config/perception.yaml
├── visualization/
│   ├── ros1_publisher.py
│   ├── lidar.rviz
│   └── run_ros1.sh
├── tools/
├── tests/
├── docs/
├── artifacts/
├── pyproject.toml
└── README.md
```

`main.py`는 설정·통신·시작과 종료를 관리합니다.
`src/Process.py`의 `WallProcessor.process(frame)`는 계산 결과 `WallResult`를 반환합니다.
LiDAR 패킷 해석·스캔 조립·시간 처리는 `LidarDecoder.py`, 소켓·송수신 큐는 `UDP.py`에 있습니다.
계산 모듈은 ROS에 의존하지 않습니다.

처리 순서: UDP 수신 → 스캔 조립 → 차량 좌표 변환 → ROI/voxel →
지면 RANSAC → 벽 지지점 검출 → 직선 경계 → 폭 기반 누락 경계 보완.
`result.boundaries` 하나를 UDP와 RViz에 전달하며 시각화에서 다시 피팅하지 않습니다.
좌표계는 base_link, 단위 m, x 전방/y 좌측/z 상방입니다.
출력은 좌우 벽 경계이며 장애물을 제외한 전체 주행가능영역이나 주행 경로는 아닙니다.

## 실행

GPSJamming 디렉터리에서 Python 3.8 이상 및 numpy/PyYAML을 사용합니다.
설치가 필요한 환경에서는 `python -m pip install -e .`로 설치할 수 있습니다.

ROS 없이 UDP 출력만 실행:

```bash
python main.py --port 2368 --boundary-host 127.0.0.1 --boundary-port 12051
```

기존 WSL의 `artifacts/ros_python` 의존성을 사용할 경우:

```bash
PYTHONPATH="$PWD/artifacts/ros_python" /usr/bin/python3 main.py --port 2368
```

ROS Noetic/RViz 디버깅은 별도 WSL 터미널에서 각각 실행합니다.

```bash
bash visualization/run_ros1.sh master
bash visualization/run_ros1.sh node _port:=2368 _boundary_udp_port:=12051
bash visualization/run_ros1.sh rviz
```

`node` 명령은 `main.py --ros`를 실행합니다. 기존 ROS private 파라미터와 토픽 이름을 유지합니다.
직접 실행할 때는 ROS 환경을 source한 뒤 `python main.py --ros --port 2368`을 사용할 수 있습니다.

```bash
/usr/bin/python3 tools/receive_wall_udp.py --full
bash visualization/run_ros1.sh audit
bash visualization/run_ros1.sh test
```

## 안내

- `config/perception.yaml`: 센서·전처리·벽 검출·보완 설정. 기존 수치 유지.
- [architecture.md](docs/architecture.md): 역할 구분 및 파일 이동 대응표.
- [wall_udp.md](docs/wall_udp.md): 플래닝용 좌표·시간·유효성 계약.
- [ros1_debug.md](docs/ros1_debug.md): ROS 토픽·RViz 사용.
- [lidar_tuning.md](docs/lidar_tuning.md): 파라미터 조정.
- `tools/audit_wall_frame.py`, `tools/replay_wall_frame.py`: 벽 프레임 저장·재현.
- `artifacts/`: 로그·저장 프레임·비교 결과·원본 백업·기존 ROS Python 의존성.

센서 시각이 매핑되지 않으면 `stamp_ns`는 발행 시각입니다. ROS 실행에서는
`stamp_source=ros_publish_time`, ROS 없는 실행에서는 `system_publish_time`입니다.
센서 취득 시각으로 사용하지 않습니다.

## 수정 범위 밖의 위치추정 연결

이전 정리에서 GPSJamming의 위치추정 기능을 제거했습니다. 외부
`AIM2026/UDP/Manager.py`와 `AIM2026/config.yaml`의 위치추정 연결은 사용자의
수정 범위 제한에 따라 그대로입니다. 그 기능을 활성화한 외부 실행은 삭제된 모듈을
불러오다 실패하며, 이 폴더의 벽 경계 단독 실행은 해당 경로를 사용하지 않습니다.
