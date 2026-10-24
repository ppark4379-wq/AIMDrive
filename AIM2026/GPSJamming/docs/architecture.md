# 벽 경계 처리 구조

## 실행과 계산

- `main.py`: 옵션·설정 로드, UDP 수신/송신기 생성, 선택적 ROS 어댑터 생성, 실행·종료.
- `src/Process.py`: `LidarPipeline`으로 전처리, `WallProcessor`로 벽 경계 계산.
  `WallResult`에는 frame/raw_boundaries/boundaries/stages/wall_clouds가 들어 있습니다.
- `src/UDP.py`: 입력 소켓/수신 스레드/유한 큐, 경계 JSON 송수신. 디코더·조립기를 호출하지만 수치 해석은 포함하지 않습니다.
- `src/LidarDecoder.py`: LidarFrame/LidarPacket, 패킷 해석, TimestampMapper, LidarAssembler.
- `src/ROIDownsampling.py`: 고정 장착 자세 변환, ROI, voxel.
- `src/Ransac.py`: 지면 제거, 벽 방향 추정, 지지점·관측 범위 선택.
- `src/Boundary.py`: 차량 쪽 대표점·직선·이격거리와 벽 폭 기반 누락 경계 보완.
- `visualization/ros1_publisher.py`: ROS 메시지 포장과 Marker 생성. 계산 결과를 수정하지 않습니다.

`main.py`는 경계 ROS 발행 → 동일 경계 UDP 송신 → 점군/진단 ROS 발행 순서를 유지합니다.
`WallProcessor`는 폭 보완 상태를 소유하며 프레임마다 새로 생성하지 않습니다.
스캔 조립 방식, 수신 큐 크기·드롭 정책, 검출 임계값, 보완 만료 조건,
ROS 노드/토픽 이름, UDP schema/좌표/단위는 유지합니다.

## 이동 대응표

| 이전 | 현재 |
|---|---|
| gps_jamming/types.py, timing/timestamps.py | src/LidarDecoder.py |
| interfaces/lidar_udp.py의 디코더·조립기 | src/LidarDecoder.py |
| interfaces/lidar_udp.py의 수신기·스레드, interfaces/wall_udp.py | src/UDP.py |
| geometry/transforms.py, lidar/ROI.py | src/ROIDownsampling.py |
| lidar/ransac.py, walls.py, wall_extent.py | src/Ransac.py |
| lidar/wall_curve.py, wall_width.py | src/Boundary.py |
| pipeline.py 및 run_ros1.py의 계산 흐름 | src/Process.py |
| lidar/runtime.py의 설정 로드 | main.py |
| lidar/runtime.py의 조립기 생성·프레임 직렬화 | src/LidarDecoder.py |
| scripts/run_ros1.py의 실행·통신 관리 | main.py |
| scripts/run_ros1.py의 메시지·Marker 생성 | visualization/ros1_publisher.py |
| configs/perception.yaml | config/perception.yaml |
| configs/lidar_ros1.rviz | visualization/lidar.rviz |
| scripts의 진단·재현 도구 | tools/ |
| run_ros1.sh | visualization/run_ros1.sh |

## 검증

현재 리팩토링 전 소스는 `artifacts/refactor_baseline_20261025/`에 보관했습니다.
`artifacts/verify_layout_refactor.py`는 이 백업과 계산 본문, 실제 프레임,
전처리 결과, 경계·메타데이터를 비교하고 처리 시간을 측정합니다.
결과는 `artifacts/refactor_verification.json`에 있습니다.
`tests/`는 원본 백업 없이도 센서 입력·전처리·벽 검출·UDP·Marker·실행기를 검증합니다.

이번 WSL 환경에서는 큰 UDP 패킷이 127.0.0.1로 전달되지 않는 현상을 관측했습니다.
프로세스 간 통합 검증은 Linux 루프백 별칭 127.0.0.2에서 수행했습니다.
제품 기본 목적지 127.0.0.1과 UDP 패킷 형식은 변경하지 않았습니다.
실제 MORAI 입력/사용자 네트워크에서의 검증은 별도로 필요합니다.
