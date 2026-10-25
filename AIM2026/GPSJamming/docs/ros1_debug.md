# ROS Noetic 벽 경계 디버깅

GPSJamming 디렉터리에서 별도 WSL 터미널마다 실행합니다.

```bash
bash visualization/run_ros1.sh master
bash visualization/run_ros1.sh node _port:=2368
bash visualization/run_ros1.sh rviz
```

MORAI LiDAR의 Destination Port는 2368입니다. 벽 경계 출력 UDP는 12051입니다.
`_boundary_udp_host`와 `_boundary_udp_port`로 경계 목적지를 바꿀 수 있습니다.

## 토픽

기본 노드 이름은 `/gps_jamming_lidar`입니다.

| 토픽 접미사 | 내용 |
|---|---|
| raw, roi, voxel, ground, nonground | 각 단계 PointCloud2 |
| wall_left, wall_right | 경계에 사용한 벽 지지점 |
| wall_representatives_left, wall_representatives_right | 직선 피팅 전 대표점 |
| wall_boundaries | 현재 프레임의 검출 경계 JSON |
| wall_boundaries_display | 폭 기반 보완을 포함한 표시 경계 JSON |
| wall_markers | 파랑(left)·주황(right) LINE_LIST |
| status | 프레임 완성도·센서 시간·수신 진단 |

EMA 진단용 `wall_boundaries_filtered`는 제거했습니다.
`wall_boundaries_display`가 RViz와 UDP의 공통 입력입니다.
보완한 선은 `inferred=true`로 표시되며 RViz에서 더 투명하게 보입니다.
Marker는 측별로 안정된 ID를 사용하고 해당 측이 사라지면 DELETE를 발행합니다.
0.5초 lifetime은 전송이 멈췄을 때 남은 선을 지우는 watchdog입니다.

```bash
rostopic hz /gps_jamming_lidar/nonground
rostopic echo -n 1 /gps_jamming_lidar/wall_boundaries_display
bash visualization/run_ros1.sh audit
/usr/bin/python3 tools/receive_wall_udp.py --full
```

센서 시간이 매핑되지 않은 경우 ROS header에는 발행 시간을 사용합니다.
`stamp_source=ros_publish_time`을 센서 취득 시각으로 해석하지 마세요.
UDP 확인 스크립트와 실제 Expert 수신기를 같은 포트에서 동시에 실행하지 않습니다.

## 검증

```bash
bash visualization/run_ros1.sh test
```

테스트는 LiDAR 입력·전처리, 벽 직선·이격거리, 보완 경계 출처,
UDP 순서·만료 및 실제 ROS Marker와 수신 좌표 일치를 확인합니다.
실제 MORAI에서의 시각 확인은 별도로 필요합니다.
