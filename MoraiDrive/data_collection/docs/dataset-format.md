# 데이터셋 구조 및 포맷

## 출력 구조

```text
Z:\DCM\data_collector\                           # DATA_ROOT
└── <작업명>/
    ├── dataset_index.txt                         # 성공 route와 프레임 수
    ├── navigation_instruction_list.txt           # 현재 빈 파일, 생성기 구현 예정
    ├── notice_instruction_list.json              # 현재 [], 생성기 구현 예정
    ├── results/
    │   └── episode_summary.json                  # route별 완료/실패와 프레임 수
    ├── maps/
    │   └── <map_id>/
    │       ├── global_info.json
    │       ├── link_set.json
    │       ├── node_set.json
    │       ├── junction_set.json                 # 제공될 때 복사
    │       ├── lane_boundary_set.json            # 제공될 때 복사
    │       ├── traffic_light_set.json            # 제공될 때 복사
    │       └── manifest.json                     # 지도 ID, 원본 경로, SHA-256
    └── data/
        └── <route>_<MM_DD_HH_MM_SS>/
            ├── route.json                        # 링크·노드·waypoint 및 지도 참조
            ├── measurements_all.json             # route 전체 프레임 요약
            ├── rgb_front/                        # %04d.jpg
            ├── rgb_left/                         # %04d.jpg
            ├── rgb_right/                        # %04d.jpg
            ├── rgb_rear/                         # %04d.jpg
            ├── lidar/                            # %04d.npy, (N,4) float32
            ├── gnss/                             # %04d.npy, raw GNSS
            ├── imu/                              # %04d.npy, raw IMU
            ├── birdview/                         # 후처리용, 현재 비어 있음
            ├── topdown/                          # 후처리용, 현재 비어 있음
            ├── 3d_bbs/                           # %04d.npy, 객체 3D box
            ├── affordances/                      # %04d.npy, 신호등/stop-sign
            ├── actors_data/                      # %04d.json, 객체 정보
            ├── measurements/                     # %04d.json, LMDrive 호환
            └── measurements_full/                # %04d.json, MORAI 전체 메타데이터
```

모든 프레임별 디렉터리는 같은 0-based 파일 번호를 사용합니다. 정상 수집에서는 RGB 4뷰,
LiDAR, GNSS, IMU, measurements, actors, affordances, 3D BB의 파일 수가 같아야 합니다.
`birdview`와 `topdown`은 raw 수집 항목이 아니므로 현재 0개가 정상입니다.

CARLA 반스캔용 `lidar_odd`와 Leaderboard의 `results/routes_*.json`은 만들지 않습니다.
Leaderboard 결과는 `results/episode_summary.json`으로 대체합니다.

## measurements 필수 필드

LMDrive loader가 기본값 없이 조회하므로 아래 필드는 모든 프레임에 존재해야 합니다.
실측값이 없는 Phase 2 필드는 빈 리스트, 0 또는 `false`로 유지합니다.

| 필드 | Phase | 값 |
|---|---|---|
| `x`, `y`, `gps_x`, `gps_y`, `theta`, `speed` | 1 | ego 위치·heading·속도 |
| `command`, `gt_command`, `x_command`, `y_command` | 1 | dense route RoadOption과 약 25m 앞 목표점 |
| `near_node_x/y`, `far_node_x/y` | 1 | near는 약 4m 앞 tracking point, far는 `x_command/y_command`와 동일 |
| `is_red_light_present` | 1.5 | 신호등 리스트 |
| `is_junction`, `stop_sign`, `affected_light_id` | 2 | MGeo 도로 의미, stop sign만 0 |
| `is_vehicle_present`, `is_lane_vehicle_present`, `is_pedestrian_present` 등 | 2 | ego 진행 경로와 충돌 가능한 actor ID |
| `actors_data` | 2 | 객체별 위치·방향·크기·속도·타입 |
| `gt_objects_raw` | 검증 | gRPC/ROS에서 변환한 차량·보행자·정적 객체 원본 |
| `traffic_lights` | 1.5/검증 | Ego 경로에 관련된 MORAI 신호 ID와 raw 상태 |
| `simulator_labels_source` | 검증 | `grpc`, `ros`, `unavailable` 중 실제 라벨 출처 |
| `grpc_label_tick_raw` | 검증 | 객체·신호 ActorStates의 원본 SyncMode tick |
| `grpc_label_timestamp_raw` | 검증 | ActorStates가 반환한 원본 simulator timestamp |
| `future_waypoints` | 2 | 현재 official dense route의 앞쪽 최대 50개 point |

`gps_x/gps_y`는 raw 위경도를 `global_info.json`의 CRS로 투영한 뒤
`local_origin_in_global`을 차감한 MGeo local 좌표입니다. 지도 정보가 없을 때만 Ego
`x/y`를 fallback으로 사용하고 경고를 출력합니다. Scenario Runner map 폴더처럼
`global_info.json`이 분리되어 있으면 `collect.py --global-info <path>`로 명시합니다.

`measurements_all.json`은 route 전체 프레임의 `gps_x`, `gps_y`, `theta`, `speed`를
모읍니다. `future_waypoints`는 수집 시점의 route index부터 최대 50개 planned point의
`[x, y, RoadOption]`으로 채웁니다.

gRPC 원본의 `frame_count`와 `elapsed_time`은 각각 `grpc_frame_count_raw`,
`grpc_elapsed_time_raw` 진단 필드로만 보존합니다. 데이터 프레임 순서는 `frame_id`, 시간축은
센서 `timestamp_ns`를 기준으로 합니다.

## 이미지

- `rgb_<view>/%04d.jpg`, JPEG quality 92
- MORAI 원본 해상도를 유지하고 resize/crop은 학습 transform에서 처리
- `rgb_full`은 만들지 않으며 LMDrive adapter가 네 뷰를 직접 읽음

## LiDAR

- `lidar/%04d.npy`: `(N,4)` float32 `[x,y,z,intensity]`
- MORAI 단일 360° 스캔을 사용하므로 `lidar_odd`를 결합하지 않음
- CARLA 전용 `-90°` 회전 보정을 적용하지 않음

## GNSS와 IMU

- `gnss/%04d.npy`: float64 `[lat,lon,alt,vel_x,vel_y,vel_z]`
- `imu/%04d.npy`: float32 `[accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z]`
