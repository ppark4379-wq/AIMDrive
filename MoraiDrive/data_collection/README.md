# MoraiDrive / data_collection

MORAI 시뮬레이터에서 gRPC(시뮬레이션 tick 제어) + ROS1(센서 토픽, message_filters로
동기화)로 받은 센서/차량 데이터를 LMDrive(vision_encoder)가 바로 읽을 수 있는 CARLA
스타일 데이터셋 포맷으로 변환하는 실시간 수집 파이프라인.

[MoraiDrive](../) 프로젝트의 세 축(데이터 수집 / 학습 / 대회 배포) 중 데이터 수집
부분입니다. 전체 프로젝트 구조는 상위 [README](../README.md) 참고.

> 이전에는 순수 UDP로 센서를 받았습니다 (`legacy_udp/` 참고, git에 보존만 하고
> 더는 안 씀). 출력 포맷(LMDrive JSON + 이미지 + lidar)은 그대로 두고, 전송 계층만
> gRPC+ROS1로 교체했습니다 — 시나리오/tick 제어는 gRPC, 센서 데이터는 ROS1으로 받고
> 카메라4+GNSS+IMU+LiDAR 동기화는 ROS1 내장 `message_filters.ApproximateTimeSynchronizer`
> 를 씁니다.

## 지금 당장 확인해야 할 일 (막혀있는 부분)

- `ros/ros_manager.py`의 `_cb_gnss`/`_cb_imu`/`_cb_objects`/`_cb_ego` 콜백은
  `third_party/morai_msgs`의 `.msg` 정의를 보고 필드명을 맞췄지만, 실제 MORAI ROS
  브릿지가 런타임에 그 필드들(특히 `header.stamp`)을 유효하게 채우는지는 라이브
  시뮬레이터 없이는 확인 못 했습니다. `rostopic echo /Ego_topic` 등으로 확인 필요.
- `config.py`의 `ROS.topics`, `GRPC.host/port/map_name/ego_vehicle_model`은 실제
  MORAI 환경값으로 채워야 합니다 (WSL에서 Windows 호스트 접속 시 `GRPC.host`는
  `127.0.0.1`이 아닐 수 있음).
- `collect.py`의 `_actor_data()`에서 `box`(바운딩박스)는 자리표시자 `[2.0, 1.0]`입니다
  — MORAI GT가 실제 크기를 준다면 `ros/ros_manager.py._cb_objects()`에서 채워서
  넘겨주세요.

## 폴더 구조

```
MoraiDrive/data_collection/
├── config.py                # ROS 토픽/gRPC 접속 정보, 수집 주기, 이미지 저장 옵션
├── schema.py                 # LMDrive가 기대하는 JSON 필드 정의 + 기본값 (변경 없음)
├── collect.py                 # 메인 실행 스크립트
├── ros/
│   ├── ros_manager.py        # ROS1 구독 + message_filters 동기화 (카메라4+GNSS+IMU+LiDAR)
│   └── sensor_types.py       # ROSManager가 주고받는 센서 데이터 컨테이너
├── grpc_ctrl/
│   └── morai_grpc_client.py  # gRPC 연결 + 동기 모드 전환 + tick (프레임 트리거)
├── third_party/
│   ├── grpc_inha_univ/       # MORAI 팀 제공 gRPC 클라이언트 (proto/api)
│   └── morai_msgs/           # MORAI ROS1 메시지 정의 (참고용, 실행 시엔 시스템 설치본 사용)
├── writer/
│   ├── image_writer.py       # rgb_<view>/%04d.jpg 저장 (변경 없음)
│   ├── lidar_writer.py       # lidar/%04d.npy 저장 (변경 없음)
│   └── dataset_writer.py     # measurements_full/%04d.json 등 (변경 없음)
├── tests/
│   └── test_ros_manager_sync.py   # rospy 없이 도는 message_filters 브리징 모킹 테스트
└── legacy_udp/                # 이전 UDP 파이프라인 보존 (collect.py/config.py/sync/udp/writer 구버전)
```

## 실행 흐름

`collect.py`가 하는 일:

1. `ROSManager.start()` — 카메라4(front/left/right/rear) + GNSS + IMU + LiDAR를
   `message_filters.ApproximateTimeSynchronizer`(slop=`ROS.sync_slop_sec`)로 묶어 구독.
   ego/gt_objects/traffic_light는 최신값만 별도 버퍼링 (자세한 설계는
   `ros/ros_manager.py` 상단 docstring 참고).
2. `MoraiGrpcClient.start_sync_mode()` — gRPC로 동기 모드 전환 (`tick_period_ms`).
3. 메인 루프: `grpc_client.tick(1)`이 프레임 트리거 — tick마다
   `ros_mgr.get_snapshot()`으로 이미 동기화된 센서 묶음을 꺼내옴
   (아직 새 동기화 묶음이 안 왔으면 직전 값을 재사용, 완전 새 데이터면 스킵).
4. 스냅샷을 `EgoState`/`ActorData`로 감싸 `schema.build_measurement_dict()`로
   LMDrive JSON 스키마에 맞게 변환 후 각 writer가 디스크에 기록.
5. 라우트 종료 시 `finalize()`가 `measurements_all.json`, `dataset_index.txt`를 마무리.

```bash
python collect.py --route route_moraiA_scenario1 --duration 120
```

## JSON 포맷에 들어가는 것 (`measurements_full/%04d.json`)

LMDrive의 `vision_encoder/timm/data/carla_dataset.py`가 `measurements["key"]` 식으로
**기본값 없이 바로 조회**하기 때문에, 아래 키들은 학습 단계와 무관하게 **모든 프레임에
항상 존재**해야 합니다. `schema.build_measurement_dict()`가 이걸 보장합니다 — Phase 2
필드는 실측값이 없으면 빈 리스트/0/False로 채워서, "그 loss를 아직 안 켰을 뿐" 상태로
안전하게 둘 수 있습니다.

| 필드 | Phase | 값 |
|---|---|---|
| `x`, `y`, `gps_x`, `gps_y`, `theta`, `speed` | 1 | ego 위치/헤딩/속도 (실측 필수) |
| `command`, `x_command`, `y_command` | 1 | 목표지점 (실측 필수, `command`는 지시문 없으면 고정값) |
| `is_red_light_present` | 1.5 | 신호등 리스트 (없으면 `[]` = 초록/없음으로 해석됨) |
| `is_junction`, `stop_sign`, `affected_light_id` | 2 | 기본 False/0/-1 |
| `is_vehicle_present`, `is_pedestrian_present` 등 | 2 | `actors_data`의 `tpe`에서 자동 파생 |
| `actors_data` | 2 | `{actor_id: {loc, ori, box, vel, tpe, sta, ...}}` |
| `future_waypoints` | 2 | **수집 시점엔 채울 수 없음** (미래 프레임이 아직 없음) — 아래 참고 |

`measurements_all.json`은 라우트 전체 프레임의 `{gps_x, gps_y, theta, speed}` 리스트로,
`command_waypoints`(미래 waypoint 시퀀스)를 나중에 뒤쪽 프레임에서 찾아올 때 씁니다.

> **주의**: `future_waypoints`는 이번 파이프라인(실시간 수집)에서는 채우지 않았습니다.
> "미래"는 정의상 그 시점엔 존재하지 않는 데이터라, LMDrive 원본도 이걸 수집 단계가
> 아니라 **후처리 단계**(`tools/data_preprocessing`, `tools/data_parsing`)에서 채웁니다.
> 우리도 라우트 수집이 끝난 뒤 `measurements_all.json`을 읽어서 각 프레임의
> `measurements_full/%04d.json`에 향후 N프레임의 위치를 되짚어 써주는 별도
> post-processing 스크립트가 필요합니다 (아직 미작성 — 다음 단계로 추천).

## 이미지 저장

- `rgb_<view>/%04d.jpg` — 뷰별 폴더 분리 (front/left/right/rear), CARLA처럼 4뷰를 한
  이미지로 합치는 스티칭은 하지 않음 (변환 로직 단순화 목적, 대신 LMDrive 학습 시 쓸
  Dataset 클래스도 개별 폴더를 읽도록 직접 작성해야 함 — `carla_dataset.py`의 주석 처리된
  개별 폴더 로딩 경로 참고)
- JPEG, quality 92 (`config.py`에서 조정 가능)
- 해상도는 MORAI가 주는 그대로 저장 — 리사이즈/크롭은 학습 시 transform 단계에서 처리
  (원본 해상도를 보존해야 나중에 카메라 스펙 차이에 따른 transform 파라미터를 다시 잡기 쉬움)

## 포인트클라우드 저장

- `lidar/%04d.npy` — `(N, 4)` float32 `[x, y, z, intensity]`
- CARLA 원본은 반스캔 특성 때문에 `lidar/` + `lidar_odd/` 두 스윕을 합쳐 360도를
  만들지만, MORAI LiDAR가 프레임당 이미 360도 풀스캔을 준다면 이 이원화는 **재현하지
  않는 게 맞습니다** — 그냥 `lidar/`만 쓰고, 나중에 만들 학습용 Dataset 클래스도
  `lidar_odd` concat 로직 없이 `lidar/%04d.npy` 하나만 읽게 작성하면 됩니다.

## 다음 단계

1. `config.py`의 `ROS.topics`/`GRPC.*`를 실제 MORAI 환경값으로 채우고, 실 MORAI SIM +
   ROS1 브릿지 환경에서 `python collect.py`로 짧게 돌려 JSON/이미지/lidar가 정상적으로
   쌓이는지 확인 (`tests/test_ros_manager_sync.py`는 이미 통과했지만, rospy 없이 도는
   모킹 테스트라 실제 토픽 신선도/필드명까지는 검증 못 함)
2. `future_waypoints` 후처리 스크립트 작성
3. LMDrive `vision_encoder/timm/data/carla_dataset.py`를 참고한 축소판 Dataset 클래스 작성
   (Phase 1: waypoint loss만 사용하는 버전) — `MoraiDrive/vision_encoder/`에 위치 예정
