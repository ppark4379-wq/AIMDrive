# MoraiDrive / data_collection

MORAI 시뮬레이터에서 UDP로 받은 센서/차량 데이터를 LMDrive(vision_encoder)가
바로 읽을 수 있는 CARLA 스타일 데이터셋 포맷으로 변환하는 실시간 수집 파이프라인.

[MoraiDrive](../) 프로젝트의 세 축(데이터 수집 / 학습 / 대회 배포) 중 데이터 수집
부분입니다. 전체 프로젝트 구조는 상위 [README](../README.md) 참고.

## 지금 당장 해야 할 일 (막혀있는 부분)

`udp/parsers.py`의 `parse_ego_info()` / `parse_object_info()`는 **아직 구현 안 됨**
(`NotImplementedError`). MORAI의 UDP 패킷 바이트 레이아웃(필드 순서/타입/오프셋)을 정확히
모르는 상태에서 struct 포맷을 추측해서 채우면, 에러 없이 조용히 잘못된 숫자가 쌓이는
게 훨씬 위험해서 비워뒀습니다. 다음 중 하나를 주시면 바로 채울 수 있습니다.

- MORAI SDK/경진대회 킷에 포함된 패킷 구조체 정의 (`ctypes.Structure` 코드나 스펙 문서)
- 또는 사용 중인 MORAI 버전/경진대회 킷 이름 (알려진 레퍼런스가 있으면 참고해서 작성)

`parse_camera_jpeg()` / `parse_lidar_points()`는 "JPEG 바이트" / "flat float32 xyzI
배열"이라는 일반적인 관례를 가정하고 작성했습니다. 이것도 MORAI 쪽 실제 인코딩과 다르면
알려주세요.

## 폴더 구조

```
MoraiDrive/data_collection/
├── config.py              # UDP 포트/IP, 수집 주기(Hz), 이미지 저장 옵션
├── schema.py               # LMDrive가 기대하는 JSON 필드 정의 + 기본값
├── collect.py               # 메인 실행 스크립트
├── udp/
│   ├── socket_receiver.py  # 범용 non-blocking UDP 수신 스레드 (완성)
│   ├── reassembler.py      # 큰 payload(이미지/라이다) 청크 재조립 (필요시 사용, 헤더 포맷 TODO)
│   └── parsers.py          # 패킷 바이트 -> dict/ndarray 파싱 (ego/object는 TODO)
├── sync/
│   └── frame_sync.py       # 비동기 UDP 토픽들을 하나의 프레임으로 정렬 (완성)
└── writer/
    ├── image_writer.py     # rgb_<view>/%04d.jpg 저장 (완성)
    ├── lidar_writer.py     # lidar/%04d.npy 저장 (완성)
    └── dataset_writer.py   # measurements_full/%04d.json, measurements_all.json, dataset_index.txt (완성)
```

## 실행 흐름

`collect.py`가 하는 일:

1. 토픽별로 `UdpTopicReceiver` 5~8개(ego, object, camera×N, lidar)를 각자 스레드로 띄움
   → 각 토픽은 "가장 최근 값 + 수신 시각"만 캐싱 (히스토리 불필요)
2. `FrameSynchronizer.tick()`을 `collect_hz`(기본 10Hz) 주기로 호출
   → ego/lidar/카메라가 전부 `max_staleness_sec` 이내에 신선하면 프레임 하나로 스냅샷,
     아니면 그 틱은 스킵 (오래된 값을 억지로 끼워맞추지 않음)
3. 스냅샷된 프레임을 `EgoState`/`ActorData`로 감싸 `schema.build_measurement_dict()`로
   LMDrive JSON 스키마에 맞게 변환 후 각 writer가 디스크에 기록
4. 라우트 종료 시 `finalize()`가 `measurements_all.json`, `dataset_index.txt`를 마무리

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

1. `udp/parsers.py`의 ego/object 파서를 MORAI 실제 스펙으로 채우기 (막힌 지점)
2. `python collect.py`로 짧게 돌려서 JSON/이미지/lidar가 정상적으로 쌓이는지 확인
3. `future_waypoints` 후처리 스크립트 작성
4. LMDrive `vision_encoder/timm/data/carla_dataset.py`를 참고한 축소판 Dataset 클래스 작성
   (Phase 1: waypoint loss만 사용하는 버전) — `MoraiDrive/vision_encoder/`에 위치 예정
