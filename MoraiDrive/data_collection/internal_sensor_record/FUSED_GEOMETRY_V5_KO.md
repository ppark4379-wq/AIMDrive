# 모델별 LiDAR 검증 박스 소스 (V5)

MORAI는 같은 actor의 박스를 두 곳에서 준다. 어느 한쪽도 모든 모델에 맞지 않는다.

| 소스 | 위치 | 크기 |
|---|---|---|
| gRPC `GetAllActorsState.bounding_box` (기존 GT) | 차량·자전거는 실제 중심, Scooter_02 등은 pivot | 보행자는 팔 벌린 기본 자세 크기(폭 ≈ 키, 1.6~1.8 m), 자전거는 1.06 m로 짧음 |
| ROS `/Object_topic` (Expert 주행 입력) | 항상 actor pivot (차량은 중심보다 1.1~1.5 m 뒤) | 보행자·자전거는 실제 크기, 일부 모델은 x/y가 반대, instance scale은 반영하지 않음 |

2026-09-25 static36 35 run과 full_route의 원본 LiDAR 대조 결과, 차량은 gRPC 중심이 98~100 %
포함했지만 ROS 위치는 46~95 %였다. Man5는 gRPC 박스가 57 %, ROS 크기(x/y 교환)가 100 %를
0.24 m²로 포함했다. 자전거는 gRPC 중심에 ROS 크기일 때만 100 %였다.

## 레지스트리

`tools/build_object_geometry_registry.py`가 봉인된 run들을 읽기 전용으로 사용한다. 12 m 이내 actor마다
주변 LiDAR 점에 대해 후보 박스(중심: gRPC 또는 V4 보정, 크기: gRPC, ROS, ROS x/y 교환)의 포함률을 계산한다.
모델·scale별로 관측 10회 이상, 포함률 중앙값 95 % 이상인 후보 중 면적이 가장 작은 것을 선택한다.
새 크기를 fit하지 않고 MORAI가 보고한 박스 중에서만 고른다.

```bash
python data_collection/internal_sensor_record/tools/build_object_geometry_registry.py \
  --run <run_dir> [--run <run_dir> ...] \
  --output data_collection/internal_sensor_record/evidence/object_geometry_registry_v1/registry.json
```

## V5 적용 규칙

- 레지스트리 승인 모델: 선택된 중심·크기로 학습한다. 카탈로그에 없는 차량도 여기서 승인된다.
- 관측은 충분하지만 어떤 후보도 기준을 넘지 못한 모델: V4가 통과시켜도 제외한다.
- 레지스트리 항목이 없거나 관측이 부족한 모델: V4 판정을 따른다. 단, 보행자는 gRPC 크기를 검증할 수
  없으므로 레지스트리 승인 없이는 제외한다.
- ROS 크기는 instance scale을 반영하지 않으므로 레지스트리는 obstacle scale별로 따로 기록한다.

V5 결과 폴더와 recovery 이름에는 레지스트리 SHA-256 앞 12자리가 붙는다
(`object_geometry_audit_v5_<tag>`, `ADMISSION_RECOVERY_FUSED_<tag>.json`). 레지스트리를 갱신해도
기존 결과를 덮어쓰지 않고 새 버전이 추가된다.

## 새 시나리오 수집 후 순서

1. 02로 수집하고 03으로 검증한다. 현재 레지스트리가 자동 적용되며, 근거가 없는 모델은 제외된다.
2. 새 run을 포함해 레지스트리를 다시 만든다(기존 파일은 보존하고 새 파일로 교체).
3. `reprocess_sensor_record_scale_geometry.py --geometry-version 5 --geometry-registry <새 파일>`로
   새 publication을 만든다.

선택된 중심·크기는 geometry frame의 `training_center_world_xy`, `training_size_wl_m`에 기록되고,
manifest는 이 값이 있을 때만 사용한다. 봉인된 GT, 원본 센서, V1~V4 결과는 바꾸지 않는다.
표준 경로에 레지스트리가 있으면 03 검증은 새 수집에 V5를 적용한다. 기존 데이터는
`reprocess_sensor_record_scale_geometry.py --geometry-version 5 --geometry-registry <path>`로 새 버전을 만든다.

LiDAR는 보이는 면만 관측하므로 가려진 면의 tightness와 카메라 노출 동시성은 보장하지 않는다.
