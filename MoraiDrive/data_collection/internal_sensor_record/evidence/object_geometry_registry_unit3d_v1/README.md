# 기본 스케일 정적 장애물 36종 3D bbox (2026-09-29)

선택된 정적 모델 **36/36종**의 중심 XYZ와 크기 L/W/H를 보정하고, 고정된 보정값을 별도 촬영으로 확인한 registry다. 스케일 `(1,1,1)`에만 적용한다. 기존 동적·확대 모델 항목은 이전 registry에서 그대로 유지하며 이번 검증 대상에 포함하지 않는다.

- 보정 자료: 기존 정면/측면 다방향 원본, 작은 모델 추가 원본, 11종 재촬영, 36종 대각선 864장.
- 독립 확인: 보정값 고정 후 36종 × 2거리(5/6m) × 4방향(22.5°부터 90° 간격) = **288장**. 이 확인 자료를 보정값 계산에 사용하지 않았다.
- 기준: 장면당 foreground LiDAR 10점 이상, 모델당 네 방향·두 거리 확보, 최소 6장, 각 면 10cm 여유 안의 점 포함률 각 장면 95% 이상. 최종 **288/288장 100%**.
- 영상 확인: 모델당 보정 검토 4방향과 독립 확인 4방향을 실제 MORAI 카메라 영상으로 확인. 유모차의 사람/아이, 탈것의 탑승자 등 분리된 구성 요소를 모두 포함한다. 근거리 영상의 보닛 가림은 보완 거리 영상과 LiDAR로 함께 검토했다.
- 실제 후처리 적용 재생: 36종 288장 모두 통과. 기존 normalized timeline의 true-ENU yaw와 카메라 projected-grid 변환 간 중심 차이는 이 지도에서 최대 0.01592m이며 10cm 검증 여유 이내다.

`registry.json`은 V5 schema를 유지하고 unit static 항목에 `geometry_3d`를 추가한다. 새 수집의 recovery는 이 registry를 기본 선택한다. 원본 GT·이전 registry·기존 publication은 변경하지 않는다. `static_unit_geometry.py`가 검증된 scale, 안정된 XYZ pivot offset, upright pose, 원본 API 크기/offset 계약을 확인하고 보정 중심·크기를 반환한다. manifest에는 `x/y/z/width/length/height`가 전달된다. 증거가 없거나 계약이 달라지면 보정 대상 객체를 무효 처리하며 빈 정답으로 승인하지 않는다.

이 승인은 **정적 unit 모델의 관측 가능한 형상 bbox**에 대한 것이다. 보이지 않는 mesh의 정확한 최소 경계, 모든 애니메이션 위상, 동적 객체, 다른 스케일, 카메라 노출 시간 동기까지 승인한 것은 아니다. 수집의 시간·인벤토리·센서 검증은 기존 별도 gate를 유지한다.

## 근거

- `profiles.json`: 고정된 모델별 중심 offset, 크기, 보정 원본 해시
- `confirmation.json`: 독립 촬영 288장 결과 및 원본 report 해시
- `camera_review.json`: 실제 검토 영상 경로·해시와 검토 범위
- `pipeline_application.json`: 실제 GT/manifest 함수 적용 결과
- 원본 및 before/after 갤러리: `/home/dydwns/workspace/MoraiDrive/local_integration/morai_visual_pretraining/validation/static36_surface_profiles_v4_20260929/index.html`

원본 gRPC `GetAllActorsState.bounding_box`가 수집 GT의 출처다. 둘이 서로 독립된 정답인 것은 아니다. 모델별 pivot/크기 오류와 이전 영상 분할의 구성 요소 누락을 보정했다. 과거 25종 BEV 통과와 이번 36종 XYZ/크기 검증은 범위가 다르다.

## 기존 수집본 적용 완료

기존 정적 35종의 9,102프레임이 새 bbox geometry audit를 통과했다. 새 publication은 8,532개 학습 레코드에 보정 좌표를 반영했고 `training_ready=true`로 seal되었다. 유효 장애물 양성 프레임은 3,984→5,896, 보정 bbox 레코드는 6,123개다. 센서 시각·파일/해시·Ego·waypoint·navigation·신호등·분할은 8,532레코드 모두 원본과 동일함을 검증했다.

기존 수집본에는 `SportBike` 주행 run이 없으므로 데이터 재처리는 35종이다. `SportBike`를 포함한 **bbox 자체의 검증·신규 수집용 등록은 36종** 모두 완료됐다. 새 publication 경로는 `republication.json`에 기록했다.

### 기반 registry 메타데이터 해석

`registry.json`의 기존 top-level `policy`, `limitation`, `module_sha256`, `script_sha256`, `center_candidates`, source run 정보는 이전 V1에서 계승한 비대상/기존 항목의 근거다. 특히 “never fits a new size”는 이전 후보 선택 방식의 설명이며, 새 unit static 36종에는 적용하지 않는다. 새 36종은 `static_unit_3d_validation`과 각 항목의 `geometry_3d` 및 위 확인 자료가 기준이다. 실제 적용 모듈 해시는 각 새 geometry audit의 `COMPLETE.json`에 저장되어 있다. 이미 생성한 seal과 연결을 보존하기 위해 registry 원본 바이트는 변경하지 않는다.
