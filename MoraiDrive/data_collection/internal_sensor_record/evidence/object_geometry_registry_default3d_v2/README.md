# 기본 스케일 3D bbox registry v2

현재 수집 설정의 보행자 계열 29/29종(사람·동물·자전거 포함), NPC 차량 36/36종의 3D 표면 bbox를 검증했다. 이전 정적 unit-scale 36종의 항목은 그대로 보존한다. 전체 101개 모델 프로파일이 `geometry_3d`를 가진다. 시뮬레이터의 모든 카탈로그 모델이나 확대 스케일 전체를 검증했다는 뜻은 아니다.

- 실제 MORAI RGB/LiDAR/TF/원본 gRPC GT로 보정 후, 형상을 고정하고 다른 방향·자세에서 독립 재촬영.
- 독립 확인 592장면, 유효 LiDAR 시야 553개. 모델별 최소 6시야·4방향·2거리에서 10cm 면 여유 내 95% 포함 기준 통과.
- 실제 GT 정규화 → 보정 → manifest 변환에서 65종 모두 통과. 적용 후 최저 포함률 97.14%, 투영 좌표 기준과 중심 차이 최대 3.66cm. 원본 API GT는 수정하지 않는다.
- 차량 가속 시 2–4도 roll/pitch가 생겨 yaw만 저장하던 경로를 보완했다. 정규화 GT에 `actor_pivot_world_xyz`, `actor_rotation_rpy_deg`, `rotation_rpy_deg`를 보존하고 시간 보간에도 반영한다. 최종 객체 레코드의 `rotation_rpy_deg`는 원본 gRPC와 같은 RPY 도 단위 관례를 사용한다.
- 보행자 애니메이션 검증 속도는 2.5/3.0m/s. 적용 범위는 noisy한 위치 차분 속도가 아니라 해시 검증된 원본 actor `velocity` 파라미터(0/9/10.8km/h)로 확인한다.
- 과거 첫 측정과 현재의 API 중심 오프셋이 다른 Man1/K7은 현재 다방향 측정의 안정된 오프셋으로 실행 계약을 고정했다. 형상 보정값은 독립 검증 중 변경하지 않았다.

`recover_sensor_record_bev_geometry.py`의 기본 registry가 이 파일이다. 기존 정적 v1과 기존 sealed 데이터셋은 덮어쓰지 않았다. 이미 생성된 timeline에 full actor pose가 없고 차량이 기울어져 있으면 보정을 거부한다. 그 데이터는 새 timeline 버전을 만들고 재검증해야 한다. 이번 작업에서 기존 학습 manifest를 재발행하지는 않았다.

이번 승인은 실제 이동 후 자세를 고정한 **모델 형상·위치** 검증이다. 병진 이동 중 센서 노출/스캔/GT의 물리적 동시성, 모든 애니메이션 프레임의 최소 외접박스, 확대 스케일을 인증하지 않는다. 상위 `policy`는 물려받은 legacy 박스 소스 선택 기준이며, full3D 항목은 각 `geometry_3d`의 범위·근거가 우선한다.

전체 모델·치수는 `dynamic65.csv`, 독립 검증은 `confirmation.json`, 영상 검토는 `camera_review.json`, 실제 변환 검사는 `pipeline_application.json`, 최종 상태는 `completion.json`에 기록했다.

원본·보정 비교 갤러리: `/home/dydwns/workspace/MoraiDrive/local_integration/morai_visual_pretraining/validation/dynamic_unit65_20260929/index.html`
