# 구현 및 검증 현황

기준일: 2026-09-03

| 항목 | 상태 | 확인 내용 |
|---|---|---|
| ROS 메시지·통신 | 완료 | 최신 `morai_msgs`, Ego/객체/신호등/센서 토픽 수신 |
| 7센서 동기화 | 완료 | 카메라 4대 + GNSS + IMU + LiDAR, 수신시각 정규화 |
| LMDrive 저장 구조 | 완료 | RGB/NPY/JSON, job/map/route 구조와 파일 수 일치 |
| GNSS localization | 완료 | WGS84→MGeo local, GNSS 이동으로 heading·speed 계산 |
| 대회 route | 완료 | 16 checkpoint, 75 links, 8,958 waypoints |
| route command | 완료 | LEFT/RIGHT/STRAIGHT/LANEFOLLOW 실주행 확인 |
| 도로 의미 라벨 | 완료 | junction, stopline, affected light, red-light 확인 |
| 객체·3D BB | 완료 | 실제 객체 크기와 3D BB 레코드 일치 |
| 미래 경로 | 완료 | `future_waypoints` 최대 50프레임 backfill |
| LMDrive adapter | 완료 | RGB 4뷰, 360° LiDAR, target/detection/heatmap 로딩 |
| navigation/notice instruction | 미구현 | MORAI route/시나리오용 생성기 필요 |
| Expert 자동 수집 | 미구현 | Expert 인터페이스와 ROS SyncMode client 필요 |

## 대표 실주행 결과

```text
route command: on-route LEFT 35, RIGHT 13, STRAIGHT 17, LANEFOLLOW 179 frames
traffic light: C1256W000052 live-state match 57, red-light 57 frames
objects: 828 / valid extents 828 / 3d_bbs 828
LMDrive: RGB 4 views, lidar (40000,4), target_point (2,), det (2500,8) PASS
```

## 알려진 제약

- `/Ego_topic` 위치·속도가 정지한 채 GNSS만 이동하는 사례가 있습니다. 학습 좌표와 route
  tracking은 GNSS를 사용하고 원본 Ego는 `ego_raw`에 보존합니다.
- `route_valid`는 route 중심선 3m 이내 여부입니다. LMDrive adapter는
  `route_valid=false` 프레임을 학습 sample에서 제외합니다.
- `stop_sign`은 대회 범위에서 제외해 항상 0/false입니다.
- 손 주행 데이터에는 신뢰 가능한 Expert 제어 정답이 없습니다. 최종 imitation-learning용
  데이터는 Expert와 ROS SyncMode를 연결한 뒤 다시 수집해야 합니다.
- 센서 adapter의 shape 검증은 통과했지만 LiDAR-카메라 시각적 overlay 검증은 남았습니다.

## 검증 명령

```bash
python3 tools/validate_route_labels.py "$ROUTE_DIR"
python3 tools/validate_semantic_labels.py "$ROUTE_DIR"
python3 tools/validate_object_labels.py "$ROUTE_DIR"
python3 tools/validate_lmdrive_adapter.py "$ROUTE_DIR"
python3 -m unittest discover -s tests -p 'test_*.py'
```

route 검증이 전체 `FAIL`이어도 `on-route ...` 값은 `route_valid=true`인 프레임만 센
결과입니다. 손 주행 중 route를 이탈한 프레임은 원본에는 보존하지만 학습에서는 제외합니다.
