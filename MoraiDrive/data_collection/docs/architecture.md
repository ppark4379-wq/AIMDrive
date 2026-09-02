# 구조와 수집 흐름

## 소스 구조

```text
MoraiDrive/data_collection/
├── config.py                 # ROS 토픽/gRPC 접속 정보, 수집 주기, 이미지 옵션
├── schema.py                 # LMDrive JSON 필드 정의와 기본값
├── collect.py                # 메인 실행 스크립트
├── geo_reference.py          # GNSS WGS84 → MGeo local 변환
├── mgeo_navigation.py        # 링크 매칭, route 탐색, command/정지선 의미
├── routes/
│   └── 2026_molit_comp_sample_route.json
├── ros/
│   ├── ros_manager.py        # ROS1 구독과 message_filters 동기화
│   └── sensor_types.py       # 센서 데이터 컨테이너
├── grpc_ctrl/
│   └── morai_grpc_client.py  # 기존 gRPC 동기 모드 실험 경로
├── writer/
│   ├── image_writer.py
│   ├── lidar_writer.py
│   └── dataset_writer.py
├── tools/                    # route 생성·backfill·검증 도구
├── tests/                    # 단위 테스트
├── docs/                     # 설계 및 운영 문서
├── third_party/              # MORAI gRPC 및 메시지 정의
└── legacy_udp/               # 더 이상 사용하지 않는 UDP 수집기
```

## 수집 흐름

1. `ROSManager.start()`가 카메라 4대, GNSS, IMU, LiDAR를 구독한다.
2. `ApproximateTimeSynchronizer`가 설정된 slop 안에서 7개 센서를 하나로 묶는다.
3. Ego, 객체, 신호등은 별도 버퍼에서 센서 기준시각과 가장 가까운 값을 결합한다.
4. GNSS를 MGeo local 좌표로 바꾸고 route tracker가 현재 링크와 진행 위치를 찾는다.
5. `route_index`, 약 25m 앞 목표점, command, junction, 정지선, 신호등 의미를 계산한다.
6. `schema.build_measurement_dict()`가 LMDrive 호환 JSON을 만든다.
7. writer가 RGB, LiDAR, GNSS, IMU, JSON, 객체 라벨을 동일 frame ID로 저장한다.
8. 종료 시 `future_waypoints`를 backfill하고 index와 episode summary를 마무리한다.

기본 `realtime` 모드는 gRPC를 사용하지 않고 새 센서 묶음마다 한 프레임을 저장합니다.
`grpc_legacy`는 이전 tick 검증 경로이며 최종 Expert 수집 방식이 아닙니다.

## MGeo route 라벨

`--route-spec`을 지정하면 각 프레임에 다음 값이 기록됩니다.

- `navigation.route_index`: 전체 MGeo 경로점에서 현재 진행 위치
- `navigation.route_link_index`, `current_link_id`: 현재 route 링크
- `navigation.route_valid`: route 중심선 3m 이내 여부
- `x_command`, `y_command`: MGeo local 좌표계의 약 25m 앞 목표점
- `command`: `1=LEFT`, `2=RIGHT`, `3=STRAIGHT`, `4=LANEFOLLOW`

명령은 현재 위치에서 30m 앞 route 링크의 MGeo `related_signal`을 사용합니다. checkpoint가
여러 진입·출발 링크의 같은 좌표에 있으면 yaw와 segment 시작점을 함께 사용합니다. 이
tie-break로 checkpoint 6→7은 `A2256W000083(left)`를 통과합니다.

MORAI `/Ego_topic` pose가 갱신되지 않는 경우가 있어 학습용 위치·속도·heading과 route
tracking은 GNSS 이동 궤적을 사용합니다. 원본 Ego는 `measurements_full.ego_raw`에 보존하고,
첫 프레임 또는 정지 시 heading fallback으로 사용합니다.

기존 수집 결과는 `tools/backfill_route_labels.py`로 라벨을 채울 수 있습니다. 동일 도로를
route에서 여러 번 지나면 `--route-start-index`로 시작 waypoint를 지정할 수 있습니다.
route를 재생성하면 index가 바뀔 수 있으므로 해당 `route.json`의 값을 기준으로 합니다.
