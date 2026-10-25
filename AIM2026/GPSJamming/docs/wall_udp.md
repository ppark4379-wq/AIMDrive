# RViz 벽 경계 UDP 인터페이스

## 실행 (WSL, GPSJamming 디렉터리)

기존 ROS master를 실행한 상태에서:

```bash
bash visualization/run_ros1.sh node _port:=2368 _boundary_udp_host:=127.0.0.1 _boundary_udp_port:=12051
```

RViz는 기존 방식으로 실행한다. ROS 토픽은 디버깅용으로 유지하며 Expert용
출력은 별도 UDP JSON이다. VE 객체 포트와 공유하지 않는다.

다른 터미널에서 수신 확인 (conda 불필요):

```bash
/usr/bin/python3 tools/receive_wall_udp.py --full
```

실제 Expert 수신기와 확인 스크립트를 같은 포트에 동시에 바인드하지 않는다.
이번 변경은 GPSJamming 내부에 한정되어 Expert 실행 코드에는 아직 연결되지 않았다.

## 데이터 계약

schema: `gps_jamming_wall_display_v1`, 기본 목적지 `127.0.0.1:12051`.
프레임당 JSON datagram 하나(최대 60000 bytes), `session`과 증가하는 `sequence` 포함.
`boundaries`는 `boundary_markers(display, ...)`에 전달한 **동일한 display 데이터**다.
재피팅, 재샘플링, 반올림, 세계 좌표 변환을 하지 않는다.

- `boundaries.left[*].points`: 파란 선 XYZ 좌표.
- `boundaries.right[*].points`: 주황 선 XYZ 좌표.
- 같은 segment 내 연속 점끼리 선을 잇는다. 다른 segment 사이를 잇지 않는다.
- 좌표계 `base_link`, 단위 m, x 전방 / y 좌측 / z 상방. RViz 선 두께가 아닌 중심선 좌표다.
- 빈 배열은 해당 측 경계 없음이다. 이전 좌표로 대체하지 않는다.
- `valid`, `complete`, 각 segment의 `inferred`와 `source`를 그대로 보존한다.
  폭 기반 보완 선은 RViz와 동일하게 전송되지만 실제 관측 경계로 간주하지 않는다.
- `sensor_timestamp_ns`, `raw_timestamp`, `timestamp_status`, `stamp_source` 등
  원본 메타데이터도 보존한다. `ros_publish_time`은 센서 취득 시각이 아니다.
- `sent_unix_ns`는 송신 시각이다. 센서 동기화에 사용하지 않는다.

## Expert에서 사용할 수신 API

GPSJamming 디렉터리를 Python 경로에 추가한 후 아래 API를 사용할 수 있다.
ROS 의존성은 없다. 이번 작업에서 Expert 파일 자체는 수정하지 않았다.

```python
from src.UDP import WallUDPReceiver, line_edges

receiver = WallUDPReceiver(port=12051, max_age_s=0.5)
try:
    # planning 주기마다 호출
    packet = receiver.poll()
    if packet is not None:
        boundaries = packet['boundaries']
        blue_edges = line_edges(boundaries, 'left')
        orange_edges = line_edges(boundaries, 'right')
finally:
    receiver.close()
```

`poll()`은 새 패킷을 읽고 최신 결과를 반환한다. 0.5초간 수신이 없으면 `None`이다.
중복/역순 패킷은 무시하며 송신 재시작 시 session 변경을 처리한다.
수신 나이는 로컬 수신 시각 기준이며 네트워크 지연이나 센서 취득 나이를 보증하지 않는다.
`None`, 빈 경계, `valid=False`를 장애물이 없는 도로로 해석하면 안 된다.
이 인터페이스는 벽 경계이며 장애물을 제외한 주행가능영역이나 주행 경로는 아니다.
Expert의 world 좌표 경로와 결합하려면 해당 스캔 시각의 자차 자세 및 VE 시각 정렬이 별도로 필요하다.
