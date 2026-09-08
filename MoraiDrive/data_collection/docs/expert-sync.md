# Expert 및 ROS SyncMode 전환 명세

최종 대회용 자동 수집은 gRPC로 SyncMode를 제어하거나 tick을 진행하지 않습니다.
MORAI Simulation Time Mode와 ROS SyncMode 서비스 기반 `collect.py --mode sync`를
사용하고, SyncMode에서 멈추는 객체·신호 publisher만 read-only gRPC로 보완합니다.

## 현재 손 주행 검증

- MORAI: `Real Time Mode`, `External Sync Off`
- 차량 제어: 키보드 또는 게임 휠
- 시간 기준: 센서 `timestamp_ns`
- 목적: 센서, writer, 좌표계, 객체, 신호등 검증
- 최종 Expert 학습 데이터로 간주하지 않음

```bash
python3 collect.py \
  --mode realtime \
  --job competition_test \
  --route manual_drive_test
```

## 최종 Expert 자동 수집

- MORAI: `Simulation Time Mode`, `Synchronous Mode On`
- 차량 제어: Expert의 tick별 steer/throttle/brake
- 시간 진행: MORAI ROS SyncMode 서비스
- 필수 시작 입력: 카메라 4대, LiDAR, Sync Ego 상태 (GNSS/IMU 미사용)
- 객체와 관련 신호 상태는 동일 tick의 gRPC `GetAllActorsState`가 필수이며,
  exact tick 또는 Ego pose가 다르면 해당 프레임을 저장하지 않음
- Sync Mode 소유자: `collect.py` (시작·제어·tick·종료 전담)
- Scenario Runner: 환경 구성과 Expert 실행만 담당하며 Sync Mode를 설정하지 않음
- 고정 `time_step=100ms`, 시뮬레이션 시간 기준 10Hz
- gRPC `SetSynchronousMode`, `Tick`, `SyncTimestamp`는 최종 루프에서 사용하지 않고
  read-only `Actor.GetAllActorsState`만 사용

필요한 ROS 서비스:

```text
/SyncModeCmd             morai_msgs/MoraiSyncModeCmdSrv
/SyncModeCtrlCmd         morai_msgs/MoraiSyncModeCtrlCmdSrv
/SyncModeWaitForTick     morai_msgs/MoraiWaitForTickSrv
```

프레임별 실행 순서:

```text
1. SyncModeCmd(start_sync_mode=true, time_step=100) 호출 및 user_id 확보
2. 현재 상태를 Expert에 입력
3. Expert가 steer/throttle/brake 계산
4. SyncModeCtrlCmd(frame=N, command=..., sensor_capture=true) 전송
5. SyncModeWaitForTick(user_id, frame=N) 호출
6. 응답 frame/vehicle_status, 해당 tick 센서, 동일 tick gRPC 객체·신호 검증
7. 센서·Ego·객체·신호·Expert command를 같은 frame_id로 저장
8. 2~7 반복
9. SyncModeCmd(start_sync_mode=false)로 정상 종료
```

`sensor_capture=true`여야 MORAI가 해당 tick의 카메라·LiDAR 센서 토픽을 발행합니다.
수신한 ROS raw data는 이 프로젝트 writer가 저장합니다.

실행 형태:

```bash
python3 collect.py \
  --mode sync \
  --job competition_train \
  --route expert_route
```

완료 조건:

- 저장 `frame_id`와 SyncMode 응답 `frame`의 1:1 대응
- 동일 tick의 센서, `vehicle_status`, gRPC 객체·신호 결합
- Expert `[steer,throttle,brake]`를 같은 frame ID로 저장
- timeout, pause, tick 실패 프레임은 저장하지 않고 episode summary에 기록
- 예외와 `Ctrl+C` 종료 시에도 sync mode 종료 서비스 호출
