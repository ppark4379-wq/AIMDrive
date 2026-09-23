# Expert 원본 제어기 공용 적용 — 2026-09-22

## 적용 결과

VLDrive의 별도 Pure Pursuit/PID 구현을 제거하고, `MoraiDrive/expert`의 제어 블록을 공용 모듈로 분리해 expert와 추론에서 함께 호출한다. **같은 경로·목표 속도·차량 상태·제어 모드가 입력되면 같은 제어 출력을 만든다.**

이관 전 원본의 초기화·전체 제어 블록·보조 메서드 4개의 AST가 이관 후와 동일함을 확인했다. 원본 대비 600개 합성 제어 조건과 최신 기록 471회 출력 비교에서 accel/brake/front_steer 차이가 모두 0이었다.

## 파일 변경

| 파일 | 변경 |
|---|---|
| `MoraiDrive/expert/src/expert/src/control/path_controller.py` | 원본 초기화·제어 블록·보조 함수 이관 |
| `MoraiDrive/expert/src/expert/src/autonomous_driving.py` | PathController 상속, 동일 초기화/제어 블록 호출 |
| `VLDrive/control/expert_source.py` | 동일 repository의 expert Python 소스 직접 import |
| `VLDrive/control/expert_controller.py` | 모델 경로·속도를 expert에 전달, 출력 UDP 표현 변환 |
| `VLDrive/control/config.py` | expert JSON 직접 읽기, 제어 파라미터 중복 설정 금지 |
| `VLDrive/configs/vldrive_control.json` | 별도 PID/차량/lookahead 값 제거, expert 설정 경로와 전송 설정만 보관 |
| `VLDrive/control/runtime.py` | expert 실시간 주기 30Hz 계산/송신, expert와 같은 실시간 dt |
| `VLDrive/utils/inference_dashboard/` | 5개 원본 점 표시, 선택 번호, 탐색 거리/실제 거리 구분, 휠각 부호 기준 일치 |

앞서 추가한 `VLDrive/control/waypoint_tracker.py`, `longitudinal_pid.py`와 해당 전용 테스트는 삭제했다. Historical LMDrive 계산 코드는 활성 제어 경로에서 사용하지 않는다.

PurePursuit, Pid, ControlInput 원본 파일과 expert config 값은 바꾸지 않았다. `AutonomousDriving`의 경로 생성·ACC·신호등 판단도 수정하지 않았다. 변경 내용은 제어 코드의 재사용 가능한 위치 이동이다.

## 설정의 단일 원본

`MoraiDrive/expert/src/expert/src/config/config.json`을 읽는다.

| 원본 변수 | 적용 값 |
|---|---:|
| `control.pid.p_gain` | .075 |
| `control.pid.i_gain` | .0001 |
| `control.pid.d_gain` | .025 |
| `control.pure_pursuit.lfd_gain` | .65 |
| `control.pure_pursuit.min_lfd` | 2m |
| `control.pure_pursuit.max_lfd` | 30m |
| `common.wheelbase` | 3m |
| `common.max_steering` | 40° |
| `common.realtime_sampling_rate` | 30Hz |
| `common.realtime_steering_sign` | -1 |
| `common.sync_steering_sign` | +1 (실시간 추론에서는 미사용) |

Expert의 `network/synchronized_inputs.control_dt`는 realtime 모드에서 고정 `1/realtime_sampling_rate`를 사용한다. 따라서 추론도 wall-clock 지터로 dt를 바꾸지 않고 1/30초를 전달한다. 송신과 제어 계산을 같은 주기로 실행하며, 추론 GPU 주기를 30Hz로 강제하는 변경은 아니다.

Expert의 `highway_lane_planning.parameters`와 `lattice_planning.parameters`도 원본 import를 공유한다. 목표 급강하 reset, 정지 처리, 가속 상승·제동 해제, roundabout 미분 모드/출발 처리, static lookahead, 차선 변경 조향 제한 등 원본 제어 분기가 공용 코드에 그대로 있다. 입력에 해당 mode/context가 있을 때 원본과 같은 분기를 실행한다.

## 모델 입력을 제어기로 연결하는 경계

상위 경로 생성기는 VLDrive다. 모델 raw `[left,-forward]`를 현재 차량 좌표로 바꾸고 expert의 `Point(forward,left)` 5개로 전달한다. 원점 추가·점 보간·재샘플링·시간 경과에 따른 waypoint 삭제를 하지 않는다. GPS pose→현재 ego 변환은 좌표 연결에 필요한 처리다.

목표 속도는 모델 첫 두 점의 거리/0.5초다. 이 입력을 expert의 `target_velocity`로 전달한다. 임의의 1/5m/s 제한이나 .75 throttle 제한은 없다. Expert 제어기 자체의 [-1,1] 출력 범위와 원본 가속 상승 처리를 사용한다.

VLDrive에는 expert planner의 `lattice_result`가 없으므로 기본 `model_waypoints` context로 동작한다. 회전교차로/장애물/차선 변경 모드를 navigation 문장만 보고 임의 생성하지 않는다. 따라서 이는 **expert의 경로 추종 제어기를 그대로 적용하는 작업**이며, expert의 global-path/ACC/신호등/장애물 planner까지 신경망 위에 실행하는 변경은 아니다. 같은 제어 코드를 사용한다고 두 전체 주행 시스템의 입력과 판단까지 같아지는 것은 아니다.

입력 만료·센서 이상·통신 단절 검사는 기존 UDP 송신 계층에 유지한다. 정상 입력의 expert 제어 출력을 추가 감쇠하거나 튜닝하지 않는다.

## 원래의 목표점 선택과 조향식

Expert PurePursuit는 다음을 그대로 수행한다.

1. `lfd = clip(lfd_gain × velocity, min_lfd, max_lfd)`.
2. 경로의 점을 순서대로 현재 pose로 변환한다.
3. 전방에 있고 기준점에서 거리가 lfd 이상인 첫 **원본 점**을 선택한다.
4. `atan2(2 × wheelbase × sin(theta), lfd)`로 조향을 계산한다.
5. expert의 steering_sign, 모드별 처리, max_steering을 적용한다.

lfd가 2m여도 선택된 점은 4m 또는 그 이상일 수 있다. 이전 구현처럼 정확히 2m 원과 경로 선분의 교차점을 만들지 않는다. 분모도 실제 점 거리로 교체하지 않고 expert의 lfd를 사용한다. 점이 없을 때 조향=0 등 원본 동작도 그대로다.

## UDP와 표시 단위

Expert realtime 출력 `front_steer`는 rad이며, 설치된 UDP EgoCtrlCmd는 반대 부호의 정규화 입력을 사용한다.

```text
wire_steer = -front_steer / radians(max_steering)
wire_accel = ControlInput.accel
wire_brake = ControlInput.brake
```

예를 들어 expert 좌회전 -20°는 UDP +0.5로 전송한다. 단위 변환에 필요한 최대 각도도 expert config에서 읽는다. expert realtime_steering_sign=-1을 UDP용 +1로 고쳐 원본 계산에 섞지 않았다.

요청 휠각 표시에는 expert `front_steer`를 degree로 사용한다. 이전 로그에서 수신 steer는 UDP 명령과 반대 부호였으므로, 이제 요청/수신 휠각은 같은 상태 기준으로 읽을 수 있다. 정규화 UDP 조향값과 각도 부호가 다름을 화면에 명시한다.

## 원본 대비 검증

### 구조 동일성

수정 전에 저장한 `AutonomousDriving` 소스를 기준으로 비교했다. 다음은 AST 동일성이 확인됐다.

- 제어 초기화 블록 전체.
- 목표 속도 reset부터 최종 front_steer까지 제어 블록 전체.
- `_configure_runtime`.
- `_recorded_lookahead_limits`.
- `_smooth_longitudinal_command`.
- `_roundabout_entry_fast_release`.

수정 전 저장본은 HEAD 원본과 줄바꿈 정규화 후 동일함도 확인했다. 원본의 계산 순서나 상수 변경 없이 이관했다.

### 600개 제어 조건

수정 전 원본 메서드/제어 블록을 별도로 실행해 이관 후 출력과 비교했다. 일반, roundabout_entry, static_obstacle, recorded 경로/복귀, highway lane change를 포함했다. 목표 속도 0/1/4/8/20m/s, 차량 속도 변화, 좌우 경로, dt=1/30/.1/.02초 및 realtime/sync 전환을 넣었다.

**600회 최대 accel/brake/front_steer 차이 = 0.0**.

### 최신 실제 로그 471회 재계산

입력: `outputs/inference/vldrive_20260922_125553_229631`의 실제 예측·pose·속도. PID update별 중복 제거, hold 시 상태 초기화. Expert realtime dt=1/30초를 동일하게 사용했다.

**471회 최대 accel/brake/front_steer 차이 = 0.0**.

| 새 expert 제어기 재계산 항목 | 결과 |
|---|---:|
| 실제 선택점 거리 최소 | 2.9992m |
| 실제 선택점 거리 평균 | 4.2996m |
| 실제 선택점 거리 최대 | 6.0638m |
| 평균 절대 요청 휠각 | 2.6932° |
| 최대 절대 요청 휠각 | 9.5541° |
| 모델 목표 속도 중앙값 | 9.2500m/s (33.30km/h) |
| 모델 목표 속도 최대 | 11.8802m/s (42.77km/h) |
| 최대 throttle 출력 | .9128 |

이번 값은 고정된 관측 기록에서 명령을 계산한 것이다. 새 명령을 적용한 실제 차량 궤적이나 폐루프 주행 성능은 측정하지 않았다. 이전 5m/s 상한보다 빠른 목표 속도가 들어간다는 점은 재실행 시 달라지는 동작이다.

### 회귀 테스트와 확인된 실패

`test_expert_control.py`, `test_waypoint_control.py`, `test_live_navigation_dashboard.py`, expert `test_longitudinal_control.py` 실행 결과 **101 passed, 1 failed**.

실패는 expert의 `test_crossing_pedestrian_is_detected_before_entering_lane`다. `AdaptiveCruiseControl.get_target_velocity(14.7,16.67)` 결과가 기대한 감속값이 아니라 16.67이었다. 이 테스트는 변경한 경로 추종 제어기를 호출하지 않는다. ACC 소스, 해당 테스트 파일, expert config가 HEAD와 그대로 동일함을 확인했다. 원본을 그대로 적용하라는 범위를 지키기 위해 이 별도 ACC 로직은 수정하지 않았다.

순수 제어와 UDP 연결의 회귀 테스트는 통과했다. 잘못된 설정, 원본 점 선택, lfd 분모, 추가 속도/throttle 제한 부재, 정지/목표점 부재의 원본 동작, 연속 입력의 원본 출력 일치, 좌표 변환, 패킷/loopback/watchdog, 오래된 입력 제동을 검증했다.

Windows Python에서 NumPy 없이 중계기 설정이 로딩되는 것을 확인했다. Headless Edge에서 원본 5점, 실제 선택점, 탐색 기준 2m/실제 목표 4m, 요청/수신 휠각을 확인했다. UI용 가상 데이터이며 차량 계측 실험은 아니다.

## 실행과 정리

기존 추론은 시작 시 설정/모듈을 읽으므로 재시작해야 적용된다. WSL lmdrive 환경의 VLDrive 폴더에서:

```bash
python inference/realtime_vldrive.py
```

현재 control.enabled=true이며 제어기와 UDP가 포함된다. 모델 체크포인트와 navigation 설정은 그대로다. 임시 원본 대조 스크립트와 원본 사본은 검증 후 삭제하고, 결과를 이 문서와 영구 회귀 테스트에 남겼다.
