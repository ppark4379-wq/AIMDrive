# VLDrive 제어: LMDrive 횡방향 PID + MORAI expert 종방향 PID

WSL `lmdrive` 환경의 `VLDrive` 폴더에서 기존 추론을 종료한 뒤 실행:

```bash
python inference/realtime_vldrive.py
```

`configs/vldrive_realtime.json`의 `control.enabled=true`이므로 제어와 시각화가 함께 시작된다. 별도 `--control.json` 인자는 필요 없다. 센서 relay는 기존 구성을 사용한다.

## 역할과 설정

| 역할 | 코드 / 설정 |
|---|---|
| 횡방향 PID | `control/lmdrive_lateral.py` |
| 지나친 구간 제외 및 조향 목표점 선택 | `control/steering_target.py` |
| LMDrive 원본 PID | `control/third_party/lmdrive/pid_controller.py` |
| 횡방향 파라미터 | `configs/vldrive_control.json` → `lateral` |
| 종방향 및 가속·제동 변환 | `MoraiDrive/expert/src/expert/src/control/path_controller.py`, `pid.py`, `control_input.py` |
| 종방향 파라미터 / 주기 | expert `config/config.json` → `control.pid`, `common.realtime_sampling_rate` |
| 두 제어기의 연결 | `control/expert_controller.py` |
| 입력 유효성 검사 / UDP 송신 | `control/runtime.py`, `transport.py`, `udp_bridge.py` |
| 상태·목표점 시각화 | `utils/inference_dashboard/` |

추론의 다섯 waypoint가 제어 경로다. Global path를 제어 목표점으로 가져오지 않는다.
이전 `control/waypoint_pid.py`의 `WaypointPID`는 실시간 제어기로 사용하지 않는다. 좌표 변환과 정지 명령 함수만 기존 runtime에서 사용한다.

## 횡방향

현재 차량 좌표 `[left, forward]`에서 예측 polyline의 가장 가까운 투영점을 찾고, 그 이전 구간을 제외한다. 이후 구간에서 속도에 맞는 거리의 앞쪽 점을 선택한다. 구간 중간은 선형 보간하며 경로 끝을 넘겨 연장하지 않는다.

`lateral.target`: `mode=remaining_path`, `time_s=.8`, `min_distance_m=4`, `max_distance_m=12`, `min_forward_m=.5`. 초기 실험값이며 실제 폐루프에서 최적화된 수치는 아니다. 비교용 `first_two_midpoint` 모드는 남겨두었다.

```text
distance = clip(실제 속도[m/s] * 0.8초, 4m, 12m)
aim = 가까운 투영점 이후 경로에서 distance 이상 떨어진 앞쪽 목표점
error = degrees(pi/2 - atan2(aim.forward, aim.left)) / 90
speed < 0.01 m/s이면 error = 0
raw = 1.25*error + 0.75*최근 40개 error 평균 + 0.3*(error - 이전 error)
steer_left_positive = clip(raw, -1, 1) * 0.8
```

LMDrive의 I는 오차 창의 평균, D는 연속 오차 차이이며 별도 dt 곱셈/나눗셈을 하지 않는다. 창은 0으로 초기화된다. 정차하면 입력 오차만 0이 되므로, 이전 I/D에 의해 잠깐 잔여 조향이 나올 수 있다.

조향 계산은 약 10Hz, 사이에는 직전 출력을 유지한다. 지연된 실행을 따라잡으려고 동일 입력으로 PID를 여러 번 돌리지 않는다. 새로운 예측이 와도 이전 조향의 예측 유효시간을 연장하지 않으며, 만료되면 유효한 최신 입력으로 조향을 갱신한다. 입력 오류로 제동하면 PID 이력도 초기화한다.

유지 중인 목표점도 30Hz마다 현재 좌표로 검사해 전방 0.5m 안쪽으로 넘어가면 즉시 다시 선택한다. 매 tick에서 최신 경로의 앞쪽 길이가 부족하면 `insufficient_remaining_path`로 가속 0 / 브레이크 1 / 조향 0을 출력한다. 별도 감속 곡선은 추가하지 않고 기존 무효 입력 정지 처리를 사용한다. 거리 하한은 조향 갱신 시점 기준이므로 유지 중에는 차량 이동만큼 줄어들 수 있다.

원본처럼 최종 배율 0.8을 적용하므로 정규화 steer 범위는 ±0.8이다. 현재 최대 휠각 40°에 대한 요청각 표현으로는 ±32°다. 이는 LMDrive 참조 설정이며 JSON의 `lateral.steering_scale`에서 확인할 수 있다. 실제 차량 응답각은 상태 패킷으로 별도 관측한다.

기존 Pure Pursuit의 곡률식은 실제 조향 출력에 사용하지 않는다. 목표점 선택은 이번에 변경했고, 선택된 점의 방향 오차를 LMDrive PID에 넣는다. PID 게인·0.8 배율·주기는 유지했다.

## 종방향 보존

`ExpertWaypointController`는 기존 `PathController.control_path`를 그대로 호출하고, 그 가속·브레이크 반환값을 그대로 사용한다. 해당 공통 함수 안에서 계산되는 Pure Pursuit 조향은 버리고 LMDrive 조향으로 교체한다. Expert 내부에서 출력되는 `[PP]` 디버그 메시지는 사용하지 않는 중간 계산이며 UDP의 조향값이 아니다. 원본 expert 프로그램 자체의 제어기는 변경하지 않았다.

- 모델 예측 속도: 첫 두 예측 점 사이 거리 / 두 점의 시간 간격. 현재 시간 간격은 0.5초.
- 현재 목표 속도: 모델 예측 속도와 `configs/vldrive_control.json`의 `max_speed_kmh=5.0` 중 작은 값. 5km/h=1.3889m/s. 모델의 더 느린 속도·정지 의도는 유지한다. `null`이면 이전 상한 없는 방식이며, 실제 차량 속도가 순간적으로 상한을 초과하지 않는다는 보장은 아니다. 설정 변경 후 추론을 재시작한다.
- expert PID: P=.075, I=.0001, D=.025.
- 계산 dt=1/30초, 종방향 30Hz, UDP 30Hz.
- 기존 적분 포화 방지, 목표 속도 하락 시 reset, 정지, 가속 상승/제동 해제 처리를 유지.
- LMDrive 종방향 PID, brake ratio, throttle .75 제한은 가져오지 않았다.

유효한 경로를 연속 사용하면 종방향 출력은 이전과 동일하다. 경로 부족으로 정지한 뒤에는 기존 PID reset이 발생하므로, 그 이후 출력까지 이전 주행 기록과 같다는 뜻은 아니다.

## 표시와 관측

현재 설치의 UDP steer는 왼쪽 양수이고, 차량 상태 휠각은 반대 부호다. 화면 요청 휠각은 `-UDP steer × 40°`로 변환해 상태값과 비교한다.

주황색은 마지막 조향 갱신에서 선택한 앞쪽 목표점이며, 유지 중에도 동일한 월드 위치를 현재 차량 좌표로 옮겨 표시한다. 선택 구간 번호·설정 거리·실제 거리·조향/종방향 frame·갱신/유지·PID 방향 오차·요청/수신 휠각을 표시한다. `control.jsonl`에는 투영 진행 거리·남은 경로 길이·보간 비율·P/I/D 항·시각·예측 나이도 기록한다.

Windows 중계 포트 19093 → MORAI `127.0.0.1:9093`. 기존 센서/예측 유효시간 검사와 watchdog은 유지된다.

## 검증 및 남은 문제

테스트 113개 통과. 기록 481 tick 재생에서 뒤쪽 목표점은 유지 tick 포함 127→0회. 공통 유효 구간의 같은 10Hz 표본에서 조향 변화 평균 4.123→1.200°, p95 26.289→4.012°. 23 tick은 경로 부족으로 정지 처리됐다. 실제 폐루프 차량 운동은 미검증이며 정지/재출발 빈도도 실주행에서 확인해야 한다.

[이번 목표점 변경 보고서](../docs/reports/REMAINING_PATH_TARGET_20260922.md) · [이전 LMDrive PID 교체 보고서](../docs/reports/LMDRIVE_LATERAL_PID_20260922.md)
