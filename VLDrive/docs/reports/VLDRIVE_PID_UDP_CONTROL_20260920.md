# VLDrive 예측 → LMDrive PID → MORAI UDP 연결 검증

> 후속 수정: 초기 `morai_steering_sign=-1`과 화면의 오른쪽 + 설명은 현재 설치 환경에 맞지 않았다. 실제 주행 로그 확인 후 **+1(왼쪽 +)**로 수정했다. 아래 원래 실험은 실제 차량에 제어하지 않은 연결 검증이며, 잘못된 부호 가정까지 검증한 것은 아니었다. [조향 수정 보고서](VLDRIVE_STEERING_SIGN_FIX_20260920.md)를 우선 참고한다.

2026-09-20. `VLDrive/control/`에 PID·전송·watchdog을 분리하고 `inference/realtime_vldrive.py --control`에 연결했다. 화면에서 제어 명령도 표시한다. 실제 모델 예측으로 Windows UDP 수신까지 검증했으며, 이번 작업에서 실제 MORAI 차량에 가속 명령을 보내 주행시키지는 않았다.

## 가져온 코드와 적용 범위

LMDrive commit: `308d8da0b4ff3463b1649fe483a9b2c56b108189`.

- [원본 PIDController](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/team_code/pid_controller.py): 파일을 그대로 복사했다. 다운로드 원본과 복사본 SHA256은 모두 `1a020ca27c7fc0231cfa8b4a55beb2b1108d4e9af560c22f32f007d339d5de1a`다.
- [control_pid 및 최종 steer×0.8](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/team_code/lmdriver_agent.py), [원본 계수](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/team_code/lmdriver_config.py)를 사용한다.
- 조향 P/I/D=1.25/0.75/0.3, 속도 5/0.5/1, 각 창 40, throttle 상한 0.75, speed-error clip 0.35, 제동 기준 0.1m/s 및 목표 속도의 1.1배다.
- 추가한 MORAI 처리: 좌표/부호 변환, 추론 지연 보정, 만료된 horizon 제외, 데이터 단절 제동, 초기 목표 속도 상한 5m/s, UDP 중계.
- global path는 자연어 navigation을 만드는 데 사용한다. PID 목표에는 Stage-2의 실제 예측만 들어가며 expert/정답/경로 waypoint로 대체하지 않는다.
- 모델 구조·checkpoint·선택 방식·loss는 바꾸지 않았다. 학습 간격의 11프레임과 무제한 추론 빈도를 유지한다.

## 좌표·단위·패킷

모델 raw `[left,-forward]`를 anchor pose를 통해 world 좌표로 옮기고 최신 GPS/status pose로 현재 ego `[left,forward]`를 만든다. anchor 수신 이후 이미 지난 horizon을 제외하고 남은 첫 두 점을 사용한다. 5개 horizon은 0.5/1/1.5/2/2.5초이며, 제어 예측 유효 시간은 1.25초다. 시각화 만료 시간 3초와 별도로 관리한다.

MORAI steer는 정규화된 -1..1로 보낸다. 초기 구현은 [MORAI 공식 ADModule](https://github.com/MORAI-Autonomous/MORAI-ADModule/blob/edddea57f1ccc4d6d80da12221891701b603ac46/autonomous_driving.py)의 변환을 참고해 `morai_steering_sign=-1`을 적용했으나, 현재 실행 환경에서는 반대였다. 현재 기본 설정은 실제 로그에 맞춰 +1이다. 이 최초 연결 실험에는 실제 차량 동적 선회 검증이 포함되지 않았다.

속도 수신에서 기존 `speed=vel_x`를 `speed=signed_vel/3.6`으로 수정했다. [MORAI UDP 문서](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/ros-1)의 signed velocity 단위는 km/h이고 PID/navigation은 m/s를 사용한다. 원시 `signed_speed`는 기존처럼 km/h로 보관한다. 181/229바이트 양쪽 상태 패킷에서 36km/h→10m/s 및 후진 -3.6km/h→-1m/s를 테스트했다. 따라서 진행 속도를 특정 축의 velocity 값으로 오인하지 않는다. 이 수정은 navigation 속도 표시에도 반영된다.

제어는 이 프로젝트에 포함된 MORAI `EgoCtrlCmd` 구조와 같은 **55바이트**다. Header `#MoraiCtrlCmd$` 14바이트, payload length=23, aux 12바이트, mode/gear/type=2/4/1, float 5개, CRLF. 직렬화한 결과를 vendor ctypes 객체의 실제 bytes와 완전 일치 비교했다. Windows Simulator가 바인딩한 제어 수신 주소는 `127.0.0.1:9093`였다.

## CPU/GPU와 통신

- Vision encoder/Stage-2는 기존 GPU 실행이다. PID·좌표 계산·JSON·UDP는 별도 CPU thread가 담당한다.
- PID 10Hz, 송신 20Hz. 20Hz가 새 모델 예측 빈도라는 뜻은 아니다.
- WSL에서 Windows gateway의 중계기 19093으로 명령을 보낸다. Windows는 MORAI localhost 9093에 공식 패킷을 보낸다.
- 최초 WSL→Windows 접속에서는 중계기의 수신 수가 0으로 남았다. Windows가 먼저 WSL 클라이언트 주소로 준비 패킷을 보내도록 바꿨고, 방화벽 규칙 수정 없이 handshake와 명령/ACK가 통과했다.
- 세션 식별자·증가 sequence·송신 peer를 검사한다. 중복·이전·다른 세션·범위를 벗어난 입력을 거부한다. ACK는 중계 수신 확인이며 MORAI 적용 확인은 아니다.
- GPS/status/필수 센서 0.3초 초과, navigation invalid, 이력 reset, 1.25초 초과 예측, 잘못된 좌표/시간이면 accel=0/brake=1. 새로운 유효 데이터로 재개할 때 PID 창을 초기화한다.
- Windows는 새 명령이 0.5초 없으면 자체 제동을 보내고 3초 단절 시 마지막 제동 후 종료한다. Ctrl+C에서도 GPU 작업 join보다 제어 종료/제동이 먼저다.

## 실제 통신 실험

모든 테스트 수신 포트는 **Windows 127.0.0.1:19094**였다. 실제 MORAI 제어 포트 9093에는 테스트 명령을 전송하지 않았다.

### WSL → Windows bridge → 테스트 수신기

`outputs/inference/control_transport_validation_final/`:

- handshake 성공, 수신 패킷 10개, 잘못된 패킷 0개.
- accel=0.25/steer=-0.3 및 brake=1/steer=+0.3 값이 float32 정밀도 내 일치.
- 명령 중단 후 Windows watchdog 제동 4개 확인.
- 마지막 패킷 accel=0/brake=1 확인.

### 기존 best-model live 출력 → PID → UDP, 약 30초

사용자의 기존 추론 프로세스 PID 34829가 센서 포트와 화면 8502를 사용하고 있었다. 별도 추론을 시작하려던 시도는 화면 포트 충돌로 중단됐고 네트워크 모델/센서 수신기를 추가로 시작하지 않았다. 기존 프로세스를 종료하거나 GPU 모델을 중복 로드하지 않고 읽기 전용 `/api/state`에서 실제 예측을 받아 제어 모듈을 검증했다. 통합 실행에서 사용할 직접 UDP snapshot 경로는 자동 테스트로 별도 검증했다.

`outputs/inference/control_existing_live_validation/summary.json`:

| 항목 | 관측값 |
|---|---:|
| 서로 다른 실제 best-model 예측 | 121개 |
| PID thread 명령 | 596개 |
| 테스트 수신 패킷, 종료 제동 포함 | 600개 |
| 잘못된 패킷 | 0개 |
| 평균 송신 빈도 | 20.001Hz |
| tracking 명령 | 324개 |
| hold 제동 명령 | 272개 |
| tracking 시 anchor 경과 평균 | 0.6844초 |
| 최대 throttle | 0.75 |

hold는 HTTP 화면에 캐시된 GPS pose가 오래된 경우 263회, status가 오래된 경우 9회였다. 화면 상태 갱신은 직접 센서 수신보다 느리다. 이 수치를 실제 제어용 직접 UDP 수신의 dropout 비율로 해석하면 안 된다. production PID는 같은 프로세스의 최신 GPS/status receiver를 직접 조회하며 HTTP를 사용하지 않는다. 이 테스트 adapter는 pose를 역변환해 GPS snapshot을 구성했고, 오래된 producer에 없는 history generation metadata는 현재 화면 reset 수로 보완했다. 실제 이력 reset 검증은 자동 테스트에 포함했다.

차량은 정지 상태였으므로 LMDrive 원본의 speed<0.01일 때 조향 오차를 0으로 하는 조건 때문에 실제 예측 테스트의 steer는 0이었다. 좌/우 비영 조향은 수식/좌표 unit test와 앞선 UDP ±0.3 패킷 테스트에서 확인했다. 이 결과를 실제 선회 주행 성능이라고 표현하지 않는다.

### 추가 구간: navigation invalid 시 제동

`outputs/inference/control_existing_live_validation_final/summary.json`:

- 기존 화면이 `navigation_unavailable / localization_invalid`로 바뀌어 유효 예측 0개.
- 따라서 tracking 100개 이상을 요구한 실험 assertion은 실패했다. 이는 성공적인 추종 실험으로 계산하지 않았다.
- 명령 597개, 수신 패킷 601개 모두 accel=0/brake=1, 송신 약 20.001Hz.
- 마지막 snapshot pose는 (-135.089, -487.410)m, 설정된 global path에서 가장 가까운 구간과 **23.823m** 차이였다. 허용 10m를 초과했고, 초기 진행 이력 없이 전체 경로에 새로 투영해도 실패했다. 따라서 이 snapshot의 localization invalid는 이전 history만의 문제로 볼 수 없다.
- 위치/경로 조건을 완화하거나 잘못된 navigation으로 출발하도록 변경하지 않았다. 현재 차량을 설정한 global path에 맞추거나 올바른 경로 설정을 사용해야 한다.

## 자동 및 화면 검사

전체 회귀 검사와 최종 제어 회귀 검사를 실행했다. PID 원본 수식 75회 연속 비교, 좌/우 부호, 정지/과속, 미래 horizon·현재 pose 변환, NaN/후방/위치 점프 거부, packet bytes, 실제 loopback UDP, 중복/잘못된 세션, watchdog, 센서/예측 단절 제동, live pipeline 결과 전달 및 제어 종료 순서를 포함한다.

최종 GPS 회귀는 제어 tick 시작 직후 도착한 최신 GPS가 미래 시각으로 오인되어 잘못 제동하지 않도록 확인한다. 최종 `tests/test_waypoint_control.py`는 **37 passed**다. 최종 전체 결과는 **356 passed, 1 skipped, 30 subtests passed (35.34초)**다.

Windows Chrome headless에서 제어 모드와 관찰 모드의 badge/panel, 가속·제동·조향 표시, 좌표 설명 전환을 확인했다. JavaScript 예외는 없었다. `control_existing_live_validation/control_ui.png`는 저장된 예측/제어 값을 이용한 UI 검사 이미지이며 실제 주행 사진이 아니다.

## 파일과 실행

- `control/`: PID wrapper, CPU thread, protocol, WSL client, Windows bridge, 원본/라이선스.
- `configs/vldrive_control.json`: PID 계수, 속도 상한, 제어 주기, timeout, IP/port.
- `inference/realtime_vldrive.py`, `live_pipeline.py`: 선택적 제어 lifecycle 및 실제 결과 전달.
- `inference/morai_udp.py`: 진행 속도의 m/s 정규화.
- `utils/inference_dashboard/`: 제어 상태 표시.
- `tests/test_waypoint_control.py`: 유지할 회귀 테스트.

VLDrive 폴더의 기존 관찰 추론을 종료한 뒤:

```bash
python inference/realtime_vldrive.py --control
```

초기 목표 속도 상한은 18km/h이고, 현재 경로가 유효할 때 11프레임을 모아 추론 후 출발한다. 종료는 Ctrl+C. 제어용 중계기는 자동 실행/종료되므로 별도 실행 명령이 필요 없다. 상세 설정은 [control/README.md](../../control/README.md)를 참고한다.

임시 실험 코드·다운로드 복사본을 제거하고 회귀 테스트와 측정 결과를 남겼다. 검증용 Windows 중계기/테스트 수신기/화면 검사 프로세스가 종료된 것을 확인했다. 사용자의 기존 관찰 추론과 센서 relay는 종료하지 않았다. 실제 MORAI의 제어 패킷 적용·동적 선회·정지·장애물 대응 및 전체 경로 완주는 다음 closed-loop 검증 범위다.
