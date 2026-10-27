# Expert 연결·실행 및 검증 도구

이 폴더는 AIM2026의 Expert에 연결하는 실행 진입점, 평가·시각화 도구와
기존 import 호환 진입점을 제공한다. Expert 본체는 `AIM2026/Expert`를 공유한다.
아래에는 기존 여섯 문서의 실행 방법과 검증 기록을 함께 보존했다.
본문의 `MoraiDrive/scenario_runner/...` 실행 명령은
`AIM2026/utils/scenario_runner/...`의 같은 진입점을 사용한다.
과거 실험 결과의 경로·수치와 개선 제안은 당시 기록이며 새 검증 결과가 아니다.

- [GT Expert UDP 실행](#gt-udp-input)
- [GPS 기반 VE Expert UDP 실행](#ve-gps-input)
- [터널 GPS 음영 구간 오차 확인](#gps-hold-error-check)
- [VE 장애물 입력·실행·계측](#ve-object-input)
- [VE ROI·RViz와 Windows 추론](#ve-roi-rviz)
- [과거 분석 및 개선 제안 (2026-10-17)](#historical-static-replan-proposal)

ROS 실행기의 반복 초기화는 `ve_ros_env.sh`의 공통 함수에서 처리한다.
이 파일을 직접 source하는 기존 ROS 설정 방식과 `../tools/project_env.sh`는 유지한다.
기존 Python·Bash·PowerShell 실행 명령과 모듈 import는 유지한다.

---

<a id="gt-udp-input"></a>

<!-- source: docs/GT_UDP_INPUT.md -->
# 촬영용 GT Expert UDP 실행

이 실행은 MORAI의 **Ego Vehicle Status 전체 GT**와 Object Info를 UDP로 받아 기존 `AutonomousDriving`과 lattice planner를 실행하고, 제어도 UDP로 보낸다. **신호등 UDP 메뉴가 없는 대회판 촬영을 위해 신호 상태만 gRPC GT로 조회하는 것이 기본값**이다. ROS master와 차량 ROS 연결은 필요하지 않다. 시나리오 러너의 gRPC는 월드·날씨·장애물 생성, 시작 위치와 Auto/D 설정, 진행·충돌 확인에도 사용된다. 자차와 장애물 인지 입력에는 gRPC를 쓰지 않는다.

## MORAI 연결

Ego/Simulator Network에서 아래 연결을 UDP로 설정한다. **Host IP는 Windows MORAI, Destination IP는 이 실행기가 있는 WSL의 실제 IPv4**이다. WSL NAT 환경에서는 양쪽에 `127.0.0.1`을 넣으면 Windows 안에서만 통신하고 WSL 수신기에 도달하지 않는다. 제어 송신 주소는 기본적으로 기존 러너의 Windows MORAI 주소를 자동 탐지한다. 필요하면 `--sim-ip`에 MORAI Host IP를 지정한다.

현재 PC에서 2026-10-16 확인한 값은 Host `172.18.32.1`, Destination `172.18.40.28`이다. WSL 재시작 후에는 주소가 달라질 수 있으므로 `hostname -I`와 Windows WSL 어댑터 주소를 다시 확인한다. rosbridge TCP의 localhost 설정과 UDP 주소는 별개다.

| 항목 | MORAI Host port | Destination port |
| --- | ---: | ---: |
| Ego Vehicle Status (전체 GT) | 8001 | 8000 |
| Object Info | 8101 | 8100 |
| Ego Ctrl Cmd | **7601 (MORAI가 수신)** | 7600 (실행기의 송신 소켓) |

신호등용 네트워크 연결은 추가하지 않는다. 기본 `--signal-source grpc`는 이번 실행 경로의 다음 신호 ID를 조회하고, 반환된 원래 상태 비트를 그대로 Expert에 전달한다. 좌회전 신호의 `32`, `33`, `48`도 보존한다. 조회 실패·다른 ID 응답·경로 이탈은 상태 0(미확정)으로 처리하며 이번 실행의 설정에서 정지 대상으로 포함한다. 강제로 초록불로 바꾸거나 VE로 자동 대체하지 않는다.

일반판에서 신호 UDP 메뉴가 제공되는 경우에만 `--signal-source udp`와 Get TrafficLight Status Host 7701 / Destination 7700을 사용한다. `--grpc-target HOST:7789`로 신호 조회 주소를 명시할 수도 있다. 기본값은 기존 러너와 같은 Windows MORAI 주소를 탐지한다.

대회용 Competition Vehicle Status는 GT 위치가 빠지므로 이 촬영용 실행의 자차 입력으로 사용하지 않는다. 현재 GT 파서는 전체 Ego 229/245 bytes, timestamp가 있는 Object Info 68/106-byte record, TrafficLight 48 bytes를 검사한다.

제어 기본값은 **55-byte 단일 조향 형식(`single_steer55`)**이다. 이 PC의 대회판에서 제동을 유지한 채 가속 없이 비교했을 때, 59-byte 명령은 조향 상태에 반영되지 않았고 55-byte의 조향값 `0.05`는 차량 상태의 `steer`와 `front_wheel_angle`에 `0.05`로 반영됐다. 기존 Expert의 조향값을 추가 배율 없이 전달한다. 이 확인은 패킷 수용과 상태 반영에 대한 것이며, 실제 주행 궤적의 조향 보정까지 검증한 것은 아니다. 다른 MORAI 빌드에서 59-byte 형식이 필요할 때만 `--control-format front_rear59`를 명시한다. 실행 중 자동으로 형식을 바꾸지 않는다.

기존 VE/Expert 주행을 Ctrl+C로 종료하고 표의 세 연결을 Connect한 뒤 설정 창을 닫고 Play로 둔다. 카메라·LiDAR의 UDP 설정은 바꾸지 않아도 된다. 이 실행은 카메라·LiDAR를 읽지 않는다.

## 실행

수신 확인만 하고 제어를 보내지 않는 명령:

```bash
cd ~/AIMDrive
bash MoraiDrive/scenario_runner/tools/run_gt_udp_full_route.sh --probe
```

기존 시나리오 러너의 풀루트 1회, GT 통과 이력이 있는 정적 2개 배치, 이동 차량, 진입 정차 차량 제외, 최대 60km/h:

```bash
bash MoraiDrive/scenario_runner/tools/run_gt_udp_full_route.sh
```

터널 진입 전까지만 촬영하려면:

```bash
bash MoraiDrive/scenario_runner/tools/run_gt_udp_full_route.sh --stop-before-tunnel
```

필요하면 `--sim-ip 172.x.x.x`, `--seed 123`, `--speed-cap-kmh 60`을 추가한다. 포트도 `--ego-port`, `--object-port`, `--signal-port`, `--control-port`, `--control-source-port`로 지정할 수 있다. `--check`는 소켓이나 MORAI 연결 없이 실제 지도·경로와 기존 Expert 초기화를 확인한다.

이 명령은 주행 준비와 실행을 담당하며 화면 녹화는 별도로 시작한다. 정상 수신·기존 Expert 시작은 `[PASS] GT 자차·장애물·제어 UDP · 신호=grpc 연결 완료`로 표시된다. 이 표시는 차량의 실제 제어 반응이나 완주를 보증하는 문구는 아니다.

로그는 `scenario_runner/.runtime/gt_udp_full_route/<실행 시각>/`에 저장된다. `expert/expert.log`, `expert/inputs.jsonl`, `expert/status.json`, `expert/report.json`에서 수신 수·오류·자차 속도, Expert 제어값과 자차가 보고한 페달·조향 상태, 제어 형식과 송신 수를 볼 수 있다. 입력이 오래 끊기면 제동하고 실행을 종료한다. 종료 시에도 짧게 UDP 제동을 보낸다.

자차 배치 후 출발 직전의 원래 gRPC 상태는 `startup_contact.json`에 보존한다. 충돌 표시가 이미 있다면 UDP Expert를 시작하기 전에 중단한다. `['Ego']`만 있는 경우도 실제 접촉 여부가 불명확하므로 무시하지 않는다. 자차 Reset 후 Play 상태에서 충돌 표시가 지워졌는지 확인한다. 주행 중 충돌 검사도 기존 러너 그대로 유지한다.

## VE로 복귀

GT 주행을 Ctrl+C로 끝낸다. GT 실행은 운영 가중치, 원래 Expert 설정, VE 실행 파일과 VE의 runtime 경로를 덮어쓰지 않는다.

현재 VE 풀루트 실행기는 자차·차량 제어에 기존 ROS 연결을 사용하므로, MORAI에서 그 차량 연결을 **기존 VE용 ROS 설정으로 복원**하고 Connect한다. 신호는 기존 VE 인식 결과를 사용한다. 카메라·LiDAR는 기존 UDP 설정을 유지한다. 자주 전환할 경우 MORAI의 Network Save 기능으로 GT 촬영용 / 기존 VE용 설정을 각각 저장해 두면 된다.

```bash
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh
```

별도 시나리오 생성 없이 이미 준비된 경로·장면에서 GT Expert만 실행하려면 `expert/tools/run_gt_udp_expert.sh --config <이번 실행 config.json> --path <path.csv>`를 사용할 수 있다.

## 프로토콜 근거

- [MORAI UDP 메시지 문서](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1)
- [MORAI 공식 UDP 예제](https://github.com/MORAI-Autonomous/MORAI-DriveExample_UDP/tree/main/network)

공개 자료의 패킷 정의와 실제 빌드가 다를 수 있다. 이 실행의 기본 제어 형식은 위에서 설명한 실제 대회판의 반영 결과를 기준으로 선택했다. 완주 여부는 별도 실주행 결과로 판단한다.
<!-- end-source: docs/GT_UDP_INPUT.md -->

---

<a id="ve-gps-input"></a>

<!-- source: docs/VE_GPS_INPUT.md -->
# GPS 기반 VE Expert UDP 실행

터널 위치 적분만 평가할 때는 **센서 전용 실행**을 사용한다. 이 모드는 GPS·IMU·차량 상태 차속만 받으며, VE 모델 가중치·PyTorch·카메라·LiDAR가 필요 없다. 빈 도로의 `full_loop_tunnel_normal` 미션에서 기존 PurePursuit/PID로 경로를 추종하며 자세한 시험·분석 절차는 [GPS_HOLD_ERROR_CHECK.md](#gps-hold-error-check)를 따른다.

```bash
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh \
  --check --mission full_loop_tunnel_normal --sensor-only \
  --gps-blackout-hold --simulate-gps-blackout
```

아래 VE 설명은 장애물·신호 인지를 함께 사용하는 기존 실행 경로에 해당한다.

`expert/tools/run_ve_gps_expert.sh`는 현재 준비된 MORAI 장면에서 기존 Expert를 실행한다. 위치는 GPS를 UTM52N으로 변환한 뒤 지도 원점 `(302595, 4124145)`을 빼서 구하고, 방향은 IMU, 속도는 차량 상태의 signed speed(km/h)를 사용한다. 차량 상태의 GT 위치·yaw는 쓰지 않는다. 장애물은 기존 P1과 추적기, 신호는 기존 빨강/초록 head와 좌회전 head를 사용한다. Expert의 waypoint 대신 기존 지도 경로와 lattice planner가 제어를 계산한다.

MORAI의 ROS 연결이나 gRPC GT 조회는 사용하지 않는다. 별도 시나리오 생성은 하지 않으므로 시험할 장면은 MORAI에서 준비한다. 자차는 현재 위치에서 시작하거나 `--from-start`로 공식 시작 위치에 배치할 수 있다. 기존 VE 시나리오 실행기는 그대로 유지한다.

## 현재 PC의 연결

기존 Windows 중계기를 이용하는 센서 설정은 다음과 같다. Host/Destination IP는 모두 `127.0.0.1`이다.

| 센서 | MORAI Host port | MORAI Destination port | 중계 후 WSL 수신 |
| --- | ---: | ---: | ---: |
| 전방 카메라 | 7501 | 9090 | 10501 |
| 좌측 카메라 | 7502 | 9090 | 10502 |
| 우측 카메라 | 7503 | 9090 | 10503 |
| 후방 카메라 | 7504 | 9090 | 10504 |
| GPS | 7506 | 9090 | 10506 |
| IMU | 7507 | 9090 | 10507 |
| LiDAR | 기존 설정 유지 | 2368 | 2368 |

차량 상태·제어는 센서 중계와 별도로 Windows Host IP와 WSL Destination IP의 직접 UDP 연결을 사용한다. 현재 주소는 WSL의 `ip route`, `ip -4 addr`로 확인한다. 차량 상태는 Host 8001 → Destination 8000, 제어는 Host 7601 ← 실행기 송신 7600이다. 대회판에서는 `Competition Vehicle Status`, 기본 MORAI에 해당 메뉴가 없으면 일반 `Ego Vehicle Status`를 선택한다. 전체 상태 메시지가 오더라도 실행기가 위치·방향 GT를 사용하지 않는다. WSL 재시작 후 IP가 바뀌면 확인해서 맞춘다.

## 확인과 실행

먼저 제어 없이 GPS와 VE 추론·Expert 계획을 20초 확인한다. 입력 준비 후 20초이며, 입력이 없으면 준비 제한 시간 후 실패한다.

```bash
cd ~/AIMDrive
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.sh --seconds 20
```

수신과 좌표를 확인한 뒤 주행한다. 기본 상한은 기존 시험과 같은 60km/h다.

```bash
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.sh --drive
```

공식 경로의 처음 위치로 배치하고 속도 상한 60km/h로 시작하려면:

```bash
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.sh --drive --from-start --sim-profile normal --speed-cap-kmh 60
```

`--from-start`에서만 출발 준비용 gRPC 설정 명령으로 cruise 해제·잔여 속도 0·시작 위치 배치를 수행한다. 월드 재시작이나 NPC 생성·삭제, GT 조회는 하지 않는다. 이후 GPS로 배치 위치를 확인한 다음 주행하며 장애물·신호는 VE, 제어는 UDP를 유지한다. 옵션이 없으면 현재 위치에서 시작한다.

Ctrl+C로 종료한다. `--drive`를 주지 않으면 제어 소켓을 만들지 않으며 제동도 송신하지 않는다. `--drive`에서는 입력 대기·누락 시 제동하고, 유효 입력이 2초 이상 끊기면 종료한다. 종료 시에도 짧게 제동한다. `--check`는 네트워크·GPU 실행 없이 경로와 설정을 검사한다.

제어용 위치·방향·속도는 추론 루프와 별도의 스레드가 최신 GPS·IMU·차량 상태에서 읽어 50Hz를 목표로 전달한다. 기존 loopback VE 수신 포트를 공유하므로 MORAI 센서 포트를 바꾸지 않는다. Expert는 이를 30Hz 제어에 사용한다. 최신 자세 수신만으로 주행 준비가 되지는 않으며, 유효한 VE 인지도 함께 있어야 한다. 제어용 자세의 가장 오래된 센서 수신 시간이 150ms를 넘거나 VE 입력이 만료되면 즉시 제동하고, 2초 이상 회복하지 못하면 종료한다.

VE의 모델 입력·객체 좌표 변환에는 기존 추론 시점의 자세를 유지한다. 객체의 지도 좌표를 최신 자차 자세로 다시 회전·이동시키지 않는다. 신호 head와 신호 판정의 유효 시간도 그대로 유지한다. `control.jsonl`의 `position`은 최신 제어 자세, `perception_position`은 추론 자세이며, `control_pose_selected_samples`와 `control_pose_age_ms`로 실제 입력 갱신과 수신 후 경과 시간을 비교할 수 있다. 이 수치는 물리적 센서 취득 지연을 뜻하지 않는다.

현재 PC에서 확인한 두 MORAI 빌드는 UDP 조향 방향이 다르다. 기본 `--sim-profile normal`은 양의 UDP 값이 오른쪽으로 꺾이는 기본판에 맞춰 `realtime_steering_sign=-1`을 적용한다. 앞서 대회판에서 확인한 방향은 `--sim-profile competition`으로 선택하며 `+1`을 유지한다. 이 옵션은 실행별 설정에만 적용하고 기존 Expert 소스·ROS 설정은 바꾸지 않는다. 프로필을 바꾸거나 다른 빌드를 쓰면 저속으로 실제 회전 방향을 확인한다.

UDP 조향에는 라디안을 직접 보내지 않고 `원하는 조향각 / 최대 조향각`의 -1~1 정규화 값을 보낸다. 최대 조향각은 이번 Expert 설정을 사용하며, 현재 기본판에서 양의 0.04 입력 → 약 -1.6도, 요청 0.04rad를 정규화한 입력 → 약 -2.29도 반영을 제동 상태로 확인했다. `plan.json`에 선택한 프로필을 기록하고, `control.jsonl`에 Expert 라디안·UDP 정규화 값·차량의 페달과 조향 상태를 함께 기록한다. [공식 UDP 규격](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/26.R1/udp-1)

로그는 `MoraiDrive/.runtime/ve_gps_expert/<시각>/`의 `status.json`, `publisher.log`, `expert.log`, `inputs.jsonl`, `report.json`이다. 신호 ID는 지도와 GPS 경로 위치로 선택하며 신호 색은 VE만 사용한다. 기존 정책대로 미지원 신호는 미확정(정지) 처리한다.

2026-10-16 기본 MORAI 연결 검증에서는 `normal` 프로필·20km/h 상한·20초로 약 81.38m 주행했다. GPS 위치와 공식 경로 선분 사이 거리는 평균 0.089m, 최대 0.237m였다. 입력 132프레임에서 GPS 위치·IMU 방향·UDP 상태 속도 사용을 확인했다. 결과는 `.runtime/ve_gps_expert/20261016_175628_223530/route_validation.json`에 있다. 출발 구간의 연결·경로 추종 검사이며 장애물·신호 성능이나 60km/h 주행을 검증한 결과는 아니다.

제어 자세 분리 후 같은 20km/h 상한·20초 검사를 반복했다. 경로 진행 5~60m의 직선 구간에서 자세 갱신은 약 6.6→29.0Hz, 사용한 자세의 평균 수신 경과 시간은 283→40ms, GPS 기준 최대 경로 오차는 0.237→0.108m였다. 각 1회 비교이며 전체 주행 성능의 보장은 아니다. `.runtime/ve_gps_expert/20261016_181058_101813/control_pose_comparison.json`에 비교 범위와 지표를 기록했다.

같은 60km/h 상한으로 35초 시험도 완료했다. 실제 최고속도는 약 40km/h였고, 약 202m를 이동했다. 이전 60km/h 상한 실행과 같은 직선 5~60m 구간에서 자세 갱신은 7.2→29.1Hz, 평균 자세 수신 경과 시간은 258→40ms, 차량 방향 표준편차는 0.80→0.26도였다. 이번 35초 전체 구간의 GPS 기준 경로 거리는 최대 0.127m였다. 전체 루트·실제 60km/h·장애물 회피 시험 결과는 아니다. `.runtime/ve_gps_expert/20261016_181237_503300/control_pose_comparison.json`을 참고한다.

GPS 수신 시점은 NMEA의 수신 시간으로 신선도를 확인한다. VE 경로에서 차량 속도는 LiDAR 원본 시각 기준 과거 50ms 안의 샘플을 선택한다. IMU에 원본 시각이 있으면 같은 기준을 적용하고, 없는 107-byte IMU는 수신 시간만 확인하며 로그에 구분한다. 물리적 동시 취득을 보증하지 않는다. `--gps-blackout-hold`를 켜면 GPS 재밍 시 마지막 정상 GPS에서 차속·IMU로 적분하며, GT 위치로 대체하지 않는다.

빈 터널의 **센서 전용 실행 경로**는 `--gps-hold-method aligned --gps-anchor-window-s 1`이 기본이다. 상태 원본 시각의 연속 적분과 IMU 시각 정렬, 기울기의 수평 투영, 진입 전 GPS 초기 위치 안정화를 적용한다. 터널 안 GPS는 비교용이며, 초기화 평균에도 사용하지 않는다. `legacy`로 기존 계산을 재현할 수 있다. 원시 센서와 평가용 상태 위치는 `driver/sensor_audit.jsonl`에 분리 기록한다. 평가용 위치는 주행 후 진단에만 쓰며 추정·제어 입력에는 포함되지 않는다. 보완 전후 5회 결과와 한계는 [GPS_HOLD_ERROR_CHECK.md](#gps-hold-error-check)에 있다.

현재 대회판 LiDAR는 한 시각당 75개 패킷이 0.4도 간격으로 한 바퀴를 구성하며 시작 각도가 0도가 아니다. 이 실행에만 `morai_cyclic_frame_timestamp`를 선택해 원형 각도와 0.01도 반올림 차이를 처리한다. 프레임 간 혼합과 중간 패킷 누락·중복·순서 뒤바뀜은 계속 거부한다. 기존 VE 실행의 기본 LiDAR 정책은 변경하지 않는다.

## 시나리오 러너와 GPS 실행기 연결

`run_ve_full_route.sh`는 이제 기존 풀루트의 장면/NPC 수명주기를 유지하면서 `run_ve_gps_expert.py`를 주행 프로세스로 실행한다. 별도 GPS 실행기를 동시에 켜지 않는다.

```bash
cd ~/AIMDrive
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh
# 터널 진입 전(고주로 종료 체크포인트 13)에 종료하려면:
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh --stop-before-tunnel
# 시뮬레이터/제어/GPU 실행 없이 설정 검사:
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh --check
```

기본은 시작 위치 배치, 기본 MORAI 조향 프로필, 60km/h 상한, GT Expert 검증된 정적 2개 배치, 이동 NPC 유지, 고주로 진입 정차 차량 제외다. `--speed-cap-kmh`(`--expert-speed-cap-kmh`), `--seed`, `--static-preset`, `--ego-trace`, `--sim-profile`을 지원한다. 충돌·주행 프로세스 종료·180초 경로 진행 없음은 실패로 종료하며 도착하면 GPS 주행 프로세스와 VE도 함께 종료한다.

주행 입력은 GPS/IMU/UDP 속도와 VE 장애물·신호이며 기존 최신 자세 전달을 그대로 사용한다. 러너는 gRPC로 장면/출발 위치/모드 설정, NPC와 보행자 신호 이벤트, 충돌·완주 판정 및 종료 정지를 수행한다. 러너가 관찰한 GT 자차·객체·신호 값은 GPS 주행 프로세스에 전달하지 않는다. 따라서 이 모드는 전체 도구가 GT를 전혀 읽지 않는 실행이 아니라, **주행 의사결정에 GT 인지를 사용하지 않는 시나리오 시험**이다.

러너가 사용한 경로/설정을 파일로 고정해 자식 실행기에 전달하고, 자식이 재구성한 경로가 일치하지 않으면 출발을 거부한다. 출발 위치는 GPS로 재확인한다. 풀루트 4,430점과 터널 전 3,530점의 오프라인 설정 검사를 통과했으며, 이 연결 방식의 실제 풀루트 완주는 별도 확인 대상이다.

로그는 `MoraiDrive/scenario_runner/.runtime/ve_gps_full_route/<시각>/`에 모인다. `plan.json`, `report.json`, `scenario_ego_trace.csv`는 시나리오 측 자료이고, `driver.console.log` 및 `driver/{inputs.jsonl,control.jsonl,expert.log,publisher.log,report.json}`은 GPS/VE 주행 자료다. `control.jsonl`에 신호 ID·상태 코드·선택 head도 함께 기록한다. 기존 캠페인에서 지정한 `--ego-trace` 경로도 유지한다.
<!-- end-source: docs/VE_GPS_INPUT.md -->

---

<a id="gps-hold-error-check"></a>

<!-- source: docs/GPS_HOLD_ERROR_CHECK.md -->
# 터널 GPS 음영 구간 위치 유지(IMU + 차속) 오차 확인

터널 입구부터 출구까지 GPS를 사용할 수 없다고 가정하고, 차량 위치를 `진입 전 정상 GPS로 초기화 + IMU 방향 + 차량 상태 속도`로 적분(dead reckoning)해 주행할 때 실제 GPS와 얼마나 차이 나는지 재는 방법이다. 실제 대회에서 GPS가 끊기고 복구되는 정확한 경계는 아직 확인되지 않았다.

이번 빈 터널 시험은 **GPS·IMU·차량 상태의 차속 세 입력만** 사용한다. `--sensor-only`를 지정하면 P1·신호 모델 가중치, PyTorch, 카메라, LiDAR가 필요 없다. 시나리오 러너는 출발 배치·빈 도로 확인·충돌/완주 관찰·반복 실행을 담당하고, 기존 PurePursuit/PID는 GPS 또는 적분 위치로 지도 경로를 추종한다. 시나리오 관찰용 GT 위치는 적분이나 GPS 오차 계산에 쓰지 않는다. 일반 차량 상태 패킷에 포함된 위치는 별도 `sensor_audit.jsonl`에 평가용으로만 기록하며, 주행 후 동일 원본 시각의 적분 위치와 비교하는 진단에 한정한다.

## 무엇을 재는가

- 지도상 터널 `A2256W000126`의 입구~출구 안에서 들어오는 실제 GPS fix를 **위치 추정 입력에서 제외하고**(`--simulate-gps-blackout`), 대신 위치 유지 값(`GPS_dead_reckoning`)으로 주행한다.
- 버린 실제 fix의 수신 시각에 해당하는 DR 기록을 보간해 `simulated_comparisons`로 남기고, 같은 수신 시각의 위치 차이를 차량 진행 방향 기준 앞뒤/옆으로 나눠 낸다. 로그에 반복된 동일 GPS 표본은 분석기가 한 번만 사용한다.
- 터널 전에는 정상 GPS를 사용한다. 보완 방식(`aligned`)은 진입 전 1초의 GPS를 연속 적분 이동량으로 같은 기준에 옮겨 평균한 초기 위치를 사용한다. 단순히 이동 중인 위치를 평균해 지연시키는 방식이 아니다. 터널 안에서는 차속과 IMU 방향만으로 적분하며, 출구를 지난 정상 GPS가 수신되면 GPS 사용으로 복귀한다.
- 이 구간과 센서 잡음 조건은 실제 대회 환경을 확인한 결과가 아닌 시험 가정이다. `--simulate-gps-blackout`은 개발용 옵션이며 대회 실행에는 쓰지 않는다.
- 위치 유지 코드: 기존 `VLDrive/inference/sensors/gps_blackout_hold.py`, 센서 전용 보완 방식 `gps_aligned_hold.py` (한도 120 s / 250 m, 초과하면 위치를 내지 않음).

## GPS 사용 불가 구간의 시험 기준

좌표는 지도 로컬 `(x, y)`, 단위는 m이다.

| 기준 | 좌표 |
| --- | --- |
| 지도상 터널 입구 | `(29.589356, -487.650273)` |
| 지도상 터널 출구 | `(-74.860745, -544.578033)` |

터널 중심선 경로 길이는 약 **118.986 m**이다. 실행기는 현재 MGeo 지도의 `A2256W000126` 첫 점과 마지막 점을 읽어 모의 차단 영역을 만든다. 입출구를 연결하는 축의 양 끝에 수직인 경계와 횡방향 ±15 m를 사용하며, 입구 전·출구 후의 종방향 여유는 추가하지 않는다. 횡방향 범위는 시험용 영역 폭이지 실제 도로·벽 폭이 아니다. 차단 여부는 수신 GPS 좌표로 판정하므로 실제 전환 위치에는 GPS 표본 주기와 잡음이 반영된다.

센서 전용 실행기는 `driver/plan.json`에 실제 사용한 차단 좌표와 센서·제어 설정을 저장한다. 기존 VE 실행 경로는 `publisher_config.json`에 기록한다. 기존 Expert `planning/Jamming/section.json`의 약 130.7 m 구간과 과거 LiDAR 구조물 경계 추정값은 이번 시험 경계로 사용하지 않는다.

## 준비

1. `git pull` 후 `~/AIMDrive`에서 작업한다.
2. MORAI는 **UDP 모드**여야 한다(ROS 모드가 아님). 센서·차량 상태·제어 포트 설정은 [VE_GPS_INPUT.md](#ve-gps-input)의 표를 따른다. GPS(7506→10506), IMU(7507→10507), 차량 상태(8001→8000), 제어(7600→7601)가 필요하다.
3. MORAI에 시험 지도를 로드하고 시나리오 실행이 가능한 상태로 둔다. 아래 터널 전용 명령은 시나리오 러너가 터널 입구 약 40 m 전의 출발 위치를 준비한다.
   화면에 `Pause — Press the ESC key`가 있으면 ESC로 해제한다. UI 일시정지 중에도 UDP 패킷·벽시계 시각은 갱신될 수 있어 수신 성공만으로 주행 가능 상태를 판정하지 않는다. 실행기는 출발 GPS와 IMU 방향이 배치 경로에 맞는지 확인한다.
4. conda 환경이 켜져 있으면 `CONDA_PREFIX`가 shapely 로드를 깨뜨린다. 실행 전에 `unset CONDA_PREFIX` 한다.

## 실행

지도상 터널 입구 약 40.19 m 전부터 출구 약 30.50 m 뒤까지 총 189.67 m를 장애물 없이 반복한다. GPS를 인위적으로 제외하는 범위는 그중 터널 입구~출구뿐이며, 접근·출구 후 구간은 정상 GPS와의 전환을 확인하는 데 사용한다.

```bash
cd ~/AIMDrive
unset CONDA_PREFIX
bash MoraiDrive/scenario_runner/tools/run_ve_full_route_campaign.sh \
  --mission full_loop_tunnel_normal --include-tunnel --sensor-only \
  --gps-blackout-hold --simulate-gps-blackout \
  --gps-hold-method aligned --gps-anchor-window-s 1 \
  --speed-cap-kmh 40 \
  --runs 5 --stall-seconds 60 --max-trial-seconds 300 --static-stuck-seconds 0
```

- 첫 줄에 출력되는 `campaign_output` 경로가 결과 폴더다(`MoraiDrive/scenario_runner/.runtime/ve_full_route_campaign/<날짜_시각_PID>`). 진행 상황은 그 안의 `status.json`, `campaign.log`.
- 오래 걸리면 맨 앞에 `--background`를 붙여 백그라운드로 돌릴 수 있다: `run_ve_full_route_campaign.sh --background --mission ...`
- 속도는 `--speed-cap-kmh`로 정한다. 20, 40, 60으로 나눠 시험하고 실제 주행 속도도 함께 확인한다.
- 센서 전용 기본값은 `--gps-hold-method aligned --gps-anchor-window-s 1`이다. 기존 계산을 재현하려면 `--gps-hold-method legacy`, 원본 시각 정렬만 비교하려면 `--gps-hold-method aligned --gps-anchor-window-s 0`을 쓴다. 이 선택은 센서 전용 실행 경로에 적용되며 VE 실행 경로 전체를 교체한 것은 아니다.
- 센서 전용 실행 연결을 정지 상태에서 확인하려면 먼저 `bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh --check --mission full_loop_tunnel_normal --sensor-only --gps-blackout-hold --simulate-gps-blackout`을 실행한다. 이 검사는 센서나 시뮬레이터에 접속하지 않는다.
- 센서 전용 시험에는 장애물 인지가 없으므로 빈 터널 미션만 허용한다. 장애물 회피 평가는 이후 별도의 인지·주행 설정으로 진행한다.
- 기존 VE 실행 경로의 입력 연결만 먼저 보고 싶으면 `bash MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.sh --seconds 20`으로 제어 없이 20초 동안 수신을 확인한다.

## 결과 보기

현재 기본 위치 유지 방식은 보완된 `aligned` DR이며, 센서 원본 시각 정렬과 진입 전 1초 GPS 앵커 안정화를 사용한다. 별도 단일 주행 `20261023_224102_277609`의 동일 수신 시각 GPS 231개 표본에서는 **평균 0.0824m / 최대 0.1766m**였다. 아래 5회 합산 결과와는 다른 표본 통계이며, 매 주행에서 같은 수치를 보장하는 것은 아니다. 해당 단일 주행의 DR 전용 HTML은 `http://localhost:8765/dr_reference/report.html`에서 볼 수 있다.

```bash
python3 MoraiDrive/scenario_runner/tools/analyze_tunnel_hold.py <campaign_output 경로>
```

시험마다 아래가 한 줄로 나온다.

| 열 | 의미 |
| --- | --- |
| 유지 | 위치 유지가 쓰인 프레임 수 |
| 유지s / 거리m | 유지한 시간 / 유지하는 동안 주행한 거리 |
| 앞뒤·옆·2D 오차 | 각 오차의 최대 / 중앙값 / 95백분위값, m |
| 출구 마지막 관측 오차 | 완주가 확인된 시험에서 마지막 비교 표본의 오차 |
| 실제 속도 | DR 구간의 실제 속도 통계, km/h |

`<campaign_output>/tunnel_hold_analysis/`에 `analysis.json`, `trials.csv`, `samples.csv`, 오차-시간/거리 그래프 `errors.png`를 저장한다. matplotlib이 없으면 JSON/CSV를 생성하고 그림 생성 불가 사유를 표시한다. 앞뒤 오차 부호가 양수면 유지 위치가 비교 GPS보다 앞서 있다는 뜻이다.

완주 시험과 부분 주행·실패 시험을 구분한다. 입력 만료를 포함한 실패 시험은 진단용 기록과 실패 횟수를 남기고, 완주 정확도 합산에서는 제외한다. 과거 로그의 `GPS 경과시간 × 현재 속도` 보정 결과는 근사치로 따로 표시하고 새 수신 시각 정합 결과와 합산하지 않는다.

## 알아둘 점

- 누적 오차는 속도·방향 편향, 시간 정합, 주행 시간과 거리의 영향을 받는다. 거리당 일정 비율이나 정지 시 무조건 증가하지 않는다고 가정하지 않는다.
- 비교 대상 GPS 자체의 잡음과 수신 지연도 오차에 포함된다. 현재 정합 기준은 monotonic 수신 시각이며 GPS·IMU·차속의 물리적 측정 시각이 정확히 같다는 뜻은 아니다.
- 센서 전용 기본 GPS lever arm은 `(0, 0)`, 수신 전 지연 보정은 0이다. 보완 방식은 현재 이 기본값만 지원한다. 실측한 지연 보정을 사용하려면 `legacy`를 선택한다. 장착 위치·지연을 실측하지 않은 상수로 결과를 작게 맞추지 않는다.
- 시간·거리 한도 초과, 오래된 IMU, 차속 누락 때는 위치를 제공하지 않고 제동한다. `sensors_expired_during_inference` 등의 입력 만료 시험도 정확도 합산에서 제외하되 원인을 남긴다.
- 허용오차는 아직 정하지 않았다. 분석기는 임의 기준으로 GPS 대체 가능 판정을 내리지 않는다. 반복 시험의 최대·95백분위·출구 오차와 전환 상태를 검토한 뒤 허용범위를 정한다.

## 2026-10-23 보완 전 센서 전용 5회 실주행 결과

캠페인 `scenario_runner/.runtime/ve_full_route_campaign/20261023_174217_4174579`에서 빈 터널 5회 모두 완주했다. 속도 상한은 40 km/h, 터널 내 실제 속도 중앙값은 각 회차 약 35.9 km/h, 최고속도는 약 37.59 km/h였다. DR 유지 시간은 약 12.1초, 거리는 약 119 m다.

동일 수신 시각으로 정렬하고 중복을 제거한 **1,197개 GPS 비교 표본**의 결과다. 초기에 정렬할 DR 후속 표본이 없어 근사 비교로 남은 5개 기록은 아래 수치에 합산하지 않았다.

| 오차 | 최대 | 중앙값 | 95백분위 |
| --- | ---: | ---: | ---: |
| 2D 위치 | 0.288 m | 0.106 m | 0.223 m |
| 앞뒤 절댓값 | 0.226 m | 0.055 m | 0.160 m |
| 옆 절댓값 | 0.282 m | 0.060 m | 0.210 m |

출구 직전 마지막 GPS 비교 표본의 2D 오차는 회차별 약 0.187 / 0.061 / 0.071 / 0.100 / 0.066 m였다. 실제 대회, 다른 속도·정지·회피 조향·다른 센서 잡음 조건까지 검증한 결과는 아니다.

첫 시도는 MORAI UI의 Pause 상태로 차량이 움직이지 않아 정상 중단했으며, 그 실행은 이 5회 결과에 포함하지 않았다. UI 일시정지 해제 후 IMU 방향 갱신을 확인하고 새 캠페인에서 측정했다.

HTML 보고서를 생성하려면:

```bash
python3 MoraiDrive/scenario_runner/tools/render_tunnel_hold_report.py <campaign_output 경로>
```

`<campaign_output>/tunnel_hold_analysis/report.html`을 브라우저에서 열면 회차·비교 방식·시간/거리 축을 바꾸며 확인할 수 있다. 보고서는 외부 네트워크 없이 열리며 분석에 사용한 표본을 포함한다.

## 2026-10-23 원인 검증·보완 후 같은 조건의 5회 재시험

보완 전 소스를 `scenario_runner/.runtime/tunnel_hold_audit/20261023_204929/before/`에 보관했다. 검토한 오차 요인은 다음과 같다.

1. GPS로 다시 초기화할 때 첫 적분 구간은 수신 시간, 이후 구간은 상태 원본 시간으로 계산하던 혼용을 제거했다. 연속 상대 이동량은 정상 GPS 수신 여부와 무관하게 상태 원본 시간으로 적분한다.
2. 차속 원본 시각에 IMU 방향을 보간하거나 최대 50ms의 자이로 전파를 적용한다. 그 시점까지 실제 수신된 데이터만 사용한다. 원본 시각이 없으면 기존 수신 시각 기준으로 돌아가며 정렬 방법을 구분한다. IMU 쿼터니언의 전방 벡터를 사용해 기울기의 수평 투영도 반영한다.
3. 마지막 GPS 한 점의 표현 해상도·잡음이 전체 구간의 초기 위치 편향으로 남을 수 있어, 진입 전 1초의 GPS를 이동량으로 보정한 뒤 평균한다. 터널 안에서 버린 GPS나 평가용 위치를 이용한 보정은 없다.

먼저 기존 방식으로 진단 주행 1회(`20261023_205039_183720`)를 기록했다. **동일 센서 기록 재생**에서 GPS 대비 평균 2D 오차는 기존 0.1029m → 시각·이동 방향 보완 0.0916m → 초기 위치 안정화 포함 0.0841m였다. 재생은 원인별 진단이며 실제 재주행 횟수에 포함하지 않는다.

그 후 보완 코드를 사용해 캠페인 `20261023_210218_196167`을 실제 실행했다. 기존과 같은 빈 터널·40km/h 상한·5회이며 모두 완주했다. 터널 속도 중앙값은 약 35.9km/h, 유지 시간 약 12.1초, 거리는 약 119m였다. 동일 수신 시각으로 정렬한 새 GPS 비교 1,207개 결과다.

| GPS 대비 2D 오차 | 기존 5회 | 보완 후 5회 |
| --- | ---: | ---: |
| 평균 | 0.1128 m | 0.0860 m |
| 중앙값 | 0.1056 m | 0.0853 m |
| 95백분위 | 0.2228 m | 0.1517 m |
| 최대 | 0.2879 m | 0.2058 m |

| 보완 후 회차 | GPS 대비 평균 2D | 최대 2D |
| --- | ---: | ---: |
| 1 | 0.0874 m | 0.1983 m |
| 2 | 0.0851 m | 0.1769 m |
| 3 | 0.0864 m | 0.2009 m |
| 4 | 0.0811 m | 0.1655 m |
| 5 | 0.0900 m | 0.2058 m |

평균 약 23.7%, 최대 약 28.5% 감소했다. 서로 다른 시점의 두 캠페인에서 관측한 개선이며 모든 조건에 대한 개선율 보장은 아니다. 옆 방향 최대도 0.2818m → 0.1570m로 줄었다. 시작 시각 정렬에 필요한 후속 적분 표본이 없어 근사 비교로만 남은 새 기록 4개는 위 합산에서 제외했다.

별도 진단으로, 새 주행의 상태 패킷 평가용 위치와 **동일 원본 시각**인 적분 위치를 비교했다. 중복을 제거한 2,426개에서 평균 0.0579m / 중앙값 0.0553m / P95 0.0940m / 최대 0.1440m였다. 이 값은 GPS 대비 오차가 아니며 두 통계를 합산하지 않는다. 기록된 DR 제어 프레임에 대응하는 평가 기준 위치의 누락은 0개였다. 센서와 상태 패킷의 기준점은 별도로 실측 검교정하지 않았다.

새 로그의 NMEA 위·경도는 분 단위 소수 4자리다. 이 지역에서 한 표현 간격은 동서 약 0.148m·남북 약 0.185m이며, 이는 실제 GPS 오차 자체를 뜻하지 않는다. NMEA 시각도 0.1초 단위라 정확한 GPS 취득 시각을 알 수 없다. **현재의 GPS 비교만으로 1cm 정확도를 검증할 수 없고, 이번 보완 후 결과도 1cm 수준은 아니다.**

보완 전후 비교와 평가용 기준 진단을 포함한 HTML을 다시 생성하려면:

```bash
python3 MoraiDrive/scenario_runner/tools/review_tunnel_hold_campaign.py \
  MoraiDrive/scenario_runner/.runtime/ve_full_route_campaign/20261023_210218_196167 \
  --baseline MoraiDrive/scenario_runner/.runtime/ve_full_route_campaign/20261023_174217_4174579 \
  --diagnostic-driver MoraiDrive/scenario_runner/.runtime/ve_gps_full_route/20261023_205040_208542/driver \
  --ablation-json MoraiDrive/scenario_runner/.runtime/tunnel_hold_audit/20261023_204929/replay_before_after.json
```

결과는 새 캠페인의 `tunnel_hold_analysis/{report.html,review.json}`이다. 기존 캠페인 파일은 보존한다. `replay_tunnel_sensor_audit.py <driver 폴더> --output <JSON 경로>`로 동일 센서 기록에 대한 세 방식의 재생 진단을 재생성할 수 있다.
<!-- end-source: docs/GPS_HOLD_ERROR_CHECK.md -->

---

<a id="ve-object-input"></a>

<!-- source: docs/VE_OBJECT_INPUT.md -->
# VE 장애물 입력으로 기존 Expert 실행

카메라 4개·LiDAR UDP → VE bbox·스칼라 속도 → 로컬 객체 추적 → UDP 12050
→ `/ve/Object_topic` → 기존 Expert의 경로 계획·제어로 연결한다.
Expert 실행 시 `/Object_topic:=/ve/Object_topic`으로 장애물 구독만 바꾼다.
자차 `/Ego_topic`과 신호등 `/GetTrafficLightStatus`는 기존 ROS 입력이다.
VE waypoint와 LLM 출력은 주행에 사용하지 않는다.

객체 클래스는 `type=-1`로 유지한다. `ve_tracked_unknown:*` 이름을 가진
객체만 기존 Expert의 ACC·회전교차로·고주로·정적 회피 분기에 추가로 허용한다.
이동/정지/불확실 상태는 위치 이력과 VE 속도로 추정하며, 시뮬레이터 객체
ID나 정답 분류를 가져오지 않는다. UDP 실행은 `perception_source=ros`와
토픽 remap을 사용한다. 별도 ROS 전용 실험 모드인 `perception_source=ve`는
다른 실행 경로이므로 UDP 실행 설정에 혼용하지 않는다.

현재 추적 정책은 `world_position_supported_speed_v2`다. 최소 4회·0.3초의
일관된 위치 변화로 이동을 확인하고, 그 속도와 맞을 때만 VE 스칼라 속도를
Expert에 전달한다. 두 값이 다르면 확인된 위치 이력 속도를 사용한다.
높은 VE 속도만으로 이동을 선언하거나 다음 bbox 위치를 예측하지 않는다.
정지는 0.5초 이상의 안정된 위치 이력으로 판단한다.

관측 초기나 bbox 흔들림으로 움직임을 확인하지 못하면 객체는 `uncertain`으로
계속 전달하고 계획용 속도는 0으로 둔다. 이것은 정지 GT나 정확한 속도 추정이
아니며, 새로 보이는 이동 차량에도 일시적인 감속·양보가 늘 수 있다.
원본 `speed_mps`, bbox 위치·크기는 보존하며 `planning_speed_mps`와
`planning_speed_source`에 Expert용 속도와 근거를 별도로 기록한다.
센서 시점 오차나 연속적인 위치 편향까지 이 추적기가 해결하지는 않는다.

LiDAR UDP 입력은 MORAI의 프레임별 microsecond-of-hour 타임스탬프와
연속된 방위각을 확인한다. 한 프레임의 마지막 패킷을 받으면 바로 출력하며,
다음 프레임을 기다리지 않는다. 패킷이 빠지거나 섞인 프레임은 제외한다.
촬영 시각은 Ego 또는 카메라의 source header를 기준으로 복원해 계측 로그의
`selected_samples.lidar.source_stamp_ns`에 기록한다. UDP VE의 ROS 자차 입력은
이 LiDAR 시각 이전 50ms 이내의 이력에서 선택하며 미래 자차 위치를 적용하지
않는다. 카메라는 여전히 각 최신값이므로 `source_spread_ms`를 확인해야 하며, 이 수정만으로
모든 센서의 동시 촬영이나 충돌 회피 성공을 보장하지 않는다.

## 실행 환경

ROS Noetic 및 프로젝트 `morai_msgs`, 기존 Expert 환경, GPU 추론용 Python이
필요하다. 기존 `internal_sensor_record_env.sh`와 PC별 환경 설정을 사용한다.
체크포인트 파일은 Git에 포함하지 않는다.

- 기본 E3: `mixed_grid_fast_20260930_152712_134039/checkpoint-best-object.pth.tar`
- 선택 E4: `continue_e4e5_20261004_050644/checkpoint-epoch-001.pth.tar`
- 체크포인트 경로·SHA256: `VLDrive/inference/realtime_ve_objects.py`의 `CHECKPOINTS`
- 센서 포트·LiDAR 변환: `VLDrive/configs/vision_encoder_realtime_best_20261003.json`

현재 기본 경로는 `/mnt/d/MoraiDrive/dataset/_training_runs/morai_transfer`다.
다른 PC에서 경로가 다르면 위 체크포인트 경로를 해당 위치로 바꾸되 SHA256은
유지한다. 센서 설정의 경로 항목에도 로컬 경로가 있으므로 사용 전에 확인한다.
UDP 장애물 실행은 `perception_only=True`여서 지도/route 파일을 읽지 않지만,
Expert의 주행 경로와 지도는 별도로 필요하다. 임의 센서 위치 변경 시 기존
LiDAR 변환을 그대로 적용하지 않는다.

MORAI는 Play 상태로 유지한다. 차량 ROS 제어 `/ctrl_cmd`, 자차·신호등 연결과
카메라·LiDAR UDP 전송이 필요하다. 다른 VE 실행이나 데이터 수집과 동시에
같은 센서 포트를 사용하지 않는다. 기존 연결 점검 명령은 다음과 같다.

```bash
cd ~/AIMDrive
bash MoraiDrive/data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
```

## 주행 및 화면

기존 시나리오 러너가 경로·NPC·Expert를 함께 관리하는 실행:

```bash
cd ~/AIMDrive
bash MoraiDrive/scenario_runner/tools/run_ve_scenario.sh roundabout
bash MoraiDrive/scenario_runner/tools/run_ve_scenario.sh highway
```

하나를 선택해서 실행한다. `--repeat`는 시나리오 반복, `--ve-checkpoint E4`는
E4 선택이다. `--check`는 오프라인 설정 확인이며 실제 주행 성공 판정이 아니다.
시작 후 `[PASS] Expert → VE 장애물 입력 연결 확인. GT 장애물 구독 없음.`을
확인한다. `[VE Hz]`에 입력·추론·ROS 발행 속도가 표시된다.

실제 장애물 정보와 비교하려면 위 명령에 `--ve-compare-gt`를 추가한다.
별도 기록기가 `/Object_topic`, `/Ego_topic`, 읽기 전용 gRPC bbox를 저장한다.
Expert의 장애물 입력은 계속 `/ve/Object_topic`만 사용한다. 종료 후 실행 폴더의
`gt_comparison_report.json`은 원래 토픽 기준, `gt_geometry_audit_report.json`은
차량 기준점과 bbox 중심의 차이를 반영한 비교 결과다. `bbox_reference_review/`에는
GT와 VE 박스를 겹쳐 보는 HTML이 있다. LiDAR의 실제 촬영 시각은 확인되지 않으므로
수신 시각으로 맞춘 수치는 비동기 오차를 포함한다. 분석 `COMPLETE`와 주행 성공은
별개이며, 시나리오 충돌·종료 원인은 콘솔 로그로 함께 확인한다.

이전에 생성한 기존 Expert 경로를 그대로 쓰는 단독 실행:

```bash
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_udp_expert.sh --drive --seconds 0
```

`AIM2026/Expert/.runtime/path.csv` 또는 `.runtime/scenario_runner/path.csv`가
필요하다. `--seconds 0`은 Ctrl+C까지 실행한다. `--drive`를 생략하면
`/ve/shadow_ctrl_cmd`로 계산 결과만 출력한다.
시나리오 러너와 단독 Expert를 동시에 실행하지 않는다.

별도 터미널에서 VE bbox·추적 상태와 Expert 경로를 BEV로 본다:

```bash
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_bev.sh
```

## 수신·추론 계측

Windows UDP 중계(`VLDrive/inference/udp_relay.py`)는 카메라 영상의 바이트와
timestamp를 유지하면서 WSL로 보내는 UDP를 최대 1,200바이트로 분할한다.
기존 65,000바이트 전송에서 확인된 Windows→WSL 카메라 누락을 줄이기 위한
처리다. 중계 코드를 갱신했으면 실행 중인 Windows 중계도 다시 시작해야 한다.
시작 로그의 `camera UDP maximum=1200 bytes`를 확인한다.
`--camera-datagram-bytes 0`은 비교 진단용 기존 전송 방식이다.
패킷 수는 증가하므로 영상 수신 속도는 `packet_hz`가 아닌
`completed_sample_hz`로 확인한다. 센서 포트와 LiDAR 전송 내용은 동일하다.

VE UDP 실행은 카메라 수신·조립을 별도 프로세스에서 처리한다. 추론 프로세스는
공유 메모리에서 최신 완성 JPEG만 가져온다. 이미지 바이트, 수신 시각과 sample ID를
유지하며, 카메라 Hz를 낮추거나 이미지를 다시 압축하지 않는다. 수신 프로세스가
종료되면 오래된 영상을 새 영상처럼 사용하지 않고 오류로 종료한다.

Expert를 종료한 상태에서 센서만 전송하고 실행한다. 아래 명령은 차량을
제어하지 않으며 카메라·LiDAR 수신, 전처리, 추론, UDP·ROS 전달을 계측한다.

```bash
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_hz_check.sh
bash MoraiDrive/scenario_runner/legacy_expert/run_ve_hz_check.sh --overnight --background
```

기본 카메라 수신 모드는 `process`다. 동일한 센서·중계 조건에서 이전 방식과
비교할 때만 `--camera-receiver-mode thread`를 지정한다. 일반 VE UDP 실행에서는
설정 파일의 `runtime.camera_receiver_mode`로 같은 옵션을 지정할 수 있다.
추론은 새 LiDAR sweep마다 한 번 수행하므로 카메라 Hz가 추론 Hz를 결정하지 않는다.

기본은 60초, overnight는 30분씩 8회다. 구간 사이에 모델 프로세스를 다시
시작한다. 시작 메시지의 경로에 `status.json`, `console.log`, 구간별
`timing_report.json`, 최종 `campaign_report.json`이 생성된다.
전체 결과는 `MoraiDrive/scenario_runner/.runtime/ve_objects/` 아래에 저장한다.
백그라운드 종료는 해당 `launch.json`의 PID에 SIGINT를 보낸다.

카메라 패킷은 같은 timestamp·연속 fragment index·정상 JPEG 경계가 확인되어야
사용한다. 추론한 LiDAR sample ID를 고정해 동일 sweep의 중복 추론을 방지한다.
이 검사는 네트워크 누락을 복구하거나 센서의 물리적 동시 촬영을 보장하지 않는다.
`COMPLETE`는 계측 완료다. 누락·지연·JPEG 오류·선택된 LiDAR 중복 항목을
별도로 확인해야 하며, 무충돌 주행이나 검출 정확도의 통과 판정이 아니다.

이번 변경에는 일반 lattice 후보의 충돌 검사 범위를 주변 객체 전체 및 경로
샘플 사이까지 확대하는 수정도 포함된다. 해당 변경은 VE 이외의 일반 lattice
사용 경로에도 적용된다. 실제 MORAI 주행 성능과 장시간 안정성은 별도 확인한다.
<!-- end-source: docs/VE_OBJECT_INPUT.md -->

---

<a id="ve-roi-rviz"></a>

<!-- source: docs/ve_roi_rviz.md -->
# VE 인지 ROI · 판제 입력 RViz

MORAI ROS 카메라 4개와 LiDAR를 Windows CUDA의 E3 가중치로 추론하고, 판제용 `/ve/Object_topic`을 그대로 RViz에 표시한다. 자차 위치·헤딩·속도는 이번 단계에서 `/Ego_topic`을 사용한다. 객체 GT는 구독하지 않으며, 이 두 실행 명령은 차량 제어를 발행하거나 Expert를 시작하지 않는다.

## 실행

### WSL / Ubuntu bash 터미널

프롬프트가 `(base) autonav@autonav:...$`라면 이 명령을 사용한다. Linux에 PowerShell을 설치할 필요가 없다. 추론 실행 스크립트가 Windows CUDA 환경을 연결한다.

터미널 1 — 추론:

```bash
cd /mnt/c/Users/dydwn/AIMDrive/AIM2026/Expert
bash tools/run_ve_ros_inference.sh
```

터미널 2 — RViz:

```bash
cd /mnt/c/Users/dydwn/AIMDrive/AIM2026/Expert
bash tools/run_ve_roi.sh
```

추론이 이미 켜져 있으면 안내만 출력하고 기존 실행을 유지한다. 종료는 각각 `bash tools/run_ve_ros_inference.sh --stop`, `bash tools/run_ve_roi.sh --stop`이다. 실행 중인 터미널에서는 Ctrl+C로 종료할 수도 있다.

### Windows PowerShell 터미널

MORAI와 ROS bridge 9090이 켜져 있고 센서가 송신 중인 상태에서 실행한다. Windows PowerShell 터미널 두 개를 사용한다.

터미널 1 — 비전 인코더 추론:

```powershell
cd C:\Users\dydwn\AIMDrive
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\run_ve_ros_inference.ps1
```

터미널 2 — ROI와 객체 RViz:

```powershell
cd C:\Users\dydwn\AIMDrive
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\run_ve_roi.ps1
```

기본 가중치는 `Y:\DCM\weights\checkpoint-best-object.pth.tar`다. SHA256이 확인된 E3만 로드한다. GPU는 Windows `vldrive` 환경을 사용하고, ROS 노드와 RViz는 `Ubuntu-20.04` WSL에서 실행한다. 모델 로드에는 수십 초가 걸릴 수 있다. 기본 실행 시간은 무제한이며 Ctrl+C로 종료한다. 이미 추론이 실행 중이면 중복 실행하지 않는다.

경로와 실행 시간을 바꾸는 예:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\run_ve_ros_inference.ps1 -Checkpoint 'Y:\DCM\weights\checkpoint-best-object.pth.tar' -Seconds 120
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\run_ve_roi.ps1 -NoTrails
```

RViz 창을 닫으면 뷰어만 종료한다. 백그라운드로 실행한 경우에도 아래 명령으로 각각 종료할 수 있다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\stop_ve_ros_inference.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\AIM2026\Expert\tools\stop_ve_roi.ps1
```

MORAI와 원래 rosbridge는 종료하지 않는다. 로그는 `artifacts/ve_live/`에 저장된다. 뷰어에 필요한 Linux pyproj는 `.runtime/ve_ros_deps`의 전용 폴더에 설치되며 시스템 Python은 변경하지 않는다.

## 화면 읽기

| 표시 | 의미 |
|---|---|
| 하늘색 ROI | 자차 X -20~+30 m, Y -25~+25 m. 보장 검출 거리가 아닌 모델 출력 영역 |
| 노란색 자차 박스 | 기존 고주로 플래너 차체 형상. 길이 4.635 m × 폭 2.0 m, 안전 여유 제외 |
| 노란 점 | 자차 좌표 원점. 플래너 형상은 원점에서 전방으로 0~4.635 m, 좌우 ±1 m |
| 초록 객체 박스 | 후처리가 moving으로 전달한 객체 |
| 주황 객체 박스 | 후처리가 stationary로 전달한 객체 |
| 보라 객체 박스 | uncertain 또는 unknown 객체 |
| 객체 숫자 | 추적 ID, 판제 속력 km/h 및 m/s, 현재 자차 기준 x/y m |
| 회색 가는 선 | 최근 0.8초 위치 이력. 현재 객체나 예측 박스가 아님 |
| 1 m 격자 | 거리 확인용. 전방은 화면 위, 좌측은 화면 왼쪽 |

자차 박스는 기존 플래너의 `highway_lane_planning/parameters.py`에서 치수를 읽고, 기존 충돌 검사·RViz와 같은 기준점 배치를 사용한다. MORAI 차량 외형을 GT로 실측한 박스나 비전 모델이 검출한 자차 박스는 아니다. 센서 기준점과 실제 차체 기준점의 별도 보정 문제는 이 시각화로 새로 확정하지 않는다.

주변 객체는 **실제 길이·폭의 2D footprint**다. 모델이 높이를 제공하지 않으므로 임의 높이의 3D 박스를 만들지 않는다. 경계에 걸친 검출 박스를 ROI에 맞춰 잘라내지 않는다.

| 상단 수치 | 의미 |
|---|---|
| NEW Hz | Windows 고해상도 시계로 측정한 최근 최대 50개 신규 추론 출력의 빈도 |
| ROS matched Hz | 뷰어가 실제 수신한 객체와 진단 정보를 센서 stamp로 매칭하고, 그 출력의 Windows 발행 시각으로 계산한 전달률. WSL 시계와 혼합하지 않음 |
| update age | 마지막 **신규 객체 메시지**를 받은 이후의 뷰어 로컬 경과 시간. 중복 메시지는 갱신으로 세지 않음 |
| GPU | CUDA event로 측정한 순수 forward 시간 |
| pipeline | 전처리 시작부터 객체 decode 완료까지 |
| input→output | 선택 입력 중 가장 오래된 Windows 수신부터 decode 완료까지. 촬영→ROS/WSL→Windows 이전 지연은 제외 |
| sim lag | 최신 Ego stamp와 객체의 원본 front sensor stamp 차이. 시뮬레이터 시간 기준 |
| ID births / missed | 최근 5초 동안 새 ID가 등장하거나 직전 프레임의 ID가 사라진 횟수. 검출 정확도 지표는 아님 |

ROS matched Hz는 수신된 프레임의 전달률이며, 수신 간격 지터를 Windows 시계로 직접 측정한 값은 아니다. `update age`는 로컬 표시 시계이므로 정밀 물리 지연의 전체 측정값으로 사용하지 않는다.

RViz 우하단의 fps는 화면 렌더링 빈도이고 상단 NEW/ROS matched Hz와 별개다. GPU를 MORAI·추론·RViz가 공유하므로 서로 영향을 줄 수 있다. 마커 구독 큐는 1개이며, 매번 전체 박스를 지우고 다시 생성하는 대신 바뀐 마커만 갱신한다. 렌더링이 느려져도 오래된 마커 메시지를 긴 큐에 쌓아 재생하지 않는다.

현재 검출 박스의 보간·평활·등속 예측은 하지 않는다. 신규 프레임에서 검출이 사라지면 그 박스를 즉시 제거한다. 수신 공백이 0.35초를 넘거나, upstream 입력 나이와 로컬 경과 시간이 합쳐 0.35초를 넘으면 현재 박스를 숨기고 `STALE`로 표시한다. 자차 pose가 오래돼도 박스를 숨긴다. 정상적인 빈 검출과 stale 입력은 구분한다.

객체가 실제로 정지했어도 현재 E3의 속도 head는 약 35 km/h를 출력할 수 있다. 뷰어는 이 오류를 숨기거나 0으로 고치지 않고 **판제에 들어가는 숫자 그대로** 보여 준다.

## 토픽과 좌표

| 토픽 | 역할 |
|---|---|
| `/image_jpeg/compressed` | Camera-1/2/3/4를 frame_id로 구분 |
| `/lidar3D` | Lidar3D-5, 학습된 장착 보정 사용 |
| `/Ego_topic` | 자차 위치·헤딩·속도 |
| `/ve/Object_topic` | 기존 판제용 `morai_msgs/ObjectStatusList`, map 좌표 |
| `/ve/diagnostics` | 원본 센서 stamp, 전달 식별자, 추론 성능과 입력 나이 |
| `/ve/roi_markers` | base_link 기준 RViz MarkerArray |
| `/ve/roi_tf_static` | RViz에만 연결하는 표시 전용 TF. 기존 `/tf`를 변경하지 않음 |

객체의 map 위치에서 **현재** Ego 위치를 빼고 현재 grid yaw의 역회전을 적용한다. Ego heading에는 학습·추론과 동일한 UTM 52N 원점 `[302595,4124145,0]`의 meridian convergence를 적용한다. 객체를 앞으로 예측하지 않으므로 이동 객체의 추론 지연은 화면에서 그대로 드러난다.

센서 입력은 ROS → WSL의 최신 데이터 relay → Windows GPU 순서다. 결과는 같은 relay에서 기존 `make_object_message`로 `/ve/Object_topic`에 발행한다. 12061은 내부 Windows↔WSL 포트이며, MORAI 센서 bridge 포트는 계속 9090이다. 원본 센서 stamp를 결과 header에 보존한다.

기존 Expert가 이 값을 사용하려면 객체 구독이 `/ve/Object_topic`으로 remap되어 있어야 한다. 뷰어는 그 토픽의 값을 보여 주며, 현재 실행 명령 자체가 고주로 주행을 시작하거나 GT 기반 Expert의 설정을 바꾸지는 않는다.
<!-- end-source: docs/ve_roi_rviz.md -->

---

<a id="historical-static-replan-proposal"></a>

> **과거 제안 보존:** 아래 내용은 2026-10-17의 분석·개선 제안이며,
> 이번 정리에서 구현한 기능이나 현재 검증 완료를 의미하지 않는다.

<!-- source: docs/VE_STATIC_REPLAN_PROPOSAL.md -->
# VE 입력 정적 장애물 2개 이상 — Expert 쪽 개선 제안

작성: 2026-10-17 · 대상: Expert(정적 장애물 planner) 담당자 · 이 문서는 코드 변경이 아니라 분석 결과와 제안입니다.

## 요약
VE(P1) 박스로 정적 장애물 2개 이상을 만날 때 경로가 안 나오는 실주행 실패 3건을 로그로 재현했더니,
**원인이 bbox 정확도만이 아니었고 planner 쪽 정책 두 가지가 결정적**이었습니다.
같은 정지 위치에서 VE 박스로 **새로 계획하면 경로가 나옵니다**(pair081, pair096). 막히는 이유는 "한 번 blocked가 되면 다시 계획하지 않는 것"과 "준비해 둔 계획을 현재 장애물과 대조하지 않는 것"입니다.

## 근거 (실주행 로그 재현)
| 실주행 | GT 박스 +δ 여유 | VE 박스로 새 계획 | 실제 원인 |
|---|---|---|---|
| attempt_020 (rot90) | +5 cm에서 이미 경로 없음 | raw 3/9 지점만 가능 | 배치 자체의 여유 부족. Expert/인지 모두 해결 어려움 |
| pair081 | +30 cm까지 가능 | 9/9 지점 가능 | 신호 대기 중 2번째 장애물이 VE 범위(~30 m) 밖이었고, blocked 시점 로그가 `selected_routes=[right_01]`, `search={}`(탐색 없음)라서 **장애물 1개 기준 preplan이 채택된 것으로 추정**(코드 경로 확인 필요) → 2개가 보이자 `remaining_path_collision` → 영구 blocked |
| pair096 | +5 cm 가능, +10 cm 불가 | 9/9 지점 가능 | 계획은 정상 확정 → 프레임마다 박스가 흔들려 `_remaining_clear`가 통과/실패를 오가며 blocked↔follow 반복 → 6~7 m 앞에서 정지 후 영구 blocked |
재현 스크립트·결과: `MoraiDrive/.runtime/ve_static_bench_20261017/` (README 참고), `ve_static_pair081_review_20261016/`, `ve_static_pair096_review_20261016/`.

## 추가 근거 (2026-10-17 캠페인 10회 × 2, 막힌 9건 전수)
`static_stuck`으로 끝난 9건 모두, 정지한 자세에서 VE 박스 전부(중복 박스 포함)로 새로 계획하면 **경로가 나옵니다**. 같은 자세에서 GT 박스, GT +5 cm도 모두 경로가 나옵니다(`classify_stuck.py`: 9/9). 즉 막힌 원인은 최종 박스의 정확도나 배치 여유가 아니라, **먼저 확정한 계획이 이후 박스 변화로 무효화된 뒤 다시 계획하지 않는 것**입니다. 막힌 회차의 `STATIC_FAILURE`는 1~7회뿐이고 이후 영구 blocked입니다. 아래 제안 1(재계획)이 가장 직접적인 해결입니다.
(오프라인 근사: 새 planner를 정지 자세에서 한 번 호출한 결과이며, 실제 Expert 상태 전이 전체를 재현한 것은 아닙니다.)

## 제안
1. **blocked 상태에서 재계획** — `정적장애물/recorded_planner.py` `plan()`의 `failure`가 `remaining_path_collision`/`tracking_error`로 `blocked`가 된 뒤, 자차가 거의 정지(`speed < ~0.3 m/s`)하고 일정 시간(예: 1 s) 지속되면 `reset()` 후 **측정된 현재 자세와 현재 장애물**로 `_prepare()`를 다시 호출. (`reset_after_teleport`처럼 이미 reset 경로가 있음.)
2. **preplan 채택 전 장애물 대조** — `plan()`에서 `self.pending.result()`를 채택할 때, 워커가 쓴 `pending_key`(= `_obstacle_key`)와 현재 `static_objects`를 비교(개수 + 위치 허용오차). 다르면 폐기하고 측정 자세에서 재계획. 현재 key는 float 정확 일치라 VE 입력과 맞지 않으므로 `_preview_obstacle_key`처럼 반올림(예: 위치 0.1 m, 각도 5°) 비교 권장.
3. **박스 흔들림 내성(선택)** — 같은 경로가 `remaining_clear` 실패를 낼 때, 연속 k프레임(예: 3) 실패해야 blocked로 전이. 또는 실패해도 기존 확정 경로를 최대 N 프레임 유지.
4. **참고: `uncertain` 트랙** — `planning_process.run()`의 `static_objects`는 `is_static`(= VE `motion_state == 'stationary'`)만 포함. 정지 물체가 `uncertain`으로 몇 프레임 깜빡이면 그 프레임엔 장애물 집합에서 빠집니다(실측: 확정 후 약 2~3% 프레임). 이전에 stationary였던 트랙은 `uncertain`이어도 정적으로 유지하는 방안도 검토 가치가 있습니다.

## 이미 인지 쪽에서 한 것 (참고)
- `VLDrive/inference/ve_object_tracking.py` `static_geometry_fusion=True`: stationary 박스를 world 좌표로 융합(시점별 샘플, 8 m 이내 갱신 중지). 로그 리플레이에서 중심 오차 −25%, 확정 경로 무효화 −65%(pair096). 기본값 off, 러너 플래그 `--static-geometry-fusion`.
- bbox 모델 재학습(파일럿 3종)은 정적 회전 박스 5~15% 개선에 그쳤고 동적·일반 정적 박스가 악화되어 채택하지 않음.

## 검증 방법
- 위 벤치마크의 `commit_test.py`(확정 후 무효화율)와 `budget.py`(GT 여유폭)로 변경 전후 비교.
- 실주행: `run_ve_full_route_campaign.sh --seed S --runs N [--static-geometry-fusion]`를 같은 `--seed`로 A/B 반복.
<!-- end-source: docs/VE_STATIC_REPLAN_PROPOSAL.md -->
