# GPS 기반 VE Expert UDP 실행

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

차량 상태·제어는 센서 중계와 별도로 Windows Host `172.18.32.1`, WSL Destination `172.18.40.28`의 직접 UDP 연결을 사용한다. 차량 상태는 Host 8001 → Destination 8000, 제어는 Host 7601 ← 실행기 송신 7600이다. 대회판에서는 `Competition Vehicle Status`, 기본 MORAI에 해당 메뉴가 없으면 일반 `Ego Vehicle Status`를 선택한다. 전체 상태 메시지가 오더라도 실행기가 위치·방향 GT를 사용하지 않는다. WSL 재시작 후 IP가 바뀌면 확인해서 맞춘다.

## 확인과 실행

먼저 제어 없이 GPS와 VE 추론·Expert 계획을 20초 확인한다. 입력 준비 후 20초이며, 입력이 없으면 준비 제한 시간 후 실패한다.

```bash
cd ~/AIMDrive
bash MoraiDrive/expert/tools/run_ve_gps_expert.sh --seconds 20
```

수신과 좌표를 확인한 뒤 주행한다. 기본 상한은 기존 시험과 같은 60km/h다.

```bash
bash MoraiDrive/expert/tools/run_ve_gps_expert.sh --drive
```

공식 경로의 처음 위치로 배치하고 속도 상한 60km/h로 시작하려면:

```bash
bash MoraiDrive/expert/tools/run_ve_gps_expert.sh --drive --from-start --sim-profile normal --speed-cap-kmh 60
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

GPS 수신 시점은 NMEA의 수신 시간으로 신선도를 확인한다. 차량 속도는 LiDAR 원본 시각 기준 과거 50ms 안의 샘플을 선택한다. IMU에 원본 시각이 있으면 같은 기준을 적용하고, 없는 107-byte IMU는 수신 시간만 확인하며 로그에 구분한다. 물리적 동시 취득을 보증하지 않는다. GPS 재밍 시 GT 위치로 대체하지 않으므로 터널 주행에는 팀원의 별도 위치 추정 연결이 필요하다.

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
