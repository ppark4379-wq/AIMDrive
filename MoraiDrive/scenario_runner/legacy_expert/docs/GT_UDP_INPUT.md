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
