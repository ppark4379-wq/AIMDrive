# AIM2026 → Expert 통합 실행

`MoraiDrive/expert/main.py` → `AIM2026/Expert/runtime.py` → `AIM2026/Inference/main.py` 자동 시작 → 센서 UDP 수신·E3 추론·GPS/IMU 지도 좌표변환 → 로컬 UDP 12050 → 기존 객체 추적·신호 정책 → Expert planning/control → 차량 제어 UDP.

계획·제어 소스는 Expert에 그대로 두고 실행·통신·어댑터는 AIM2026에 둔다.
인지 시점 객체 좌표와 최신 제어 pose는 분리하며, 지도 좌표로 변환된 객체를 다시 변환하지 않는다.

## 센서 정답지와 포트
기준은 사용자가 제공한 `sensor_set_competition.json`이다. 원본 JSON을 그대로 보관했다.
`config.yaml`, `config_local.yaml`의 센서 포트·장착 위치·회전·카메라 규격을 이 파일에 맞춘다.
IP는 모두 127.0.0.1이며 현재 WSL 명령도 Windows Python을 실행한다.

| 항목 | MORAI Host Port | MORAI Destination Port |
|---|---:|---:|
| 전방 카메라 | 7501 | 9090 |
| 좌측 카메라 | 7502 | 9090 |
| 우측 카메라 | 7503 | 9090 |
| 후방 카메라 | 7504 | 9090 |
| GPS | 7506 | 9090 |
| IMU | 7507 | 9090 |
| LiDAR | 2369 | 2368 |
| 차량 상태(기존 별도 설정) | 8001 | 8000 |
| 차량 제어(기존 별도 설정) | 9097 | 9098 |

카메라/GPS/IMU는 하나의 9090 수신 소켓을 공유한다.
`UDP/SharedPort.py`가 송신 IP/Host Port로 분리한 뒤 기존 센서별 파서 스레드로 전달한다.
카메라별 JPEG 조립 상태는 독립적이다. 모르는 송신원은 버린다.
큐는 센서별 최대 128개이며 넘침은 shared_queue_drops로 기록한다.
대기열 지연을 최신 센서 데이터로 오인하지 않도록 소켓 수신 시각을 보존한다.

차량 제어는 **MoraiCmdController / UDP / AV-ExternalCtrl**로 설정한다.
TrajectoryTrackingCmd는 현재 가속·브레이크·조향 패킷과 다른 프로토콜이다.
제어 데이터는 Python 9098 → MORAI 9097로 간다.
센서 JSON에는 차량 상태/제어 포트가 없으므로 이 값은 JSON에서 가져온 것으로 표시하지 않는다.
내부 인지·pose 브리지 12050도 JSON에 없으며 MORAI에서 설정하지 않는다.
JSON의 rosConfig/RosBridgePort는 UDP 수신 설정이 아니다.

## 실행
MORAI Sensor Settings에서 위 JSON을 로드·연결한 후 실행한다.
이미 실행 중인 Expert는 이전 설정 사본을 사용하므로 Ctrl+C 후 재시작한다.
포트 변경만으로 실행 중인 시뮬레이터 설정이 자동으로 바뀌지는 않는다.

WSL Bash:
```bash
cd /mnt/c/Users/dydwn/AIMDrive
/mnt/c/Users/dydwn/anaconda3/envs/vldrive/python.exe -u -B MoraiDrive/expert/main.py --check
/mnt/c/Users/dydwn/anaconda3/envs/vldrive/python.exe -u -B MoraiDrive/expert/main.py --visualize --drive
```

별도 WSL 터미널에서 RViz:
```bash
cd /mnt/c/Users/dydwn/AIMDrive
bash AIM2026/Visualization/run_rviz.sh
```

실행 설정은 기본적으로 config_local.yaml을 사용한다. 정답지 센서 기하·포트만 동기화했으며,
E3 가중치 선택과 normal 조향 부호(-1), 기존 판제 게인은 바꾸지 않았다.
독립 GPSJamming 센서 뷰어도 같은 포트를 사용하므로 통합 Expert와 동시에 켜지 않는다.

## 검증과 기록
실행 기록은 AIM2026/outputs/expert/<시각>/에 남고 Git에서는 제외한다.
control.jsonl에는 실제 사용한 pose와 시각/dt도 기록한다.
기존 포트 구성의 실제 주행은 사용자가 확인했다.
새 공용 포트 구성은 임시 포트의 실제 UDP 분배 테스트와 설정 대조를 통과했으며,
이 구성으로 MORAI 실주행한 결과는 아직 확인하지 않았다.

판제 비교·기존 테스트 실패·좌표변환 검증 결과는
`MoraiDrive/expert/docs/verification.md`를 참고한다.
