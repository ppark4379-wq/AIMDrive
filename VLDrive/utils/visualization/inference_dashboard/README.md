# VLDrive live waypoint display

`inference/realtime_vldrive.py`가 자동 실행하는 로컬 읽기 전용 화면이다. 이 폴더는 HTTP 화면과 그 표시 상태만 담당한다. 모델/UDP 수신은 `inference/`, 경로와 자연어 생성은 `navigation/`, 실행 설정은 `configs/`에 있다.

WSL `(lmdrive)`의 VLDrive 폴더에서:

```bash
python inference/realtime_vldrive.py --config ../AIM2026/Inference/vldrive_realtime.json
```

브라우저 기본 주소는 `http://127.0.0.1:8502`이며 실행 시 자동으로 열린다. `--no-browser`는 자동 열기만 끈다. `visualization.enabled=false`는 서버를 끈다. `--probe-only --duration 8 --no-browser`는 모델을 로드하지 않고 센서 수신만 확인한다.

화면에는 같은 anchor의 front JPEG/waypoint, 모델에 실제 입력한 navigation 문장, global path와 차량 위치, token 수, 시간, 센서 수신 상태가 표시된다. 모든 좌표는 기존 `forward_right()`에서 변환하며 x=right, y=forward, m 단위의 동일 축척이다. 5개 점은 미래 0.5/1/1.5/2/2.5초다.

Navigation 설정은 `configs/navigation/navigation_observer.json`의 `route`와 `position`을 재사용한다. 별도 observer 프로세스는 실행하지 않는다. 같은 GPS/status receiver를 공유한다. CLI `--instruction "..."`를 명시하면 해당 실행만 고정 문장 모드가 된다. 기본은 `navigation.mode=route`다.

실시간 이력은 학습과 같은 0.2초 간격의 실제 11프레임이다. Offline 테스트에서 사용한 interval=1과 구분한다. 처음에는 `n/11`을 표시하며, 누락 프레임을 복제하거나 zero-padding하지 않는다. 잘못된 중간 샘플은 건너뛰고, 정상 입력 간격이 0.2±0.06초 범위를 벗어나거나 simulator 시간이 되돌아가면 이력을 새로 모은다.

GPS/status가 오래되거나 route에서 벗어나면 navigation을 사용할 수 없다고 표시한다. 수집 시 센서는 `runtime.max_age_s=0.3`초 이내여야 한다. 이 조건과 별도로, 이미 계산한 예측은 anchor 수신부터 `visualization.max_prediction_age_s=3.0`초까지만 표시하고 경과 시간을 함께 보여준다. 이는 제어용 지연 기준이 아니라 관찰용 화면의 만료 시간이다. 모델 계산 중 이력이 초기화되면 해당 예측도 표시하지 않는다. UI 연결이 끊겨도 과거 예측을 현재 결과로 계속 표시하지 않는다.

기본 관찰 모드의 소켓은 MORAI **센서 수신 UDP**와 브라우저의 **localhost TCP**다. `--control`을 추가하면 `control/`의 LMDrive PID와 Windows UDP 중계기를 연결한다. 화면에는 활성 모드와 가속·제동·조향·목표 속도·hold 이유를 표시한다. 화면 자체에는 제어 API/버튼이 없고 POST 요청을 받지 않는다. 창을 닫아도 추론과 활성화된 제어는 계속된다. **터미널의 Ctrl+C로 종료**하면 제어 모드에서 제동을 전송하고 수신기와 화면 서버도 종료한다. 설정과 연결 방법은 [control/README.md](../../../control/README.md)를 참고한다.

입력 이력 수집은 5Hz를 목표로 하고, waypoint 계산은 `runtime.prediction_hz=null`로 빈도 상한 없이 실행한다. 새로운 유효 anchor가 있을 때 계산하며, 같은 프레임을 반복해 Hz를 부풀리지 않는다. 숫자(예: `0.5`)를 넣으면 해당 Hz 상한을 다시 사용할 수 있다. 실제 속도는 MORAI와 GPU를 공유하는 부하 및 센서 상태에 따라 달라진다.

사용할 수 없는 중간 프레임은 JPEG/LiDAR 전처리 전에 걸러 CPU 중복 작업을 줄인다. `runtime.separate_encoder_stream=true`이면 encoder와 VLDrive를 별도 CUDA stream에서 실행하고, 필요한 특징의 준비 완료 event만 기다린다. CPU 모드에서는 이 옵션을 사용하지 않는다. 입력 이력을 확인하는 대기 루프에서는 GPU tensor를 매번 다시 쌓지 않고, 실제 추론할 때만 `[1,11,106,256]`을 만든다. 모델 구조/정밀도/선택 방식은 유지한다.

수집과 encoder는 분리하며 대기 큐는 `runtime.encoder_queue_size=3`개로 제한한다. 큐가 차면 가장 오래된 입력을 버리고, 실제 시각의 공백이 이력 허용 범위를 넘으면 다시 11프레임을 모은다.

최근 11개 특징/anchor 문맥, 최대 3개 전처리 대기 입력, 최대 3개 표시 JPEG를 보관하며 무한히 누적하지 않는다. 화면 조회는 기본 5Hz이며 모델 계산 횟수와 다르다. 별도 프레임마다 PNG/PDF를 만들지 않는다. `waypoints.jsonl`, `navigation.jsonl`, `planned_route.json`, `inference_report.json`은 실행별 `outputs/inference/vldrive_<시각>/`에 저장한다. Log의 instruction/카메라 id/센서 id/history timestamp는 출력 anchor에 연결된다.

현재 MORAI가 Windows localhost로 보내고 WSL에서 추론하는 중계 구성은 `configs/inference/vldrive_udp_relay.json`이다. 기존 중계가 실행 중이면 중복 실행하지 않는다. 새로 시작할 때는 Windows의 VLDrive 폴더에서:

```powershell
python inference/udp_relay.py --config configs/inference/vldrive_udp_relay.json
```

이 중계는 JPEG 내용은 그대로 두고 카메라 datagram만 1,400 bytes 이하로 나눠 IP 단편화에 따른 누락을 줄인다. LiDAR/GPS/status/IMU 내용은 바꾸지 않는다. `to`는 현재 WSL 주소이며 재시작으로 주소가 바뀌면 `wsl -d Ubuntu-20.04 -- hostname -I` 결과에 맞춰 설정한다. 별도의 기존 GPS 전용 중계가 필요한 현재 센서 설정은 유지한다.

추론 수신 포트는 realtime config의 `ports`에서 관리한다: front 10501, left 10502, right 10503, rear 10504, LiDAR 2368, status 10909, GPS 10506, IMU 10507. GPS/status 포트는 navigation observer config와도 일치해야 한다. 다른 센서 수신 프로그램이 같은 포트를 사용하면 데이터 수신이 나뉠 수 있으므로 함께 실행하지 않는다.
