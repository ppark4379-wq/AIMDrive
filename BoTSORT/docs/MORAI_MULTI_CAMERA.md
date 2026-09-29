# MORAI 카메라 3대 검출·추적 (WSL)

전방·좌측·우측 RGB JPEG를 받아 YOLO 모델 하나와 카메라별 BoT-SORT 세 개로 처리한다. 기본 입력은 **UDP**이며 후방 카메라는 수신하지 않는다. 시뮬레이터의 객체 목록·GT 박스·차량 위치는 검출·추적 입력에 사용하지 않는다. ROS 입력은 이미 연결된 환경에서 개발 결과를 비교할 때만 명시적으로 선택한다.

## 환경 준비

저장소의 `BoTSORT` 디렉터리에서 실행한다. Python 3.10 환경과 공식 YOLOv7 가중치가 필요하다. 가중치 출처·SHA256은 `pretrained/yolov7.source.json`에 있다. `.venv`, 가중치, `runs/` 결과는 Git에서 제외된다.

```bash
conda create --prefix "$PWD/.venv" python=3.10 pip -y
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-morai-wsl.txt
.venv/bin/python -m pip check
```

ROS 개발 입력을 사용할 때만 `source /opt/ros/noetic/setup.bash`가 추가로 필요하다. UDP 실행은 ROS master나 rosbridge를 사용하지 않는다.

## MORAI UDP 연결

`configs/morai_three_cameras.yaml`의 포트와 실제 센서 Destination Port를 맞춘다.

| 카메라 | MORAI Host Port | WSL Destination Port | 원본 영상 | 설정 주기 |
|---|---:|---:|---|---:|
| front / Camera-1 | 9290 | 9291 | 1280 × 720 | 20 Hz |
| left / Camera-2 | 9292 | 9293 | 640 × 480 | 20 Hz |
| right / Camera-3 | 9294 | 9295 | 640 × 480 | 20 Hz |

WSL NAT 구성에서는 다음 주소를 사용한다. 재시작 후 주소가 달라질 수 있으므로 실행할 때 확인한다.

```bash
ip -4 addr show eth0  # MORAI Destination IP = WSL eth0 IPv4
ip route             # MORAI Host Sensor IP = Windows의 WSL 가상 어댑터 주소
```

2026-09-27 환경에서는 Host `172.19.0.1`, Destination `172.19.0.55`로 실제 UDP 영상 수신을 확인했다. Host를 `127.0.0.1`로 바인딩하면 이 NAT 경로로 송신하지 못했다. Windows의 실제 어댑터 주소도 함께 확인한다. WSL 수신기의 `bind_ip: 0.0.0.0`는 로컬 수신 주소이지 MORAI에 입력할 Destination IP가 아니다.

카메라 위치·각도·FOV는 대회 고정값을 유지한다. Camera / Ground Truth None / Viz Bounding Box 2D·3D 해제 상태에서 UDP를 선택하고 Connect한다. 모델 입력 `--img-size`는 센서 FOV·해상도를 바꾸는 옵션이 아니라, 수신한 영상의 검출 전처리 크기다.

현재 설정을 MORAI Sensor Save로 먼저 저장한 뒤 새 프리셋을 만들 수 있다. 다음 도구는 파일만 만들고 시뮬레이터에 적용하지 않는다. 후방·LiDAR·GPS·IMU와 차량 제어/상태 통신은 변경하지 않으므로 이 명령이 전체 팀 네트워크 전환을 완료하는 것은 아니다.

```bash
.venv/bin/python tools/prepare_camera_udp_preset.py \
  --sensor-file runs/before_udp.json --host-ip 172.19.0.1 \
  --destination-ip 172.19.0.55 --output runs/camera_udp_preset.json
```

생성 파일을 Windows MORAI의 `SaveFile/Sensor/버전` 폴더에 별도 이름으로 복사한 후 Sensor File Manager에서 로드한다. 원본 프리셋을 보관한다. 테스트한 설치 환경에서 gRPC `LoadSensorFile`에 WSL UNC 전체 경로를 전달했을 때 성공 응답과 실제 로드 상태가 일치하지 않았으므로, 위 GUI 경로와 실제 JPEG 수신 확인을 사용한다.

[MORAI 공식 센서 패킷 설명](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35)의 24.R1 이후 `MOR + seconds + nanoseconds + index + size + JPEG + AI/EI` 형식을 사용한다. 다른 형식의 구버전 패킷은 지원하지 않는다. index 기본값은 0이며 실제 송신 형식이 1부터 시작할 때만 설정에서 변경한다.

### Windows → WSL 큰 UDP 패킷 손실 진단과 선택적 중계

2026-09-27의 해당 WSL NAT 환경에서 별도 송신 시험은 1472바이트 UDP 400개 중 400개, 1473바이트는 400개 중 130개를 수신했다. 양쪽 인터페이스 MTU는 1500이었다. IP 단편화와 관련된 경로 문제를 의심할 근거이며, 특정 방화벽이나 드라이버를 원인으로 확정한 결과는 아니다. 영상의 JPEG 조각 누락을 검출기 추가 학습으로 해결하려 해서는 안 된다.

`tools/morai_udp_relay.py`는 **Windows의 Python**에서 MORAI UDP를 받아 JPEG를 재조립한 뒤, JPEG 바이트와 원본 timestamp를 바꾸지 않고 최대 1200바이트의 UDP 패킷으로 WSL에 전달한다. 표준 라이브러리만 사용하며 ROS는 필요 없다. 전송 방식은 양 구간 모두 UDP이다. 별도 반복 송신 시험에서 카메라별 100장씩 총 300장의 JPEG SHA256과 timestamp가 모두 보존됐으며 수신 누락은 없었다. 이 시험은 실제 MORAI 장시간 운용 검증과 구분한다.

Windows PowerShell에서 `BoTSORT` 디렉터리로 이동해 실행한다. 예시 IP는 실제 WSL IP로 바꾼다.

```powershell
py -3.11 -B tools/morai_udp_relay.py --destination-ip 172.19.0.55 `
  --duration 1800 --stop-file runs/relay_001.stop --output runs/relay_001
```

중계 모드의 MORAI 설정은 Host IP `127.0.0.1`, Destination IP `127.0.0.1`, Host Port `9290/9292/9294`, Destination Port `19291/19293/19295`이다. 중계기가 Windows loopback에서 받은 뒤 WSL의 `9291/9293/9295`로 전달한다. WSL 검출기 명령은 그대로 사용한다. 중계기가 먼저 `READY`를 출력한 후 센서를 연결한다. 직접 연결 프리셋을 별도로 보관하고, 중계 종료 시 센서가 원래 WSL 목적지로 자동 복구되지는 않는다는 점에 유의한다.

`--duration`이 끝나거나 중계 프로세스가 중단되면 영상 전달도 멈춘다. WSL 입력 상태를 함께 확인한다. 중계 전 수신·재조립 시간은 WSL의 `receive_to_result_ms`에 포함되지 않으므로 직접 연결과 이 지연 수치만 비교해 전체 지연이 같다고 판단하지 않는다. 이 선택 기능은 시스템 MTU나 방화벽 설정을 변경하지 않는다.

중계 실행 시간을 수신·추론 준비와 시험 시간보다 길게 지정한다. `status.jsonl`에는 약 1초마다 카메라별 수신 패킷, 중계한 영상, 재조립 실패 및 마지막 원본 timestamp가 기록된다. 종료 시 `summary.json`에 전체 카운터와 종료 이유가 저장된다. `--stop-file`을 지정했다면 해당 파일을 만들면 다음 상태 점검에서 정상 종료한다. 새 실행에서는 남아 있는 종료 파일을 사용하지 않도록 별도 이름을 지정한다.

WSL의 `incomplete_frames`는 일부 조각만 들어와 조립에 실패한 영상 수이고, `overwritten_frames`는 처리 대기 중 최신 영상으로 교체한 수다. 패킷 전체가 사라진 영상은 재조립 실패 카운터만으로 알 수 없으므로 실제 수신율과 원본 시간 간격도 함께 확인한다. 20Hz 수신이 곧 카메라별 20FPS 검출·추적을 뜻하지는 않는다.

## 기록·실행·재생

출력 디렉터리는 매번 새 경로를 지정한다. 기존 결과를 덮어쓰지 않는다.

```bash
# 모델 없이 수신과 기록부터 확인
.venv/bin/python tools/record_cameras.py --duration 15 --output runs/capture_001

# 전방 한 대, 5분
.venv/bin/python tools/morai_multi_camera_mot.py \
  --weights pretrained/yolov7.pt --fp16 --cameras front \
  --duration 300 --save-every 100 --run-dir runs/front_001

# 3대 동시, 10분
.venv/bin/python tools/morai_multi_camera_mot.py \
  --weights pretrained/yolov7.pt --fp16 --duration 600 \
  --save-every 100 --run-dir runs/three_001

# 동일한 기록 영상 재생: 원본 카메라·timestamp·epoch를 유지하며 최대 속도로 처리
.venv/bin/python tools/morai_multi_camera_mot.py \
  --weights pretrained/yolov7.pt --fp16 --replay runs/capture_001 \
  --save-every 20 --run-dir runs/replay_001

# 선택 사항: 현재 ROS 연결의 개발 영상 비교
.venv/bin/python tools/morai_multi_camera_mot.py \
  --transport ros --weights pretrained/yolov7.pt --fp16 \
  --duration 30 --run-dir runs/ros_comparison_001
```

재생은 실시간 입력 시험을 대체하지 않는다. `record_cameras.py`도 카메라별 최신 프레임 슬롯을 사용하므로 디스크가 느리면 일부 프레임을 생략한다. 기록 결과의 `overwritten_frames`와 저장 개수를 확인한다. `--save-every`로 저장한 추론 결과의 영상은 간격을 둔 표본이므로 연속 추적 품질 평가에는 별도 연속 기록을 사용한다.

### YOLOv7의 960 입력 실행 최적화

`--yolov7-runtime cuda-graph`는 동일한 YOLOv7 가중치에 channels-last 메모리 배치와 CUDA Graph 실행을 적용한다. 모델 학습, 센서 해상도·장착값, 검출·추적 임계값은 이 옵션으로 바뀌지 않는다. CUDA GPU가 필요하며 YOLO26에는 적용하지 않는다. 기본값 `eager`는 기존 실행 방식이다.

```bash
# --weights에는 사용할 가중치 경로를 지정한다. 클래스 목록은 가중치를 따른다.
.venv/bin/python -B tools/morai_multi_camera_mot.py \
  --weights /path/to/model.pt --img-size 960 --fp16 \
  --yolov7-runtime cuda-graph --iou-thres 0.5 \
  --track-buffer 90 --match-thresh 0.85 \
  --duration 600 --save-every 100 --run-dir runs/three_960_001
```

설정된 카메라 영상 크기를 시작할 때 준비하고, 수신·처리율 측정은 준비가 끝난 뒤 시작한다. 현재 3대 설정에서 모델 입력은 전방 `1×3×544×960`, 좌우 `1×3×736×960`이며 좌우가 같은 실행 버퍼를 순차 사용한다. 원본 비율을 유지한 크기 조정과 패딩을 포함한 값이다. 검출기 호출은 기존처럼 작업자 하나가 순차 수행해야 한다.

준비하지 않은 크기가 들어오면 해당 프레임은 기존 방식으로 계산한다. 실행 중 그래프를 계속 추가하지 않는다. `manifest.json`과 `summary.json`의 `detector_runtime`에서 실행 방식, 준비된 크기, `eager_fallback_frames`를 확인한다. 정상적인 현재 3대 입력의 fallback 수는 0이어야 한다. 가중치나 장치·입력 설정을 바꾸면 프로그램을 재시작하고 품질·처리율을 다시 비교한다.

위 대체 실행은 검출기 API의 동작이다. 3대 실행기는 원본 영상 크기가 카메라 설정과 다르면 기존처럼 프레임 오류로 거부한다. 실행 중 센서 해상도가 바뀌어도 자동으로 설정을 변경하는 기능은 아니다.

최적화 전후 출력은 부동소수점 연산 순서에 따라 조금 달라질 수 있다. 같은 영상의 검출·ID 지표로 비교해야 하며, 다른 가중치에서도 동일 성능 향상을 보장하는 옵션은 아니다. 문제가 있으면 같은 명령에서 `--yolov7-runtime eager`로 되돌려 비교한다.

선택적으로 위 명령에 `--postprocess-device cpu`를 추가하면 YOLOv7의 신경망 계산은 GPU에서 수행하고, 예측을 CPU로 옮겨 NMS와 원본 영상 좌표 변환을 수행한다. 기본값 `model`은 모델과 같은 장치에서 후처리한다. 기존 PyTorch/torchvision만 사용하며 TensorRT 설치는 필요 없다. CPU 부하와 GPU 전송 비용에 따라 효과가 달라지므로 실제 사용할 장비에서 비교한다. GPU/CPU 연산 차이로 출력 수치가 완전히 같지는 않을 수 있다. 이 옵션도 YOLO26에는 적용하지 않는다.

`detector_runtime.postprocess_device`에 후처리 장치를 기록한다. 후처리만 원복하려면 `--postprocess-device model`을 사용하고, 최적화 전체를 비교 기준으로 돌리려면 `--yolov7-runtime eager --postprocess-device model`을 함께 사용한다. 입력 크기·가중치·임계값은 같은 값으로 유지한다.

### 선택 사항: TensorRT FP16 실행

객체 검출을 실행하는 PC의 Python 환경에 TensorRT가 필요하다. 인지 결과만 받아 제어하는 별도 PC에는 필요 없다. WSL에서 인지를 실행하면 WSL 환경에 설치한다. 현재 검증 조합은 Linux x86-64/WSL, RTX4060, PyTorch2.7.1+cu128, TensorRT10.9.0.34, ONNX1.17.0이다. 기존 PyTorch 실행 경로는 TensorRT를 가져오지 않으며 설치 없이 유지된다.

아래는 기존 환경의 패키지를 대체하지 않는 선택적 설치 방법이다. 앞의 기본 WSL 환경이 준비되어 있어야 한다. ONNX의 protobuf 의존성도 준비한다. 큰 NVIDIA 라이브러리 wheel은 최초 설치 때 내려받고, 매번 실행할 때 다운로드하지 않는다. 명시한 wheel은 Linux x86-64용이다.

```bash
.venv/bin/python -m pip install protobuf==7.36.2
.venv/bin/python -m pip install --no-deps --target .venv/runtime-accel \
  onnx==1.17.0 tensorrt-cu12==10.9.0.34 tensorrt-cu12-bindings==10.9.0.34 \
  'https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.9.0.34-py2.py3-none-manylinux_2_28_x86_64.whl#sha256=4a82f0bda2874596f202f6edc8dae99b86a3c4ec2fa142a9c847c4d3a57864a0'
PYTHONPATH="$PWD/.venv/runtime-accel" .venv/bin/python -c \
  'import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)'
```

실제로 사용할 GPU에서 가중치와 카메라 설정에 맞는 엔진을 만든다. 생성 도구는 설정된 원본 영상 크기에서 전처리 모양을 계산하며, 좌우처럼 같은 모양에는 엔진 하나를 공유한다. 새 출력 디렉터리만 허용한다. TensorRT10.x 생성 API를 사용하고10.9.0.34에서 검증했다. 실행 중 엔진 생성이나 추가 학습은 하지 않는다.

```bash
PYTHONPATH="$PWD/.venv/runtime-accel" .venv/bin/python -B tools/build_morai_tensorrt.py \
  --weights /path/to/model.pt --img-size 960 \
  --config configs/morai_three_cameras.yaml --output runs/engines_960_001

PYTHONPATH="$PWD/.venv/runtime-accel" .venv/bin/python -B tools/morai_multi_camera_mot.py \
  --weights /path/to/model.pt --img-size 960 --fp16 \
  --yolov7-runtime tensorrt --engine-manifest runs/engines_960_001/engines.json \
  --postprocess-device cpu --iou-thres 0.5 --track-buffer 90 --match-thresh 0.85 \
  --duration 600 --save-every 100 --run-dir runs/three_trt_001
```

`--timing-cache /path/to/tactics.cache`로 같은 환경의 생성 캐시를 재사용할 수 있다. 원본 캐시는 수정하지 않으며 호환되지 않으면 캐시 없이 새 출력 경로로 생성한다. 모든 엔진의 생성·로딩·준비가 성공해야 최종 `engines.json`을 만든다. 실패한 폴더의 부분 엔진을 임의로 연결하지 않는다.

가중치와 엔진 SHA256, 입력 크기, 클래스 수, TensorRT 버전, GPU 모델명, OS/아키텍처 및 엔진 입출력을 시작할 때 확인한다. 불일치나 미준비 영상 크기는 명시적으로 거부하며 다른 실행 방식으로 자동 변경하지 않는다. 같은 환경이라도 새 엔진은 같은 영상의 검출·ID와 실제 처리율을 검증한 후 사용한다. 가중치·입력 모양이 바뀌거나 제어기 PC의 GPU/환경이 달라지면 해당 환경에서 재생성한다. [NVIDIA 엔진 생성·실행 안내](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).

TensorRT 옵션은 기존 YOLOv7 전처리·NMS·원본 좌표 변환과 BoT-SORT를 재사용한다. 엔진 실행은 전용 CUDA 스트림에서 수행하며 PyTorch 입력/후처리 스트림과 순서를 맞춘다. 검출기 단일 작업자의 순차 호출을 전제로 한다. 실행 정보와 엔진 해시는 `detector_runtime`에 기록한다. `peak_torch_gpu_memory_mib`는 PyTorch 할당량이며 TensorRT 엔진/작업공간의 전체 메모리가 포함된 값이 아니다.

TensorRT 문제가 있으면 `--engine-manifest`를 빼고 `--yolov7-runtime cuda-graph --postprocess-device cpu`로 명시적으로 바꿔 비교한다. 최초 기본값은 계속 eager/model이다. 두 방식 모두 인지 가중치와 입력960을 별도로 지정해야 한다.

## 융합 파트에 전달할 결과

- `tracks.jsonl`: 프레임마다 원본 timestamp, 카메라 ID, 원본 픽셀 좌표의 검출·추적 결과.
- `status.jsonl`: 약 1초마다 카메라별 입력/출력 상태. 입력이 없을 때 빈 장애물 목록을 반복 송신하지 않는다.
- `manifest.json`: 모델 SHA256, 클래스 목록, 설정, 실행 옵션, 환경 버전.
- `summary.json`: 카메라별 처리율, 지연, 프레임 오류, 수신·재조립·덮어쓰기 카운터.
- 카메라 폴더: `--save-every` 지정 시 원본 JPEG와 추적 박스 이미지.

객체를 식별하는 키는 **`(session_id, camera_id, stream_epoch, track_id)`**다. front의 1번과 left의 1번은 서로 다른 트랙이다. 현재 출력은 카메라 간 동일 객체 ID를 보장하지 않는다. 박스 좌표는 `xyxy_pixels`, 각 카메라 원본 영상 좌표다. 이미지 왼쪽 위가 원점이며 x는 오른쪽, y는 아래쪽이다. 렌즈 왜곡 보정과 LiDAR 투영은 수행하지 않는다.

`tracks`와 표시 영상은 기본적으로 BoT-SORT가 확정한 ID만 내보낸다. 미확정 트랙은 내부에서 계속
관리하고, 다음 처리 프레임에서 다시 연결되면 같은 ID로 출력한다. 따라서 실행 중 새로 나타난 객체는
검출 박스보다 ID 표시가 보통 한 처리 프레임 늦다. 실제 시간은 처리율·프레임 누락에 따라 달라진다.
각 카메라/epoch의 첫 처리 프레임은 원본 BoT-SORT의 즉시 확정 규칙을 유지한다.
모든 검출 결과는 `detections`에 남으며, `--include-tentative-tracks`는 이전처럼 미확정 ID도
`confirmed: false`로 출력하는 진단 옵션이다. 확정 ID만 출력해도 오래 지속되는 오검출·중복이나
낮은 검출 점수 때문에 시작하지 못한 트랙까지 해결되는 것은 아니다.

`timestamp.sec/nanosec`는 수신 헤더 정수를 그대로 보존한다. `timestamp_domain`과 `transport`를 함께 확인한다. 다른 센서와 같은 기준 시계라고 가정하지 말고 융합 시 검증한다. `received_monotonic_ns`는 WSL 로컬 시계이며 센서 timestamp와 직접 빼지 않는다. `receive_to_result_ms`는 완성된 JPEG 수신부터 결과가 준비될 때까지로, 센서 노출·전송·재조립 이전의 지연과 파일 쓰기 시간은 포함하지 않는다. 재생 지연은 처리 비용 측정값이며 실제 센서 지연이 아니다.

`tracks: []`는 **해당 시각에 처리한 영상에서 내보낼 트랙이 없었다**는 의미다. 장애물이 없다는 보증이 아니다. `no_frames`, `stale_input`, `stale_output`는 별도로 처리해야 한다. 상태 임계값 1초는 현재 개발용 고장 표시 기준이며 주행용 정지 판단 기준은 팀과 별도로 합의해야 한다. JSONL 파일 전달은 개발용 인터페이스이며 실제 융합 프로세스 연결 방식은 아직 합의 대상이다.

## 중단·복구와 한계

카메라 하나가 끊기면 나머지 카메라는 계속 처리한다. 다시 연결된 영상은 timestamp가 계속 증가하면 수신한다. 시뮬레이터 시간 초기화로 timestamp가 역행하면 오래된 패킷과 구분할 수 없으므로 자동 초기화하지 않는다. CLI를 새 `--run-dir`로 재시작하면 새 session ID가 부여된다. Python API `source.reset(camera_id)`는 해당 카메라 epoch와 재조립 상태만 초기화하며, 이미 소비 중인 프레임이 없는 경계에서 호출해야 한다.

BoT-SORT의 Kalman 예측과 lost buffer는 여전히 처리 프레임 단위다. `expected_hz`는 원본 코드의 buffer 계산에만 쓰이며 timestamp 간격에 따라 Kalman dt를 바꾸지 않는다. 입력 누락·처리 지연이 있으면 품질을 별도로 평가해야 한다. 현재는 ReID를 사용하지 않는다. 긴 가림, 완전히 화면 밖으로 나갔다가 재등장, 재시작 후에는 새 ID가 생길 수 있다.

COCO 가중치는 MORAI 전용 모델이 아니다. 일반 상자·콘 등에는 클래스 자체가 없고, 자기 차량 보닛 검출과 배경 오검출도 실제 확인됐다. 모든 NPC·장애물 검출 성공이나 정량 정확도는 아직 검증되지 않았다. 작은 객체 미검출을 고치려면 검출기와 입력 크기부터 평가하고, 필요한 클래스의 정답 데이터로 추가 학습한다. ID 변경만으로 추가 학습 필요성을 판단하지 않는다.

선택적으로 카메라 설정의 `ego_mask_polygons`에 원본 픽셀 좌표의 차체 실루엣을 지정할 수 있다. 검출과 GMC 입력에만 중성 회색을 채우며 원본 JPEG와 출력 박스 좌표는 유지한다. 기본 설정에는 마스크가 없다. 차종·장착 위치·해상도가 달라지면 다시 보정해야 하며, 근접 장애물이 가려지지 않는지 검증한 후 적용한다. 일부 선택된 장면에서 오검출이 줄었다는 사실만으로 전체 품질 개선을 주장하지 않는다.

## 정답이 있는 영상의 검출 평가

`tools/evaluate_camera_detections.py`는 직접 라벨링한 프레임의 클래스·박스를 일대일 IoU 매칭한다. 박스에 ID가 있더라도 클래스가 틀리면 성공으로 세지 않는다. `--kind detections`는 검출기 결과, `--kind tracks`는 실제로 내보낸 트랙 박스를 평가한다. 출력이 없는 프레임과 정상 입력에서의 빈 검출을 구분하며, 추적 HOTA/IDF1 평가는 별도다.

정답 JSON의 `frames` 각 행에 `camera_id`, `stream_epoch`, `received_frame_id`, `timestamp: {sec, nanosec}`, `objects: [{class_name, bbox_xyxy}]`를 기록한다. `scope`에는 대상 클래스, 라벨 범위와 표본 선정 방식을 적는다. 완전히 라벨링한 평가 프레임만 넣고 모델 수정에 쓴 영상과 최종 평가 영상을 분리한다.

```bash
.venv/bin/python tools/evaluate_camera_detections.py \
  --annotations runs/evaluation/annotations.json --run-dir runs/evaluation_inference \
  --kind detections --score-threshold 0.25 --iou-threshold 0.5 \
  --output runs/evaluation_metrics.json
```

```bash
.venv/bin/python -m unittest discover -s tests -v
```

단위 테스트는 세 포트 격리, 조각 순서·중복·누락, timestamp 역행, 카메라별 초기화, ID 독립성, 빈 검출에서의 추적 갱신, 기록 재생을 확인한다. 이 테스트의 통과가 실제 주행 인지 품질 통과를 뜻하지는 않는다.
