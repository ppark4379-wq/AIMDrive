# MORAI UDP 카메라 2D MOT

ROS 없이 **UDP JPEG → 기존 YOLOv7 가중치 → BoT-SORT → track_id + 2D bbox**를 실행합니다.
한 프로세스에 카메라 한 대, 입력 20 Hz 기준입니다. 학습이나 가중치 변경은 하지 않습니다.
현재 엔트리포인트는 이 저장소에 포함된 YOLOv7 모델용입니다. YOLOv8/YOLO11 등의 가중치는 호환되지 않습니다.

## 준비

기존 README의 BoT-SORT/YOLOv7 추론 환경을 사용하세요.
PyTorch/torchvision, OpenCV, NumPy, scipy, lap, cython_bbox 및 YOLOv7 의존성이 필요합니다.
ROS나 catkin 빌드는 필요 없습니다. ReID는 꺼져 있어 FastReID 환경과 별도 ReID 가중치는 필요 없습니다.
이 저장소에는 YOLO 가중치가 포함되어 있지 않으므로 기존 .pt 경로를 지정해야 합니다.

이 저장소의 원래 torch 1.11 계열 환경을 기준으로 합니다. 최신 PyTorch에서
기존 YOLOv7 checkpoint 로딩 방식의 호환성은 별도로 확인해야 합니다.

## MORAI 설정

- Camera 종류: 일반 RGB 카메라, Ground Truth=None.
- 영상 크기: 1280×720, Horizontal FOV=90°, Compression Ratio=90.
- 한 카메라의 UDP 스트림만 수신합니다. 화면의 Active List (7)는 이 코드의 카메라 수 설정이 아닙니다.
- 주기: 20 Hz.
- Network: UDP.
- 제공된 설정: Host Sensor IP/Port=127.0.0.1:5010, Destination IP/Port=127.0.0.1:15001.
- 같은 네트워크 환경에서 실행하면 이 설정을 그대로 사용합니다. Host Sensor Port는 송신 측 설정이며, 코드는 Destination Port에 맞춰 수신합니다.
- Destination Port: 15001 (코드 기본값).
- 센서 송신 Connect 활성화.

--bind-ip 0.0.0.0은 로컬 모든 인터페이스에서 수신한다는 뜻입니다.
MORAI의 Destination IP에 0.0.0.0을 넣는 것은 아닙니다.
Windows MORAI → WSL2의 기본 NAT 구성에서는 Windows의 127.0.0.1이 WSL 수신 주소를 뜻하지 않습니다.
WSL에서 실행하려면 MORAI Destination IP를 WSL의 도달 가능한 IP로 설정하고 --bind-ip 0.0.0.0을 사용하세요.
WSL IP는 WSL 터미널의 hostname -I로 확인할 수 있습니다. 포트는 양쪽 모두 15001입니다.
미러 네트워킹 사용 여부에 따라 localhost 연결 조건이 달라지므로 실제 수신을 확인하세요.
[Microsoft WSL 네트워크 안내](https://learn.microsoft.com/en-us/windows/wsl/networking)
동일 포트를 다른 카메라 수신 프로그램이 사용하면 안 됩니다.
미러 네트워킹에서는 Windows MORAI의 Host Sensor Port와 WSL 수신 포트가 충돌할 수 있습니다.
Host Sensor Port=5010, Destination Port=15001으로 서로 다르게 설정하세요.

지원 패킷은 24.R1 이후 문서의 timestamp 포함 형식입니다:
MOR(3) + seconds(uint32) + nanoseconds(uint32) + index(uint32)
+ JPEG block size(uint32) + JPEG block + AI/EI(2), little endian.
인덱스는 기본 0부터 시작하며 필요하면 --index-base 1로 지정합니다.
23.x의 다른 timestamp 표현이나 더 오래된 헤더 형식은 지원 대상으로 가정하지 않습니다.
시뮬레이터 시간을 초기화하면 이 프로그램도 다시 실행하세요.

공식 근거:
- [센서 통신 프로토콜](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35)
- [MORAI 공식 카메라 수신 예제](https://github.com/MORAI-Autonomous/MORAI-SensorExample/blob/main/src/sensor_example/CAMprocess.py)

## 실행

저장소 루트에서 기존 추론 Python 환경을 활성화한 다음:

```bash
python tools/morai_udp_mot.py \
  --weights /absolute/path/to/yolov7.pt \
  --bind-ip 0.0.0.0 --port 15001 \
  --device 0 --img-size 1280 \
  --view --show-detections \
  --output runs/morai/tracks.jsonl
```

GUI 없는 환경에서는 --view --show-detections를 빼세요.
CPU 실행은 --device cpu, CUDA FP16은 --fp16을 추가합니다.
화면에서 q 또는 ESC, 터미널에서 Ctrl+C로 종료합니다.
--output을 생략하면 프레임별 JSON을 stdout에 출력합니다.
지정한 출력 파일은 실행 때 새로 쓰므로 실험별 다른 경로를 사용하세요.

추가 학습 판단용 원본 프레임도 남기려면:
```bash
python tools/morai_udp_mot.py \
  --weights /absolute/path/to/yolov7.pt --port 15001 --device 0 \
  --output runs/morai_trial1/tracks.jsonl \
  --save-images runs/morai_trial1/images
```
JSON의 frame_id와 000001.jpg 파일 번호가 대응합니다.
저장 대상은 실제 처리한 프레임입니다. 저장 비용도 전체 처리율에 영향을 줍니다.

## 출력

프레임마다 JSON 한 줄을 출력합니다. 객체가 없으면 tracks는 빈 배열입니다.
예시(일부 필드):
```json
{
  "frame_id": 1,
  "timestamp": {"sec": 10, "nanosec": 150010000},
  "image_size": [1280, 720],
  "bbox_format": "xyxy_pixels",
  "tracks": [
    {"track_id": 7, "bbox_xyxy": [100.0, 200.0, 350.0, 480.0],
     "class_id": 2, "score": 0.91, "confirmed": true}
  ]
}
```

bbox는 원본 영상 픽셀의 [왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y]입니다.
class_id는 사용한 가중치의 클래스 정의를 따릅니다.
ID는 해당 실행/카메라에 한정됩니다. 원래 mc_bot_sort 출력에 맞춰 미확정 track도
유지하되 confirmed 필드로 구별합니다.

전체 JSON에는 NMS 후 YOLO detections, 처리 시간 processing_ms,
카메라 timestamp 간격 source_dt, 수신 진단 receiver도 포함합니다.
화면의 녹색은 track, 주황색은 YOLO 검출(--show-detections)입니다.

## 20 Hz와 성능 확인

- 추적기는 한 번 생성하고 frame_rate=20을 전달합니다.
- 원래 track_buffer=30 설정은 이 코드에서 20 update, 정상 20 Hz 입력 시 약 1초입니다.
- 추적기의 매칭, GMC, Kalman 예측 로직은 원래 동작을 유지합니다.
- ReID는 비활성화한 BoT-SORT baseline입니다.
- YOLO 입력 크기 기본값은 카메라 폭에 맞춘 1280입니다. 원래 데모 기본값은 1920입니다.
  1280×720 영상은 모델 stride에 맞게 letterbox 패딩되며 bbox는 원본 1280×720 좌표로 복원됩니다.
  속도가 부족하면 --img-size 640으로 낮출 수 있지만 검출 성능도 달라집니다.
- FOV와 JPEG 압축률은 MORAI에서 설정합니다. 2D MOT는 FOV를 입력받지 않으며 수신 JPEG를 그대로 디코딩합니다.
- 원래 데모의 confidence/NMS/추적 임계값을 기본값으로 사용합니다.
  --fuse-score도 원래 데모의 mot20 매핑을 그대로 따릅니다.
- UDP 수신은 추론과 분리한 스레드에서 수행합니다. 순서가 바뀐 조각을 재조립하며
  누락/잘못된 패킷은 완성 영상으로 사용하지 않습니다.
- 추론이 느리면 대기열을 쌓지 않고 최신 완성 프레임을 사용합니다.
  overwritten_frames는 이 때문에 처리하지 않은 완성 프레임 수,
  incomplete_frames는 일부 조각을 받았지만 완성하지 못한 영상 수입니다.
  **아예 수신하지 못한 프레임은 이 카운터에 잡히지 않습니다.**
- source_dt는 정상일 때 약 0.05초입니다. processing_ms > 50 또는 프레임 누락이
  있으면 20 Hz 전체 프레임 평가가 아닙니다. 해상도/FP16/하드웨어와 저장 비용을 확인하세요.
  원래 프레임 단위 Kalman 모델은 누락 시간에 맞춰 수정하지 않았습니다.
- processing_ms는 JPEG 디코딩부터 추적 결과 생성까지이며 파일 쓰기/화면 표시,
  네트워크 전송 지연은 포함하지 않습니다.

YOLO 박스부터 놓치는지, YOLO 박스는 있는데 ID가 바뀌는지 구분해서 보세요.
confidence 임계값 영향도 확인한 뒤 추가 학습 필요성을 판단할 수 있습니다.
MOTA/IDF1/HOTA 같은 정량 평가는 정답 bbox와 정답 ID가 별도로 필요합니다.

## 수신부 검증

```bash
python -m unittest discover -s tests -p 'test_morai_udp.py' -v
```

합성 UDP 패킷의 순서 변경, 누락, 중복, 헤더 오류, 크기 제한과
실제 JPEG의 로컬 UDP 수신/복원을 확인합니다. 실제 MORAI/가중치 추론을 대체하는 검증은 아닙니다.
