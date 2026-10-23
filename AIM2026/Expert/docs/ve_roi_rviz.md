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
