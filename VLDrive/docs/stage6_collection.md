# MORAI Stage6E 수집·검증

이 팀 checkout의 `VLDrive/experiments/morai_stage6`에서 실행한다. ROS1 Noetic,
`morai_msgs`, rosbridge와 catkin setup이 필요하다. 학습은 [LAVIS 안내](../../../LAVIS/README_MORAI.md)의 별도 환경을
사용한다. `/Ego_topic`은 학습 teacher이며 모델 입력으로 전달하지 않는다.

## 센서와 확인된 원인

Camera-1~4는 `/image_jpeg/compressed`의 frame ID로 구분한다. LiDAR `/lidar3D`의
receive monotonic time을 anchor로 Camera/GPS/IMU/teacher를 인과적으로 선택한다.
각 sensor header/source time은 따로 보존하며 cross-sensor clock으로 빼지 않는다.
teacher trajectory의 보간은 Ego source clock 내부에서 수행한다.

실측한 Ego 장시간 지연의 원인은 rosbridge가 MORAI의 binary JSON frame을 Python3
`str(bytes)`로 처리해 JSON parse/recovery 비용이 증가하고 FIFO에 과거 메시지가 누적된 것이다.
Ego seq/timestamp가 계속 증가해도 현재 차량보다 오래된 상태를 처리할 수 있었다.
최대 594초 queue 대기를 관측했고, UTF-8 text 경로로 연결한 뒤 production 주행에서는
8개 stream queue 최대 5.579ms, GPS/Ego moving 56구간 불일치 0을 확인했다.
모든 과거 MORAI session의 이상을 같은 원인으로 확정한 것은 아니다.

`src/aimdrive/ros_training/rosbridge_ingress.py`는 JSON 모드 binary frame만 수정하고 BSON,
FIFO 정책과 message 내용은 유지한다. `production_bridge.py`가 bounded health를 제공하며
recorder는 bridge identity·queue·publisher·freshness와 독립 GPS/Ego motion을 검사한다.
설치된 `/opt/ros`를 수정하지 않는다. 수집 품질은 `scripts/validate_stage6e_run.py`,
teacher motion은 `scripts/analyze_stage6e_integrity.py`로 검사한다. 과거 병목 재현 도구는
로컬 실험 기록에 보존하고 실행용 Git 패키지에서는 제외한다.

## 연결과 수집

1. ROS host 주소와 catkin setup을 확인한다. 아래 주소는 기존 실험 설정이므로 환경에 맞춰 바꾼다.
2. bridge를 한 개만 실행하고 MORAI 차량 정보·센서를 그 host의 9090에 연결한다.
3. 별도 터미널에서 pipeline을 실행한다. 정상 moving preflight가 끝난 뒤 지정 시간만큼 수집한다.

```bash
# 터미널 A: morai_stage6 폴더에서 실행한다.
export ROS_IP=172.18.40.28
/usr/bin/python3 scripts/start_stage6e_bridge.py \
  --master http://172.18.40.28:11311 --ros-setup "$HOME/catkin_ws/devel/setup.bash"

# 터미널 B: 실제 LMDrive custom timm이 설치된 vision 환경의 python 경로를 지정한다.
/usr/bin/python3 scripts/stage6e_pipeline.py \
  --master http://172.18.40.28:11311 --ros-ip 172.18.40.28 \
  --bridge-url ws://172.18.40.28:9090 --ros-setup "$HOME/catkin_ws/devel/setup.bash" \
  --vision-python /path/to/lmdrive-env/bin/python \
  --checkpoint assets/pretrained/vision_encoder/vision-encoder-r50.pth.tar \
  --runs-root data/stage6e_runs --duration 120
```

collection wrapper는 vision subprocess의 PYTHONPATH를 `src`로 설정하므로 vision 환경 자체에
팀 custom timm이 설치돼 있어야 한다. 새 별도 vision 환경에서는 이 폴더 기준으로
`python -m pip install --no-deps -e ../../vision_encoder`를 사용할 수 있다.
LAVIS MORAI trainer는 backend bootstrap을 사용하므로 이 editable 설치 없이도 상위 코드를 읽는다.
기존 frozen cache loader의 Memfuser 초기화에는 torchvision/ImageNet 초기 weight cache 또는
최초 다운로드 연결이 필요할 수 있다. LAVIS MORAI trainer의 중복 초기 다운로드 생략과는 별도 경로다.

센서 프리셋 검사에는 `--sensor-preset /path/to/SensorInfo.json`을 추가한다.
활성 UI에서 확인한 경우에만 `--confirmed-fovs 90,130,130,90 --confirmation-note '활성 UI 확인'`을
추가한다. ROS TF/해상도와 FOV 확인은 exposure/scan-time calibration 완성을 의미하지 않는다.
MORAI LiDAR +Y left는 adapter에서 LMDrive +Y right로 변환하며 cache policy를 기록한다.

## 후처리와 학습 승격

wrapper는 raw → future teacher XY/speed → local route → 실제 vision feature cache →
training manifest와 registry 검증을 수행한다. raw 학습도 같은 Stage6E 검사·manifest를 재사용한다.
종료 후 teacher invalid 또는 단절된 run은 보존하면서 격리하고 자동 학습 승격하지 않는다.
reset/teleport 전후를 하나의 정상 주행으로 연결하지 않는다.

```bash
/usr/bin/python3 scripts/validate_stage6e_run.py \
  --run data/stage6e_runs/run_NNNN --require-features

# 최소 두 독립 정상 주행을 수집한 뒤 validation run을 명시한다.
/usr/bin/python3 scripts/build_stage6e_run_splits.py \
  --runs-root data/stage6e_runs --val-runs run_NNNN \
  --output data/stage6e_runs/splits_train_val.json
```

학습 설정에는 `dataset: data/stage6e_runs`, `split_file: splits_train_val.json`을 지정한다.
이전 axis policy의 cache를 혼합하지 않는다. 동일 주행의 revision을 독립 validation으로 세지 않는다.
raw만 끝난 run은 같은 wrapper에 `--run data/stage6e_runs/run_NNNN`과 vision 인자를 주어
후처리하며, 기존 완성 파일을 덮어쓰지 않는다.

GPS도 MORAI 출력이므로 외부 실측 GT 인증은 아니다. WebSocket 입구 이전 capture/send 지연의
절대값, 모든 센서가 함께 지연되는 경우, runtime localization과 closed-loop 제어는 별도 검증 대상이다.

수집기와 LAVIS는 같은 VLDrive checkout으로 유지한다. 공유 데이터 검증의 기준 구현은
`LAVIS/lavis/datasets/datasets/morai`이며 수집 환경에서는 NumPy 기반 검증만 import한다.
