# 경로 구간 notice와 RViz 확인

사용자가 지정한 시작·끝 XY를 기존 전역경로에서 가장 가까운 **실제 vertex**에
맞춘다. 보간점이나 사용자가 입력한 XY를 실행 경계로 사용하지 않는다.
기준 경로는 `MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt`다.
인덱스는 0부터 시작하고 시작·끝 vertex를 모두 포함한다. 끝 다음의 서로 다른 위치부터 해제한다.
MGeo 링크 접점은 같은 좌표가 연속 중복될 수 있어 누적 경로거리로 같은 위치를 동일하게 처리한다.
예를 들어 신호등 1의 끝점 424번과 425번은 같은 좌표이며 실제 이탈은 426번이다.

## 설정과 연결

- `configs/notice_zones/kcity_points.json`: 사용자 입력 좌표, 종류, 문장.
- `configs/notice_zones/kcity.json`: 생성된 실제 경로점, 인덱스, 원래 좌표,
  snap 거리, 경로 SHA256. 원본 경로 변경 시 재생성이 필요하다.
- `utils/notice_policy.py`: 위치→현재 구간, 중첩 notice 결합, 진입·이탈 판정.
- `models/vldrive.py`: `instruction`과 선택 입력 `notice_text`를 줄바꿈으로 결합.
- `models/language_model.py`: 기존 tokenizer/embedding으로 변환해 시각 토큰 앞에 붙여 TinyLlama에 전달.
- `inference/notice_conditioner.py`: 현재 위치를 받아 동일한 모델 입력을 구성하는 실시간 어댑터.
- `tools/rviz_notice_zones.py`: 같은 판정 모듈을 사용해 RViz 마커와 현재 문장을 발행.

신호등 안내는 구간 주의 정보다. 현재 빨간불/초록불을 주장하지 않는다.
정적 구간에서는 신호등 안내가 먼저 활성화되고 정적장애물 시작점부터 두 문장이 함께 나온다.
동적 구간은 신호등·동적장애물 안내가 동시에 활성화된다.
회전교차로는 진입점부터 출구까지 유지한다. 구간 밖에서는 notice가 빈 문자열이다.
겹치는 동일 문장은 한 번만 넣는다.

## 좌표 확정 및 구간 생성

저장소 루트에서 실행한다. 경로에서 5m 넘게 떨어진 입력은 좌표 확인 오류로 중단한다.
신호등·정적장애물 공통 끝점은 사용자가 확인한 `(-59.2, -92.58)`이다.
가장 가까운 경로점은 771번 `(-59.8253, -92.5310)`이며 원본 경로의 정밀도로 저장한다.

```bash
python VLDrive/tools/build_notice_instructions.py \
  --points VLDrive/configs/notice_zones/kcity_points.json
```

## RViz에서 주행으로 확인

ROS Noetic 환경과 기존 expert를 실행한다. expert가 발행하는 `/odom`을 읽으며,
표시 노드는 차량 제어 명령을 발행하지 않는다.

```bash
source /opt/ros/noetic/setup.bash
bash MoraiDrive/expert/src/expert/rviz_lattice.sh
```

이 스크립트는 notice 표시 노드와 수정된 RViz 설정을 함께 실행한다.
ROS 패키지를 사용할 수 있는 Python을 쓰려면 `NOTICE_PYTHON=/usr/bin/python3`를 지정한다.
RViz를 이미 열어 두었다면 표시 노드를 따로 실행하고 `MarkerArray` 표시를 추가한다.

```bash
python3 VLDrive/tools/rviz_notice_zones.py
# RViz: Add -> MarkerArray -> Topic: /vldrive/notice_markers
rostopic echo /vldrive/notice_text
rostopic echo /vldrive/notice_state
```

전체 경로를 한 화면에서 보려면 `rviz -d VLDrive/configs/notice_zones.rviz`를 사용한다.
Views의 `Follow Vehicle and Read Notice`로 전환하면 차량을 따라가며 현재 문장을 읽을 수 있다.

색 선은 적용 경로이고 양끝 구는 실제 경계점이다. 라벨에 구간 ID와 경로 인덱스가 나온다.
진입 시 해당 선이 밝아지고 차량 위에 `NOTICE ACTIVE`와 **LLM에 넣는 것과 같은 notice 문장**이 나온다.
이탈하면 다른 활성 notice로 바뀌거나 `NOTICE: none`이 나온다.
`notice_state`에는 `entered`, `exited`, `zone_ids`, `route_index`와 현재 문장이 들어 있다.
위치가 경로에서 12m보다 멀거나 odometry가 2초 넘게 끊기면 활성 notice를 해제한다.
정지·시뮬레이션 일시정지 중에도 stale 검사는 벽시계로 수행한다.
RViz 좌표 프레임은 기본 `map`이다. 다른 프레임의 odometry는 변환 없이 사용하지 않는다.

## 학습 데이터와 LLM

기존 expert cache 경로에서는 원본 측정 파일을 읽기만 하여 notice JSONL을 만든다.
각 주행의 현재 `gps_x/gps_y`로 구간을 구한다. 미래 궤적·객체 정답은 읽지 않는다.
파일은 네비게이션과 같은 `route_id/start_frame/stop_frame/text` 구조이며,
빈 notice 구간도 저장하여 누락 라벨과 일반 구간을 구분한다.

```bash
python VLDrive/tools/build_notice_instructions.py \
  --dataset-root MoraiDrive/dataset_local/competition_train \
  --output VLDrive/data/processed/instructions/full_loop_notices.jsonl
python VLDrive/train.py vldrive --config VLDrive/configs/vldrive_train.json \
  --notices VLDrive/data/processed/instructions/full_loop_notices.jsonl \
  --output VLDrive/outputs/logs/vldrive-notice
```

또는 학습 JSON의 `notices`에 VLDrive 기준 상대 경로를 추가한다.
Sensor Record 경로에서는 기존 `build_sensor_record_clips.py` 명령에
`--notice-zones VLDrive/configs/notice_zones/kcity.json`을 추가하고 새 `--output`을 사용한다.
기존 네비 클립 안에 프레임별 notice 범위를 저장하며, 학습 loader가 자동으로 읽는다.
Sensor Record의 경로점은 notice 기준 경로와 일치해야 한다. 다른 경로의 인덱스를 그대로 재사용하지 않는다.

두 loader 모두 notice 전환을 가로지르는 시각 이력을 보존한다.
현재 네비·notice 조합이 유효한 연속 suffix만 `frame_loss_mask`로 학습한다.
진입 이후 notice를 진입 이전 프레임의 주행 정답에 적용하지 않는다.
tokenizer는 런타임에 실행되며, 텍스트와 시각 토큰의 기본 LLM 구조는 동일하다.

```python
from inference.notice_conditioner import NoticeConditioner

conditioner = NoticeConditioner()  # 차량 스트림당 하나
inputs, state = conditioner.model_inputs(
    visual_feature, "Proceed along this route.", current_x, current_y)
# visual_feature: [1, T, N, D], 기존 cache/raw encoder 결과
output = model.predict(inputs)
# inputs = {"visual_feature": ..., "instruction": [navigation], "notice_text": [notice]}
# tokenizer 입력: navigation + "\n" + notice (notice가 없으면 기존 navigation 그대로)
```

표시 확인은 구간과 문장 전달을 검증한다. notice 조건으로 학습한 주행 성능 검증은 별도다.

## 재현 가능한 검증

```bash
python VLDrive/tools/verify_notice_zones.py \
  --tokenizer /path/to/local/TinyLlama-tokenizer \
  --output VLDrive/docs/reports/route_notice_verification --plot
```

전체 경로를 두 바퀴 재생해 정확한 진입·이탈, RViz 문장과 모델 입력 일치,
토큰 ID·길이, snap 좌표를 `verification.json`에 기록한다. `--plot`은 경로 검토용 PNG를 만든다.
`--tokenizer`를 생략하면 좌표와 이벤트만 확인한다.

```bash
# ROS Noetic: 새 private master에서만 가짜 위치를 발행한다.
python3 VLDrive/tools/verify_notice_ros.py \
  --output VLDrive/docs/reports/route_notice_verification/ros.json
python -m pytest VLDrive/tests/test_route_notices.py \
  VLDrive/tests/test_notice_llm_input.py VLDrive/tests/test_morai_expert_dataset.py \
  VLDrive/tests/test_morai_sensor_record_clips.py -q
```

LLM 통합 테스트는 로컬에서 만든 작은 Llama와 실제 DDIA/언어모델/waypoint 코드를 실행한다.
배포용 1.1B 체크포인트의 주행 성능을 측정하는 테스트는 아니다.

검증 결과는 `docs/reports/route_notice_verification/`에 저장했다.
전체 경로 4,430점 × 2바퀴의 8,860개 위치, 실제 경계 62개 ROS 상태와 stale 해제,
관련 pytest 41개가 통과했다. 실제 TinyLlama tokenizer의 검토용 결합 문장은 최대 39토큰이다.
