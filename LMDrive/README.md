# LMDrive (학습 코드 재구성판)

[LMDrive: Closed-Loop End-to-End Driving with Large Language Models](https://arxiv.org/abs/2312.07488) (CVPR 2024)
의 학습 코드를 표준 딥러닝 프로젝트 구조로 정리한 것입니다. 원본은 LMDrive 저장소 안에 LAVIS 전체와
timm 포크, CARLA leaderboard, scenario_runner 가 함께 들어 있었고 학습에 실제로 쓰이는 파일은 그중
일부였습니다. 여기서는 **학습과 오프라인 평가에 필요한 코드만** 남기고 이름을 역할이 드러나게 바꿨습니다.

모델 파라미터 이름은 그대로라서 [공개 체크포인트](https://huggingface.co/collections/OpenDILabCommunity/lmdrive-658b3d9f0ceb0dcd6e7b0a3e)
가 그대로 로드됩니다. 학습 하이퍼파라미터와 손실, 데이터 전처리도 원본과 같은 값을 씁니다.

## 구조

```
LMDrive/
├── train.py                  두 단계 학습 진입점 (YAML 설정으로 단계 선택)
├── evaluate.py               체크포인트 오프라인 평가 (시뮬레이터 불필요)
├── configs/
│   ├── vision_encoder_model.yaml    비전 인코더 구조
│   ├── vision_encoder_train.yaml    1단계 학습 설정
│   ├── lmdrive_model.yaml           주행 모델 구조 (비전 인코더 + Q-Former + LLM)
│   └── lmdrive_train.yaml           2단계 학습 설정
├── models/
│   ├── resnet.py             카메라 CNN 백본 (ResNet-D)
│   ├── transformer.py        DETR 방식 인코더/디코더와 위치 인코딩
│   ├── pointpillars.py       LiDAR PointPillars 인코더
│   ├── vision_encoder.py     Memfuser 비전 인코더 (멀티뷰 카메라 + LiDAR -> 토큰/인지 출력)
│   ├── qformer.py            BLIP-2 Q-Former (LAVIS 원본 그대로)
│   ├── llama.py              LLaMA 구현 (LAVIS 원본, import 누락 1줄만 수정)
│   └── lmdrive.py            언어 조건 주행 모델 (2단계)
├── datasets/
│   ├── carla_perception_dataset.py   1단계용 프레임 + 인지 라벨
│   ├── carla_instruction_dataset.py  2단계용 클립 + 지시문
│   ├── perception_labels.py          BEV 점유도와 검출 라벨 생성
│   ├── transforms.py, lidar.py, sensor_io.py   전처리와 파일 입출력
│   ├── instruction_templates.json    instruction_id 별 문장 템플릿
│   └── build_dataset_index.py        dataset_index.txt 생성 도구
├── utils/
│   ├── config.py, distributed.py, logger.py, metrics.py
│   ├── losses.py             1단계 손실 (검출 + 웨이포인트 + 신호등 + 정지선)
│   ├── checkpoint.py         저장, 불러오기, 재개
│   ├── schedulers.py         단계별 학습률 스케줄
│   └── visualization.py      TensorBoard 이미지 (카메라, 검출, 웨이포인트)
├── tools/
│   ├── route_paths.py        dataset_index.txt 경로 해석 (도구와 학습 코드 공통)
│   ├── preprocess/           수집 데이터 정리
│   │   ├── build_dataset_index.py           dataset_index.txt 생성
│   │   ├── merge_camera_and_measurements.py 4개 카메라를 rgb_full 로, 측정값을 measurements_full 로 병합
│   │   ├── merge_measurements.py            프레임별 측정값을 measurements_all.json 로 병합
│   │   ├── remove_split_camera_folders.py   병합 후 남은 rgb_front/left/right/rear 삭제
│   │   ├── find_blocked_frames.py           정차로 막힌 구간 탐색 (blocked_stat.txt)
│   │   ├── remove_blocked_frames.py         해당 구간 삭제
│   │   └── renumber_frames.py               삭제 후 프레임 번호 재정렬
│   └── labeling/             2단계 지시문 라벨 생성 (CARLA 패키지 필요)
│       ├── parse_navigation_instructions.py  navigation_instruction_list.txt 생성
│       ├── parse_notice_instructions.py      notice_instruction_list.json 생성
│       ├── parse_misleading_instructions.py  misleading_data.txt 생성
│       ├── turn_rules.py, follow_rules.py, notice_rules.py, other_rules.py  지시문 규칙
│       └── carla_maps/*.xodr                 규칙이 참조하는 CARLA 도로망
├── data/                     raw/ processed/ external/  (git 추적 제외)
└── outputs/                  checkpoints/ logs/         (git 추적 제외)
```

## 설치

```bash
pip install -r requirements.txt
```

Python 3.10, torch 2.8.0+cu128, torchvision 0.23.0, transformers 4.31.0 에서 확인했습니다.
`torch-scatter`, `accelerate`, `peft`, `tensorboard` 는 없어도 동작합니다(각각 순수 PyTorch 대체 구현,
저메모리 로딩 생략, LoRA 비활성, CSV/JSON 로그만 기록).

## 데이터

[HuggingFace LMDrive 데이터셋](https://huggingface.co/datasets/OpenDILabCommunity/LMDrive)을 받아
`data/raw/lmdrive_carla/` 아래에 둡니다.

```
data/raw/lmdrive_carla/
├── dataset_index.txt                 "<경로>/ <프레임 수>"  (1단계)
├── navigation_instruction_list.txt   주행 지시문 클립       (2단계)
├── notice_instruction_list.json      알림 지시문            (2단계)
└── data/routes_town01_short_w2_.../
    ├── rgb_full/0000.jpg             전방·좌·우·후방 800x600 을 세로로 이어 붙인 이미지
    ├── lidar/0000.npy, lidar_odd/0000.npy
    ├── measurements_full/0000.json, measurements_all.json
    └── ...
```

공개 데이터셋은 이 파일들이 이미 들어 있습니다. 직접 수집한 데이터를 쓰거나 일부만 받았다면
`tools/` 로 같은 파일들을 만듭니다. 순서는 아래와 같습니다.

```bash
# 1. 수집 직후 파일 정리 (선택 단계는 건너뛰어도 학습됩니다)
python tools/preprocess/merge_camera_and_measurements.py data/raw/lmdrive_carla   # rgb_full, measurements_full
python tools/preprocess/remove_split_camera_folders.py   data/raw/lmdrive_carla   # 병합 후 원본 삭제
python tools/preprocess/find_blocked_frames.py           data/raw/lmdrive_carla   # 막힌 구간 탐색
python tools/preprocess/remove_blocked_frames.py         data/raw/lmdrive_carla
python tools/preprocess/renumber_frames.py               data/raw/lmdrive_carla
python tools/preprocess/merge_measurements.py            data/raw/lmdrive_carla   # measurements_all.json

# 2. 인덱스 (1단계 학습이 읽습니다)
python tools/preprocess/build_dataset_index.py data/raw/lmdrive_carla

# 3. 지시문 라벨 (2단계 학습이 읽습니다)
python tools/labeling/parse_navigation_instructions.py data/raw/lmdrive_carla
python tools/labeling/parse_notice_instructions.py     data/raw/lmdrive_carla \
    --scenarios-dir data/external/carla_scenarios
python tools/labeling/parse_misleading_instructions.py data/raw/lmdrive_carla    # 선택
```

3번 라벨 생성은 도로망을 봐야 해서 `carla` 파이썬 패키지가 필요합니다(`pip install carla==0.9.10.1`).
알림 지시문은 추가로 CARLA 시나리오 정의 `town<번호>_all_scenarios.json` 을 읽습니다. 이 저장소에는
포함하지 않았으니 LMDrive 원본의 `leaderboard/data/scenarios` 에서 받아 `data/external/carla_scenarios/`
에 두거나 `--scenarios-dir` 로 경로를 지정하세요. 규칙 파일(turn/follow/notice/other_rules.py)이 곧
지시문 종류의 정의이고, 각 규칙의 `instruction_id` 가 `datasets/instruction_templates.json` 의 문장
목록을 가리킵니다.

사전학습 가중치는 `data/external/` 에 둡니다.

- `LMDrive-vision-encoder-r50-v1.0/vision-encoder-r50.pth.tar` (2단계에서 고정된 비전 인코더)
- `llava-v1.5-7b` 또는 `vicuna-v1.5-7b`, `llama-7b` (2단계 언어 모델)
- Q-Former 초기화용 `bert-base-uncased` (기본값은 HuggingFace에서 내려받음)

## MORAI 등 다른 시뮬레이터에서 수집한 데이터 쓰기

폴더 구조와 `measurements` 필드가 위와 같으면 CARLA가 아니어도 그대로 학습됩니다. MORAI 수집분처럼
경로 이름에 town/weather가 없고 카메라가 합쳐져 있지 않아도 됩니다. 설정만 바꾸면 됩니다.

```yaml
data:
  root: ../MoraiDrive/dataset_local/competition_train
  lidar_rotation_deg: 0        # 프레임마다 360도를 다 담는 수집이면 0 (CARLA 반 스윕은 -90)
  train:
    towns: null                # 경로 이름의 town/weather 로 거르지 않음
    weathers: null
    routes: ["full_loop_*_01_*", "full_loop_*_02_*"]   # 학습에 쓸 경로 폴더 (glob 패턴)
  val:
    towns: null
    weathers: null
    routes: ["full_loop_*_03_*"]                        # 검증에 쓸 경로 폴더
```

`routes` 를 비우면 모든 경로를 씁니다. 경로가 하나뿐이면 학습과 검증이 같은 데이터가 되므로, 수집할 때
여러 주행으로 나눠 두고 폴더 이름으로 학습·검증을 가르는 것을 권합니다.

차이가 나도 되는 것과 반드시 있어야 하는 것을 정리하면 다음과 같습니다.

| 항목 | 필요 여부 |
| --- | --- |
| `rgb_full/` 합본 이미지 | 없어도 됨. `rgb_front/left/right/rear` 를 직접 읽습니다 |
| `lidar_odd/` | 없어도 됨. 있으면 두 스윕을 합칩니다 |
| 경로 이름의 town·weather | 없어도 됨 (`towns: null`, `weathers: null`, 분할은 `routes` 패턴) |
| `dataset_index.txt` 의 `data/` 접두사 | 있어도 없어도 됨 |
| 카메라 해상도 | 달라도 됨. 고정 크기로 리사이즈 후 크롭합니다 |
| `measurements_full/*.json` 과 `measurements_all.json` | **필수**. x, y, theta, gps_x, gps_y, x_command, y_command, speed, command, is_junction, is_red_light_present, stop_sign, future_waypoints, actors_data 를 씁니다 |
| `actors_data` 의 주변 객체 | 비어 있어도 학습은 되지만 검출 손실이 0이 되어 BEV 토큰이 학습되지 않습니다 |

2단계를 돌리려면 `navigation_instruction_list.txt` 가 필요합니다. `tools/labeling` 의 파서는 CARLA 도로망
(xodr)과 town 번호를 전제로 하므로 MORAI 데이터에는 그대로 쓸 수 없습니다. MORAI용 생성기를 만들 때는
아래 형식의 JSON 을 한 줄에 하나씩 쓰면 됩니다. `town_id` 와 `weather_id` 는 필터를 쓰지 않으면 생략해도
됩니다.

```json
{"route_path": "data/full_loop_basic_drive_2026_09_09_01_44_15", "start_frame": 120,
 "end_frame": 210, "instruction": "Turn-01-L", "instruction_id": 0, "instruction_args": [],
 "route_frames": 4240}
```

`instruction_id` 는 `datasets/instruction_templates.json` 의 문장 목록을 가리키고, 규칙 이름과 번호의
대응은 `tools/labeling/parse_navigation_instructions.py` 상단 표에 있습니다. 알림 지시문을 쓰려면
`notice_instruction_list.json` 을 `{경로: [{"frame_id": n, "instruction_id": k}, ...]}` 형태로 만듭니다.

전처리 도구 중 `merge_camera_and_measurements.py` 는 CARLA 해상도(800x600)를 전제로 하므로 MORAI
데이터에는 쓰지 않습니다. 나머지(인덱스 생성, 측정값 병합, 막힌 구간 정리)는 그대로 쓸 수 있습니다.
CARLA 데이터를 전혀 쓰지 않는다면 `tools/labeling/carla_maps/` (21MB)는 지워도 됩니다. 규칙 코드는
남겨 두는 편이 좋습니다. 어떤 상황을 어떤 지시문으로 라벨링했는지가 그 안에 정의되어 있습니다.

## 학습

1단계, 비전 인코더 사전학습:

```bash
python train.py --config configs/vision_encoder_train.yaml
torchrun --nproc_per_node=8 train.py --config configs/vision_encoder_train.yaml   # 다중 GPU
```

2단계, 지시문 미세조정:

```bash
torchrun --nproc_per_node=8 train.py --config configs/lmdrive_train.yaml
```

설정값은 명령줄에서 덮어쓸 수 있습니다.

```bash
python train.py --config configs/vision_encoder_train.yaml \
    --options data.batch_size=8 scheduler.epochs=10 "data.val.towns=[5]"
```

학습률은 원본과 같이 `lr * (배치 크기 * GPU 수) / 512` 로 자동 조정됩니다. 1단계는 `scheduler.epochs`
뒤에 `cooldown_epochs` 만큼 최소 학습률로 더 학습하므로 기본 설정에서는 25 + 10 = 35 에폭을 돕니다.

산출물은 다음 위치에 쌓입니다.

- `outputs/checkpoints/<실행 이름>/` : `last.pth.tar`, `model_best.pth.tar`, `checkpoint-<epoch>.pth.tar`
  (2단계는 `checkpoint_best.pth`, `checkpoint_last.pth`)
- `outputs/logs/<실행 이름>/` : TensorBoard 이벤트, `summary.csv` 또는 `log.txt`, `config.yaml`, `train.log`

중단한 학습은 `training.resume` 에 체크포인트 경로를 주면 에폭과 옵티마이저 상태까지 이어서 돕니다.

## 평가

```bash
python evaluate.py --config configs/vision_encoder_train.yaml \
    --checkpoint outputs/checkpoints/<실행 이름>/model_best.pth.tar

python evaluate.py --config configs/lmdrive_train.yaml \
    --checkpoint outputs/checkpoints/<실행 이름>/checkpoint_best.pth
```

1단계는 검출·웨이포인트·신호등·정지선 손실과 신호등/정지선 정확도를, 2단계는 웨이포인트 L1 손실과
지시문 종료 플래그의 손실·정확도를 출력하고 `outputs/logs/eval-<단계>-<시각>.json` 에 저장합니다.
주행 점수(Driving Score)를 재는 폐루프 평가는 CARLA 시뮬레이터가 필요하므로 이 저장소에는 없습니다.

## 원본 코드와의 대응

| 원본 | 현재 |
| --- | --- |
| `vision_encoder/timm/models/memfuser.py` | `models/vision_encoder.py`, `models/transformer.py` |
| `vision_encoder/timm/models/resnet.py`, `features.py`, `helpers.py` | `models/resnet.py` |
| `vision_encoder/timm/models/pointpillar.py` | `models/pointpillars.py` |
| `vision_encoder/timm/data/carla_dataset.py` 등 | `datasets/carla_perception_dataset.py`, `transforms.py`, `lidar.py`, `sensor_io.py` |
| `vision_encoder/timm/data/heatmap_utils.py`, `det_utils.py` | `datasets/perception_labels.py` |
| `vision_encoder/train_pretrain.py` (1861줄, 108개 인자) | `train.py` 1단계 + `utils/losses.py`, `utils/schedulers.py`, `utils/checkpoint.py` |
| `LAVIS/lavis/models/drive_models/drive.py` | `models/lmdrive.py` |
| `LAVIS/lavis/models/blip2_models/Qformer.py`, `modeling_llama.py` | `models/qformer.py`, `models/llama.py` |
| `LAVIS/lavis/datasets/datasets/carla_dataset_llm.py` | `datasets/carla_instruction_dataset.py` |
| `LAVIS/lavis/tasks/drive.py`, `runners/runner_base.py`, `train.py` | `train.py` 2단계 + `evaluate.py` |
| `LAVIS/lavis/projects/lmdrive/*.yaml` | `configs/lmdrive_*.yaml` |
| `tools/data_preprocessing/` | `tools/preprocess/` |
| `tools/data_parsing/` | `tools/labeling/` |
| `leaderboard/`, `scenario_runner/`, `data_collection/`, `langauto/` | 제거 (시뮬레이터 구동과 폐루프 평가 전용) |
| `tools/generate_*.py`, `tools/sample_junctions.py`, `tools/interpolate_route.py` | 제거 (CARLA 경로 생성, 데이터 수집용) |

## 원본 대비 달라진 점

동작을 바꾼 곳(모두 원본에서 단일 GPU 실행이나 최신 라이브러리에서 실패하던 부분입니다).

- 단일 GPU 실행 복구: 학습률 스케일과 `dist.barrier()` 가 분산 초기화를 강제하던 부분, 검증 루프에서
  비분산일 때 정의되지 않은 변수를 쓰던 부분을 고쳤습니다.
- `torch.load` 에 `weights_only=False` 를 명시했습니다. torch 2.6부터 기본값이 바뀌어 공개 비전 인코더
  체크포인트를 읽지 못했습니다.
- `models/llama.py` 에 `torch.nn.functional` import 를 추가했습니다. 원본은 이 이름을 쓰면서 import 를
  빠뜨려 일부 경로가 실행되면 바로 오류가 납니다.
- 선택 의존성 처리: `torch-scatter` 가 없으면 동일 연산의 순수 PyTorch 구현을, `tensorboard` 가 없으면
  CSV/JSON 로그만 씁니다. `accelerate` 가 없으면 저메모리 LLM 로딩을 끕니다.
- 체크포인트 저장은 파일 교체 실패 시 재시도 후 복사로 대체합니다(Windows 마운트 대응).
- 다중 GPU 검증 지표를 모든 랭크에서 합산합니다. 원본은 0번 랭크가 본 일부만 평균했습니다.
- 2단계 체크포인트에서 학습하지 않는 가중치를 뺐습니다. 비전 인코더와 LLM 은 각자 파일에서 옵니다.
- `dataset_index.txt` 경로를 학습 코드와 `tools/` 가 같은 규칙(`tools/route_paths.py`)으로 해석합니다.
  원본은 도구가 `<root>/<경로>`, 학습 코드가 `<root>/data/<경로>` 를 봐서 한쪽이 어긋났습니다.
- 라벨 생성 스크립트에 argparse 를 붙이고(`--workers`, `--scenarios-dir`), 지도용 `carla` import 를
  실제 사용 시점으로 미뤘습니다. CARLA 없이도 도움말과 규칙 확인이 가능합니다.

의도적으로 원본 동작을 유지한 곳.

- 지시문 템플릿의 `[x]`, `[y]` 치환은 원본에서 결과를 버려 실제로는 치환되지 않습니다. 공개 모델이
  그 상태로 학습됐으므로 기본값을 그대로 두고, `data.fill_instruction_placeholders: true` 로 켤 수
  있게만 했습니다.
- 1단계의 최적 체크포인트 선택 지표 `l1_error` 는 이름과 달리 전체 검증 손실입니다. 같은 값을 `loss`
  라는 이름으로도 기록합니다.

## 원본과의 동일성 검증

재구성 뒤, 재구성 전 원본 코드(커밋 `1d12636` 시점의 `vision_encoder/`, `LAVIS/`)와 같은 입력에 대해
같은 값을 내는지 텐서 단위로 비교했습니다.

1단계 — CPU, float32, 공개 비전 인코더 체크포인트, CARLA 프레임 3장: **113개 항목 전부 일치**

- 데이터셋과 로더가 만든 입력 12종과 정답 8종
- 시각 토큰 106개와 검출·웨이포인트·신호등·정지선 출력 (최대 오차 0.0)
- 같은 시드로 만든 모델의 state_dict 888개 텐서
- 손실 5종, 평가 지표 평균, 학습 2스텝 뒤 전체 가중치, 35+1에폭 학습률 스케줄

2단계 — GPU, fp16 autocast, 소형 LLaMA, 클립 3개: **35개 항목 전부 일치**

- 지시문 데이터셋이 만든 센서 텐서와 지시문·알림 문장
- state_dict 키 1251개와 같은 시드에서의 초기값
- 평가·추론 출력, 알림 삽입 경로와 프레임 프롬프트 경로
- 학습 1스텝 뒤 학습 대상 텐서 341개, 체크포인트 저장과 복원

추가로 공개 비전 인코더 체크포인트를 새 평가 경로로 CARLA 검증 구간(41 샘플)에 돌려 총손실 0.072,
웨이포인트 L1 0.085, 신호등·정지선 정확도 100%를 확인했습니다. 두 단계 모두 학습 → 검증 → 체크포인트
저장 → 평가 → 재개를 실제로 실행해 확인했습니다.

## 알려진 제약

- `models/llama.py` 는 grouped-query attention 을 쓰지 않습니다(원본이 그렇게 고정해 두었습니다).
  LLaVA-v1.5-7B, Vicuna-v1.5-7B, LLaMA-7B 는 문제없지만 TinyLlama, Llama-3 계열은 그대로는 못 씁니다.
- LoRA(`use_lora: true`) 경로는 `peft` 가 필요하며 이 환경에서는 검증하지 못했습니다.
- 최상위 `datasets/` 패키지 이름이 HuggingFace `datasets` 와 겹칩니다. 이 저장소는 HuggingFace
  datasets 를 쓰지 않지만, 같은 환경에 설치해 두었다면 프로젝트 루트에서 실행할 때 주의하세요.

## 라이선스

원본 LMDrive 는 Apache-2.0, 포함된 LAVIS 코드는 BSD-3-Clause, timm 코드는 Apache-2.0 입니다.
각 파일 상단의 저작권 표시와 `LICENSE` 를 그대로 유지했습니다.
