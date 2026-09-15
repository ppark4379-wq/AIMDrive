# VLDrive (MORAI 학습 코드)

VLDrive 논문 구현을 MORAI 시뮬레이터에서 수집한 데이터로 학습하도록 정리한 저장소입니다.
구조는 자매 프로젝트 [`../LMDrive`](../LMDrive) 와 동일하게 맞췄습니다.
설계 근거와 구현 기록은 [`docs/`](docs/) 에 있습니다.

원래는 LAVIS 전체(BLIP/ALBEF/CLIP 등)와 timm 포크, CARLA leaderboard·scenario_runner 가
함께 들어 있었고 실제로 학습에 쓰이는 파일은 그중 일부였습니다. 여기서는
**학습과 오프라인 평가에 필요한 코드만** 남기고 이름을 역할이 드러나게 바꿨습니다.

## 모델 개요

```
                 [1단계] 비전 인코더 사전학습                [2단계] VLDrive 학습
  MORAI 4-view 카메라 + LiDAR                        얼린 인코더 토큰 [T,106,256]
            │                                                  │
      Memfuser 인코더  ── 검출/웨이포인트/신호등 헤드            CCDP  (Gumbel 토큰 가지치기, 약 30% 유지)
            │                                                  │
     106 x 256 시각 토큰  ──────── 캐시 ───────────────▶   MEFA  (Modified Q-Former, 프레임 메모리 Z=10)
                                                               │
                                                    TinyLlama-1.1B + DDIA (모달리티 인지 어텐션)
                                                               │
                                                    웨이포인트 5점 + 종료 플래그

  손실:  L_way + 10 · L_prun + 1 · L_rec
```

1단계는 LMDrive 의 비전 인코더 구조 그대로이고, 공개 LMDrive 체크포인트가 `strict=True` 로
로드됩니다. 다만 **공개 체크포인트로 시작하지 않고** ResNet-50d ImageNet 백본에서 출발해
MORAI 전문가 주행으로 PointPillars·융합·쿼리·헤드를 학습합니다.

## 구조

```
VLDrive/
├── train.py                  두 단계 학습 진입점 (하위 명령으로 단계 선택)
├── evaluate.py               체크포인트 오프라인 평가 (시뮬레이터 불필요)
├── configs/
│   ├── vision_encoder_train.json     1단계 설정 (학습 파라미터의 유일한 출처)
│   ├── vision_encoder_train_smoke.json  2배치 1에폭 스모크
│   ├── vldrive_train.json            2단계 기본 설정 (80/20 분할, 15에폭)
│   ├── vldrive_train_fullroute.json  전체 경로 인코더 + 95/05 분할
│   ├── vldrive_train_smoke.json      2윈도우 2에폭 스모크
│   └── splits/                       시간순 train/purge/validation 분할 파일
├── models/
│   ├── resnet.py             카메라 CNN 백본 (ResNet-D)
│   ├── transformer.py        DETR 방식 인코더/디코더와 위치 인코딩
│   ├── pointpillars.py       LiDAR PointPillars 인코더 (torch_scatter 없으면 순수 PyTorch)
│   ├── vision_encoder.py     Memfuser 비전 인코더 (멀티뷰 + LiDAR -> 토큰/인지 출력)
│   ├── frozen_vision_encoder.py  1단계 인코더를 얼려서 2단계에 붙이는 래퍼
│   ├── ccdp.py               Gumbel 토큰 가지치기 (유지율 목표 0.3)
│   ├── qformer.py            BLIP-2 Q-Former (LAVIS 원본 그대로)
│   ├── qformer_mefa.py       MEFA - 프레임 메모리 Z=10 을 가진 Q-Former 커넥터
│   ├── ddia_attention.py     DDIA - 시각/지시문/패딩을 구분하는 어텐션 바이어스
│   ├── multimodal_sequence.py  시각·지시문·패딩 시퀀스 배치
│   ├── tinyllama_backbone.py TinyLlama 로딩 + DDIA 설치
│   ├── visual_token_interface.py  [B,T,N,256] 시각 토큰 규격 검증
│   ├── language_model.py     TinyLlama + DDIA + 웨이포인트/종료 헤드
│   └── vldrive.py            전체 학습 모델 (인코더 -> CCDP -> MEFA -> 언어 모델)
├── datasets/
│   ├── morai_perception_dataset.py  1단계용 동기화 프레임 + 인지/웨이포인트 라벨
│   ├── morai_expert_dataset.py      2단계용 캐시 토큰 윈도우 + 지시문 + 전문가 궤적
│   ├── sensor_adapter.py            원시 MORAI 프레임 -> 얼린 인코더 입력
│   ├── split_manifest.py            시간순 분할 파일 생성/해석
│   ├── traffic_labels.py            신호등 라벨 사이드카 생성/검증
│   ├── transforms.py, lidar.py      카메라 전처리, LiDAR 전처리
│   └── loader.py                    DataLoader 와 카메라 변환 부착
├── utils/
│   ├── runtime.py            오프라인 캐시 설정, sha256 출처 기록, 수치 어댑터, 경로 해석
│   ├── vision_encoder_config.py  1단계 config 로더 (frozen, 누락·미지 키·타입 거부)
│   ├── vision_encoder_training.py  1단계 학습 루프 (에폭·체크포인트)
│   ├── trainer.py            2단계 학습/preflight/평가/체크포인트
│   ├── training.py           배치·autocast·시드·그래디언트 보고 공통 함수
│   └── metrics.py            궤적 지표 (ADE, FDE, 구간별 오차)
├── tools/                    분할·라벨·지시문·토큰 캐시·시각화 (tools/README.md 참고)
├── tests/                    148개 단위 테스트
├── docs/                     설계 노트와 실험 보고서
├── data/                     raw/ processed/ external/  (git 추적 제외)
└── outputs/                  checkpoints/ logs/         (git 추적 제외)
```

## 설치

```bash
pip install -r requirements.txt
```

Python 3.10, torch 2.8.0+cu128, transformers 4.31.0, RTX 5060 Ti 16GB 에서 확인했습니다.
`torch-scatter` 는 없어도 됩니다(PointPillars 가 순수 PyTorch 대체 경로를 씁니다).
`matplotlib` 은 `tools/render_vision_tokens.py` 에만, `huggingface-hub` 은
`tools/prepare_assets.py --download-*` 에만 필요합니다.

transformers 는 **4.31.0 고정**입니다. 이 버전이 LLaMA GQA 를 지원하면서도 DDIA 가 교체하는
3-value `LlamaAttention` API 를 유지합니다. 상위 버전을 쓰려면 `models/ddia_attention.py` 에
어댑터가 필요합니다.

## 준비물

```bash
# 1. 사전학습 가중치 (최초 1회) -> data/external/
python tools/prepare_assets.py --download-language --download-qformer

# 2. 시간순 분할 -> configs/splits/
python tools/create_split.py ../MoraiDrive/dataset_local/competition_train \
    configs/splits/my_split.json --train-fraction 0.8 --purge-frames 25
python tools/audit_split.py ../MoraiDrive/dataset_local/competition_train \
    configs/splits/my_split.json

# 3. 신호등 라벨 사이드카 -> data/processed/traffic_labels/
python tools/build_traffic_labels.py \
    --dataset-root ../MoraiDrive/dataset_local/competition_train \
    --split-file configs/splits/my_split.json \
    --output data/processed/traffic_labels/my_split.json \
    --report data/processed/traffic_labels/my_split_report.json

# 4. 주행 지시문 -> data/processed/instructions/
python tools/build_instructions.py --dataset-root ../MoraiDrive/dataset_local/competition_train \
    --output data/processed/instructions/my_split.jsonl
```

MORAI 수집 데이터는 CARLA 와 같은 형식입니다. 경로마다 `rgb_front/left/right/rear`,
`lidar`, `measurements_full`, `measurements_all.json` 이 있고 루트에 `dataset_index.txt`
가 있습니다. LiDAR 축은 MORAI 가 FLU(전방-좌-상), LMDrive 인코더가 FRU(전방-우-상)라서
`lidar_axis_policy="morai_flu_to_lmdrive_fru"` 가 경계에서 y 부호만 뒤집습니다.
원본 기록과 궤적/경로 라벨은 손대지 않습니다.

## 학습

### 1단계 — 비전 인코더

```bash
python train.py vision-encoder --config configs/vision_encoder_train.json
```

**모든 학습 파라미터는 config 파일에서만 옵니다.** 명령줄은 `--config` 하나뿐이고,
불러온 config 는 frozen 이며, 학습 함수들은 config 객체가 아닌 값을 거부합니다. 즉
명령줄에서도 다른 모듈에서도 설정값을 덮어쓸 수 없습니다. 다른 설정으로 돌리려면
config 를 고치거나 복사하세요.

config 는 빠진 키, 모르는 키, 잘못된 타입, 범위를 벗어난 값을 모두 거부합니다.
따라서 "안 적었으니 기본값" 같은 경로가 없고, 오타가 조용히 무시되지도 않습니다.
`configs/vision_encoder_train_smoke.json` 은 2배치·1에폭 스모크용입니다.

주요 키:

| 키 | 뜻 |
|---|---|
| `dataset.root`, `dataset.split_file` | 수집 데이터 루트와 시간순 분할 파일 |
| `dataset.traffic_label_sidecar` | 감사된 신호등 라벨 마스크 (없으면 `null`) |
| `augmentation.rgb_hue_jitter` | 0 이 기본. 신호등은 색으로 구분하므로 hue 증강을 끄고 밝기·대비·채도만 씁니다. 0.4 는 legacy 재현용 |
| `schedule.epochs`, `warmup_epochs`, `accumulation_steps` | 에폭 수와 누적 스텝 |
| `initialization.warm_start` + `warm_start_sha256` | 가중치만 가져와 새 학습 시작. SHA-256 을 함께 선언해야 합니다 |
| `initialization.resume` | 옵티마이저·에폭까지 이어받기. `warm_start` 와 동시 지정 불가 |
| `output.dir` | 결과 디렉터리 |

출력: `checkpoint-last.pth.tar`, `model-best.pth.tar`, `metrics.jsonl`,
`run_provenance.json`, `feature_contract.json`. `run_provenance.json` 과 체크포인트에는
사용한 config 의 경로·SHA-256·내용이 그대로 기록됩니다.

### 두 visual encoder 경로의 구분

`vision-encoder`는 논문 학습 recipe를 목표로 유지하는 비교 경로입니다. 가공된
MORAI dataset에서 ImageNet RGB 초기화와 새 LiDAR·fusion·head를 학습하며,
현재 검증된 loss는 waypoint와 신호등입니다.

`vision-encoder-morai`는 실제 Sensor Record publication을 사용하는 MORAI 맞춤
후보 경로입니다. 원본 LMDrive visual checkpoint 전체에서 시작해 GPS 속도와
객체·객체 속도·waypoint·신호등을 함께 학습합니다. 두 경로는 같은
`models/vision_encoder.py`를 사용하지만 dataset, 초기화, loss, 실행 기록을
서로 섞지 않습니다.

### MORAI 맞춤 전이학습 실행

Sensor Record의 검증 완료된 여러 campaign을 합쳐, 원본 LMDrive 전체 가중치에서
ResNet·PointPillars와 네 가지 예측 과제를 학습하는 경로입니다. 위의 논문 기반
학습 경로를 유지하며, PC별 manifest와 가중치 경로는 Git에서 제외된
`configs/morai_transfer.local.json`에 지정합니다.

```bash
bash tools/train_visual_morai.sh --epochs 1
```

`--preflight`는 실제 데이터의 forward/backward만 검사하고 가중치는 갱신하지 않습니다.
입력·loss·데이터 결합과 기존 실험과의 차이는
[전용 안내](docs/MORAI_TRANSFER_TRAINING_KO.md)를 참고하세요.

### 1.5단계 — 시각 토큰 캐시

```bash
python tools/cache_vision_tokens.py \
    --dataset-root ../MoraiDrive/dataset_local/competition_train \
    --split-file configs/splits/morai_full_loop_80_20.json \
    --checkpoint outputs/checkpoints/vision_encoder/run1/model-best.pth.tar \
    --output data/processed/vision_tokens/run1 --batch-size 1 --workers 2
```

2단계는 이 캐시만 읽습니다. 인코더는 학습 중에 아예 돌지 않으므로 GPU 메모리와 시간이
크게 줄고, 캐시가 어느 체크포인트에서 나왔는지 sha256 으로 묶여 검증됩니다.

### 2단계 — VLDrive

```bash
python train.py vldrive --config configs/vldrive_train.json --output outputs/logs/run1
```

먼저 preflight 가 돌아 그래디언트가 흐르는지, 미래 프레임이 과거로 새지 않는지,
금지된 교사 입력이 들어오지 않는지 확인하고 실패하면 학습을 시작하지 않습니다.
`--mode preflight` 로 검사만, `--mode evaluate --checkpoint ...` 로 평가만 할 수 있습니다.
학습이 끝나면 얼린 가중치가 정말 안 바뀌었는지(`frozen_unchanged`) 확인합니다.
출력: `model-best.pt`, `model-last.pt`, `checkpoint.pt`, `training.jsonl`,
`epoch_metrics.jsonl`, `provenance.json`, `report.json`.

전체를 한 번에 돌리려면 `python tools/run_pipeline.py` 를 씁니다.

## 평가

```bash
# 1단계: 웨이포인트 L1, ADE/FDE, 신호등 정확도
python evaluate.py vision-encoder --config configs/vision_encoder_train.json \
    --checkpoint outputs/checkpoints/vision_encoder/run1/model-best.pth.tar

# 2단계: 궤적 지표 + val_predictions.json
python evaluate.py vldrive --config configs/vldrive_train.json \
    --checkpoint outputs/logs/run1/model-best.pt --output outputs/logs/run1-eval
```

시뮬레이터를 띄우는 폐루프 주행 평가는 이 저장소에 없습니다. 모델 성능이 어느 정도
나온 뒤에 MORAI 와 연결합니다.

## 인코더가 무엇을 보는지 확인

```bash
python tools/inspect_vision_tokens.py --checkpoint <ckpt> --label run1 \
    --frames 991,1591,2491 --output outputs/logs/tokens/run1
python tools/render_vision_tokens.py --probe-root outputs/logs/tokens --labels run1 \
    --frames 991,1591,2491 \
    --dataset-root ../MoraiDrive/dataset_local/competition_train \
    --output outputs/logs/tokens/run1.png
```

언어 모델을 거치지 않고 인코더 토큰만 뽑아 프레임별 BEV 토큰 세기를 그립니다.

## 재구성 전 코드와의 대응

| 재구성 후 | 재구성 전 |
|---|---|
| `train.py vision-encoder`, `utils/vision_encoder_training.py` | `vision_encoder/train_morai_pretrain.py` |
| `train.py vldrive` | `LAVIS/train_vldrive.py` |
| `models/vision_encoder.py`, `resnet.py`, `transformer.py`, `pointpillars.py` | `vision_encoder/timm/models/{memfuser,resnet,...,pointpillar}.py` |
| `models/qformer.py` | `LAVIS/lavis/models/blip2_models/Qformer.py` |
| `models/{ccdp,qformer_mefa,ddia_attention,multimodal_sequence,tinyllama_backbone,visual_token_interface}.py` | `LAVIS/lavis/models/vldrive/` 의 같은 이름 |
| `models/frozen_vision_encoder.py` | `LAVIS/lavis/models/vldrive/visual_encoder.py` |
| `models/language_model.py` | `LAVIS/lavis/models/vldrive/vldrive_model.py` |
| `models/vldrive.py` | `LAVIS/lavis/models/vldrive/training_model.py` |
| `datasets/morai_perception_dataset.py` | `vision_encoder/timm/data/carla_dataset.py` 의 `MoraiExpertDataset` |
| `datasets/{transforms,lidar,loader}.py` | `vision_encoder/timm/data/{transforms_carla_factory,carla_loader}.py` |
| `datasets/{split_manifest,traffic_labels}.py` | `vision_encoder/{morai_data,morai_traffic_labels}.py` |
| `datasets/morai_expert_dataset.py`, `sensor_adapter.py` | `LAVIS/lavis/datasets/datasets/morai/` |
| `utils/runtime.py` | `LAVIS/lavis/common/morai_runtime.py` |
| `utils/{trainer,training,metrics}.py` | `LAVIS/lavis/runners/{vldrive_trainer,morai_trainer,morai_metrics}.py` |
| `configs/vldrive_train*.json` | `LAVIS/lavis/projects/vldrive/morai_paper*.json` |
| `configs/vision_encoder_train*.json`, `utils/vision_encoder_config.py` | `vision_encoder/train_morai_pretrain.py` 의 argparse 인자 |
| `configs/splits/` | `vision_encoder/configs/` |
| `data/external/` | `LAVIS/assets/pretrained/`, `weights/`, `.runtime/lmdrive_official/` |
| `data/processed/vision_tokens/`, `instructions/` | `LAVIS/.runtime/morai_vldrive/` |
| `data/processed/traffic_labels/` | `vision_encoder/.runtime/traffic_labels_v2/` |
| `outputs/checkpoints/vision_encoder/` | `vision_encoder/output_morai/`, `LAVIS/assets/pretrained/vision_encoder/` |

## 재구성 전과 달라진 점

- **LAVIS 프레임워크를 걷어냈습니다.** 우리 코드가 LAVIS 에서 실제로 쓰던 것은 Q-Former,
  `registry`, `BaseModel` 세 가지뿐이었습니다. Q-Former 는 그대로 옮겼고, 레지스트리 등록과
  `BaseModel` 상속은 없앴습니다(`nn.Module` 직접 상속). 쓰이지 않던
  `BaseModel.load_checkpoint_from_config` 경로도 제거했습니다.
- **timm 포크를 걷어냈습니다.** 200여 개의 무관한 모델 대신 인코더에 실제로 필요한 4개
  파일만 남겼습니다. 파라미터 이름은 그대로라 기존 체크포인트가 계속 로드됩니다.
- **`stage6e` 데이터 형식을 제거했습니다.** 현재 모든 설정이 `morai_expert_cache_v1` 을
  쓰고, ROS 로 직접 수집하던 초기 경로는 더 이상 사용하지 않습니다(설계 기록은
  `docs/stage6_*.md`).
- **신호등 라벨 사이드카를 다시 생성했습니다.** 사이드카는 자기를 만든 코드의 해시를
  기록하고 로드할 때 전체 재현 비교를 합니다. 파일이 옮겨져 해시가 바뀌었으므로
  다시 만들었고, 라벨 레코드 자체는 이전과 **바이트 단위로 동일**합니다.
- **체크포인트 출처 기록 경로가 바뀌었습니다.** `utils/training.py:source_hashes` 가
  이제 `models/`, `datasets/`, `utils/`, `train.py`, `evaluate.py` 를 해시합니다.
  이전 체크포인트는 출처 불일치로 재개되지 않습니다(이전 학습 결과는 모두 삭제했습니다).
- 진단·재현용 일회성 스크립트 약 25개를 지웠습니다. 그 결과 보고서는 `docs/reports/` 에
  그대로 있습니다.

## 재구성 전후 동일성 검증

원본 코드를 지우기 전에, 같은 입력을 넣어 텐서 단위로 비교했습니다.

- **1단계 데이터셋 44개 항목 완전 일치** — train/validation 각각에 대해 샘플 수, 프레임
  목록, 9개 입력 텐서(카메라 5, LiDAR, 점 개수, 속도, 목표점)와 5개 타깃 텐서.
- **1단계 인코더 완전 일치** — 실제 MORAI 체크포인트를 양쪽에 `strict=True` 로 올려
  state_dict 키 888개가 같고, 검출/웨이포인트/신호등/정지선/BEV 5개 출력과 106x256 시각
  토큰이 CPU 결정론 모드에서 비트 단위로 같습니다.
- **테스트 148 passed, 1 skipped** — 남긴 테스트 파일 17개의 테스트 개수가 이동 전과
  하나도 다르지 않습니다(지운 일회성 스크립트의 테스트 42개만 함께 제거).
- **실제 학습 동작 확인** — 1단계: 4배치 학습 + 검증 + 체크포인트 + feature contract
  검증(`strict_load: All keys matched`, 토큰 [1,106,256]). 2단계: preflight 통과,
  4 업데이트, 학습 대상 가중치는 변하고 얼린 가중치는 그대로, 검증 FDE 28.868 -> 28.020.
  `evaluate.py` 가 같은 체크포인트에서 같은 값을 재현합니다.

## 알려진 제약

- **2단계 성능은 아직 목표에 못 미칩니다.** 위 스모크 값(FDE 약 28 m)은 2윈도우 2에폭
  결과라 성능 지표가 아닙니다. 본 학습 결과와 원인 분석은 `docs/reports/` 를 보세요.
- **데이터가 한 경로(4240프레임)뿐입니다.** 시나리오를 늘려 더 수집해야 합니다.
- `tools/labeling`, `tools/preprocess`, `tools/carla` 는 CARLA 시절 스크립트라 지금
  파이프라인에서 실행되지 않습니다. 유지보수 대상 판은 `../LMDrive/tools/` 입니다.
- `LAVIS/.pytest_cache` 디렉터리가 WSL 권한 문제로 남아 있습니다(12KB). 관리자 권한
  터미널에서 `rd /s /q LAVIS` 로 지우면 됩니다.

## 라이선스

`LICENSE` 참고. Q-Former 는 LAVIS(BSD-3-Clause), 비전 인코더 구조는 LMDrive(Apache-2.0),
TinyLlama 는 Apache-2.0 입니다.
