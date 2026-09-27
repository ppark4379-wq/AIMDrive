# 학습 코드

비전 인코더·커넥터·LLM별로 학습 구성을 나누고, 공동 학습과 공통 기능을 별도 폴더에 모았습니다.
각 `setup.py`는 설정과 학습 준비를 함께 다룹니다. 기존 학습 계산·기본값·호출 순서는 유지합니다.

```text
training/
├─ README.md
├─ vision_encoder/
│  ├─ setup.py
│  ├─ recipes/
│  │  ├─ pretrain.py
│  │  └─ transfer.py
│  ├─ losses/morai_transfer.py
│  ├─ utils/morai_transfer_metrics.py
│  ├─ scripts/
│  │  ├─ pretrain.sh
│  │  ├─ transfer.sh
│  │  └─ object_head.sh
│  └─ README.md
├─ connector/
│  ├─ setup.py
│  ├─ utils/
│  └─ README.md
├─ llm/
│  ├─ setup.py
│  ├─ utils/
│  └─ README.md
├─ joint/
│  ├─ setup.py
│  ├─ training.py
│  ├─ scripts/train.sh
│  └─ README.md
├─ common/
│  ├─ batching.py
│  ├─ precision.py
│  ├─ reproducibility.py
│  ├─ optimizers.py
│  ├─ checkpoints.py
│  ├─ logging.py
│  ├─ training_helper.py        # 이전 helper import 호환
│  └─ README.md
├─ experiments/paired_hue_finetune.py
├─ tests/test_training_layout.py
├─ scripts/                    # 이전 셸 실행 경로 보존
├─ training.py                 # joint/training.py 호환 import
├─ vision_encoder_training.py  # recipes/pretrain.py 호환 import
└─ morai_transfer_training.py  # recipes/transfer.py 호환 import
```

패키지용 `__init__.py`는 위 트리에서 생략했습니다.

## 학습 흐름

- `vision_encoder/setup.py`: 기존 설정 클래스와 검증 함수를 재사용하고 두 recipe에 있던 모델 초기화·동결·optimizer·LR 설정 함수를 모았습니다.
- `vision_encoder/recipes/pretrain.py`: ImageNet 백본에서 시작하는 MORAI 학습입니다. 제로샷 추론이 아닙니다.
- `vision_encoder/recipes/transfer.py`: LMDrive 또는 기존 MORAI 인코더 가중치에서 시작합니다. CARLA 데이터를 자동으로 혼합하지 않습니다.
- `connector/setup.py`, `llm/setup.py`: 기존 모델 생성자와 설정을 제공하는 학습용 import 입구입니다. 모델 구현과 동결 정책을 복제하지 않습니다.
- `joint/setup.py`: 기존 공동 설정·모델 생성자·설정 로더·optimizer 구성을 제공합니다.
- `joint/training.py`: 커넥터와 LLM을 기존처럼 하나의 학습 루프에서 학습합니다.

실제 커넥터·LLM 생성과 동결 순서는 기존 `models/vldrive.py`의 생성자를 사용합니다.
각 setup은 기존 객체를 그대로 재사용하며 독립 학습 루프나 새 기본값을 만들지 않습니다.
데이터·모델 담당 팀원의 `data/src/loaders/`, `models/ve/`, `models/connector/`, `models/llm/` 구조를 유지합니다.

## 실행

VLDrive 루트에서 실행합니다. 셸은 기존 `train.py`에 인자를 그대로 전달합니다.

```bash
bash training/vision_encoder/scripts/pretrain.sh --config training/vldrive_train.json
bash training/vision_encoder/scripts/transfer.sh --help
bash training/vision_encoder/scripts/object_head.sh
bash training/joint/scripts/train.sh --config training/vldrive_train.json
```

`pretrain.sh`와 `joint/scripts/train.sh`는 요청한 전체 구조에 맞춰 추가한 얇은 셸 진입점입니다.
객체 head 스크립트는 기존처럼 로컬 설정과 체크포인트가 필요합니다.
기존 `training/scripts/`와 `tools/train_visual_morai.sh`, `tools/train_visual_morai_object_head.sh`,
`tools/paired_hue_finetune.py` 경로도 유지합니다. paired hue 실험은 기존 비활성 상태입니다.

## 이동·복사 내역

| 기존 파일/기능 | 새 위치 | 처리 |
|---|---|---|
| 최초 `utils/trainer.py` → `training/training.py` | `joint/training.py` | 학습·평가·preflight 루프 유지, 이전 경로는 호환 import |
| `vision_encoder_training.py` | `vision_encoder/recipes/pretrain.py` | 학습 루프 이동, 이전 경로는 호환 import |
| `morai_transfer_training.py` | `vision_encoder/recipes/transfer.py` | 전이학습 루프 이동, 이전 경로는 호환 import |
| pretrain의 동결·optimizer·LR 설정, transfer의 모델 복원·동결 범위·학습 모드 | `vision_encoder/setup.py` | 함수 본문 그대로 이동 |
| `utils/config/vision_encoder_config.py` | `vision_encoder/setup.py`에서 재사용 | 원본 설정 클래스·검증 함수 import |
| 기존 커넥터 설정·생성자 | `connector/setup.py` | 동일한 모델 클래스 재노출 |
| 기존 공동 설정·언어 모델·TinyLlama 로더 | `llm/setup.py` | 동일한 설정·모델·로더 재노출 |
| 기존 토큰·마스크 / 언어 시퀀스·파라미터 통계 | `connector/utils/`, `llm/utils/` | 원본 보조 함수 재사용 |
| 기존 공동 모델·설정 로더·optimizer | `joint/setup.py` | 기존 객체 재사용 |
| `utils/training/morai_transfer_objective.py` | `vision_encoder/losses/morai_transfer.py` | 복사, 원본 유지 |
| `utils/evaluation/morai_transfer_metrics.py` | `vision_encoder/utils/morai_transfer_metrics.py` | 복사, 원본 유지 |
| `training_helper.py`의 배치·장치 이동·subset + joint의 stride/balanced 선택 | `common/batching.py` | 함수 본문 그대로 이동 |
| helper의 autocast | `common/precision.py` | 함수 본문 그대로 이동 |
| helper의 시드 고정 + joint의 RNG 보존 | `common/reproducibility.py` | 함수 본문 그대로 이동 |
| joint의 CPUAdamW·파라미터 그룹·cosine LR·optimizer 생성 | `common/optimizers.py` | 함수·클래스 본문 그대로 이동 |
| helper와 joint의 체크포인트·warm start·해시 | `common/checkpoints.py` | 본문 유지, 새 소스 경로 검증 목록 반영 |
| helper의 gradient 진단 + 기존 telemetry | `common/logging.py` | 함수 이동·기존 telemetry import |
| 기존 `train_visual_morai.sh`, `train_visual_morai_object_head.sh` | `vision_encoder/scripts/transfer.sh`, `object_head.sh` | 복사, 기존 경로 유지 |

공통 helper의 최초 경로는 `utils/training.py`이며, `utils/training/training_helper.py` 원본도 유지합니다.
`common/training_helper.py`는 분리된 함수들을 다시 제공하는 호환 파일입니다.

## 검증과 호환성

`training/tests/`는 setup 객체의 동일성, 기존 import 연결, 동결 정책 및 새 소스 파일의 해시 수집을 검사합니다.
기존 학습·체크포인트 테스트는 루트 `tests/`에 그대로 둡니다.
소스 검증 메타데이터는 `utils/common/source_relocations.json`에 기록합니다.
학습 대시보드는 기존 `utils/visualization/training_dashboard/`를 사용합니다.
이번 구조 완성 작업은 `training/`과 승인된 소스 검증 메타데이터만 수정합니다.
