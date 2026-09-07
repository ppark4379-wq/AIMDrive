# MORAI 소규모 학습·검증

4 Camera + LiDAR → LMDrive visual encoder → route-conditioned CCDP → causal
MEFA/Q-Former → pretrained TinyLlama/DDIA → 미래 XY·speed를 학습하는 실험 코드다.
`/Ego_topic`은 학습 teacher/GT 생성·검증에만 사용한다. 최종 구조나 주행 성능을
확정한 상태는 아니며, visual encoder 정책을 같은 데이터로 비교할 수 있다.

| 작업 | 안내 |
|---|---|
| MORAI 연결, 수집·동기화·teacher 검증 | [수집 절차](docs/COLLECTION.md) |
| 전체 목표 구조, preflight·학습·checkpoint 재개 | [Goal stack](docs/STAGE6H_GOAL_STACK.md) |
| Git 포함·제외 기준 | [파일 범위](docs/GIT_SCOPE.md) |

## 환경과 데이터 준비

이 폴더는 팀 저장소의 `VLDrive/experiments/morai_stage6` 위치에 둔다. 실제 backend는
상위 `VLDrive/vision_encoder`의 custom timm과 `VLDrive/LAVIS`를 사용한다.
일반 PyPI timm으로 교체하지 않는다. 기존 `AIMDrive/MoraiDrive`에는 통합하지 않았다.

```bash
cd VLDrive/experiments/morai_stage6
conda env create -f environment_goal_stack.yml
conda activate morai_goal_stack
export PYTHONDONTWRITEBYTECODE=1

# 실제 LMDrive r50 checkpoint의 경로를 지정한다. SHA256과 LM revision을 검증한다.
python scripts/prepare_goal_assets.py \
  --vision-checkpoint /path/to/vision-encoder-r50.pth.tar --download-language
```

데이터셋·pretrained weight·가상환경·실험 출력은 Git에 포함하지 않는다. 데이터는
별도 전달받거나 [수집 절차](docs/COLLECTION.md)로 생성한다. `configs/*.json`의
`dataset`은 이 실험 폴더 기준 경로, `split_file`은 dataset 기준 경로다. 제공 설정의
`diagnostics/stage6f_calibration_20260906/dataset_axis_v2`는 기존 실험의 로컬 위치이므로
새 데이터에는 두 값을 변경해야 한다. run별 split과 teacher·route·feature manifest를
포함한 완성된 Stage6E 데이터가 필요하다.

## 검사와 실행

```bash
# 데이터·weight 준비 후 실제 센서/LM gradient와 causal preflight.
python scripts/train_goal_stack.py --config configs/goal_stack_small.json \
  --mode preflight --output reports/my_goal_preflight

# 기본 raw 입력, visual fusion 학습, LM 마지막 층 학습, train 8 / val 16 windows.
python scripts/train_goal_stack.py --config configs/goal_stack_small.json \
  --output reports/my_goal_training
```

`output`에는 항상 새 폴더를 지정한다. 기존 로컬 설치에서는 환경 생성 대신
`.runtime/stage6h_env/bin/python`을 사용할 수 있다. 새 clone에는 이 private 환경이 없다.

단위 테스트·visual probe·ablation과 이전 실험 도구는 Git에서 제외하고 로컬에 보존한다.
`goal_stack_small.json`은 fusion, `goal_stack_frozen_visual.json`은 frozen,
`goal_stack_all_features.json`은 RGB/LiDAR backbone까지 학습하는 전체 모델 설정이다.
소규모 두 주행의 결과만으로 정책·일반화 성능을 확정하지 않는다.
현재 파일 구성과 기존 checkpoint 재개 조건은 [파일 범위](docs/GIT_SCOPE.md)를 참고한다.
