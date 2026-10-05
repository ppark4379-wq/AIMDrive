# Reconstruction-only 추가 학습 결과

LLM/waypoint loss 없이, 복원 모듈만 추가 학습해 reconstruction loss를 낮췄다.
CCDP 선택기, 입력 정규화, MEFA, Q-Former 및 language projection은 고정했다.
모델 구조는 변경하지 않았다. 이전 검사에서 보인 낮은 복원 품질은
적어도 상당 부분 추가 학습으로 개선될 수 있었다. 원인을 특정 구조 결함으로 단정하지 않는다.

## 조건

- 시작 가중치: `outputs/logs/connector_mefa_dynamic_anneal/model-last.pt`
- 학습 대상: `reconstruction_align`, `reconstruction_transformer`, `mask_embedding`,
  `reconstruction_position_embedding` (변경된 state key 54개)
- 목표: 기존과 동일한 encoder 특징 공간의 제거 토큰·채널 평균 MSE
- 고정된 argmax 선택 및 정상 과거로 만든 현재 프레임 connector 출력을 캐시해 사용
- 학습: 기존 4개 주행의 32개 현재 프레임, batch 4, AdamW lr=3e-4, weight decay=.01
- 200 epoch / 1600 update, seed 607. 검증은 5 epoch마다 수행 (첫 epoch 포함).
- 기존 2개 validation 주행/16개 현재 프레임의 Lrec로 195 epoch를 선택
- 이전에 이미 검사한 20개 주행/160개 프레임은 epoch 선택에 사용하지 않고
  최종 가중치로 한 번 재평가. 완전히 새 독립 test set으로 부르지 않는다.

## 결과

| 평가 집합 | 추가 학습 전 | 추가 학습 후 | 학습 위치별 평균 기준 |
|---|---:|---:|---:|
| 기존 validation 2개 주행/16프레임 | 0.31801453 | **0.03101666** | 0.08740759 |
| 이전 검사 20개 주행/160프레임 | 0.31407523 | **0.02610798** | 0.08691510 |

Validation 오차는 약 90.2%, 20개 주행 오차는 약 91.7% 감소했다.
단순 평균 기준보다 낮은 오차를 보인 프레임은 validation 16/16,
20개 주행 157/160이었다. 모든 조건에서 동일한 프레임의 동일한 제거 위치를 평가했다.

입력 정보 활용도 다시 확인했다. 같은 모델에서 connector 출력값만 0으로 바꾸면
validation 오차는 0.13029208, 20개 주행 오차는 0.13459849로 증가했다.
따라서 이번에는 위치별 평균보다 잘 복원하면서 정상 입력의 이점도 보였다.

## 해석

복원 모듈을 MEFA 평가에 쓰기 위한 기초 기준이 이전보다 강화됐다.
단, 작은 학습 집합에서 단일 seed로 수행했고, 기존에 확인한 주행을 사용했으므로
일반적인 복원 성능의 충분한 검증이나 실제 주행 성능 향상을 주장하지 않는다.
이전 복원 모듈의 MEFA 과거 유무 delta는 새 복원 모듈에 그대로 적용할 수 없다.
새 복원 모듈로 과거 유무를 다시 평가해야 한다.

## 산출물과 검증

`outputs/logs/reconstruction_finetune_v1/`:

- `model-best.pt`: 195 epoch의 inference용 전체 connector state, optimizer 미포함
- `model-last.pt`: 200 epoch state와 reconstruction optimizer/RNG
- `result.json`: train/validation/20개 주행 결과 및 입력 0 비교
- `metrics.jsonl`, `learning_curve.png`: 학습 곡선
- `experiment.json`: 학습 설정
- `verification.json`: strict reload 후 validation 결과 재현, 변경된 state key 목록

저장된 best checkpoint를 새 모델에 strict load하여 validation loss를 재현했다.
이전 체크포인트와 모든 state tensor를 비교해 복원 관련 54개 key만 바뀐 것을 확인했다.
기존 가중치는 보존했다. 원본 센서 데이터는 변경하지 않았다.

실행기: `training/connector_only/train_reconstruction.py`.
실행 폴더가 이미 있으면 덮어쓰지 않고 중단한다. 다른 실험을 하려면 별도 출력 경로를 지정하도록
실행기를 수정하거나 분리해야 한다. 일반 connector/LLM 학습 실행기를 대체하는 스크립트는 아니다.
