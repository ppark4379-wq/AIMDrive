# MORAI 동적 CCDP 학습과 검증

`morai_dynamic.json`은 MORAI의 route·speed 조건을 사용하는 **오프라인 지도학습 설정**이다.
visual encoder는 frozen이며 CCDP·MEFA/Q-Former, TinyLlama 마지막 decoder layer와 norm,
waypoint·speed 예측기를 학습한다. visual encoder 자체를 학습하거나 MORAI 차량을 폐루프로 제어하지 않는다.

## 목적함수와 논문 경로의 구분

현재 MORAI 목적함수는 다음 네 항이다.

```
SmoothL1(미래 XY) + SmoothL1(미래 speed) + 10 × pruning loss + reconstruction mean loss
```

XY와 speed는 시퀀스 마지막 프레임의 0.5–2.5초 teacher를 사용한다. pruning과 reconstruction은
시퀀스의 모든 프레임에서 계산한다. reconstruction은 제거된 토큰의 feature 성분별 제곱 오차를
평균한다. target은 기존 구현의 `LayerNorm(F).detach()`이며 raw encoder feature와 구분해야 한다.
복원 loss가 selector까지 전달되는 gradient는 유지한다.

이는 `train_vldrive.py` + `morai_paper.json`의 수식 재현 경로와 별도다.
논문 경로의 Gumbel-Softmax, Eq.8, Eq.13 sum, 가중치 10/1은 변경하지 않았다.
MORAI 설정의 평균 정규화와 잡음 스케줄을 논문의 공개되지 않은 구현이라고 주장하지 않는다.

## 수정 이유

실제 MORAI feature의 통제 실험에서 sum reconstruction의 크기와 selector gradient가 pruning보다
훨씬 컸고, 모든 106개 토큰을 유지해 복원 loss를 0으로 만드는 상태에 빠졌다.
TinyLlama와 주행 예측기를 제외한 connector 단독 실험에서도 재현됐다.

mean 정규화만 적용하면 학습 Gumbel 선택은 약 32개여도 추론 argmax에서는 1–5개만 남았다.
각 토큰의 keep 확률을 약 0.3으로 맞추는 것으로도 무작위 학습 선택 비율은 만족시킬 수 있지만,
argmax에서는 drop이 우세할 수 있기 때문이다. softmax temperature만 줄여서는 hard Gumbel의
선택 잡음 크기가 줄지 않는다.

`selection_noise_anneal_updates: 64`는 다음과 같이 선택 잡음을 줄인다.

```
noise_scale = max(0, 1 - global_update / 63)  # global_update는 0부터 시작
```

첫 update는 기존 Gumbel, 64번째 이후에는 argmax와 같은 hard 선택에 straight-through gradient를
사용한다. BF16 softmax의 반올림으로 서로 다른 점수가 같은 확률이 될 수 있으므로,
hard 선택은 softmax 이전 점수에서 결정한다. 유지 개수를 top-k로 고정하지 않는다. 처음부터 잡음을 0으로 하는 설정도 별도 실험에서
실패했으므로 warm-up을 생략하지 않는다. 스케줄은 목표 총 update 수와 독립적이며 재개 시에도
저장된 global update로 복구한다. `0`은 기존 Gumbel 학습을 유지하는 호환 설정이다.

## 실행과 실패 판정

완성된 Stage6E 데이터와 독립 run split을 준비한 뒤 다음 진입점을 사용한다.

```bash
python train_morai.py \
  --config lavis/projects/vldrive/morai_dynamic.json \
  --dataset-root <검증된_데이터셋_루트> \
  --asset-root <pretrained_가중치_루트> \
  --output outputs/morai_dynamic_run01
```

`--mode preflight`는 gradient·인과성·teacher 입력 차단 검사다. 선택 비율의 학습 성공까지
판정하지는 않는다. 새 설정은 새로운 초기화와 output에서 시작한다. 기존 collapse checkpoint를
그대로 재개한 효과는 검증하지 않았으며 strict 코드·설정 hash 검사를 완화하지 않았다.

`training.jsonl`에는 잡음 크기, 원본/가중 loss, 학습 중 선택 토큰 수가 남는다.
평가 보고서의 `token_retention`은 **잡음 없는 실제 추론**에서 모든 window/frame 발생 횟수를
기준으로 계산한다. 서로 겹치는 window는 프레임이 중복 집계될 수 있다.

`require_healthy_retention: true`에서는 최종 평가 후 다음 중 하나라도 해당하면 오류로 종료한다.
checkpoint와 보고서는 오류 전에 저장한다.

- 평균 유지 비율이 목표 30%에서 ±15%p를 벗어남.
- 95% 이상의 프레임에서 106개 전부 유지.
- 95% 이상의 프레임에서 1개만 유지.

이 기준은 collapse 검출을 위한 실험 기준이다. 장면별 중요 토큰 선택이나 최종 주행 성능을
증명하지 않는다. 복원 loss 0은 모든 토큰 유지 상태에서도 나올 수 있으므로 단독 성공 지표로 쓰지 않는다.
작은 검증에서는 동일한 진단용 마스크의 전후 복원 MSE도 별도로 비교한다.

독립 주행의 waypoint 성능, 다양한 상황의 토큰 선택 품질, encoder fine-tuning 및 폐루프 주행은
추가 검증 대상이다. 모델이 낮은 loss를 얻었다는 이유만으로 구조 확정이나 대규모 수집을 결정하지 않는다.
