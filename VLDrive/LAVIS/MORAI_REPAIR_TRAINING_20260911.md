# Frozen epoch25 encoder를 유지한 후속 모델 수정 실험

사용자 승인에 따라 vision encoder를 다시 학습하지 않고 마지막 epoch25 checkpoint와 기존 frozen token cache를 재사용한다.
원본 dataset, 기존 instruction JSONL, 이전 학습 결과는 보존한다. 수정·신규 도구는 LAVIS에 한정된다.

이 경로는 **논문 동일 재현이 아니라, 문제 진단에 기반한 명시적 MORAI 수정 실험**이다.
기존 `morai_paper_last_encoder.json`의 기본 모델 동작은 유지하고 새 config에서만 변경을 켠다.

## 변경 사항

- Global path의 앞쪽30m에서 좌·우회전을 미리 안내한다. 현재 pose와 저장된 route만 사용하며 미래 ego GT는 사용하지 않는다.
- 실제 turn metadata가 없지만 도로가 굽으면 앞쪽20m의 route 접선 변화20°를 기준으로 좌·우 굴곡을 설명한다.
- 지시 변경 전 history도 보존한다. 현재 지시를 입력하고, 이전 지시 구간의 waypoint 예측은 `frame_loss_mask`로 loss에서 제외한다.
- Train을 정지·직진·좌곡선·우곡선 네 범주로 균형 재추출한다. 실제 미래 ego GT는 train sampling과 loss에만 사용한다.
- 매 full epoch640개 draw를 seed+epoch로 다시 추출한다. 중복 clip은 새로운 주행 사건으로 계산하지 않는다.
- 기존 후속 모델 가중치를 명시적으로 이전하되, 포화된 `connector.ccdp.`만 새로 초기화한다. optimizer/RNG는 이전하지 않는다.
- Selector 학습률은 기본1e-5의10배다. 이전 모델에서 사용된 TinyLlama·MEFA·Q-Former·waypoint head는 재사용 후 학습한다.
- CCDP reconstruction은 dropped element mean을 사용하고, loss를 가중하는 `(1-M)`에만 stop-gradient를 적용한다.
  Eq.8 및 retained-query를 통한 reconstruction→selector gradient는 유지한다.
- 단순 정규화만으로는 toy에서 재붕괴하여, `BCEWithLogits(keep_logit-drop_logit, target=.3)` 안정화 보조항을 weight100으로 추가했다.
  이는 논문 Eq.12를 대체하지 않는 **실험적 추가 loss**이며, token 확률을0.3 쪽으로 모아 점수 대비를 줄일 수 있는 tradeoff가 있다.
- 추론은32/106 고정 budget top-k다. 이는 **학습된 동적 유지율30%를 달성했다는 뜻이 아니다.** 학습은 Gumbel이며 유지율·확률·포화율을 따로 기록한다.

## 짧은 검증 후 전체 학습

처음에는 각 범주8개, 총32개 train clip으로 최대640 update까지 overfit 검사한다.
160 update마다 다음 gate를 모두 검사하며 실패하면 전체 학습을 시작하지 않는다.

- 좌/우 곡선 ADE≤1.5m이고 시작 모델보다20% 이상 감소.
- 좌/우 각각80% 이상의 clip에서 올바른 좌우 부호와 최소0.5m 횡이동 예측.
- 최근64 update의 실제 Gumbel 유지율 및 soft 유지율15–50%.
- keep확률>.999인 token 비율≤5%.

Gate는 강제 top-k 개수를 성공 근거로 사용하지 않는다. Pilot 통과 시 전체 train을15 epochs×640 updates 학습한다.
Train/purge/validation 경계는 그대로이며, 이 overfit 실험의 학습·gate·모델 선택에는 validation을 사용하지 않는다.
전체 train 성능은 중복 없는 chronological stride5 목록670개로 따로 평가한다.

제한: 작은 pilot의 성공은 전체 경로 성공이나 simulator closed-loop 안전 주행을 보장하지 않는다.
Vision encoder의 정보가 부족한 것으로 후속 검증에서 드러나면, 그때 encoder 추가 학습을 별도 판단한다.

## 검증 및 실행

실제106×256 token connector CPU toy에서 detach+mean만으로는 유지율97.5%로 재붕괴했다.
Logit 안정화 weight100에서는 마지막20 update의 sampled 유지율37.52%, soft36.80%를 관측했고 reconstruction mean은2.004→1.060으로 감소했다.
원본 모델과 다른 작은 Q-Former를 사용한 CPU 안정성 실험이며 MORAI 성능 결과가 아니다.

통합 회귀 테스트63개 통과. Instruction 생성기 관련 추가 tests도 통과했다.
학습 실행 중에는 checksum 대상 모델·dataset·runner source를 변경하지 않는다.

실행:

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive/LAVIS
PYTHONDONTWRITEBYTECODE=1 LAVIS_IMPORT_PROFILE=morai \
/opt/miniforge3/envs/vldrive-ve/bin/python tools/morai/launch_repair_training.py
```

현재 실행: `outputs/morai_repair_epoch25_20260910T192642Z`.
진행 상태: [progress.json](outputs/morai_repair_epoch25_20260910T192642Z/progress.json).
Config: [morai_repair_epoch25.json](lavis/projects/vldrive/morai_repair_epoch25.json).
진입점: [train_morai_repair.py](train_morai_repair.py).

진행 로그는 `.runtime/training_runs/<run>/pipeline.log`에 기록된다.
Pilot 실패는 `pilot_gate_failed`로 명시하고 모델/optimizer checkpoint를 남긴다.
Full 학습은200 update 간격의 두 개 교대 checkpoint와 epoch별 model-last를 저장한다.
추가 보조 손실의 최소값이 약61.1이므로 전체 loss 숫자만 이전 실험과 비교하지 않는다. Waypoint·reconstruction·retention을 분리해서 본다.

Full 단계가 재부팅 등으로 중단되면 기존 output을 덮어쓰지 않고 복구한다:

```bash
/opt/miniforge3/envs/vldrive-ve/bin/python resume_morai_repair.py \
  --checkpoint outputs/<기존_run>/checkpoint_slot0.pt \
  --output outputs/<새_resume_run>
```

실제 저장된 slot0/1 중 무결성 검사를 통과한 최신 snapshot을 선택해야 한다.
복구 CLI는 통과한 pilot gate, 동일 코드·설정·encoder/cache 해시를 요구하고 모델·optimizer·RNG·epoch 내 sampling 위치를 복원한다.
복구 계약과 중복 실행 잠금 CPU tests11개 통과. 실제 GPU 강제 중단/재복구 시험은 하지 않았다.
