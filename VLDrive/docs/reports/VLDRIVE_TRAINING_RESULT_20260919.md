# VLDrive 학습 완료 분석 및 best checkpoint offline 추론

## 학습 결과

분석 대상은 `outputs/logs/vldrive_20260919_182054_301289`이다. `report.json`, `epoch_metrics.jsonl`, 5,595행의 `training.jsonl`, 실제 학습 설정과 split을 읽었다. 학습은 **15 epoch / 5,595 update**를 완료했다.

**Best는 마지막 epoch가 아니라 epoch 10 / update 3,730이다.** `train.py`는 validation FDE가 가장 작은 checkpoint를 `model-best.pt`로 저장한다. 파일 크기는 2,712,974,099 bytes, 저장 sidecar의 SHA256은 `7b369b0349ede0c2f8511e677ea53b4dba73a498fde6dfebf2aacea580c022c8`이며 추론 로더에서 실제 파일과 다시 검증한다.

| 검증 지표 | 학습 전 | Best: epoch 10 | 마지막: epoch 15 |
|---|---:|---:|---:|
| ADE: 5개 미래 지점의 평균 위치 오차 [m] | 14.0913 | 2.0334 | 2.1373 |
| FDE: 2.5초 뒤 최종 위치 오차 [m] | 23.8254 | 3.3778 | 3.6679 |
| 좌표별 waypoint L1 [m] | 7.5386 | 1.2013 | 1.2543 |
| Argmax 유지 token 수 / 106 | 3.7042 | 9.6473 | 11.7430 |

학습 전 대비 best ADE는 85.57%, FDE는 85.82% 감소했다. Best 이후 FDE는 마지막 epoch에서 약 8.59% 커졌다. 추가 학습이 검증 성능 개선으로 이어지지 않았으므로 요청한 best 선택이 타당하다. 이것만으로 과적합 원인을 확정할 수는 없다.

Best 기준은 **FDE**다. ADE만 보면 epoch 6의 1.9646m가 epoch 10보다 낮다. 따라서 "모든 지표에서 최상"이라는 뜻은 아니다.

### Best의 예측 시간별 오차

| 미래 시점 [s] | 위치 거리 오차 [m] | 전후 방향 MAE [m] | 좌우 방향 MAE [m] |
|---|---:|---:|---:|
| 0.5 | 0.7840 | 0.7776 | 0.0645 |
| 1.0 | 1.3842 | 1.3323 | 0.2322 |
| 1.5 | 1.9716 | 1.8468 | 0.4685 |
| 2.0 | 2.6492 | 2.3296 | 0.8618 |
| 2.5 | 3.3778 | 2.8480 | 1.2516 |

먼 미래일수록 오차가 커진다. 특히 전후 방향 오차가 좌우보다 크다. 즉 방향뿐 아니라 "얼마나 멀리 이동할지"에 개선 여지가 크다. 이 수치만으로 속도 입력이나 특정 구조가 원인이라고 단정하지 않는다.

### CCDP

| 학습 로그 구간 평균 | 처음 100 update | 마지막 100 update |
|---|---:|---:|
| Gumbel keep ratio | 49.389% | 33.120% |
| mean p_keep | 0.49405 | 0.32940 |
| p_keep std | 0.00449 | 0.17287 |
| 선택 token / frame | 52.35 | 35.11 |
| L_way | 3.61184 | 0.83111 |
| L_prun | 0.04009 | 0.00309 |
| L_rec | 0.57647 | 0.17489 |
| Total loss | 4.58922 | 1.03693 |

이 표는 학습 batch의 11개 시점에 대한 loss 평균이며, 위 validation 표는 window의 마지막 시점만 평가한다. 따라서 두 표의 L_way/L1을 같은 집계 방식으로 해석하지 않는다.

전체 token을 KEEP하던 문제는 이 학습의 Gumbel 통계에서 관찰되지 않는다. 유지율은 목표 30%보다 약 3.1%p 높고, 확률의 표준편차가 증가해 모두 비슷한 확률을 갖던 초기 상태에서 분화했다. 단, std는 각 update의 전체 window/token에 걸친 값이므로 의미상 중요한 물체를 정확히 선택했다는 증거는 아니다. 이번 기존 로그에는 개별 p_keep/logit histogram이 없어 그 분포를 임의로 복원하지 않았다.

**Gumbel 33%와 argmax 약 9%는 같은 값이 아니다.** 학습은 확률적으로 선택하고, 저장된 validation은 `argmax`로 평가했다. Best validation의 유지 token 수는 평균 9.65/106 = 9.10%, 범위 6~30이다. 마지막 validation은 11.74/106 = 11.08%, 범위 6~36이다. 둘 다 모든 frame이 1개/106개로 붕괴한 상태는 아니지만 argmax가 30% 예산을 만족하는 것도 아니다. 초기 epoch 1~2는 rescue 후 모든 frame이 1개였다.

이번 offline inference는 앞서 요청된 **eval + hard Gumbel**을 사용한다. 따라서 위 validation 수치를 이번 offline 설정의 성능으로 재사용하지 않는다.

### 학습 정상 동작 및 시간

- Update 1~5,595가 연속이며, 주요 loss/retention/gradient/time 값에 NaN/Inf가 없었다. Telemetry error도 없었다.
- `trainable_changed=true`, `frozen_unchanged=true`. Preflight의 gradient 및 causal history 검사 통과.
- 마지막 gradient 기록(update 5,500)의 pre-clip L2: selector 0.3762, MEFA 0.3638, Q-Former 1.8148, projection 0.8421, TinyLlama 2.7294, waypoint head 11.6516. 수치의 크기를 모듈 중요도로 해석하지 않는다.
- 학습 장치: NVIDIA GeForce RTX 5060 Ti / CUDA. Peak allocated GPU memory 5.77 GiB.
- Training loop elapsed 12,071.7초(약 3시간 21분), update 계산 시간 합 11,275.6초, update 중앙값 2.068초. 초기 준비와 최종 checkpoint 저장까지의 전체 시간은 별도이며, run 생성~report 저장 파일 시각으로는 약 3시간 30분이다.

### 평가 범위와 한계

Train 373개 / validation 75개의 마지막-frame anchor로 평가했다. 같은 주행 1회에서 시간을 나누고 미래 horizon 경계를 purge한 split이다. **별도 주행·별도 경로의 일반화 검증이나 MORAI 폐루프 주행 평가는 아니다.** Train epoch 15 ADE/FDE는 1.8545/3.2376m, validation은 2.1373/3.6679m이다. Best를 선택하는 데 사용한 validation을 독립 test라고 부르지 않는다.

### 분석 산출물

- [학습 추세 PNG](../../outputs/logs/vldrive_20260919_182054_301289/analysis_after_training/training_summary.png)
- [학습 추세 PDF](../../outputs/logs/vldrive_20260919_182054_301289/analysis_after_training/training_summary.pdf)
- [전체 집계 JSON](../../outputs/logs/vldrive_20260919_182054_301289/analysis_after_training/training_summary.json)

## 이번 offline inference 설정

`configs/vldrive_offline_inference.json`의 `vldrive_checkpoint`를 위 best 파일로 지정했다. `visualization.show_actual_trajectory=true`를 켜서 기존 audited pose 기반 미래 궤적을 그림에만 겹친다. 실제 궤적이나 미래 sensor는 model input에 넣지 않는다. 가중치, loss, architecture, 학습 설정은 수정하지 않았다.

- Frozen stage-1: 캐시 생성과 같은 `morai_visual_encoder.pth.tar`.
- Stage-2 input: raw에서 재계산한 `[1,11,106,256]` visual features + 고정 navigation instruction.
- Instruction: `Continue straight along the current road.`
- 사용자 확인대로 원본 연속 11개, interval=1. **학습 interval=2와 다름**을 manifest에 기록한다.
- Eval + hard Gumbel, seed=607 한 번 설정, reconstruction branch 비활성화.
- Raw frame 0~150을 모두 읽고, 유효한 anchor 10~150을 모두 추론한다. 그림은 10/30/50/70/90/110/130/150의 8개만 저장한다.
- 이 첫 추론은 초반 약 15초 구간의 동작 확인이다. 고정 instruction과 이력 간격이 학습 평가와 다르므로 전체 경로의 정식 점수와 구분한다.

## 실제 offline 실행 결과

**완료: 141개 window 추론, 8개 anchor × 2종 그림 × PNG/PDF = 32개 파일.** 실제 model-best의 update 3,730, 학습 run metadata, checkpoint 파일/가중치 hash, frozen assets 검증을 통과했다. Manifest의 `status=completed`, `vldrive.loaded=true`, `training_complete=true`다. 모든 history가 원본 연속 11개인지, 저장된 그림 32개의 SHA256이 manifest와 같은지도 확인했다.

실행에는 dataset 검사와 모델 로딩, 파일 저장을 포함해 173.38초가 걸렸다. 추론 구간의 순수 계산 중앙값은 encoder 40.53ms, Stage-2 74.97ms, 두 단계 합 116.48ms였다. 합의 중앙값은 각 중앙값을 더한 값과 다를 수 있다. 이 값은 센서 파일 읽기/전처리/그림 저장/초기화를 제외하므로 실시간 end-to-end 처리 속도로 해석하지 않는다.

| 이번 초반 141개 window의 결과 | 측정값 |
|---|---:|
| 실제 미래 궤적이 확보된 window | 141 / 141 |
| ADE | 2.2424 m |
| FDE | 3.9541 m |
| FDE 중앙값 / 90 percentile | 3.8959 / 5.2650 m |
| Anchor별 Gumbel 유지 token 평균 | 34.70 / 106 = 32.74% |
| Anchor별 유지 token 범위 | 25~47 |
| 2.5초 전후 방향 MAE | 3.0709 m |
| 2.5초 좌우 방향 MAE | 1.7392 m |

위 오차는 **학습에 사용된 주행의 초반 구간, 서로 겹치는 window, 고정 직진 지시, interval=1, Gumbel** 조건에서 실제 주행 기록과 비교한 값이다. Training validation의 75개 held-out anchor/장면별 지시/interval=2/argmax와 조건이 다르다. 두 ADE/FDE의 차이를 모델 성능의 개선/악화로 단정하지 않는다.

### 그림에서 실제 확인한 문제

- **Frame 30:** 2.5초 뒤 전방 위치를 14.75m로 예측했지만 기록상 전방 위치는 19.56m다. 전방 이동량을 약 4.81m 짧게 예측했다. 최종 위치 거리 오차는 4.95m다.
- **Frame 110:** 실제 기록은 2.5초 뒤 왼쪽 3.47m, 전방 9.14m인데 예측은 오른쪽 0.26m, 전방 7.69m다. 실제 좌회전 궤적을 따라가지 못한 결과이며 최종 오차는 3.99m다. 다만 이 frame에도 `Continue straight along the current road.`를 입력했으므로 **실제 좌회전 행동과 지시가 맞지 않는다.** 이 결과만으로 모델이 올바른 좌회전 instruction도 따르지 못한다고 결론 내릴 수 없다.
- **가장 큰 FDE:** Frame 34에서 7.0849m. 초반 직진/감속 구간의 전후 이동량 오차가 크다. 감속 원인이나 입력 이력 간격의 기여도는 이번 실행으로 분리하지 않았다.

현재 결론은 **학습과 실제 best inference 연결은 정상 작동하고 학습 오차도 크게 감소했지만, 이동량 예측과 지시/실제 행동 일치 여부를 추가로 확인해야 한다**는 것이다. 전체 경로 주행 성능이 검증됐다는 결론은 아니다.

다음 성능 비교에서는 동일 held-out anchor와 기존 장면별 navigation label을 사용하고, 먼저 학습 간격 2에 맞춘 결과를 비교하는 것이 좋다. 지금 기본 config의 interval=1은 사용자가 명시적으로 선택한 값이므로 그대로 유지했다. 학습이나 loss를 추가로 바꾸지 않았다.

### 실제 결과 파일

- [Frame 30 직진 비교 PNG](../../outputs/logs/vldrive_offline/basic_drive/frame_000030_vldrive_overview.png)
- [Frame 110 좌회전 구간 비교 PNG](../../outputs/logs/vldrive_offline/basic_drive/frame_000110_vldrive_overview.png)
- [Frame 110 비교 PDF](../../outputs/logs/vldrive_offline/basic_drive/frame_000110_vldrive_overview.pdf)
- [Inference manifest](../../outputs/logs/vldrive_offline/basic_drive/inference_manifest.json)
- [모든 prediction JSONL](../../outputs/logs/vldrive_offline/basic_drive/waypoints.jsonl)
- [사후 오차 분석 및 frame별 실제 궤적 JSON](../../outputs/logs/vldrive_offline/basic_drive/inference_analysis.json)

기존 offline 테스트의 미지정 checkpoint fixture를 사용자 선택값과 분리했으며, **20개 테스트가 모두 통과**했다. 이번 분석에만 사용한 임시 스크립트는 삭제하고 보고서/집계/그림을 남겼다. 기존 inference 결과 보호 동작은 유지하므로 같은 명령을 다시 실행할 때는 `output.dir`을 새 경로로 지정해야 한다.
