# 로컬 connector 초기 학습 결과 — 2026-10-05

학습된 connector 체크포인트를 생성했다. 그러나 이번 실험에서 CCDP의 유용한
토큰 선택 능력은 검증되지 않았다. 이를 완성된 CCDP나 MEFA 성능 검증 결과로 사용하면 안 된다.

## 실행 조건

- GPU: NVIDIA GeForce RTX 5070, PyTorch 2.7.1+cu128, float32
- Vision encoder: `F:/dataset_label/checkpoint-best-object.pth.tar`
  - 현재 구현에 strict load 성공, 모든 parameter key 일치
  - encoder 고정, 특징 `[106,256]`, Ego 속도는 source time 기준 과거 50ms 이내
- 데이터: `F:/dataset_label`, 원본 변경 없음
  - train: dynamic lap 001/004, static lap 001/002
  - validation: dynamic lap 100, static lap 097
  - train 32개 / validation 16개 시퀀스, 각 시퀀스 11프레임, 원본 frame interval 2
  - connector 기준 주행 분리; encoder 사전학습 주행과의 중복은 미확인
- CCDP·MEFA·Q-Former·projection·복원 모듈 공동 학습, LLM 미사용
- Q-Former: 기본 768차원, 12층, 무작위 초기화
- 학습 10 epoch / 320 optimizer updates, seed 607, AdamW lr=1e-5
- 마지막 프레임의 `10 * Lprun + Lrec`로 학습
- `Lrec`: encoder 공간에서 제거된 토큰·채널 평균 MSE (`mean`)
  - 기존 로컬 MORAI 설정과 동일한 reduction; 논문의 sum 숫자와 직접 비교 불가

## 학습 및 평가 결과

| 항목 | 결과 |
|---|---:|
| 첫 epoch train Lrec | 0.791832 |
| 마지막 epoch train Lrec | 0.331638 |
| 마지막 epoch 학습 중 평균 유지 비율 | 37.53% |
| 마지막 epoch argmax 평가 유지 비율 | 0.9434% (1/106) |

CCDP, temporal encoder, Q-Former, 복원 transformer 모두 유한한 nonzero gradient를 확인했다.
Gumbel 학습에서의 유지 비율과 argmax 평가에서의 유지 비율이 크게 다르다.
모든 검증 현재 프레임에서 argmax는 최소 보존 수준인 1개 토큰만 남겼다.
학습 loss 감소는 있으나 실제 평가 선택 정책은 목표 30%를 충족하지 못했다.

| 평가 조건 | 정상 과거 Lrec | 과거 0 Lrec | 차이 (0 − 정상) |
|---|---:|---:|---:|
| argmax, 1개 유지 | 1.185472868 | 1.185874641 | +0.000401773 |
| top-k 진단, 32개 유지 | 0.315122777 | 0.315188607 | +0.000065831 |

두 과거 조건은 같은 가중치·선택 마스크·선택 인덱스·복원 정답을 사용했다.
검증은 과거 10프레임이 확보된 마지막 프레임만 대상으로 했다.
argmax와 top-k 사이에는 복원 대상이 달라지므로 두 행의 loss 차이를
MEFA 효과나 같은 문제에 대한 성능 개선으로 해석하면 안 된다.

Top-k의 정상 과거 우세는 16개 중 8개 프레임에만 나타났다.
주행별 delta는 dynamic -0.000026982, static +0.000158643으로 방향도 일치하지 않았다.
현재 작은 pilot에서 MEFA의 안정적인 이점을 확인했다고 판단할 수 없다.

## CCDP 점수 순위의 추가 진단

학습된 점수의 top-k와 같은 수의 무작위 토큰을 선택했다. seed 11/23/47을 사용했으며,
복원 대상 난이도가 달라지는 문제를 줄이기 위해 두 마스크에서 **공통으로 제거된 토큰**만 비교했다.

| 선택 방법 | 공통 복원 대상의 평균 MSE |
|---|---:|
| 학습된 점수 top-k | 0.314810065 |
| 무작위 동일 개수 | 0.312549654 |

학습된 점수의 우세는 확인되지 않았다. 무작위 마스크는 학습 완료 모델에 대한
입력 변경 진단이며, 무작위 선택으로 별도 학습한 모델과의 비교는 아니다.

## 저장 위치와 확인

모든 실행 산출물은 `VLDrive/outputs/logs/connector_mefa_local_pilot/`에 있다.

- `model-last.pt`: 학습 완료 connector, optimizer, RNG, 설정, 데이터 hash
- `manifest.json`, `features/`: 실제 encoder 특징과 분리된 주행 목록
- `metrics.jsonl`: 10개 epoch 학습 결과
- `evaluation-last.json`: 원래 argmax 정책의 과거 유무 비교
- `evaluation-topk-diagnostic.json`: top-k 조건 및 무작위 선택 비교
- `run_verification.json`: 파일 hash, 환경, source hash, 재로드 검증

새 checkpoint를 strict reload한 별도 평가에서 마지막 epoch와 동일한 결과를 재현했다.
기존 connector 검사 2개와 추가 검사 4개를 직접 실행해 모두 통과했다
(이 Python 환경에는 pytest가 없어 테스트 함수를 직접 호출).

다음 학습 실험에서는 우선 Gumbel/argmax 선택 비율 불일치를 해결하거나
고정 예산 top-k를 명시적인 정책으로 정의해야 한다. 더 많은 주행과 여러 seed에서
선택 순위의 이점을 확인하기 전에는 이번 가중치를 검증된 selector로 취급하지 않는다.
