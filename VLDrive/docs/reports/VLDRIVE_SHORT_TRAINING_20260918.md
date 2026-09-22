# VLDrive 실제 40-update 진단 학습 — 2026-09-18

## 결과

**실제 optimizer update 40회를 완료했다. 궤적 오차는 감소했지만, 평가 시 토큰을
거의 전부 남기는 상태가 관측됐다. 현재 결과로 본학습 확대를 권하지 않는다.**

이는 초기 forward/backward 검사만 수행한 이전 결과와 다르다. 학습 가능한
파라미터의 hash 변경과 고정 파라미터 hash 유지를 모두 확인했다.

## 실행 조건

- 모델·loss: `vldrive_train_navigation.json`의 현재 논문 기준 구현 그대로.
  TinyLlama 전체 decoder + CCDP/MEFA/Q-Former + reconstruction + waypoint head.
  ST 우회 경로, top-k, 복원 mask detach, loss 평균화 등의 실험 옵션은 활성화하지 않음.
- 고정 visual encoder의 실제 `basic_drive_morai` 캐시와 navigation token cache를 사용.
- 입력: 11프레임, 원본 2프레임 간격, 현재 데이터 기준 2초 이력.
- AdamW: lr=1e-5, weight decay=.06, CPU FP32 master/state, BF16 연산.
  원래 5,595-update cosine schedule을 유지하여 40회 만에 학습률이 0이 되지 않도록 함.
  전체 gradient clipping=10도 기존 trainer 설정 그대로 사용.
- seed=607. 40 update의 마지막 로그 시각은 약 254초이며, 중간 평가를 포함하고
  모델 로딩·최종 평가·체크포인트 저장 시간은 제외한 값이다.
- PyTorch peak allocated GPU memory 약 **5.76GiB**. 운영체제/다른 프로그램의
  GPU 메모리까지 합한 수치는 아니다. 학습 중 확인한 WSL CPU 메모리는 약 19GiB,
  swap 사용은 0이었다.

## 데이터 선정과 평가 한계

알려진 navigation v2의 교차로·신호등·고속도로 출구 의미 오류를 학습 진단과
섞지 않기 위해 일반 직진/도로 따라가기 문구(rule ID 38, 42)만 사용했다.

자동 선정 조건:

- 원본 label의 시간적 유효성 및 split 검증 통과.
- 입력 이력의 ego heading 및 기록된 route의 방향 변화 ≤10도.
- 각 입력 프레임의 실제 미래 궤적 lateral 절대값 ≤1.25m.
- 마지막 입력의 2.5초 미래 forward 거리 ≥3m, 후진 궤적 제외.
- 선택 window 시작점 간격 ≥4.6초: 2초 입력 + 2.5초 미래 정답의 중복을 피함.

최대 train 16 / validation 8개를 요청했으나 조건을 통과한 비중복 구간은
**train 9개 / validation 3개**였다. 전부 기존 데이터에서 선택했고 정답이나
지시문을 새로 만들지 않았다. 각 구간의 epoch 0 문구를 고정하여 비교했다.
이것은 geometry 기반 자동 일치 검사이며 전체 장면의 수동 의미 검증은 아니다.

**검증 구간은 학습에 넣지 않았지만 같은 한 바퀴의 다른 시간 구간이다.**
회전, 정지, 장애물 회피, 새로운 경로, MORAI 폐루프 주행 성능의 검증은 아니다.
고정된 문구와 직진만 사용했으므로 LLM의 다양한 지시 이해가 좋아졌다는 증거도 아니다.

## 같은 validation 3개 구간의 변화

| Optimizer update | Waypoint L1 (m) | ADE (m) | FDE (m) | 평균 유지 token / 106 |
|---:|---:|---:|---:|---:|
| 0 | 7.3870 | 14.3136 | 25.1020 | 4.5758 |
| 10 | 5.8873 | 11.4945 | 20.7510 | 22.5455 |
| 20 | 5.4307 | 10.3545 | 18.2776 | 74.1818 |
| 40 | **4.9116** | **9.4795** | **16.6344** | **105.5758** |

Waypoint L1은 미래 다섯 점의 x/y 절대 오차 평균이다. ADE는 다섯 점의
Euclidean 거리 오차 평균, FDE는 마지막 2.5초 점의 거리 오차다.
독립 window 평가에서는 마지막 입력 프레임의 예측만 집계한다.

Validation waypoint L1은 **33.5% 감소**했다. 같은 train 9개 구간의 평가 L1도
7.7681m → 4.2291m로 감소했다. 실제 학습 경로가 동작하고 궤적 오차를 줄이는
초기 변화는 확인했으나, 여전히 절대 오차가 크며 수렴/주행 가능 상태는 아니다.

## 토큰 선택 문제

목표 유지율 R=0.3은 평균 약 **31.8 / 106 token**에 해당한다. 모든 프레임에서
정확히 32개를 강제해야 한다는 뜻은 아니지만, 최종 평가의 평균은 다음과 같다.

- Train: **105.596 / 106**, 범위 102–106.
- Validation: **105.576 / 106**, 범위 103–106.
- Validation의 33 frame occurrence 중 **26개는 106개를 모두 유지**했다.
  33개는 3개 window의 내부 프레임 집계이며 33개의 독립 주행을 뜻하지 않는다.
- 반면 마지막 train Gumbel sample 유지율은 약 **49.66%**였고,
  마지막 validation의 평균 soft keep probability는 약 **0.524**였다.

즉, 학습 Gumbel 선택과 평가 argmax 선택 사이에 큰 차이가 있으며, 이번 40-step
끝에서는 실질적인 token 압축이 거의 사라졌다. 확률이 1로 포화된 것은 아니다.
이 상태가 긴 학습에서도 계속될지까지 확정할 수는 없지만, 무작정 본학습을
늘리기보다 pruning/복원 학습 균형과 train/eval 선택 정책을 먼저 비교할 근거다.

같은 train 9개를 한 번씩 방문한 첫 회전(update 1–9)과 네 번째 회전(28–36)의
로그 평균에서 reconstruction loss는 15,157.59 → 6,266.42로 줄었지만,
pruning loss는 0.03873 → 0.04677로 증가했다. 이들은 각 update의 서로 다른
가중치/Gumbel 상태에서 얻은 값이므로 고정 모델 평가와 구분해야 한다.

이전 감사에서 관측한 복원 gradient 우세와 일관된 현상이지만, 그것만이 원인임을
분리 실험으로 입증한 것은 아니다. 이번에는 학습 후 waypoint-only selector gradient를
별도로 재측정하지 않았다. “전체 loss가 감소했다”를 “중요한 토큰을 잘 고른다”로
해석하면 안 된다.

## 산출물과 재현

- 당시 실행 도구: `short_navigation_training.py` (완료된 일회성 진단이므로 2026-09-19 정리에서 삭제).
- 당시 설정: `vldrive_short_diagnostic.json` (같은 정리에서 삭제).
- [학습 결과 JSON](../../outputs/logs/vldrive_short_diagnostic_20260918/report.json).
- [구간 선정과 실제 문구](../../outputs/logs/vldrive_short_diagnostic_20260918/selection.json).
- [모든 업데이트 로그](../../outputs/logs/vldrive_short_diagnostic_20260918/training.jsonl).
- [0/10/20/40회 평가](../../outputs/logs/vldrive_short_diagnostic_20260918/evaluations.json).
- [40회 모델 가중치](../../outputs/logs/vldrive_short_diagnostic_20260918/model-last.pt),
  [무결성 hash](../../outputs/logs/vldrive_short_diagnostic_20260918/model-last.pt.sha256.json).

체크포인트는 약 2.71GB이며 **모델 가중치만 저장**했다. optimizer/RNG state가
없으므로 정확한 이어 학습용 checkpoint는 아니다. 기존 vision 체크포인트와
본학습 config는 변경하지 않았다. 실제 text token cache hit 92 / 새 토큰화 0회였다.

아래는 당시 사용한 명령의 기록이다. 진단 도구와 config를 삭제했으므로
현재 실행 명령이 아니다. 위 결과 파일과 checkpoint는 보존했다.

```bash
export PYTHONPATH="$PWD/VLDrive/.runtime/stage2_wsl_deps:$PWD/VLDrive${PYTHONPATH:+:$PYTHONPATH}"
python VLDrive/tools/short_navigation_training.py \
  --config VLDrive/configs/vldrive_short_diagnostic.json
```

다음 비교에서는 같은 선정 구간·초기 가중치·seed를 유지하고 수정 후보를 한 번에
하나씩 적용해야 한다. 우선순위는 복원/압축 목적함수의 실효 균형과 selector의
학습·평가 차이다. 현재 40-update 모델을 장시간 이어 학습하는 것은 보류한다.
