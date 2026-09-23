# VLDrive 논문 명시사항 수정 및 미공개 구현 선택 보고 — 2026-09-18

## 범위와 근거

사용자 요청에 따라 논문 명시사항과 다른 부분을 수정했다. 논문에 없는 개선안을
저자의 방법인 것처럼 적용하지 않았다. 이전 측정은
[수정 전 감사](VLDRIVE_PAPER_AUDIT_20260918.md), 이번 결과는
[`vldrive_paper_fixes_20260918`](../../outputs/logs/vldrive_paper_fixes_20260918/)에 보관한다.

기준은 사용자 첨부 `C:/Users/autonav009/Desktop/VLDrive.pdf`의 §3.1–3.6,
식 (1)–(13), §4.1과 공식 부록 §3.2다.
[본문](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VLDrive_Vision-Augmented_Lightweight_MLLMs_for_Efficient_Language-grounded_Autonomous_Driving_ICCV_2025_paper.pdf),
[부록](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_VLDrive_Vision-Augmented_Lightweight_ICCV_2025_supplemental.pdf).
실제 읽은 본문·부록 사본과 원문 hash는 이전 감사 출력 폴더에 있다.

## 수정 1: 학습 구간 내부의 sampling interval = 2

부록 §3.2의 2프레임 sampling interval을 반영했다.

- `configs/vldrive_train_navigation.json`, `configs/vldrive_train.json`:
  `frame_sampling_interval: 2`를 명시했다. `train_window_stride`와 별도 설정이다.
- `NavigationCacheDataset`, `MoraiExpertVLDriveDataset`:
  원본 21프레임 구간에서 index `[0,2,...,20]`의 11프레임을 선택한다.
  source가 다른 dataset도 설정된 interval을 사용하며 로더의 기본값 1은 기존 호출과의 호환용이다.
- 샘플링 전 전체 source 구간의 연속성, split, instruction 유효성을 확인한다.
  NavigationLabels는 건너뛴 프레임의 invalid 상태·누락·timestamp gap·지시 단절도 검사한다.
  기존 max sensor gap 150ms를 200ms 입력 간격 때문에 완화하지 않는다.
  각 원본 프레임의 100ms 간격을 검사한 뒤 sampling한다.
- MORAI `current_with_history` 실험 경로도 원본 구간에서 최신 instruction의 연속
  suffix를 구한 다음 frame loss mask를 sampling한다. A→B→A에서 B가 건너뛴 프레임에
  있다는 이유로 과거 A에 현재 A의 supervision이 적용되지 않는다.
- text cache 생성기·학습 진입점·감사 도구에 같은 설정을 전달했다.
- 기존 visual feature와 미래 궤적은 프레임별이므로 재사용한다.
  미래 궤적의 시간 horizon `[0.5,1,1.5,2,2.5]`초는 바뀌지 않는다.

현재 MORAI 데이터는 10Hz이므로 11개 입력의 간격은 **200ms**, 이력 범위는 **2초**다.
논문 데이터 자체의 raw Hz가 동일하다고 주장하지 않는다.

새 navigation text cache는 `data/processed/navigation_tokens/basic_drive_morai_interval2`에
생성하고 학습 config도 연결했다. 기존 `basic_drive_morai` text cache는 보존했다.
새 유효 window는 train **1,857**, validation **371**개이며,
15 epoch에 필요한 unique text **4,778**개, 최대 길이는 **32 token**이다.
window stride 5 및 마지막 유효 window 포함 정책을 적용한 실제 epoch당 선택 수는
train **373**, validation **75**개다.
clip의 시간 범위가 길어져 지시 변경에 걸리는 window가 더 제외되는 것은 의도된 결과다.

## 수정 2: 식 (8)/(13)의 Fi를 원본 encoder 특징으로 사용

논문 §3.1과 부록 §3.1은 Fi를 고정된 visual encoder 출력으로 정의하고,
식 (13)은 Fi와 복원 특징의 오차를 사용한다. 기존 구현은 이 대신 학습 가능한
`visual_input_norm(Fi)`를 target으로 사용했다. `detach()`해도 LayerNorm 파라미터가
업데이트되면 target 값이 바뀐다.

이번 수정은 **공개된 Fi 정의와 식 (13)에 직접 맞추는 것**이다. 저자 코드 내부의
정규화 위치까지 확인했다는 뜻은 아니다. 이전 감사에서 정규화 정책 확인 필요로
분류했던 부분을 원문 Fi 정의에 맞춰 정리했다.

- `model.reconstruction_feature_space: encoder`를 config에 명시했다.
- 식 (8)의 원본 특징 자리와 식 (13)의 reconstruction target에 동일한 raw Fi를 사용한다.
- target은 `Fi.detach()`이며, connector의 학습 가능한 LayerNorm과 독립적이다.
- 기존 `normalized` 방식은 명시적인 실험 옵션으로 남겼다. Stage 2의
  `paper_independent`/`paper_objective_st_probe` 설정에서는 허용하지 않는다.
- 선택 query/K,V 경로의 LayerNorm, Q-Former 구조·초기화는 바꾸지 않았다.
  이들의 상세 위치·정책은 공개 논문에 없어 아래 독립 구현 선택으로 보고한다.
- loss 계수 10/1, token/channel 합산, R=0.3, Z=10, 4-layer 복원기,
  DDIA 및 주행 L1 목적함수는 유지했다.

이 변경은 학습 목적함수의 target과 window 구성을 바꾸므로 기존 학습과 동일한
이어 학습으로 취급하면 안 된다. 이번에는 checkpoint를 수정하거나 optimizer를 실행하지 않았다.

## 논문 미공개 세부사항과 남은 문제

| 우선순위 | 현재 구현 선택 / 관측 | 왜 문제가 될 수 있는가 | 이번 처리 |
|---|---|---|---|
| 높음 | Hard gather한 retained feature × scalar ST mask → LayerNorm | LayerNorm이 scalar 변화 영향을 상쇄해 waypoint loss에서 selector로 가는 gradient가 매우 작음. reconstruction 경로는 연결됨 | 임의 ST bypass·top-k를 켜지 않음. 재측정 결과와 함께 보고 |
| 높음 | Gumbel train, argmax eval, 최소 1개 token rescue; temperature=1 | 논문은 Gumbel 사용만 명시하고 이 세부정책을 공개하지 않음. 초기 train/eval 선택 수 차이가 크고 pruning 목표 비율이 보장되지 않음 | 초기 측정은 수렴/학습 후 collapse 증거와 구분. 정책 유지 |
| 높음 | 복원 오차의 channel 합산 및 batch/frame 평균 | 식 (13)은 token 합을 명시하지만 channel/batch reduction의 코드 세부는 공개하지 않음. 현재 raw-feature scale에서 복원 gradient가 pruning/task 신호를 압도할 수 있음 | 임의 평균화·loss weight 변경 없이 수치 보고 |
| 중간 | BERT에서 attention 등을 초기화하지만 query 전용 FFN은 random | Q-Former의 pretraining/초기화 recipe 미공개. 사전학습 FFN을 활용하지 못해 초기 학습에 영향을 줄 수 있음 | 복사 초기화를 임의 적용하지 않음 |
| 중간 | Q-Former의 사용하지 않는 일반 text FFN도 trainable | 72 tensor, 56,687,616 parameter가 전체 loss에서 gradient 없음. CPU AdamW FP32 master 약 216.25MiB 낭비 | 이번에는 보고만. optimizer 제외는 출력 의미를 바꾸지 않는 후속 정리 후보 |
| 중간 | 각 frame의 마지막 valid visual token을 waypoint MLP readout으로 사용 | 논문은 LLM feature 뒤 MLP만 설명. 선택 mask에 따라 마지막 token의 위치/종류가 달라져 readout이 불안정할 수 있음 | 독립 설계 선택으로 명시; 별도 pooling/readout 비교 필요 |
| 확인 필요 | CCDP/MEFA 입력의 추가 trainable LayerNorm, projection의 LayerNorm, 복원기의 추가 position embedding | 논문에 정확한 위치·구조가 없음. feature scale·selector gradient에 영향. 복원 target 변경만으로 이 경로가 해결되지는 않음 | 유지하고 보고 |
| 평가 한계 | sequence_length=11, 매 window의 memory reset, 단일 GPU/CPU optimizer, MORAI 한 바퀴 시간 분할 | 논문 Z=10은 memory 크기이며 학습 clip 길이 T=11을 의미하지 않음. 현재 초기 frame들은 충분한 history가 없고 긴 문맥/새 주행의 일반화를 확인할 수 없음 | 자체 실행 설정으로 구분 |

Navigation v2의 회전교차로·고속도로 출구·신호등 참조 의미 오류는 별도의 알려진
학습 데이터 문제다. 이번 변경은 instruction을 scene에 연결하는 시간축을 수정했으며
문장의 의미 오류를 고쳤다는 뜻은 아니다.

## 검증

- 모델·수식·navigation·두 dataset 경로 관련 **134개 테스트 통과**.
  [JUnit 결과](../../outputs/logs/vldrive_paper_fixes_20260918/regression.xml).
- 새 독립 식 (13) 검사는 값이 알려진 encoder feature에서 직접 합산한 정답과
  loss를 비교한다. LayerNorm weight/bias를 바꿔도 target이 원본과 정확히 같은지 확인한다.
- sampling 검사는 feature/trajectory/timestamp/instruction의 동일 장면 연결,
  건너뛴 source 프레임의 지시 변경·누락·시간 gap 거부, split 경계를 포함한다.
- 실제 전체 모델의 11프레임 입력에서 시간 간격은 전부 **200ms**, history span은 **2초**였다.
  복원 target과 encoder 원본의 최대 절대 차이는 **0**, target의 `requires_grad=False`를 확인했다.
  [실측 JSON](../../outputs/logs/vldrive_paper_fixes_20260918/model_audit.json).

### 수정 후에도 남는 selector 학습 문제

새 config로 초기화한 실제 모델, 실제 train window 0, 동일 Gumbel seed 902에서
각 loss만 따로 backward했다. 모든 측정에서 gradient는 finite다.

| 항목 | loss 값 | CCDP parameter gradient L2 |
|---|---:|---:|
| Waypoint L1 | 7.970661 | **6.8344e-11** |
| 10 × pruning | 0.393608 | 2.290587 |
| Reconstruction | 19,559.603516 | **97,861.669243** |
| 전체 | 19,567.968750 | 97,859.878617 |

복원 gradient / weighted pruning gradient 비율은 **42,723배**다.
원본 특징 복원으로 맞췄다고 해서 selector의 주행 학습 신호 문제가 해결된 것은 아니다.
이번에는 sampling과 reconstruction target이 함께 바뀌었으므로 이전 수치와의 차이를
target 수정 하나의 효과로 해석하면 안 된다.

초기 train 선택 평균은 **52.55 / 106** token, 서로 다른 5개 window의 초기 argmax eval
평균은 **2.55 / 106** token(55 frame occurrence, 범위 1–10)이다.
학습 후 수렴/붕괴를 측정한 것이 아니며, optimizer update는 **0회**다.
사용하지 않는 일반 text FFN의 trainable parameter **56,687,616개**도 재확인했다.

Training entrypoint preflight도 **통과**했다.
[결과 JSON](../../outputs/logs/vldrive_paper_fixes_preflight_20260918/preflight.json):
trajectory shape `[1,11,5,2]`, future→past 최대 변화 **0**, history dependency 변화
**0.076171875**, 실제 text cache hit **4** / runtime tokenization **0**,
optimizer update **0회**다. 이는 연결·인과성 검사이며 위의 waypoint→selector 학습
신호 문제나 실제 주행 성능을 해결/검증했다는 뜻은 아니다.

## 재현

AIMDrive 루트, 기존 WSL lmdrive 환경:

```bash
export PYTHONPATH="$PWD/VLDrive/.runtime/stage2_wsl_deps:$PWD/VLDrive${PYTHONPATH:+:$PYTHONPATH}"
python VLDrive/tools/cache_navigation_text_tokens.py \
  --config VLDrive/configs/vldrive_train_navigation.json
python VLDrive/tools/audit_vldrive_paper.py \
  --config VLDrive/configs/vldrive_train_navigation.json \
  --output VLDrive/outputs/logs/vldrive_paper_fixes_repeat
python VLDrive/train.py vldrive \
  --config VLDrive/configs/vldrive_train_navigation.json \
  --output VLDrive/outputs/logs/vldrive_paper_fixes_preflight_repeat \
  --mode preflight
```

감사/preflight는 새 output 경로를 요구하며 optimizer update는 수행하지 않는다.
