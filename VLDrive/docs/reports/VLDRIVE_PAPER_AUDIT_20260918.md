# VLDrive 논문 대비 현재 모델 검증 — 2026-09-18

## 판정

**핵심 모듈과 forward 구조는 구현되어 있다. 그러나 논문과 동일한 재현을 완료했다고
판정할 수 없으며, 현재 설정으로 본 학습을 진행하기 전에 CCDP 학습 신호를 보완·검증해야 한다.**

이전 preflight 통과는 실제 데이터가 연결되고 전체 loss의 역전파가 가능하다는 뜻이다.
주행 loss가 selector를 충분히 학습시키거나, 모든 trainable parameter가 사용되거나,
목표 token 압축률이 학습된다는 뜻은 아니다. 이번 검사에서 이 차이를 실제 수치로 확인했다.

검증 대상은 `configs/vldrive_train_navigation.json`, 지정된 MORAI encoder로 생성한
`basic_drive_morai` feature cache, navigation v2 및 실제 TinyLlama/BERT 자산이다.
검증 도구와 테스트·보고서만 추가했으며 모델 가중치/학습 설정은 변경하지 않았다.
진단 모델의 optimizer update는 0회다.

## 근거와 재현 범위

- 사용자 첨부 `C:/Users/autonav009/Desktop/VLDrive.pdf` 11페이지를 직접 추출했다.
  본문 §3.1–3.6, 식 (1)–(13), §4.1을 대조했다.
  [원문 hash와 경로](../../outputs/logs/vldrive_paper_audit_20260918/paper_source.json).
- [공식 보충자료](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_VLDrive_Vision-Augmented_Lightweight_ICCV_2025_supplemental.pdf)
  §3.1–3.2도 다운로드하여 확인했다. 로컬 사본은 같은 진단 출력 폴더의 `supplemental.pdf`다.
- [저자 저장소](https://github.com/ReaFly/VLDrive)는 확인 시점에 README만 공개되어 있다.
  저자 구현과 tensor 단위 비교는 불가능하다. 미공개 세부사항은 구현 선택으로 분류했다.
- 현재 코드의 핵심 모듈/학습 회귀 검사 **69개 통과**, 독립 scalar 식 (9) 검사 **1개 통과**.
  [회귀 결과](../../outputs/logs/vldrive_paper_audit_20260918/regression.xml),
  [식 (9) 결과](../../outputs/logs/vldrive_paper_audit_20260918/equation9.xml).
- 전체 구성의 실제 11프레임 입력으로 loss별 backward를 실행했다.
  [상세 수치·코드 hash](../../outputs/logs/vldrive_paper_audit_20260918/model_audit.json).

## 1. 우선 해결할 문제: CCDP의 주행 loss gradient가 거의 사라진다

현재 학습 경로는 `selector_task_gradient_mode=legacy`다. 선택한 feature에
straight-through scalar mask를 곱한 다음 LayerNorm을 통과시킨다.
LayerNorm은 feature 전체의 양의 scalar 배율에 거의 불변이므로 이 mask 경로의
gradient가 약해진다. Hard gather로 버린 token은 이 query 경로에 참여하지 않는다.

근거 코드: `models/qformer_mefa.py` 705행 `_pack_by_mask`, 787–800행 mask 곱셈,
500행 `retained_query_projection`의 LayerNorm, 1358–1368행 연결.

실제 train window 0, 같은 초기 가중치와 같은 Gumbel seed=902의 분리 검사:

| 역전파한 항 | loss 값 | CCDP parameter gradient L2 |
|---|---:|---:|
| Waypoint L1만 | 7.791618 | **1.3910e-10** |
| 10 × pruning | 0.393608 | 2.266287 |
| Reconstruction만 | 28,415.3125 | **232,555.0645** |
| 전체 loss | 28,423.4980 | 232,552.6240 |

따라서 현재 검사 사례의 selector 학습 신호는 복원 loss가 지배한다. 전체 loss에서
selector gradient가 nonzero라는 사실만으로 주행 목표와 연결이 충분하다고 판정하면 안 된다.
이는 현재 초기화의 한 실제 window에서 측정한 수치이며 전체 학습의 실패율을 뜻하지 않는다.
또한 식 (8)을 통한 복원→selector gradient는 실제 존재하므로 CCDP 전체가 끊겼다는 뜻도 아니다.

논문은 hard packing의 backward 구현을 명시하지 않는다. 따라서 이 구현의 약화 문제를
확인한 것과 저자의 구현이 반드시 특정 surrogate를 사용한다고 주장하는 것은 구분한다.
기존 `retained_output_st` 옵션도 자체 실험안이며 이번에 활성화하지 않았다.

## 2. 복원 loss와 압축 loss의 실효 규모가 크게 다르다

현재 loss 형식은 `Lway + 10*Lprun + Lrec`이고 식 (13)의 token/channel squared error를
합산한 뒤 batch/frame 평균을 취한다. 계수와 합산 해석은 논문 식과 맞추려는 구현이다.
그러나 위 사례에서 reconstruction의 CCDP gradient는 weighted pruning보다 약
**102,615배** 크다. 손실값뿐 아니라 같은 parameter 집합의 gradient를 비교한 결과다.

초기화 상태에서 서로 다른 train window 5개, 총 55 frame occurrence를 argmax 평가했을 때:

- 유지 token 평균 **1.418 / 106**, 범위 **1–4**.
- 55개 중 약 **81.8%**가 1개만 유지한다.
- 학습 Gumbel 선택은 위 진단 window에서 평균 **52.545 / 106**이었다.

이는 **미학습 초기화의 train/eval 선택 차이**다. 학습 후 collapse가 확정됐다는 뜻이 아니다.
R=0.3은 penalty의 목표이므로 모든 프레임에서 정확히 32개여야 하는 것도 아니다.
그러나 현 preflight는 이런 상태에서도 통과하므로 학습 중 retention과 주행 성능을 별도로
검사해야 한다. 임의로 mean loss나 top-k로 바꾸면 동일한 논문 설정이라는 주장을 유지할 수 없다.

## 3. 확인된 논문 설정 차이: 프레임 sampling interval

공식 보충자료 §3.2는 학습 중 고정 **2프레임 sampling interval**을 명시한다.
현재 `NavigationCacheDataset`은 연속 원본 프레임으로 window를 만들고,
`NavigationLabels.eligible`도 원본 index 차이 1을 요구한다.

- 실제 입력 timestamp 간격: **100 ms × 10구간**.
- sequence_length=11, Z=10일 때 현재 이 MORAI 데이터의 history span: **1.0초**.
- config의 `train_window_stride=5`는 window 선택 간격이며 내부 프레임 간격이 아니다.

근거 코드: `datasets/navigation_cache_dataset.py` 48–54행,
`navigation/sampling.py` 43행, `configs/vldrive_train_navigation.json` 10–12행.

논문의 raw 데이터 Hz를 이 수치로 단정하지 않는다. 같은 MORAI 10Hz 데이터에서
내부 interval=2를 적용하면 11프레임 history span은 2초가 된다. 엄밀한 재현을 위해서는
내부 sampling interval을 별도 설정으로 지원하고, window 연속성 검사와 text token cache를
함께 재생성해야 한다. 이미 저장한 프레임별 visual feature는 재사용 가능하다.

## 4. Q-Former의 미사용 parameter와 부분 사전학습 초기화

실제 전체 loss backward 후, `requires_grad=True`인데 gradient가 없는 parameter는
72개 tensor, **56,687,616개 scalar parameter**였다. 모두 Q-Former의 일반 text FFN
(`intermediate`, `output`)이다. 현재 query-only 경로는 별도의 `intermediate_query`,
`output_query`를 사용한다. `utils/trainer.py`의 CPUAdamW는 requires_grad 기준으로
master copy를 만들므로 미사용 부분에도 약 **216.25 MiB**의 FP32 master 메모리를 할당한다.
gradient가 없으므로 Adam의 moment state가 모두 생성된다고 주장하지는 않는다.

또 BERT의 일반 FFN은 로딩되지만 query FFN으로 복사하는 처리는 없다. 실제 첫 layer의
두 weight 차이 L2는 **65.2016**이며 query branch는 random initialization이다.
Attention 등 BERT의 사용되는 다른 가중치는 로딩되므로 Q-Former 전체가 random이라는 뜻은 아니다.

근거: `models/qformer.py` 396–400, 429–483행;
`models/qformer_mefa.py`의 `build_qformer`;
`models/vldrive.py` 148–151행 freeze 목록;
`utils/trainer.py` 139–155행 CPU master 구성.

논문은 Q-Former의 상세 초기화 recipe를 공개하지 않아 query FFN의 random 초기화 자체를
명백한 논문 위반으로 단정할 수 없다. 다만 미사용 FFN은 optimizer에서 제외하고,
query 초기화 정책은 독립적인 구현 선택으로 명시·검증해야 한다.

## 5. 수식/구조 대조

| 항목 | 논문 기준 | 현재 상태와 판정 |
|---|---|---|
| Vision encoder | LMDrive 구조, 106 token, Stage 2 고정 | 100 BEV + 1 light + 5 waypoint feature, 고정 캐시 사용. 구조 일치. MORAI 학습 가중치는 도메인 변경 |
| CCDP 식 (1)–(4) | local C/2, global 평균, keep/drop 확률, Gumbel mask | forward 기본 구조 일치. hard gather·최소 1개 rescue·eval argmax는 자체 구현 선택 |
| MEFA 식 (5)–(7) | 이전 Z개 평균, MLP temporal encoding, retained Q / full enhanced K,V | Z=10, 현재/미래를 history에서 제외, retained feature query 사용. 인과성 검사 통과. 시간 sampling 차이는 별도 문제 |
| Reconstruction 식 (8) | 원래 위치에 정렬, keep mask와 mask embedding, 4-layer Transformer | 해당 구조 구현. replacement의 ST backward, 추가 position embedding은 미공개 세부사항에 대한 구현 선택 |
| Reconstruction target | 본문의 encoder feature Fi | 현재는 **학습 가능한 LayerNorm(Fi).detach()**. raw feature 그대로의 복원과 다름; 정규화를 Fi 정의에 포함할지 명시 필요 |
| DDIA 식 (9) | text 양방향, visual→text 위치 분리, visual 과거 attention, 공동 분모 | 독립 scalar 계산과 일치. RoPE FP32 삼각함수/캐스팅 회귀 검사 통과 |
| Loss 식 (10)–(13) | L1 + 10×pruning + reconstruction | 목적함수 형태·계수 일치. 현재 수치 규모와 selector task gradient 문제는 위 진단 참조 |
| Language model | TinyLLaMA 또는 4-layer LLaMA* | TinyLlama 변형 사용, decoder와 text embedding 학습. 사용하지 않는 vocab/end head 고정 |
| Waypoint readout | LLM feature를 MLP로 trajectory 예측 | 매 frame 마지막 valid visual token을 readout. 상세 readout 위치는 논문에 없어 구현 선택 |
| 학습 schedule | AdamW, cosine, lr 1e-5, wd .06, 15 epochs | 해당 설정 일치. 단일 GPU/CPU optimizer offload와 clip 길이는 자체 실행 설정 |
| 학습 sampling | interval 2 frames | 현재 interval 1. 확인된 불일치 |
| 성능 검증 | CARLA LangAuto closed loop | 현재는 MORAI 한 바퀴의 시간 분할 + preflight. 논문 성능 재현/폐루프 검증은 미완료 |

복원 target 차이의 실제 수치: raw feature RMS=0.609571, normalized feature RMS=0.999986,
두 target의 차이 RMS=0.390936. `visual_input_norm`은 학습 가능하므로 `.detach()`는
이번 backward를 차단할 뿐 target을 학습 전체에 걸쳐 고정하지 않는다.
논문에 추가 LayerNorm의 위치가 공개되지 않았으므로 이 차이를 정규화 정책의
확인 필요 항목으로 분류한다.

## 6. 다음 수정·검증 순서

1. CCDP에 대해 waypoint-only gradient 크기, 전체 loss의 gradient 구성,
   실제 train/eval retention을 검사하는 학습 gate를 추가한다.
2. Hard packing / normalization / selector surrogate를 통제 비교한다.
   논문 원래 목적함수와 자체 실험안을 구분하고 단순 nonzero 여부만으로 통과시키지 않는다.
3. 내부 frame interval=2를 config로 지원하여 명시된 sampling recipe를 맞춘다.
   instruction 구간 유효성·future target 분할을 그대로 검사한다.
4. 미사용 Q-Former FFN을 optimizer에서 제외하고 query 초기화/복원 target 정규화 정책을 정리한다.
5. Navigation v2의 기존 의미 오류를 수정한 후 짧은 실제 optimizer 실행에서
   loss 감소·turn 방향 반응·압축률·메모리를 확인한다. 이후 본 학습 및 독립 주행 평가로 진행한다.

이번 검증에서는 원래 모델의 수식이나 학습 설정을 임의로 변경하지 않았다.
기존 preflight 결과는 유지하며, 위 문제를 찾은 추가 감사 결과로 해석한다.

## 실행 명령

AIMDrive 루트, WSL lmdrive 환경:

```bash
export PYTHONPATH="$PWD/VLDrive/.runtime/stage2_wsl_deps:$PWD/VLDrive${PYTHONPATH:+:$PYTHONPATH}"
python VLDrive/tools/audit_vldrive_paper.py \
  --config VLDrive/configs/vldrive_train_navigation.json \
  --output VLDrive/outputs/logs/vldrive_paper_audit_new
```

이미 결과가 있는 output을 덮어쓰지 않는다. 모델은 저장된 Stage 2 학습 완료 가중치가 아니라
현재 학습 config의 시작 상태를 생성한다. 실제 data cache와 사전학습 자산을 쓰며 optimizer를
만들거나 업데이트하지 않는다.
