# 근본 원인 재감사 — 논문 방법을 바꾸기 전 확인한 사실

2026-09-11. 대상: `outputs/morai_paper_last_encoder_20260910T055054Z/model-last.pt`의 9,540-update 최종 모델.
SHA256: `a63a619b2565d2bb0fba13ddfe305779f344bf67d0fbd9aafa7f6476c9f14b9a`.

Train만 사용했다. 원본 데이터·기존 checkpoint·모델/학습 코드는 변경하지 않았다.
본 모델 optimizer update는 0회다. CPU 산술 검증용 작은 tensor의 AdamW 대조는 별도다.
이전 off-paper repair 학습은 SIGSTOP 일시정지 상태를 유지했다.
이번 변경은 진단 도구와 보고서다. 진단 중 입력/gradient 경로를 비교한 것은 배포 모델 수정이 아니다.

## 결론과 확실성

**작은 데이터 때문이라고만 설명할 근거는 부족하다. 신호 보조 라벨의 정책 불일치, RoPE 위치 계산의 수치 결함, 자체 구현의 학습 신호 약화 경로를 구체적으로 확인했다.**
그러나 회전 실패·압축 실패·첫 녹색 반응 지연을 한 원인으로 설명하거나, 아래 문제 하나를 고치면 모두 해결된다고 결론내리지 않는다.

| 문제 | 이번에 확인한 내용 | 아직 확인하지 못한 내용 |
|---|---|---|
| 회전 예측 | 회전 GT는 실제 학습에 포함됨. 좌우 지시 변경이 최종 방향에 거의 반영되지 않음 | 어느 초기화·최적화·표현 문제가 이 상태를 만들었는지 |
| token 압축 | Hard gather/LayerNorm을 통한 query-side 선택 gradient가 거의 사라짐 | 이를 바로잡으면 원래 목적함수로 정상 압축되는지 |
| 위치 계산 | RoPE 각도를 BF16으로 내린 후 삼각함수를 계산하여 원래 LLaMA와 다른 큰 오차 발생 | 이 오류를 바로잡는 것이 회전 학습에 기여하는 크기 |
| 신호 보조 라벨 | Expert의 주행 허용 판정과 collector red/should_brake 판정이 다른 상태 존재 | 해당 보조 라벨이 최종 궤적 실패에 기여한 크기 |
| 첫 녹색 반응 | 기존 encoder traffic head부터 0.2~0.4초 늦는 네 사례 | 지연을 만든 센서 표현·학습 데이터·감독 문제의 기여율, 실제 자력 출발 성능 |

## 1. 기존 데이터 설명 중 정정할 부분

- Train **3,352개** cached future trajectory를 원본 ego 위치에서 독립 재계산했다. 최대 float32 차이 **0m**.
  Global-path 곡선을 미래 이동 정답으로 잘못 사용한 문제가 아니다. 시간 간격과 좌표 부호도 함께 검사했다.
- 우회전 instruction 사건의 **56/60**, 좌회전 사건의 **46/52** 프레임이 실제 loss에 들어갔다.
  경계 때문에 일부 window anchor가 제외된 것과 회전 초반 프레임이 전부 미학습인 것은 다르다.
- 명시적 좌·우 지시 사건은 각각 한 번이지만, 2.5초 뒤 횡변위 절댓값 ≥2m인 곡선 GT는 **461프레임**이다.
  그중 **437개 고유 프레임**이 실제 감독됐다. 연속 기준으로 나뉜 9개 구간은 독립 교차로 9개라는 뜻이 아니다.
- 실제 loss 위치의 정지는 **65.35%**, 곡선 GT는 **12.61%**다. 좌우 지시 비율 **2.83%**를 곡선 정답 비율로 오해하면 안 된다.

같은 train 정답을 instruction별 중앙 궤적으로 대표한 서술 통계:

| 동일한 train anchor | 정답들의 중앙 궤적 FDE | 실제 최종 모델 FDE |
|---|---:|---:|
| 우회전 | 0.850m | 3.958m |
| 좌회전 | 1.201m | 3.998m |

새 모델 학습이나 독립 평가가 아니다. 적어도 **명시적 회전 정답 자체가 서로 모순되어 학습 불가능한 상태는 아니다**라는 반증이다.
데이터 불균형이 최적화를 어렵게 할 가능성은 남지만, 데이터 양만으로 현재 실패를 충분히 설명하지 못한다.

또한 LMDrive 논문과 공식 parser에는 회전 command 시작부터 instruction clip을 만드는 방식도 있다.
따라서 "회전 시작에 지시가 붙어서 논문 위반" 또는 "30m lookahead만이 논문 정답"이라는 해석은 철회한다.
공식 데이터는 거리형 지시, 시작 구간 확장 등 더 풍부한 문맥을 제공하지만 이 차이의 영향은 별도 비교가 필요하다.
[LMDrive §3](https://arxiv.org/html/2312.07488v2#S3), [공식 parser](https://github.com/opendilab/LMDrive/blob/main/tools/data_parsing/turn_rules.py).

근거: [데이터 감사](outputs/diagnosis_root_20260911/data_audit_original_final.json), [코드·논문 비교](outputs/diagnosis_root_20260911/data_audit_sources.md).

## 2. 신호 라벨은 실제 expert 정책과 별도 감사해야 함

Collector의 `is_red_light_present`는 모든 주변 신호를 뜻하지 않는다. MGeo가 고른 앞 정지선의 신호 ID와 gRPC ego 신호 ID를 맞춘 뒤 생성된다.
그러나 현재 expert의 신호 정지 컴포넌트는 status `33`을 정지 대상으로 처리하지 않고, collector는 red bit가 있으면 red/should_brake로 기록한다.
Train에서 **112프레임**이 이 상태며 실제104프레임은 throttle>0,8프레임은 brake>0이었다. 이것을 expert의 모든 제어 조건이33에서 무조건 GO라는 뜻으로 해석하지 않는다.
112개 모두 속도4.43~5.89m/s로 움직였고, 실제2.5초 뒤 변위는11.63~14.05m였다. 별도 ACC 조건은 존재하므로 dedicated 신호 정지 정책과 전체 expert 정책도 구분한다.
또 관련 신호 ID는 있으나 상태를 확인하지 못한 **70프레임**을 not-red로 만들면서 traffic supervision을 활성화한다.
70개 모두 실제 색이 틀렸다는 뜻이 아니라, 확인 불가능한 상태를 확정적인 음성 정답으로 감독하는 유효성 문제다.

이것은 **신호 보조 정답의 정의·유효성 문제**다. 실제 미래 ego waypoint GT까지 잘못됐다는 뜻은 아니다.
후속 모델은 `should_brake`를 정답으로 사용하지 않고 실제 미래 ego 궤적을 사용한다.
물리적 red 색 분류와 해당 경로에서 정지해야 하는지의 분류는 구분해 정의해야 한다.
참고로45m를 넘는 정지선이 먼저 선택되는 guard 순서도 발견했지만, 멀리 있는 빨간 신호 앞에서 현재 가속한다는 사실 자체는 주행 오류가 아니다.
근거와 실제 신호 전환/ID 변경 구분: [신호 라벨 감사](outputs/diagnosis_root_20260911/traffic_label_audit.md).

## 3. Encoder부터 존재하는 첫 녹색 지연

이미 학습된 encoder의 traffic head를 strict load하고 기존 cached traffic token에만 적용했다. 재학습하지 않았다.
아래 첫 녹색은 동일한 관련 신호 ID의 유효한 green 상태가 기록된 시점이다.

| 첫 녹색 frame | Encoder가 처음 not-red로 예측한 frame | 지연 | 실제 속도 >0.1m/s가 된 frame |
|---|---|---:|---|
| 991 | 994 | 0.3초 | 994 |
| 1591 | 1593 | 0.2초 | 1594 |
| 2491 | 2494 | 0.3초 | 2494 |
| 2791 | 2795 | 0.4초 | 2794 |

이 네 시점은 unknown/신호 ID mismatch로 설명되지 않는다. 따라서 첫 녹색의 문제가 CCDP 뒤에서만 생겼다는 가설은 성립하지 않는다.
그러나 head 출력은 downstream에 직접 연결되지 않으며, 지연과 실제 차량 움직임의 상관관계만으로 속도 의존이 유일한 원인이라고 말할 수 없다.
기존 속도 교체 실험도 속도 하나만의 원인을 지지하지 않았다. Closed-loop 영구 정지 여부는 아직 검증하지 않았다.

근거: [전체 train feature/head 감사](outputs/diagnosis_root_20260911/feature_audit_with_departures.json).

## 4. Encoder token이 전부 무의미한 것은 아님

Train 3,352개 cache feature의 hash는 모두 달랐고, BEV/traffic/waypoint token 모두 시간에 따른 변화를 보였다.
동일한 기존 GRU head에서 입력만 바꾸는 진단을 추가했다:

- 곡선 frame 566개(final |lateral|>1m)에서 원래 target point를 사용하면 FDE **1.620m**.
- target point의 좌우 성분만 0으로 바꾸면 **4.535m**.
- 원래 target point를 유지하되 모든 waypoint token을 train 평균값으로 바꾸면 **13.745m**.

숫자 경로는 회전에 큰 역할을 했고, token도 관측별 이동 정보를 담고 있다. "encoder는 완벽"과 "token에 정보가 전혀 없음" 둘 다 지지되지 않는다.
원래 GRU head는 numeric target point를 사용하지만 downstream token은 이를 포함하지 않는 것이 원래 LMDrive 구조이므로, 이것 자체를 새 오류로 분류하지 않는다.

Train feature의 최근접 이웃 정답을 이용한 진단에서는 인접 프레임을 허용할 때 곡선 FDE1.413m, ±25프레임을 제외하면5.291m였다.
이는 temporal redundancy/coverage를 보여주는 서술 비교이지 일반화 성능이나 실제 모델의 학습 가능성 보장이 아니다. 후속 모델에 이웃 정답을 입력하지 않았다.

## 5. 실제 LLM 추론에서 좁힌 방향 정보 경로

Frame2865·2970·3320의 11-frame 입력으로 기존 모델을 GPU 추론했다. 저장된 baseline prediction과 최대 차이는 세 사례 모두 **0m**.
추가 repair 옵션은 끄고 original config 기본값을 사용했으며, trainable/frozen weight hash를 검증했다.
현재 일부 파일은 이전 repair 옵션 추가로 source hash가 달라져 있으므로 이를 "전체 코드 byte-identical 재실행"이라고 표현하지 않는다.

- 좌/우 단어 token과 embedding은 서로 다르다.
- 마지막 visual readout이 지시문에 주는 attention mass는 일부 중간 층에서 약 **39~47%**다. 문장을 아예 안 보는 것은 아니다.
- 좌우 문장 교체로 최종 readout vector는 상대 L2 약 **0.31~0.40%** 바뀌지만, 2.5초 endpoint 좌우 차이는 FP32 head로 계산해도 **0.45~1.90mm**뿐이다.
- 최종 MLP를 FP32로 계산해도 이 현상이 유지되어, 출력 BF16 반올림 하나로 설명되지 않는다.
- 실제 visual embedding RMS는 **0.747~0.784**, 좌우 지시문 embedding RMS는 약 **0.0101**이다.
  큰 입력 scale 차이는 확인됐지만 residual/정규화/학습 영향이 있으므로 이 비율만으로 원인 확정 또는 임의 rescale 수정의 근거로 쓰지 않는다.

현재까지는 입력 wire 단절보다 **학습된 표현에서 방향 의미가 최종 궤적으로 전달되지 않는 상태**로 좁혔다. 그 상태를 만든 원인은 아직 분리 검증 중이다.
근거: [관찰 hook 추론](outputs/diagnosis_root_20260911/language_token_audit.json).

## 6. CCDP의 자체 backward 선택에서 확인한 약화 경로

현재는 hard gather한 token에 ST scalar mask를 곱하고 LayerNorm을 통과시킨다.
LayerNorm은 scalar 배율 변화에 거의 불변이므로 이 query 경로의 mask gradient가 크게 약해진다.
버린 query는 hard gather 때문에 이 경로 gradient가 0이다. Pruning/reconstruction의 다른 gradient가 전부 0이라는 뜻은 아니다.

- 실제 최종 projection의 동일-forward 국소 비교: 현재 mask gradient L2 **0.000872**, 비교 경로 **86.953**. 약10만 배 차이이며 중앙차분으로도 확인했다.
- 실제 768폭/12층 전체 connector의 동일-forward 선형 probe: 현재 **1.679e-7**, 출력에 별도 ST gate를 둔 진단 대조 **3.1296**.
- 첫 LayerNorm 뒤로 gate만 옮겨도 Q-Former 내부 LayerNorm이 있어 개선되지 않았다. 단순 위치 이동이 해결이라는 증거는 없다.

이는 실제 구현의 학습 신호 문제 후보를 확인한 것이다. 선형 probe이므로 실제 waypoint loss/주행 실패의 단독 인과를 입증하지는 않았다.
DynamicViT 공식 구현은 훈련 중 attention policy를 사용하고 평가에서 gather한다. VLDrive 본문은 정확한 packing backward를 명시하지 않아 우리의 선택을 저자 구현과 같다고 주장할 수 없다.
[DynamicViT 공식 코드](https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py), [VLDrive §3.2–3.4](https://arxiv.org/html/2511.06256v1).

이전 보고서는 `(1-M)` loss-weight의 직접 gradient를 과도하게 강조했다.
새 synthetic graph에서 그 직접 경로는 reconstruction selector-bias gradient의 약5.7%였고 나머지는 Eq.8 입력/query 경로였다.
따라서 loss-weight detach만으로 해결된다고 볼 수 없다. 이 비율은 원래 전체 학습의 기여율이 아니다.
근거: [CCDP 수치 감사](outputs/diagnosis_root_20260911/ccdp_audit_full.json).

## 7. 초기화·optimizer 감사

- CPU master AdamW는 표준 FP32 AdamW와 400-step 산술 대조에서 오차0. 실제 최종 master와 BF16 model도 일치했다.
- 실제 q/k/v 및 left/right embedding은 초기값에서 변경됐다. Frozen optimizer/embedding 문제는 아니다.
- DDIA의 checkpointing on/off 수치와 causal-mask 검사는 통과했다. 이는 모든 수치 세부 구현의 정확성을 보장하지 않는다.
- BERT의 일반 text FFN은 로드되지만, 실제 query branch FFN에 복사되지 않아 해당 branch는 random initialization이다.
  미사용 일반 FFN 56.7M parameters가 optimizer master 공간을 차지한다. 경로 단절이 아니라 초기화와 자원 사용의 차이다.
  BLIP-2 stage1은 query FFN으로 복사한다. VLDrive의 초기화 세부는 미명시이므로 paper violation 또는 단독 원인으로 단정하지 않는다.
  [공식 BLIP-2 초기화](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py#L69).

근거: [산술·가중치 감사](outputs/diagnosis_root_20260911/training_audit.json), [query 초기화 감사](outputs/diagnosis_root_20260911/training_audit_projection.json).

## 8. 추가로 확인한 실제 RoPE 수치 구현 결함

현재 `ddia_attention.apply_rope`는 FP32 각도를 먼저 BF16으로 변환한 후 `cos/sin`을 계산한다.
공식 HF Transformers4.31 LLaMA는 **FP32에서 cos/sin을 계산하고 그 결과를 BF16으로 변환**한다.
큰 각도를 일찍 반올림하면 삼각함수 입력 자체가 달라져, 나중에 높은 정밀도로 계산해도 복구되지 않는다.

Head dimension64, position0~1200에서 같은 synthetic Q/K를 비교했다:

| 공식 HF 기준 오차 | FP32 | FP16 | 실제 학습 정밀도 BF16 |
|---|---:|---:|---:|
| 최대 각도 오차 | 0 | 0.25rad | **4rad** |
| 회전된 Q 상대 L2 오차 | 0 | 2.71% | **27.64%** |
| visual→visual attention score 상대 L2 오차 | 0 | 3.84% | **36.26%** |

첫 주파수에서는1,201개 위치가535개 각도로 뭉친다. 이 값은 synthetic 연산 오차이며 주행 성능 오차가 아니다.
원래 run은 106 tokens×11 frames+text로 실제 약1,176개 token을 처리하므로 검사 위치 범위와 관련이 있다.
Visual→instruction raw dot product 자체는 이 RoPE를 쓰지 않아 직접 오차가0이지만, 동일한 softmax에서 경쟁하는 visual→visual score는 영향을 받는다.

이는 논문 구조나 loss를 바꾸자는 제안이 아니라, **원래 위치 인코딩 계산을 낮은 정밀도에서도 정확히 수행했는지의 구현 문제**다.
CUDA autocast도 이미 BF16으로 반올림된 각도의 정보를 복구할 수 없다. 다만 전체 주행 실패의 기여율은 아직 입증하지 않았다.

실제 최종 모델 frame2865의 GPU 추론에서도 확인했다. Q/K dtype은 BF16, sequence length는1,176이었다.
각 layer의 동일한 실제 Q/K에서 위치 계산만 HF 순서로 비교했을 때 마지막 readout의 attention 분포 차이(TV)는
층별 평균 head 기준 **0.00818~0.09946**였다. 실제 모델의 계산에도 오차가 발생하며 synthetic 사례에만 국한되지 않는다.
이 관찰은 원래 forward를 변경하지 않았고, baseline waypoint prediction은 저장된 값과0m 차이였다.
**TV0.09946은 attention 분포 차이이며 주행 성능9.95% 악화라는 뜻이 아니다.**
이전 층의 출력까지 교체한 재추론이나 이 계산을 고친 재학습 결과도 아니다.

근거: [RoPE 수치 감사](outputs/diagnosis_root_20260911/training_audit_rope.json),
[공식 HF4.31 구현](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py#L95),
[현재 구현](lavis/models/vldrive/ddia_attention.py:38), [실제 GPU Q/K 비교](outputs/diagnosis_root_20260911/language_rope_audit.json).

## 다음 검증 원칙

1. 논문에 없는 BCE나 고정 top-k로 결과를 맞추지 않는다. 기존 loss와 동적 선택 목표를 기준으로 삼는다.
2. 신호 보조 정답은 원본 signal 상태·관련 ID·경로 maneuver·expert 정책과 유효성 mask부터 일치시키는 설계를 검토한다. 실제 ego 미래 GT는 유지한다.
3. 확인된 RoPE 계산 순서 문제를 우선 대상으로 하고, gradient/초기화 문제와 섞지 않은 짧은 비교 학습으로 영향을 확인한다. 이는 별도 수정 승인 후 수행하며 이번에는 모델 함수를 바꾸지 않았다.
4. 그 비교 전에는 encoder 재학습 불필요, 데이터 추가만으로 해결, 또는 한 구현 수정으로 전부 해결된다고 약속하지 않는다.

이번 진단만으로 재수집이나 대규모 재학습을 시작할 근거는 없다. 원본에 정보가 남아 있는 라벨 문제는 우선 재수집 없이 검증할 수 있다.
