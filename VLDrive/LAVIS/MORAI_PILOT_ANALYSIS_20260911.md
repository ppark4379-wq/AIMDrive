# MORAI 후속 학습 pilot 결과 및 원인 분석 — 2026-09-11

## 결론

이번 결과는 **주행 정답을 조건별로 충분히 모방하지 못한 초기 학습 결과**입니다. 단순히 회전 방향만 틀리는 것이 아니라, 움직여야 할 장면에 거의 정지한 미래 경로를 출력하는 문제가 큽니다.

CCDP가 토큰을 줄이지 못하는 현상은 원인을 상당히 좁혔습니다. 현재 수치 스케일에서는 복원 목적이 토큰을 더 남기는 방향으로 선택기 gradient를 지배합니다. 이전에 추가한 task-gradient 우회 경로만으로는 이 경쟁을 해결하지 못했습니다. 다만 **압축 실패 원인을 확인한 것과 주행 실패 전체의 단일 원인을 확정한 것은 다릅니다.**

검사한 미래 waypoint GT는 실제 미래 ego 위치와 일치했습니다. 따라서 이번 실패를 global-path GT 오류, NPC 부재, 정지 데이터 과다 중 하나로 단정할 근거는 없습니다. 학습 예산, 언어 지시 표본 구성, 신호 전환 평가 범위, 공유 gradient clipping의 영향도 구분해야 합니다.

이번 턴에서는 production 모델/학습 코드·원본 데이터·기존 checkpoint를 수정하지 않았습니다. 진단 스크립트와 보고서만 추가했습니다. 저장된 모델의 GPU 추론과 loss별 backward를 실행했지만 **optimizer update는 0회**이며, 검사 전후 가중치 hash가 동일합니다. 추가 학습은 시작하지 않았습니다.

## 1. 무엇을 비교했는가

- 실행: `morai_root_pilot_20260910T214044Z`, 2026-09-11 06:40–07:42 KST.
- A: 기존 `legacy` backward 경로. B: `retained_output_st` 실험 경로.
- B는 forward 결과를 유지하면서 선택기에 waypoint gradient를 추가 전달하는 독립적인 backward 실험입니다. 저자 공개 구현을 복원했다고 주장하지 않습니다.
- 두 모델 모두 동일한 pretrained TinyLlama/BERT에서 시작했습니다. 과거 9,540-update 최종 모델을 이어 학습한 것이 아닙니다.
- 기존 MORAI vision encoder epoch25와 그 feature cache는 동결했습니다. 이번 비교는 vision encoder 재학습 결과가 아닙니다.
- 동일한 초기 가중치 hash, 초기 출력, 데이터/코드 provenance, seed, optimizer, 실제 샘플 순서를 확인했습니다.
- 각각 320 update, batch 1개의 11-frame clip. 32개 train window를 반복 사용했습니다. 정지/직진/좌곡선/우곡선 anchor 각 8개이며, 각 window는 6–17회 사용됐습니다.
- 아래 평가는 **그 32개 train window**에 대한 것입니다. 전체 train 평균이나 validation 결과가 아닙니다.
- 상태는 `comparison_gate_failed`; 선택된 후보가 없으며 자동 후속 학습도 시작하지 않았습니다. 두 pilot 자체는 320/320을 정상 완료했습니다.

근거: [.runtime 상태](.runtime/training_runs/morai_root_pilot_20260910T214044Z/status.json), [training audit](outputs/diagnosis_pilot_20260911/training_audit.md).

## 2. 실제로 나타난 주행 예측 문제

ADE는 0.5/1.0/1.5/2.0/2.5초 미래 위치의 평균 거리 오차, FDE는 마지막 2.5초 위치 오차입니다.

| 동일 train 표본, 320 update | A 기존 경로 | B ST 후보 |
| --- | ---: | ---: |
| 전체 32개 ADE | 4.46 m | 3.87 m |
| 전체 32개 FDE | 7.82 m | 6.95 m |
| 정지 8개 FDE | 0.58 m | 0.37 m |
| 직진 8개 FDE | 11.76 m | 9.51 m |
| 좌·우곡선 16개 FDE | 9.46 m | 8.96 m |
| 유효 좌회전 예측 | 0/8 | 0/8 |
| 유효 우회전 예측 | 3/8 | 4/8 |

유효 회전은 마지막 좌우 변위의 부호가 맞고, 절댓값이 0.5 m 이상인 경우입니다. 좌회전 0/8은 8개 모두 반드시 반대 부호라는 뜻이 아니라, 올바른 방향으로 충분히 꺾은 예측이 없다는 뜻입니다. 이 기준은 실제 폐루프 주행 성공률도 아닙니다.

### 이동거리 과소예측이 큰 비중을 차지함

- 이동 24개 중 양 모델 모두 8개에서 미래 종점이 현재 위치로부터 1 m 이내였습니다.
- 이동 표본의 `예측 종점 변위 / 실제 종점 변위` 중앙값은 A 28.1%, B 46.6%입니다.
- 이동 표본의 종점 **제곱오차 합** 중 종방향 비중은 A 91.1%, B 89.1%입니다. 평균 FDE 자체를 이 비율로 나눈 수치는 아닙니다.
- 명시적 우회전 지시가 있는 frame2850에서 실제 2.5초 변위는 약 12.25 m인데 A/B 예측은 약 0.35/0.37 m였습니다.
- 명시적 좌회전 지시가 있는 frame2980도 실제 약 12.25 m에 비해 A/B는 약 0.17/0.38 m였습니다. 이 두 window는 실제 학습에 각각 12회, 11회 들어갔습니다.

그렇다고 전진 거리만 고치면 되는 것도 아닙니다. 곡선 GT의 마지막 좌우 변위 절댓값 평균은 약 3.94 m인데 예측은 A 0.394 m, B 0.452 m에 그칩니다. B의 좌곡선 8개에서 왼쪽으로 가장 크게 간 예측도 0.110 m였습니다.

### 정지를 배우는 동안 회전 성능이 후퇴함

| 160 → 320 update | A 기존 경로 | B ST 후보 |
| --- | ---: | ---: |
| 정지 FDE | 9.34 → 0.58 m | 7.65 → 0.37 m |
| 곡선 FDE | 6.86 → 9.46 m | 7.82 → 8.96 m |
| 유효 좌회전 | 6/8 → 0/8 | 7/8 → 0/8 |
| 유효 우회전 | 0/8 → 3/8 | 0/8 → 4/8 |

이는 조건별 주행을 안정적으로 구별했다기보다, 학습 중 전진 길이와 좌우 편향이 함께 이동하는 현상에 가깝습니다. B의 최종 곡선 FDE 개선은 A 대비 5.30%지만 좌우 오차 개선은 1.70%에 그쳐, 이를 회전 지시 이해의 회복으로 표현하면 과장입니다.

대표 비교 그림: [실제 expert GT와 A/B 160·320 궤적](outputs/diagnosis_pilot_20260911/saved_trajectory_comparison_v2.png). 정지 첫 표본, 최고속 직진 표본, 첫 명시적 우회전·좌회전 표본을 표시했습니다. 무작위 표본은 아닙니다. 각 패널의 가로·세로는 동일한 m 스케일이며, 패널 간 범위는 다릅니다.

## 3. CCDP 압축 실패: 복원과 가지치기의 실제 gradient 경쟁

현재 목표는 약 30% token 유지입니다. 하지만 고정된 320-update checkpoint에서 검사한 모든 token의 keep 확률은 약 0.630–0.790, 평균 0.745였습니다. 모든 확률이 0.5를 넘으므로 argmax 추론에서 **106개를 전부 유지**합니다. 160 update에서도 전부 유지했습니다.

이는 확률이 1에 포화됐다는 뜻이 아닙니다. 학습의 Gumbel 선택은 확률적으로 일부를 버리지만 추론의 argmax는 전부 남깁니다. 마지막 학습 구간 평균 유지율 약 70%와 고정된 최종 checkpoint의 약 74.5%도 서로 다른 집계입니다.

처음 64 update 평균과 마지막 64 update 평균을 비교하면, A의 복원 loss는 14,248 → 2,969로 감소했지만 weighted pruning loss는 0.406 → 1.642로 증가했습니다. 압축 목표에서는 멀어진 것입니다.

### loss 숫자만 비교하지 않고 실제 모델을 backward로 확인함

저장된 최종 모델에 좌곡선·우곡선·정지 11-frame clip을 넣고 waypoint, weighted pruning, weighted reconstruction을 각각 역전파했습니다. optimizer는 호출하지 않았습니다.

다음은 B의 좌곡선 clip에서 **동일한 selector 파라미터 그룹에 전달된 gradient L2 norm**입니다.

| 학습 목적 | 실제 loss | selector gradient norm |
| --- | ---: | ---: |
| 주행 waypoint | 4.067 | 0.264 |
| 토큰 가지치기, 가중치 포함 | 2.061 | 8.122 |
| 버린 token 복원, 가중치 포함 | 2,283.850 | 21,435.873 |

복원 gradient는 waypoint보다 약 8.1만 배 컸습니다. 세 clip 모두 pruning과 reconstruction gradient의 cosine은 약 -0.997~-1.000으로 거의 반대였고, 합산 gradient 방향은 reconstruction 쪽과 거의 같았습니다. 마지막 2-class score bias의 gradient 부호까지 확인하면 복원은 keep 쪽, pruning은 drop 쪽으로 압력을 줍니다.

따라서 **현재 checkpoint와 검사한 입력에서는 복원 쪽이 선택기를 더 남기는 방향으로 지배한다**는 근거가 있습니다. 다만 gradient norm 비율을 AdamW의 최종 parameter update 비율로 그대로 해석해서는 안 됩니다.

현재 구현에서 모든 token을 남기면 복원할 대상이 없어 reconstruction loss는 0입니다. 반면 weighted pruning 비용은 `10 × (1 - 0.3)^2 = 4.9`뿐입니다. 현재 수천 단위 복원 비용에 비해 토큰을 많이 남기는 방향이 수치적으로 저렴합니다. 완벽히 복원하면서 30%만 유지하는 해는 더 낮은 loss를 가질 수 있으므로, all-keep이 유일한 전역 최적해라거나 실패가 필연적이라고 주장하지 않습니다.

또한 복원 loss 감소를 전부 회피라고 보면 안 됩니다. 평균 dropped token 수는 53.46 → 31.64로 줄었고, dropped token당 squared error도 266.51 → 93.85로 줄어 복원 자체도 학습됐습니다.

근거: [selector audit](outputs/diagnosis_pilot_20260911/selector_audit.json), [실제 full-graph 진단](outputs/diagnosis_pilot_20260911/full_graph.json), [복원 구현](lavis/models/vldrive/qformer_mefa.py).

## 4. 이전 task-gradient 수정과 추론 선택 방식만으로 설명할 수 없는 이유

### ST 후보는 경로를 복구했지만 선택 정책을 바꾸지 못함

초기 동일 입력에서 waypoint→selector gradient는 A의 1.44e-10에서 B의 0.01475로 커졌습니다. 그러나 당시 reconstruction gradient는 277,908이었습니다. 최종 320 update에서도 위와 같은 압도적 차이가 남았습니다.

A/B 최종 token 확률 차이는 평균 0.0001147, 프레임 내 score 상관 평균은 0.99999923이었습니다. 경로가 생긴 것은 확인됐지만 실제 선택 정책을 유의하게 바꾼 결과는 아닙니다. B를 해결책으로 채택할 근거가 부족합니다.

이 실험은 기존 reconstruction/pruning 목적과 gradient를 유지했습니다. 로그에 `retention_logit_loss`가 보이더라도 이번 설정의 가중치는 0이므로 BCE를 학습 목적에 추가한 실험은 아닙니다. 강제 top-k나 reconstruction mask detach도 사용하지 않았습니다.

### 같은 가중치에서 argmax를 Gumbel로 바꿔도 주행 오류는 유지됨

B 최종 모델의 동일 32개 window에 대해 선택 방식만 바꿨습니다.

- argmax: 106개 전부 유지, FDE 6.949 m.
- Gumbel, seed 607/608/609: 평균 약 74% 유지, FDE 6.950–6.989 m.
- 예측점의 평균 변화는 약 3–4 cm였습니다.
- A에서도 같은 경향이었으며, argmax 재실행은 양 모델 모두 저장된 예측을 정확히 재현했습니다.

따라서 학습/추론의 mask 차이는 실제 문제지만, 그것만 없애면 수 m의 주행 오류가 사라지는 상황은 아닙니다. 강제 top-k로 바꾸면 해결된다고 결론내릴 수 없습니다.

또한 동일 시각 입력의 2개 clip에서 지시문만 left/right로 교체했을 때, B의 마지막 좌우 예측 변화는 약 4–5 mm였습니다. 이 제한된 검사는 언어에 따른 좌우 조절이 약하다는 보조 증거입니다. 시각 정보와 충돌할 수 있는 가상 지시에 유일한 GT를 부여한 시험은 아니므로, 모델 전체가 모든 언어를 무시한다고 일반화하지 않습니다.

## 5. 데이터와 평가 범위에서 확인한 문제

### 이번 pilot을 '정지 데이터가 너무 많다'로 설명하면 안 됨

실제 320 update × 11 frame의 loss 적용 위치 3,520개를 재구성했습니다. 정지 23.9%, 직진 35.8%, 좌곡선 19.8%, 우곡선 20.5%였습니다. 곡선 정답은 40.3%입니다. 후반 160 update에서 곡선 loss 위치가 사라진 것도 아닙니다.

과거 전체 데이터의 높은 정지 비율과 이 균형 pilot은 다른 실험입니다. 이번 정지 편향 출력의 충분한 설명으로 과거 비율을 재사용하면 안 됩니다.

### 검사한 waypoint GT는 실제 미래 ego로 만들어져 있음

32개 window의 345개 고유 supervision frame에 대해 원본 동기 수집 ego 위치로 0.5–2.5초 미래 GT를 다시 계산했습니다. cache의 float32 GT와 최대 차이는 0.0 m였습니다. 검사 범위에서 global path를 미래 이동 정답으로 잘못 넣거나 좌표축/미래 frame offset을 틀린 증거는 없었습니다.

이 검증이 전체 dataset의 모든 label, 센서 영상 내용, frozen visual token의 의미 품질까지 보장하는 것은 아닙니다. NPC를 의도적으로 넣지 않은 것도 현재 빈 도로의 회전·전진 실패를 직접 설명하지 않습니다.

### 지시문 학습 표본은 여전히 제한적임

32개 anchor 중 명시적인 turn 지시는 3개뿐입니다. 우회전 2개는 같은 maneuver에서 나온 것이고 좌회전은 1개입니다. 곡선 loss 위치의 65.1%는 일반적인 route-follow 문장, 10.9%는 다음 교차로 직진 문장, 24.0%는 명시적인 좌/우 지시였습니다.

도로가 휘는 것과 다음 교차로에서 회전하는 것은 다르므로 이 분포 자체를 annotation 오류로 단정할 수 없습니다. 다만 회전 형상을 전부 turn 단어에서 얻을 수 없고, 기존 visual feature가 나머지 도로 정보를 제공해야 합니다. 현재 결과만으로 encoder 정보 부족과 downstream 정보 활용 부족을 분리하지 못했습니다.

### 첫 초록불 출발 문제는 이번 pilot의 검증 범위에서 빠짐

알려진 첫 초록불 frame991/1591/2491/2791은 이번 345개 supervision frame과 32개 평가 anchor에 없습니다. 따라서 이번 결과로 신호 전환 후 출발 문제가 개선됐거나 실패했다고 판정할 수 없습니다.

frame2490처럼 현재는 빨간불·정지 상태인데 곧 초록불이 되어 2.5초 GT가 전진하는 표본은 포함됐습니다. 이것은 실제 미래 ego 정답이며, 이미 보이는 초록불을 무시하는 것과 다른 문제입니다. 입력에 신호 잔여 시간이 없는 상황의 미래 상태 예측 모호성을 별도로 다뤄야 하며, 모든 빨간불 GT를 0으로 바꾸는 것은 실제 주행 기록을 왜곡합니다.

새 traffic-label-v2 보정도 이 pilot의 기존 epoch25 cache에는 반영되지 않았습니다. 계획된 label 보정 encoder 미세학습과 cache 재생성은 비교 gate 이후 단계였으므로 아직 실행되지 않았습니다.

근거: [data sampling audit](outputs/diagnosis_pilot_20260911/data_sampling_audit.md), [정량 결과](outputs/diagnosis_pilot_20260911/data_sampling_audit.json).

## 6. optimizer와 학습 예산: '실제로 업데이트 안 됨'은 아님

- Adam state가 있는 532개 tensor는 모두 step320에 도달했습니다.
- 검사한 31개 tensor/모델에서 저장 모델 값은 FP32 master를 해당 dtype으로 변환한 값과 정확히 같았습니다.
- 160→320 사이 waypoint head, 사용된 instruction embedding, query FFN, MEFA, projection, selector의 실제 weight가 바뀌었습니다.
- 비활성인 ordinary BERT text FFN 72개 tensor와 실제 사용하는 query FFN을 구분했습니다. 미사용 파라미터가 있다고 전체 Q-Former가 끊겼다고 볼 수 없습니다.

다만 양 모델 모두 320 step 전체에서 global gradient clipping 상한 10에 걸렸습니다. clipping 이전 norm은 약 1.8만–30만입니다. reconstruction은 LLM에 직접 gradient를 보내지 않지만, 공유 clipping은 LLM gradient에도 영향을 줍니다.

실제 저장된 Adam state에서 layer0 Q/K의 많은 좌표는 `sqrt(v_hat)`가 epsilon=1e-8보다 작았습니다. A Q/K는 97.3%/98.7%, B는 80.7%/91.6%였습니다. 이 좌표에서는 epsilon 항의 영향이 상당합니다. 따라서 전체 stack이 명목상 같은 LR로 균형 있게 학습된다고 가정하기 어렵습니다. 하지만 이것만으로 clipping이 회전 실패의 유일한 원인이라거나 epsilon 변경이 해결책이라고 결론내릴 수는 없습니다.

또한 320 update는 9,540-update schedule의 3.35%입니다. LR는 약 1e-5로 유지되고 있으며 0으로 사라진 문제가 아닙니다. 초기 모델 대비 오차는 줄었습니다. gate 미통과는 다음 단계 자동 진행을 막을 근거이지, 더 학습해도 불가능하다는 수렴 증명은 아닙니다.

과거 9,540-update 모델은 같은 32개 anchor에서 ADE 1.286 m/FDE 2.928 m였습니다. 새 pilot보다 낫지만, 학습량·sampling·과거 RoPE 구현이 함께 달라 이 차이를 구조 수정 효과만으로 해석할 수 없습니다.

## 7. 논문 동일성과 관련해 확정할 수 있는 범위

VLDrive 본문은 trajectory L1, pruning 및 reconstruction 목적, 유지 목표 0.3, memory 10, 가중치 10/1을 제시합니다. 현재 코드는 waypoint의 10개 좌표 오차는 평균내고 reconstruction은 frame별 token/channel을 합산합니다. 이 scalar reduction과 optimizer 세부까지 본문이 완전히 특정하지는 않습니다. 따라서 관측된 스케일 문제를 곧바로 저자 구현과 다른 버그라고 단정할 수 없습니다. [VLDrive 본문 및 수식](https://arxiv.org/html/2511.06256v1#S3.S6)

확인 시점의 저자 저장소는 전체 trainer를 제공하지 않아 세부 동등성을 직접 대조할 수 없었습니다. '논문이 잘못됐다'와 '우리가 공식 구현을 정확히 재현했다' 모두 이번 결과에서 도출되지 않습니다. B의 독립적인 backward surrogate 역시 공식 구현으로 부르면 안 됩니다. [저자 저장소](https://github.com/ReaFly/VLDrive)

추가로 현재 BERT 로더는 pretrained ordinary FFN을 `_query` FFN으로 복사하지 않아 query FFN은 무작위 초기화됩니다. Salesforce BLIP-2 초기화는 그 복사를 수행합니다. 이는 확인된 초기화 차이지만 VLDrive 위반이나 실패 원인으로 확정한 것은 아닙니다. [Salesforce BLIP-2 초기화](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py#L69)

## 8. 다음 판단의 우선순위 — 이번 턴에서는 실행하지 않음

1. **원래 목적함수 안에서 수치 계약을 먼저 명확히 하기.** 복원/주행 loss의 scalar reduction, target feature 스케일, 공유 gradient clipping, 초기화 계약을 문서·코드로 대조합니다. 공식 확인이 불가능한 선택은 가정으로 표시하고 분리 실험해야 합니다. 현재 증거만으로 BCE·강제 top-k를 해결책으로 도입하지 않습니다.
2. **평가 묶음의 빈틈 보완.** 정지와 이동, 전진 길이와 좌우 변위를 따로 평가하고, 실제 명시적 회전 및 첫 초록불 전환 window를 기존 train 데이터에서 포함시켜야 합니다. 이는 곧바로 재수집을 요구하는 문제가 아닙니다.
3. **원인별 비교 예산을 맞추기.** 동일 초기값·샘플·update 수로 비교하고, 개선이 압축 수치만인지 실제 이동/회전인지 구분합니다. 초기 320 update gate로 장기 수렴 여부까지 판단하지 않습니다.
4. **vision encoder 재학습 필요성은 별도 판정.** 현재 frozen token에 필요한 도로·신호 정보가 있는지와 downstream이 이를 활용하는지를 분리해야 합니다. 이번 A/B는 encoder를 바꾸지 않았으므로 encoder 문제를 배제하지도, 반드시 재학습해야 한다고 입증하지도 못했습니다.

현재 가장 단단한 결론은 **CCDP의 복원/가지치기 경쟁이 불균형하고, 작은 task-gradient 수정만으로는 해결되지 않았으며, 주행 예측에는 이동거리 과소예측과 방향 조건 학습 미완성이 함께 남아 있다**는 것입니다. 이들을 하나의 원인으로 합쳐 설명하지 않는 것이 다음 실험의 출발점입니다.

## 재현 자료

- [Full-graph 진단 스크립트](tools/morai/diagnose_pilot_full_graph.py): 같은 저장 가중치의 loss별 gradient, 선택 방식 및 지시문 민감도 검사. optimizer update 없음.
- [GPU 진단 JSON](outputs/diagnosis_pilot_20260911/full_graph.json): complete=true, optimizer_updates=0, 양 모델 trainable/frozen hash 불변.
- [CCDP audit](outputs/diagnosis_pilot_20260911/selector_audit.json): 원본 optimizer checkpoint 4개의 전체 SHA 검증, 초기/160/320 selector 비교.
- [Optimizer audit](outputs/diagnosis_pilot_20260911/training_audit.json): 실제 업데이트, FP32 master, Adam denominator 및 LR 확인.
- [Data/trajectory audit](outputs/diagnosis_pilot_20260911/data_sampling_audit.json): 원본 GT 재계산, 실제 sampling과 잔차 분해.

Gradient cosine의 1을 아주 조금 넘는 값은 float32 norm/내적의 반올림 오차이며 보고서에서는 약 1로 해석했습니다. Gradient 진단용 clip 3개와 instruction 진단용 clip 2개는 제한된 표본이므로 전체 데이터의 인과 검증으로 일반화하지 않습니다.
