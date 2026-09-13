# 최종 모델의 회전·CCDP 오류 원인 진단

> 후속 재감사: [근본 원인 재감사](MORAI_ROOT_CAUSE_AUDIT_20260911.md).
> 아래의 관측 결과는 보존하되, 데이터 부족/늦은 지시 및 reconstruction의 직접 mask 경로를 주원인처럼 해석한 부분은 후속 보고서의 반증과 함께 읽어야 한다.
> 새로 신호 보조 라벨의 정책 불일치와 RoPE 수치 구현 결함을 확인했다. Off-paper 수정 실험은 일시정지 상태다.

2026-09-11. Train만 분석. 학습·원본 dataset·기존 모델 가중치는 변경하지 않았다.
대상은 `outputs/morai_paper_last_encoder_20260910T055054Z/model-last.pt`, 9,540 updates 완료 모델이다.
SHA256: `a63a619b2565d2bb0fba13ddfe305779f344bf67d0fbd9aafa7f6476c9f14b9a`.

## 1. 회전 지시의 의미를 출력 궤적에 반영하지 못함 — 직접 확인

동일한 train 시각 feature와 history를 유지하고 instruction만 교체했다.
아래 값은 2.5초 뒤 **오른쪽 이동 거리**다. 실제 head 좌표 `[left,-forward]`에서 부호를 변환했다.

| Frame | 실제 expert | 우회전 지시 | 좌회전 지시 | 직진 지시 |
|---|---:|---:|---:|---:|
| 2865 | 4.457 m | 0.271 m | 0.271 m | 0.268 m |
| 3320 | 8.110 m | 0.605 m | 0.605 m | 0.605 m |

방향 단어가 바뀌어도 실제 좌우 회전이 바뀌지 않는다. 다만 일반 문장으로 바꾸면 전진 거리는 달라진다.
예를 들어 frame3320에서 교차로 직진 문장은 전진16.25m, 일반 경로 따라가기 문장은22.375m를 예측한다.
따라서 instruction 입력 경로가 통째로 끊겼다는 뜻은 아니다. **문장에 반응하되 방향 의미를 제대로 grounding하지 못했다.**
반사실 입력에는 그에 맞는 새 GT가 없으므로, 교체 문장의 ADE/FDE를 새로운 주행 성능으로 해석하지 않는다.

History를 같은 현재 frame으로 채우거나 과거 순서를 뒤집으면 일부 출력도 변한다.
예를 들어 frame280은 과거 순서를 뒤집으면 endpoint 전진4.219m→5.594m로 변한다.
따라서 MEFA/과거 입력이 완전히 단절된 현상으로도 설명되지 않는다.

원자료: [counterfactuals.json](outputs/morai_paper_last_encoder_diagnosis_20260911/counterfactuals.json).
Inference allowlist: [training_model.py:140](lavis/models/vldrive/training_model.py:140).

## 2. Vision encoder의 낮은 waypoint 오차에는 숫자형 경로 입력의 도움이 있었음 — 직접 확인

Encoder의 기존 GRU waypoint head는 `target_point`를 받아 hidden state를 초기화한다.
반면 token 반환은 이 head를 실행하기 **전**에 일어난다. Token은 RGB·LiDAR·현재 속도에는 의존하지만
`target_point`에는 의존하지 않는다. 후속 모델의 입력은 token과 텍스트 지시문뿐이다.

근거: [memfuser.py:201](../vision_encoder/timm/models/memfuser.py:201),
[memfuser.py:863](../vision_encoder/timm/models/memfuser.py:863),
[training_model.py:140](lavis/models/vldrive/training_model.py:140).

같은 최종 encoder checkpoint의 GRU head를 strict load하고, pooling되지 않은 cached waypoint token 5개를 넣어 비교했다.
숫자형 target point의 **전방 성분은 유지한 채 좌우 성분만 0으로 바꾸는** 추가 실험도 했다.

| Frame | 실제 오른쪽 이동 | 기존 head + 원래 target point | 같은 head + target point 좌우 성분 0 | 후속 모델 |
|---|---:|---:|---:|---:|
| 2865 | 4.457 m | 3.949 m | 0.513 m | 0.271 m |
| 3320 | 8.110 m | 7.619 m | 0.764 m | 0.605 m |

Target point의 좌우 부호만 뒤집으면 같은 token에서도 기존 head는 각각 **왼쪽2.663m/4.333m**를 예측한다.
즉 이 두 장면의 기존 head 회전 성능에는 숫자형 경로 입력이 큰 역할을 했다.
원래 target point를 쓴 기존 head의 FDE는0.778m/0.526m, 후속 모델은4.208m/7.871m다.

이 실험은 숫자형 경로가 token에 자동으로 들어 있다고 가정하면 안 된다는 증거다.
다만 두 head의 입력이 다르므로 **encoder는 완벽하고 connector만 고장났다고 결론낼 수는 없다.**
시각 token에 곡률 정보가 전혀 없다는 증거도 아니다. 현재 downstream은 시각 feature와 언어에서 곡률을 다시 배워야 한다.
기존 head의 성공을 후속 모델의 성공으로 곧바로 연결한 기대가 지나쳤다.

원자료: [vision_head_probes.json](outputs/morai_paper_last_encoder_diagnosis_20260911/vision_head_probes.json),
[vision_lateral_target_probes.json](outputs/morai_paper_last_encoder_diagnosis_20260911/vision_lateral_target_probes.json).

## 3. 데이터 구성·지시문 시점이 회전 학습에 불리함 — 사실 확인, 기여율은 미확정

실제 학습 loss에는636 clip×11=6,996개 frame 위치가 들어간다. 겹치는 clip을 포함한 수이며 고유 frame은3,282개다.

| 구분 | 학습 frame 위치 수 | 비율 |
|---|---:|---:|
| 미래2.5초 동안 현재 위치에서0.5m 이상 벗어나지 않음 | 4,572 | 65.35% |
| 일반 경로 따라가기 지시 | 6,523 | 93.24% |
| 교차로 직진 지시 | 275 | 3.93% |
| 우회전 지시 | 110 | 1.57% |
| 좌회전 지시 | 88 | 1.26% |

정지와 지시문 분류는 서로 겹친다. 마지막 네 지시문 행만 합이100%다.
우회전10 clip은 단 한 번의6.0초 사건, 좌회전8 clip도 단 한 번의5.2초 사건을 나눈 것이다.
18개의 독립적인 회전 사건이 아니다. 반복 epoch를 늘려도 독립 사건이 늘지는 않는다.

또한 instruction builder는 앞쪽 회전을 미리 찾아 알리는 것이 아니라 **현재 링크의 `related_signal`**을 문장화한다.
Frame2830은 미래2.5초 궤적이 이미 우측2.58m 꺾이는데 아직 일반 주행 지시다.
2835에서 우회전 문장이 시작하고, 이미 차량이 돌고 있다.
11-frame 내 instruction 변경을 금지하므로2835–2844를 마지막 frame으로 하는 window는 제외된다.
첫 우회전 anchor2845는 진입1초 후이며 heading이89.2°에서77.1°까지 바뀐 뒤다.

2835 이후 frame 자체가 loss에서 전부 사라지는 것은 아니다. 2845 window의 앞부분에는 포함된다.
빠지는 것은 **회전 진입 전 history와 새 지시를 함께 보는 window**다.
3,342개 후보 중166개가 이런 instruction 변경으로 제외된다.

근거: [mgeo_navigation.py:439](../../MoraiDrive/data_collection/mgeo_navigation.py:439),
[mgeo_navigation.py:592](../../MoraiDrive/data_collection/mgeo_navigation.py:592),
[build_global_path_instructions.py:72](tools/morai/build_global_path_instructions.py:72),
[expert_vldrive_dataset.py:223](lavis/datasets/datasets/morai/expert_vldrive_dataset.py:223).

Frame3320의 직진 문장은 MGeo 링크`A2256W000151`의`related_signal="straight"`에서 정상 변환되었다.
하지만 교차로 maneuver의 직진은 현재 도로의 곡률이0이라는 뜻이 아니다.
단순 좌우 부호 오류가 아니라 **거친 지도 표지를 문장으로 바꿨을 뿐 정확한 경로 곡선을 전달하지 못하는 한계**다.

데이터 편향과 늦은 지시는 회전 의미를 덜 학습한 결과를 강하게 뒷받침한다.
다만 균형 sampling/지시문 개선을 각각 적용한 비교 학습을 하지 않았으므로 원인별 기여율이나 단독 인과는 확정하지 않는다.

## 4. CCDP는 현재 loss의 전부 유지 탈출 경로로 붕괴 — 코드·로그·가중치 직접 확인

현재 reconstruction은 버린 token에 대해서만 비용을 부과한다.
한 frame에서 token j의 재구성 제곱오차 합을 E_j, 유지 mask를 M_j라 하면:

`L_rec = sum_j (1-M_j) E_j`

Mask 자체에도 gradient가 흐른다. 모두 유지하면 reconstruction 값과 reconstruction decoder에 흐르는 해당 loss의 gradient는0이 되지만,
mask에 대한 직접 gradient `-E_j`는 남아 해당 token을 더 유지하게 한다.
반면 pruning은 `10*(mean(M)-0.3)^2`이므로 전부 유지 지점의 mask gradient는 약`+0.132/token`이다.

Frame2865의11-frame window에서 직접 계산한 E_j는 최소140.5, 중앙172.3, 최대268.9였다.
즉 현재 지점에서는 reconstruction 쪽 직접 gradient가 pruning보다 훨씬 강하게 keep을 선호한다.
이는 정상적인 재구성 성공이 아니라 **문제를 복원하지 않고 평가 대상에서 없애는 경로**다.

- 첫 update: reconstruction31,317.52, weighted pruning0.258, waypoint0.377.
- 전체9,540 update 중8,724개에서 reconstruction0/pruning0.49.
- Update2573부터 마지막까지6,968회 연속 전부 유지.
- 최종 train671 frame/71,126 token CPU 검사: keep확률 최소0.99999285,97.8868%는 FP32에서도 정확히1.0.
- Weighted pruning loss는4.9인데 pruning만의 selector gradient L2는3.075e-5로 작다.
- 추론 argmax뿐 아니라 seed607 Gumbel 선택에서도100% 유지다.

따라서 inference threshold 문제나 pruning loss 누락이 아니다. **목적함수의 경쟁 방향·크기와 확률 포화 문제**다.
LayerNorm weight는0.992–1.016으로 norm scale이0이 된 현상도 아니다.

근거: [qformer_mefa.py:1084](lavis/models/vldrive/qformer_mefa.py:1084),
[ccdp.py:254](lavis/models/vldrive/ccdp.py:254),
[training_model.py:118](lavis/models/vldrive/training_model.py:118),
[training.jsonl](outputs/morai_paper_last_encoder_20260910T055054Z/training.jsonl).

토큰이 전부 보존되었으므로 **회전에 필요한 token을 CCDP가 버려서 직진한다는 설명은 틀리다.**
압축 실패는 별개의 확정 문제이며, 초기 큰 auxiliary loss가 trajectory 학습에 끼친 추가 영향은 비교 학습 없이는 수치화할 수 없다.
이 결과를 논문 자체의 오류나 공개 저자 구현과의 차이라고 단정하지 않는다. 현재 독립 구현에서 관측한 현상이다.

## 5. 출발 사례는 신호 전환 전후를 구분해야 함

Frame990은 빨간불이고0.1초 뒤991에서 녹색으로 바뀐다. Native/gRPC tick은 일치한다.
990에서 expert의 미래2.5초 위치는10.22m 앞이지만 현재·과거 입력에는 아직 녹색이 없다.
그 시점의10.23m FDE만으로 녹색 신호를 무시하거나 영구 정지한다고 판단할 수 없다.

다만991의 RGB에는 실제 녹색이 보이는데 모델은 아직 정지를 예측한다.
995에서는 출발 경로를 예측하고 FDE는0.577m다. 이때 실제 expert가 이미 움직이므로,
단순히995의 성공만으로 모델이 신호를 보고 스스로 출발할 수 있다고 결론내려서도 안 된다.
현재 속도 의존 여부를 별도 offline intervention으로 확인했다. 동일한 RGB·LiDAR·지시문을 유지하고
11-frame의 속도를0으로 바꾸거나 마지막 속도만 바꾸어 raw encoder부터 다시 추론했다.

| 장면·입력 교체 | 후속 모델의2.5초 endpoint 거리 |
|---|---:|
| 995 원래 속도 | 15.39m |
| 995 history 전체 속도0 | 14.15m |
| 991 원래 속도 | 0.00037m |
| 991 현재 속도만995의0.60246m/s로 교체 | 0.01713m |

995는 속도를0으로 해도 출발을 예측하고991은 속도만 높여도 거의 정지다.
따라서 **현재 속도만 따라가기 때문에 출발하지 못한다는 단독 설명은 지지되지 않는다.**
기존 vision head도995의속도0에서11.83m를 예측하므로 속도 외 시각 장면에 따른 encoder 표현 차이도 존재한다.
991→995 사이에는 녹색 유지 history와 실제 차량 이동에 따른 RGB/LiDAR 변화가 함께 있다.
어느 변화가 지연 반응을 해소했는지는 이 실험으로 완전히 분리하지 않았다.
따라서 첫 녹색 frame의 정지 예측은 확인했지만, 자력 출발 성공/실패나 영구 정지는 closed-loop 검증 없이 확정하지 않는다.

실험 기록: [departure_speed_probes.json](outputs/morai_paper_last_encoder_diagnosis_20260911/departure_speed_probes.json).

원본 [0991.jpg](../../MoraiDrive/dataset_local/competition_train/data/full_loop_basic_drive_2026_09_09_01_44_15/rgb_front/0991.jpg),
[0995.jpg](../../MoraiDrive/dataset_local/competition_train/data/full_loop_basic_drive_2026_09_09_01_44_15/rgb_front/0995.jpg).

## 수정 우선순위와 한계

1. 회전·이동·정지, 신호 전환 구간을 따로 평가하고 학습 sampling 비중을 분리한다. 평균 ADE만으로 통과시키지 않는다.
2. 수집된 global path에서 다음 maneuver를 미리 추출해 instruction 시점을 개선하고, 전환 history를 보존할 정책을 정한다.
   기존 dataset의 annotation 재생성으로 가능한 작업이며 원본 sensor를 바로 다시 수집해야 한다는 뜻은 아니다.
3. 숫자형 target point가 없는 후속 단계에서 방향 의미를 배웠는지 좌/우 지시 교체와 곡률별 train 오차로 확인한다.
   논문 경로를 유지하려면 숫자형 경로 입력을 몰래 추가하는 대신 language/vision 학습과 데이터 구성을 개선해야 한다.
4. CCDP는 현재 포화 selector를 그대로 오래 학습시키는 것보다 loss-mask gradient/정규화/retention 목적을 작은 실험에서 먼저 검증한다.
   재구성오차를 가중하는 `(1-M)`에만 stop-gradient하는 실험, 정규화, 강제 top-k 등의 선택은 서로 다르며 논문 재현 설정과 구분해야 한다. 이번에는 적용하지 않았다.
5. 초록불+속도0에서 출발하는지 offline counterfactual과 이후 simulator closed-loop로 검증한다.

원본 GT는 실제 미래 ego 위치와 일치했고, raw sensor 재인코딩과 token cache 예측도 이미 일치했다.
따라서 global-path 미래 waypoint를 잘못 정답으로 쓴 오류나 cache만의 오류를 주된 원인으로 볼 증거는 없다.
NPC 부재가 위의 loss 붕괴나 좌우 지시 무반응을 직접 유발했다는 증거도 없다.

분석 도구: [diagnose_train_predictions.py](tools/morai/diagnose_train_predictions.py),
[diagnose_departure_speed.py](tools/morai/diagnose_departure_speed.py).
이번 변경은 진단 도구와 보고서 추가에 한정된다. 모델 구조·loss·학습 데이터·가중치 수정이나 재학습은 하지 않았다.
