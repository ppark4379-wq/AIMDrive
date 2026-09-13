# MORAI 라벨·시간 정합성 및 조건별 학습 실패 재진단

2026-09-11. 대상 데이터는 정상 보존된 4,240프레임이며, 모델 진단은 기존 9,540-update 최종 checkpoint를 대상으로 한다. 최근 320-step pilot 결과와 혼합하지 않는다. Validation 주행 성능은 평가하지 않았으며 전체 데이터 파일의 무결성만 검사했다.

## 결론

- 현재 정상 데이터에서 **프레임 혼합, 센서 파일 오연결, ego/신호 라벨의 100ms 단위 시간 밀림을 주원인으로 볼 증거를 찾지 못했다.** 메타데이터뿐 아니라 실제 원본 파일과 영상의 신호 전환도 대조했다.
- **라벨 정의·유효성에는 별도 문제가 남아 있다.** 신호 상태를 모르는 표본을 유효한 not-red 정답으로 쓰는 70프레임과 red/green 허용이 함께 있는 112프레임의 의미를 구분해야 한다. 후자는 물리적인 red 성분과 해당 경로의 정지 의무가 다르다는 문제이며 112개 모두 색이 틀렸다는 뜻은 아니다.
- 회전 방향 GT는 반전되지 않았다. 최종 모델은 좌/우 입력 차이를 출력으로 거의 전달하지 않는 상태다. L1·표본 비율·학습된 특징/출력 가중치가 만드는 학습 압력을 추가 측정했으나, 단일 원인이나 논문 구조의 결함으로 확정하지 않는다.
- 첫 녹색에서 실제 이미지와 신호 라벨은 맞는다. 과거를 제거해 현재 프레임만 주어도 출발 실패가 남았으므로 단순히 MEFA가 빨간불 history로 녹색을 덮는다는 설명은 충분하지 않다.
- 원본 데이터, production 코드, checkpoint는 수정하지 않았다. 실행한 GPU 작업은 재추론뿐이며 optimizer update는 0이다. 아래 학습 실험은 제안이며 시작하지 않았다.

## 1. 데이터 생산부터 학습 입력까지 검증한 범위

| 검사 | 범위 | 결과 |
|---|---|---|
| Dataset frame → native tick | 4,240프레임 | `native = 161 + 5 * frame_id`, gap/중복 없음 |
| aggregate/full/compact/original-prefix/manifest/spool 대조 | 4,240프레임 | 검사한 필드 불일치 0 |
| 같은 tick의 ROS ego ↔ gRPC ego | 4,240프레임 | 위치·heading 차이 정확히 0 |
| 원본 PNG → 저장 JPG | 카메라 4대 × 4,240 = 16,960장 | JPEG quality 92 재인코딩 bytes와 전부 동일 |
| 원본 BIN → 저장 NPY | 4,240개 | float32 배열 bytes와 전부 동일 |
| 속도와 위치 변화량 위상 검사 | -0.5~+0.5초 shift | 0초가 최선. ±0.1초로 옮기면 오히려 크게 악화 |
| 신호 색의 영상 전환 ↔ gRPC 전환 | 첫 녹색 4사건 주변 36장 | 991/1591/2491/2791에서 같은 100ms 수집 프레임에 전환 |
| Cache feature/target 파일 | 4,190쌍 = 8,380개 파일 | hash·경로 frame 이름·shape·finite 검사 통과 |
| 원본 → encoder → cache 재계산 | train 24프레임, batch 1 및 원래 batch 4 | 모든 재계산 특징의 비교 후보 중 같은 frame cache가 가장 가까움. 모든 GT 정확히 일치 |

Raw simulator timestamp는 tick당 약 20,000 단위 증가한다. 20ms tick과 대조하면 microsecond 단위에 부합하며 전체 데이터의 누적 차이는 0~20 raw units다. 이를 20ms 혹은 100ms 밀림과 혼동하지 않는다.

속도와 중앙 위치 차분의 0-lag RMSE는 약 0.01475m/s, 이동 중 상관계수는 0.9999967이다. ±100ms 속도 shift 시 RMSE는 약 0.16m/s로 악화된다. 이는 상수적인 한 데이터 프레임 지연 가설에 반대되는 증거이지, 모든 센서 내부 동작의 지연을 직접 측정한 것은 아니다.

캐시 재계산은 bitwise 동일성을 모두 통과한 것은 아니다. 원래 batch 4에서는 24개 중 3개가 완전히 같고 나머지 최대 성분 차이는 0.001953125, 표본별 평균 절대 차이는 최대 약 0.000219다. 같은 frame의 feature L2 차이는 0~0.061인 반면 검사한 다른 시점 후보들은 훨씬 멀다. 따라서 frame swap 증거는 없지만 작은 수치적 재현 차이의 정확한 원인까지 단정하지 않는다. Batch 1보다 batch 4에서 오차가 작아졌다. Cache 생성 이후 dataset source에 sidecar 옵션이 추가됐다는 점도 hash 보고서에 명시했다.

보존된 native PNG에는 독립 촬영 timestamp chunk가 없다. 파일 번호와 실제 시뮬레이터 내부 render 시점 사이에 숨은 1 tick(20ms) 차이가 전혀 없다고 증명할 수는 없다. 다만 확인한 신호 사건에서 1 dataset frame(100ms) 지연은 없었다.

근거: [metadata](outputs/diagnosis_label_alignment_20260911/collector_metadata.json), [전체 sensor payload](outputs/diagnosis_label_alignment_20260911/collector_payload_summary.json), [속도/제어](outputs/diagnosis_label_alignment_20260911/ego_kinematic_control_audit.json), [신호 pixel 전환](outputs/diagnosis_label_alignment_20260911/traffic_pixel_timing.json), [cache batch4](outputs/diagnosis_label_alignment_20260911/cache_sample_alignment_batch4.json).

## 2. 후처리 라벨: 정상적인 것과 정의가 필요한 것

### 미래 ego trajectory, heading, speed

Train 3,352개 trajectory는 이전 전수 감사에서 실제 미래 ego +5/+10/+15/+20/+25프레임을 기준으로 float32 오차 0으로 재계산했다. 이번 raw dataset loader 재계산 24개도 정확히 같았다. 원본의 global-path `future_waypoints` 필드를 GT로 잘못 사용하지 않는다.

Heading degree→radian, ego 위치, speed의 3차원 속도 norm 변환도 원본과 일치한다. 이동 중 현재 heading과 바로 다음 위치의 진행방향 차이 중앙값은 약 1.35도로, 90/180도 축 오류를 지지하지 않는다. 사용 좌표는 `[left, negative_forward]`, 단위는 m다.

`gps_x/gps_y`라는 이름은 이 데이터에서 실제 GNSS 센서가 아니라 simulator ego.position의 별칭이다. GNSS/IMU 폴더에는 파일이 없고 현재 VE/후속 모델도 해당 폴더를 읽지 않는다. 따라서 GPS와 ego의 일치를 서로 독립된 센서 증거로 계산하지 않았다. 미래 ego GT에 simulator 위치를 사용하는 것 자체는 현재 정답 정의와 일치한다.

- 우회전 지시 2835~2894: 실제 heading 변화 -85.31도, 60/60 미래 종점 lateral 음수.
- 좌회전 지시 2947~2998: 실제 heading 변화 +88.04도, 52/52 미래 종점 lateral 양수.

따라서 학습한 좌회전 GT 자체가 우회전으로 뒤집혀 있다는 가설은 맞지 않는다.

### 신호 보조 라벨

기존 train에서 관련 신호 상태를 확인하지 못한 **70프레임**도 `traffic_available=True`, not-red로 처리됐다. 이는 unknown을 확정 정답으로 사용하는 유효성 문제다.

상태 33의 **112프레임(2835~2946)**은 red 성분과 green 허용 성분이 함께 있다. 기존 라벨은 red=True지만 차량은 움직였고 expert 신호 정지 정책도 이 상태를 단순 red-stop으로 처리하지 않는다. 물리적 red 존재와 ego 경로의 GO/STOP을 분리해야 한다. 특히 우회전 지시 60프레임 전체가 이 구간에 포함된다. 하지만 이것이 waypoint GT 오류 또는 회전 실패의 인과라는 뜻은 아니다.

기존 보수적 sidecar는 이 70+112=182프레임을 다른 클래스로 바꾸지 않고 traffic loss 대상에서 제외한다(3,352→3,170 유효 표본). 기존 epoch25 encoder 및 cache에는 아직 반영되지 않았다. 새 데이터를 동일한 후처리로 만들면 같은 정의 문제가 재발할 수 있다.

### Navigation instruction의 시간과 의미

실제 yaw-rate가 5도/s를 넘는 회전 시작과 v1 지시 시작은 해당 두 사건에서 일치한다. 반면 2.5초 미래 궤적이 lateral 2m를 넘기 시작하는 것은 지시보다 0.7/0.6초 빠르다. 이는 현재 회전 command와 미래 trajectory horizon의 시간 범위가 다르기 때문이다. 현재 회전보다 지시가 수초 늦거나 좌우가 반대로 붙었다는 결론은 아니다.

`Please proceed straight at the next intersection.`이 붙은 일부 프레임에서 미래 궤적은 상당히 휜다(예: 3320, lateral -8.11m). 그러나 교차로의 위상적 직진은 곡선 차선을 따라가는 것과 양립할 수 있다. 지도·교차로 topology를 별도로 검증하지 않았으므로 문구 오류로 확정하지 않는다. 현재 link에 붙은 command를 'next intersection'이라고 번역하는 적용 범위와 예고 문맥은 감사할 후보다.

### 제어·객체 필드

Throttle/brake는 기록된 applied control과 일치한다. 명령 N은 N→N+1 단계에 적용되므로 같은 N의 ego actuation 대신 다음 tick과 더 잘 맞을 수 있다. 이 표본에서 다음 tick 값과 차이는 약 3e-8 이하다. Callback age=0이 expert가 정확히 어떤 센서 frame을 보고 명령을 만들었는지까지 증명하지는 않는다.

기존 후속 모델의 GT는 실제 미래 ego trajectory이며 throttle/brake/should_brake/target_speed를 정답으로 사용하지 않는다. 현재 type1 accel/brake 제어에서 target_speed=0인 필드는 정지 명령의 증거가 아니다. 객체 raw와 actors가 빈 것은 사용자가 NPC 없이 수집한 시나리오와 일치하며, 이를 수신 실패나 잘못된 객체 GT로 판정하지 않는다.

근거: [라벨 의미 감사](outputs/diagnosis_label_alignment_20260911/labels_semantics.json), [제어 위상 감사](outputs/diagnosis_label_alignment_20260911/ego_kinematic_control_audit.json).

## 3. 빨간불 직전의 긴 미래 GT는 시간 오류가 아닌 예측 모호성

첫 녹색 직전 990/1590/2490/2790은 이미지와 라벨 모두 빨간불이지만 +2.5초의 실제 이동 거리는 각각 10.23/10.09/9.90/3.83m다. 그 사이에 실제로 녹색이 되고 출발했기 때문이다. 같은 표본의 +0.5초 이동은 0.032~0.056m에 불과하다.

확인된 빨간불에서 완전히 정지한 train 2,079프레임 중 미래 이동이 0.5m를 넘는 개수:

| 미래 시점 | +0.5s | +1s | +1.5s | +2s | +2.5s |
|---|---:|---:|---:|---:|---:|
| 해당 표본 수 | 0 | 6 | 26 | 46 | 66 |

예를 들어 961과 990은 최근 1초 동안 위치·heading·빨간불·지시가 같지만 미래 종점은 10.23m 차이난다. Feature bytes는 같지 않으므로 수학적으로 동일 입력에 모순 정답을 붙였다고 말할 수는 없다. 다만 남은 신호 시간을 입력받지 않는 모델은 비슷한 빨간불 정지 장면에서 미래 녹색 전환을 미리 예측해야 한다.

이는 실제 ego 미래 trajectory 라벨을 사용하는 방식에서 생기는 예측 모호성이다. GT를 timestamp 오류라고 간주하거나 빨간불이면 미래 5개 좌표를 강제로 0으로 덮어쓰지 않는다. 그런 수정은 실제 expert 미래 기록이라는 정답 정의를 바꾼다. 또 긴 horizon에 출발이 포함된다고 현재 즉시 가속하라는 제어 정답이 되는 것도 아니다.

## 4. 모델의 입력 활용을 직접 바꿔 확인한 결과

최종 모델에서 회전 4표본(2865/2870/2970/2975), 첫 녹색 4표본(991/1591/2491/2791)에 총 44회 forward를 수행했다. 과거 학습 당시 RoPE를 진단 프로세스 안에서만 재현했고 회전 baseline은 저장 예측과 정확히 일치했다. Trainable/frozen hash와 production source hash 불변, optimizer update=0이다.

- **같은 회전 영상에 left/right 지시만 교체:** 네 표본의 최종 lateral 변화는 0~0.001953m. 전진값은 BF16 한 단계만큼 달라지는 경우도 있지만 좌우 방향을 정상적으로 바꾸지는 않았다.
- **반대 회전의 current visual token으로 교체:** 원래 history와 지시를 유지하면 종점 변화는 약 0.079~0.096m. 입력 전체를 아예 무시하는 것은 아니지만 검사한 두 회전 장면의 차이를 충분한 회전 차이로 사용하지 못한다. 이 swap은 속도·위치·scene도 바꾸므로 순수 곡률의 인과 실험이 아니다.
- **첫 녹색에서 빨간불 history 제거, 현재 frame만 제공:** 네 경우 모두 사실상 정지를 유지했다. 따라서 과거 평균이 녹색을 덮는 문제 하나로 설명할 수 없다. 이 비교는 history/MEFA/position을 함께 바꾸므로 특정 모듈 하나의 기여율은 구분하지 못한다.
- **첫 녹색 current feature를 바로 전 빨간불 feature로 교체:** 종점 변화는 0.54~1.13mm. Traffic token(index100)만 교체해도 약 0.31~0.72mm에 불과했다. 이 token은 순수 신호 색 숫자가 아니라 학습된 융합 특징이므로 color-only 개입으로 해석하지 않는다.

이 실험은 입력 민감도 진단이지 변경된 입력의 정답 궤적을 안다는 뜻이 아니다. Counterfactual 결과의 ADE/FDE로 주행 성능을 주장하지 않는다.

근거: [44회 입력 개입 결과](outputs/diagnosis_label_alignment_20260911/context_input_use.json), [재현 도구](tools/morai/diagnose_context_input_use.py).

## 5. 회전 학습이 약한 현재 메커니즘

기존 636개 train anchor의 좌회전34/우회전47을 비교하면 GT 종점 lateral 평균 차이는 7.964m인데 실제 출력 차이는 0.056m다. 전진 방향 평균 차이는 GT -2.258m/출력 -2.198m로 상대적으로 잘 보존된다.

최종 lateral weight가 0이어서 생긴 것은 아니다. 그 weight와 좌/우 평균 feature 차이 방향이 거의 직교(cosine0.0107)한다. 마지막 계산에서 좌우 차이를 출력으로 읽지 않는 상태다. 평균 특징 분석으로 모든 개별 회전 정보가 사라졌다고 단정하지 않는다.

L1에서는 오차가 0이 아니면 좌표 미분 크기는 오차량이 아니라 부호에 의해 결정된다. 매우 작은 정지 오차도 학습 항이 되고 큰 회전 오차라고 자동으로 더 큰 좌표 미분을 만들지 않는다. 실제 6,996개 감독 위치의 약65%가 정지이고, 입력을 무시하는 상수 predictor의 10좌표별 최적 중앙값은 모두0이다. 실제 모델은 이동 입력을 사용하므로 이 상수 해가 전체 모델의 최적해라는 뜻은 아니다.

기존 anchor의 최종 affine layer에서 좌회전/우회전 lateral gradient cosine은 -0.995다. 공유 회귀에서는 반대 방향 신호가 정상적으로도 생길 수 있다. 다만 현재 표현과 L1·표본 분포가 만들어내는 압력은 구체적으로 관측됐다. 이 계산은 anchor-only CPU 국소 미분이며 6,996개 실제 Gumbel training graph 전체의 parameter update가 아니다.

출력 frame indexing/teacher forcing에 잘못된 한 프레임 shift를 찾지 못했고 FP32 master optimizer도 실제 업데이트했다. 기존 기록의 weight-decay-only 수축은 약0.286%이므로 가중치가 decay 때문에 거의0이 됐다는 설명은 맞지 않는다.

CCDP의 압축 실패와 selector gradient 불균형은 별도로 남아 있다. 최종 일부 입력에서 reconstruction이 전체 주행 gradient를 압도하지 않았다는 결과가 CCDP 정상 판정은 아니다. 또한 이를 회전 실패의 단독 원인이라고 하지 않는다.

근거: [모델 기전 감사 및 실험안](outputs/diagnosis_label_alignment_20260911/model_mechanism_review.md), [실제 마지막 출력층](outputs/diagnosis_label_alignment_20260911/model_readout_direction.json).

## 6. 다음 실험 계획 — 아직 실행하지 않음

### A. 모든 감독 위치에서 출력층을 실제로 학습할 수 있는지

1. 기존 636×11=6,996개 frame occurrence의 LLM hidden, head feature, GT를 추출한다. 같은 frame도 history 위치가 다르면 별도 occurrence로 유지한다. 기존 anchor-only ridge 검사의 빈틈을 없애는 읽기 전용 단계다.
2. 정지/직진/좌/우 각각8개 anchor의 32개 window를 예측 성능을 보지 않고 사전 선택한다. 각 window의 11개 GT를 전부 사용한다. 실제 frame-level 비율도 별도 보고한다.
3. VE/connector/LLM을 고정하고 **기존과 같은 MLP 구조**만 별도 복사본에서 학습한다. 현재 head 초기값과 동일 구조 reset head를 비교하되 optimizer/schedule/LR/표본 순서는 같게 한다. 예: 사전 고정 LR1e-4, 최대1,000update. 학습 실패시 무제한 조정하지 않고 원인을 보고한다.
4. 진단용 암기 통과 기준 예: 전체 유효 frame L1<0.05m, 회전 FDE<0.5m, 방향정확도≥95%, 정지 FDE<0.1m. 이는 안전 주행/논문 성능 기준이 아니다.

현재 head도 성공하면 기존 최적화/노출 방식 문제를 더 강하게 의심한다. Reset만 성공하면 초기화·조건화 문제 후보가 커진다. 둘 다 실패해도 정보 부재를 즉시 확정하지 않고 마지막 affine만/기존 MLP 전체의 차이와 수치·입력 충돌을 분리한다.

### B. 표본 비율과 보조 loss는 한 번에 하나씩

A를 통과한 뒤 같은 데이터·초기값·update 수에서 기존 비율과 균형 비율을 비교한다. 필요할 때만 원래 세 loss와 waypoint-only를 별도 진단 ablation으로 비교한다. 보조 loss를 끈 실험은 논문 동일 학습이 아니므로 정식 수정으로 몰래 채택하지 않는다. L1을 MSE로 바꾸거나 query/구조/라벨/초기화를 동시에 바꾸지 않는다.

### C. 신호 색 정보가 어디까지 학습 가능한지

Ego가 아직 멈춰 있는 첫 녹색과 직전 빨간불을 고른다. 원본 색과 raw gRPC가 일치하는 표본만 사용한다. Frozen traffic/waypoint/all-token 각각에서 작은 분류 probe가 색을 읽을 수 있는지, 기존 traffic head와 구분해 측정한다. Probe는 진단 도구이며 production 입력/모델 교체가 아니다.

성공하면 downstream/기존 head가 정보를 읽지 못하는 문제 쪽으로 좁힌다. 실패만으로 정보가 없다고 확정하지 않고 입력 해상도 및 encoder 표현을 추가 검사한다. 이어 동일 초기 encoder에서 기존 라벨 유효성 정책 vs 검증된 sidecar mask만 바꾸는 제한 비교로 182개 불확실/모호 표본의 영향을 분리한다. 두 실험 모두 아직 실행하지 않았다.

### D. 신호 horizon 및 지시문을 대조

신호 전환 평가를 +0.5초와 +2.5초로 나누고 빨간불 유지/전환 직전/첫 녹색을 따로 비교한다. 정답은 계속 실제 미래 ego로 유지한다. 지시문은 현재/다음 교차로 topology와 적용 구간을 먼저 검증하고, 수정이 필요하다고 확인됐을 때 같은 model/data에서 지시문 생성 정책만 비교한다. 미래 ego GT에서 좌/우 명령을 역으로 만들면 미래 정보 누출이 되므로 사용하지 않는다.

전체 경로 수집량과 train 사용 범위도 구분한다. 이미 수집된 후반 경로가 시간순 split 때문에 빠진 문제는 대량 수집만으로 자동 해결되지 않는다. 전체 경로 암기용 학습 범위와 독립 평가용 범위를 별도 설계해야 한다.

## 산출물 및 무결성

이번 신규 작업은 `tools/morai/diagnose_cache_sample_alignment.py`, `tools/morai/diagnose_context_input_use.py`, 본 보고서 및 `outputs/diagnosis_label_alignment_20260911/`의 진단 파일뿐이다. 원본 dataset이나 모델에 보정 라벨, probe 계수, 새로운 checkpoint를 적용하지 않았다.
