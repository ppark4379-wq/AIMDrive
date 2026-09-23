# 첫 직진 구간 GT 및 best 모델 진단 (2026-09-20)

## 결론

검사한 구간에서 저장 GT의 좌표/시간 생성 오류는 발견하지 못했다. 원본 ego 위치로 독립 재계산한 좌표와 최대 0.0000009504 m 차이였다. 반면 학습에 실제 포함된 장면에서도 best 모델의 좌우 오차가 남아 있다. 데이터 부족만으로 단정할 수 없으며, 현재 학습/표현이 해당 장면을 충분히 재현하지 못한다는 증거다. Encoder 자체의 표현 품질과 connector/LLM 학습 문제는 아직 분리하지 않았다.

## 범위와 방법

- 모델: model-best.pt, epoch 10, update 3730. 원래 30m 학습 labels와 캐시를 보존했다. 학습/모델/제어 설정 변경 및 UDP 송신 없음.
- 첫 좌회전 시작 route progress 72.629 m, 녹화 frame 90. 출발 녹화 frame 0은 이미 약 7.02 m/s 주행 중이므로 정지 출발 장면이 아니다.
- 회전 전 90개 원본 frame 중 캐시가 있는 89개 GT를 검사했다. 누락 frame을 임의로 채우지 않았다.
- quality/active_signals.json의 EgoVehicleStatus 위치를 source timestamp에 따라 독립 보간하고 현재 heading의 회전행렬로 [left, -forward]를 만들었다. 미래 0.5/1/1.5/2/2.5초를 비교했다. 기존 future_target 함수를 재사용하지 않았다.
- 위치 timeline 자체도 원본 ego와 최대 5.86e-14 m 차이였다. 사용 heading source의 시간 차이는 최대 5.0 ms. 이는 생성 파이프라인의 수치적 정합성 검사이며, 카메라/LiDAR의 물리적 동시성이나 차선 중심의 정확성을 별도 입증하지 않는다.
- 저장 GT는 vision encoder 예측이 아니다. datasets/navigation_sensor_record.py:future_target → tools/cache_sensor_record_tokens.py → NavigationCacheDataset.training_targets.future_trajectory → models/vldrive.py:loss(F.l1_loss)로 연결된다. Encoder는 cached visual features를 제공한다.
- 기존 학습 window_indices의 창 중 마지막 anchor부터 미래 2.5초까지 첫 회전 전인 창은 3개. frame [20, 57, 62]. epoch=0 instruction sampling으로 재현했다. 이것은 학습 프레임이 3개뿐이라는 뜻이 아니다. 모델은 창 안의 여러 프레임에 대해 supervision을 받는다.
- 추가 실시간 방식 재생: 실제 연속 11프레임(0.2초 간격) 12개 anchor, frame [20, 22, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]. 이력 전체에 캐시 누락이 없는 창만 사용. 현재 anchor의 active instruction을 입력했다. 실제 UDP 주행 평가가 아닌 녹화 feature 재생이다.
- Gumbel과 argmax에 동일 센서·문장 적용. 30m/20m 비교 시 Gumbel seed=607로 고정해 mask 난수 차이를 통제했다. 별도 학습창에서는 seed 607/608/609도 비교했다.

## 실제 학습창 3개: 기존 30m instruction

Gumbel seed607 좌우 MAE(m): [0.13853888461987177, 0.3053473209341367, 0.6112235337495804, 0.8391680320103964, 1.037845512231191]. 평균 위치 오차 ADE 0.994 m, 2.5초 FDE 1.452 m.
Argmax 좌우 MAE(m): [0.13194708774487177, 0.2915075992544492, 0.5994640936454138, 0.819311261177063, 1.0313350955645244]. ADE 0.963 m.
Gumbel seed 3종의 ADE: [0.994, 0.987, 0.989]. 이 표본에서는 선택 무작위성이 오차의 주원인이라는 근거가 약하다.

구체적 예: frame20, route26.10m. 2.5초 GT 좌우 -0.533m(오른쪽), 예측 +1.117m(왼쪽), 좌우 차이 1.650m. 따라서 제어기와 무관하게 예측 자체의 방향/형상 오차가 있다. 직진이어도 GT는 차량 heading 및 실제 미세 조향 때문에 좌우 0으로 고정되지 않는다.

## 현재 지시문을 넣은 녹화 재생

각 열은 미래 시간별 좌우 절대오차 평균(m). ADE는 전후 오차까지 포함한 전체 위치 오차다. 이 표본의 95백분위는 소표본 기술통계이며 일반화 보장이 아니다.

| 조건 | 0.5s | 1s | 1.5s | 2s | 2.5s | ADE |
|---|---:|---:|---:|---:|---:|---:|
| 30m_argmax | 0.185 | 0.370 | 0.817 | 1.211 | 1.525 | 1.123 |
| 30m_gumbel | 0.187 | 0.390 | 0.820 | 1.196 | 1.509 | 1.157 |
| 20m_argmax | 0.178 | 0.365 | 0.751 | 1.123 | 1.424 | 1.119 |
| 20m_gumbel | 0.183 | 0.382 | 0.756 | 1.108 | 1.414 | 1.150 |

![직진 오차](../../outputs/analysis/straight_gt_20260920/live_style_errors.png)

원본 수치: outputs/analysis/straight_gt_20260920/audit.json, live_style.json, errors.csv. 그래프의 +오차는 GT보다 왼쪽이다. 연결선은 샘플 사이 안내선이며 누락 구간에서 측정했다는 뜻이 아니다. 우하단 좌우/전방 축 비율은 다르므로 각도는 수치로 판단해야 한다.

## 학습 부족 해석과 한계

현재 20m+Gumbel의 12개 anchor 평균 좌우 오차는 0.5초 0.183m, 1초 0.382m, 1.5초 0.756m, 2초 1.108m, 2.5초 1.414m다. 30m+Gumbel의 2.5초 1.509m에서 약 0.095m 감소했지만, 전체 위치 ADE는 1.157m → 1.150m로 거의 그대로다. 지시 타이밍 변경만으로 해결됐다는 근거는 없다. 20m 결과의 2.5초 좌우 signed bias는 +1.341m로 대부분 GT보다 왼쪽이다. 예측점과 GT 선분 사이의 거리도 2.5초 평균 1.395m여서, 같은 길을 속도만 다르게 예측한 문제로만 설명되지 않는다.

학습 window_indices를 통해 실제 supervision 프레임의 합집합도 확인했다. 첫 90개 프레임 중 62개가 적어도 한 창의 학습 대상이며, frame21~36은 포함되지 않는다. 일부 미포함은 2칸 sampling의 결과이고, frame24의 cache exclusion 및 전체 창에 공통 instruction이 필요하다는 조건도 초반 학습 창을 제한한다. 이 누락이 오차의 원인이라고 확정하지는 않는다. frame20은 실제 학습 대상인데도 2.5초 좌우 오차 1.650m가 있었다.

Encoder 입력 recipe의 velocity_input과 target_point_input은 constant_zero이다. 속도가 명시적으로 제공되지 않는 대신 시간에 따른 시각 feature로 움직임을 추론해야 한다. 검사한 실제 학습창 anchor 속도는 4.64~11.07m/s, 현재 제어 설정의 속도 상한은 5m/s다. 이는 추가 실시간 비교에서 맞춰야 할 조건이며, 이번 결과만으로 속도 입력을 원인으로 단정하거나 변경하지 않았다.

전체 검증 ADE는 epoch1 4.480m → best epoch10 2.033m → epoch15 2.137m였다. 추가 epoch가 전체 성능을 계속 개선하지 않았으므로 단순 시간 부족으로 확정하지 않는다. 이것은 직진 구간 전용 학습 곡선이 아니며 중간 모든 checkpoint가 남아 있지도 않다.

이 구간은 training split이고 별도의 동일 출발 구간 validation lap은 없다. 실시간 자유주행 상태와 동일 위치/속도의 비교, encoder 자체 waypoint head 품질, 이미지·LiDAR 물리적 시간 정렬, 차선 중심 대비 횡오차는 이번 검사만으로 확정하지 않았다. GT는 사람이 실제로 간 미래 경로이며 이상적인 global path와 동일하다는 보장은 없다.

다음 최소 실험은 작은 고정 직진 학습 묶음을 충분히 맞출 수 있는지 확인하는 것이다. 모델/정답 연결은 맞는데 소량 장면조차 맞추지 못하면 encoder 표현·connector/LLM·최적화 문제를 더 분리해야 한다. 소량 장면은 맞추지만 다른 상태에서 실패하면 상황 다양성과 일반화 쪽 우선순위가 올라간다. 이번 단계에서는 재학습하지 않았다.
