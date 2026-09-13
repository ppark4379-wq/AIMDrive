# 마지막 encoder 기반 후속 학습 결과와 train offline 추론

후속 원인 진단: [MORAI_TRAIN_DIAGNOSIS_20260911.md](MORAI_TRAIN_DIAGNOSIS_20260911.md).
특히 frame990의 큰 오차는 녹색 전환 직전의 예측이므로 그 자체를 자력 출발 실패의 증거로 해석하면 안 된다.

학습은 2026-09-11 03:39:50 KST에 최종 저장까지 정상 완료했다.
이번 보고는 train 데이터만 분석한다. 마지막 25번째 epoch의 Vision Encoder를 고정하고,
CCDP·MEFA/Q-Former·TinyLlama/DDIA·waypoint MLP를 15 epochs, 9,540 updates 학습했다.
재시작부터 완료까지 약 12시간 49분이 걸렸다.

실행: `outputs/morai_paper_last_encoder_20260910T055054Z`.
Offline replay: `outputs/morai_paper_last_encoder_train_replay_20260911`.
사용 모델: [model-last.pt](outputs/morai_paper_last_encoder_20260910T055054Z/model-last.pt).
SHA256: `a63a619b2565d2bb0fba13ddfe305779f344bf67d0fbd9aafa7f6476c9f14b9a`.
최종 optimizer/RNG checkpoint도 저장됐다. 학습 대상 가중치 변경과 frozen 가중치 보존이 확인됐다.

## Train 정확도

Train 3,352 frame에서 명령 경계를 넘지 않는 11-frame window를 만들고, 학습 설정과 동일한
stride 5로 선택한 636 window의 마지막 frame을 평가했다. 각 지점에서 0.5/1/1.5/2/2.5초 후의
실제 ego 위치와 비교했다. 서로 겹치는 history frame을 평가 표본으로 중복 집계하지 않았다.

| 시점 | ADE | FDE | Waypoint L1 |
|---|---:|---:|---:|
| 후속 학습 전 | 3.987 m | 6.661 m | 2.200 |
| 후속 학습 완료 | 0.421 m | 0.935 m | 0.246 |

매 epoch의 학습 waypoint L1 평균은 1.450 → 1.243 → 0.844로 감소했고, 15번째에는 0.246이었다.
학습 loss는 window의 11개 frame 전체에 적용되므로 마지막 frame 평가 지표와 집계 단위가 다르다.

전체 평균에는 정지 장면 비율의 영향이 크다. 평가한 636지점 중 416개(65.4%)는
향후 2.5초 동안 ego가 현재 위치에서 0.5m 이상 멀어지지 않는 정지·거의 정지 장면이다.

| 장면 분류 | 표본 수 | ADE | FDE |
|---|---:|---:|---:|
| 정지·거의 정지 | 416 | 0.006 m | 0.008 m |
| 이동, 조향 작음 | 172 | 0.984 m | 2.236 m |
| 이동, 조향 큼 | 48 | 2.001 m | 4.316 m |
| 이동 전체 | 220 | 1.206 m | 2.690 m |

이동 장면은 기록된 `abs(steer)>0.1` 여부로 나눴다. 정밀한 도로 형태 라벨이 아니라 진단용 분류다.
현재 속도가 0이어도 미래에 출발하면 이동 장면에 포함된다. 이동 전체 행은 앞의 두 이동 행의 합이다.
이동 장면의 FDE 90백분위는 4.972m이며, 2.5초 뒤 오차가 1m 미만인 비율은 25%였다.
이동 장면의 2.5초 뒤 평균 절대 좌우 오차는 1.712m, 전후 오차는 1.415m였다.

정지 패턴은 매우 잘 맞췄지만 회전과 출발까지 대회 경로의 expert 정책을 충분히 외웠다고 보기는 어렵다.
전체 ADE 0.42m만으로 움직이는 차량의 궤적도 같은 수준으로 맞춘다고 해석하면 안 된다.

## 실제 offline 추론

최종 checkpoint를 새 프로세스에서 복원해 train 구간에 균등하게 분포한 32개 window를 다시 추론했다.
진단용으로 각 장면 분류의 중앙값·최악 오차 사례 6개를 추가해 총 38개를 실행했다.
진단 사례는 좋은 결과만 고르거나 전체 평균에 섞어 해석하지 않도록 별도로 표시했다.

| 새 추론 표본 | ADE | FDE |
|---|---:|---:|
| 균등 선택 32개 | 0.384 m | 0.891 m |
| 진단 6개 포함 38개 | 0.622 m | 1.376 m |

38개 모두 저장 당시 train 예측과의 최대 좌표 차이가 0이었다.
GT는 원본 `measurements_all.json`의 GPS 위치와 heading으로 다시 계산했다.
미래 native tick 간격도 확인했으며, cache의 GT와 최대 차이는 0이었다.
저장된 global-path `future_waypoints`를 미래 위치 정답으로 쓰지 않았다.

원본 4방향 RGB 카메라·현재 LiDAR·현재 속도에서 encoder부터 다시 실행하는 비교도
정지(frame 1180)와 우회전(frame 2865), 두 개의 11-frame window에 수행했다.
Encoder는 strict load를 통과했고, 두 장면의 raw sensor 재추론과 cached token 추론의
최대 waypoint 좌표 차이는 각각 0.000832m, 0.002930m였다.
따라서 아래 우회전 오차는 token cache를 읽는 경로에서만 발생한 문제가 아니다.

| Frame | 실제 수집 기록과 예측 비교 | ADE / FDE |
|---|---|---:|
| 1180 | 실제 정지, 예측도 거의 제자리 | raw 추론 0.0008 / 0.0016 m |
| 280 | 실제 2.5초 뒤 전방 2.98m, 예측 전방 4.22m | 0.806 / 1.238 m |
| 990 | 현재 정지 상태에서 expert는 이후 전방 10.22m까지 이동, 모델은 계속 정지를 예측 | 4.145 / 10.233 m |
| 2865 | 실제 전방 11.99m·우측 4.46m, 예측 전방 11.56m·우측 0.27m | raw 추론 1.978 / 4.208 m |
| 3320 | 실제 전방 13.88m·우측 8.11m, 예측 전방 16.25m·우측 0.61m | 4.021 / 7.871 m |

Frame 2865의 입력 instruction은 `Proceed ahead and make a right turn.`이었다.
우회전 지시가 있음에도 옆으로 휘는 정도가 크게 부족했다.
Frame 990은 현재 정지 상태에서 이후 출발하는 미래 예측 오차다. 이 결과 하나로 실제 제어에서
영원히 출발하지 못한다고 확정할 수는 없으며, 출발 전환 상황의 별도 연속 평가가 필요하다.

아래 그림에서 파란 선은 실제 ego 미래 위치, 주황 선은 모델 예측이며 좌표축 단위는 m다.

![수집 RGB와 실제/예측 궤적](outputs/morai_paper_last_encoder_train_replay_20260911/train_scenes.png)

## CCDP 압축 결과

Train 평가의 모든 636 window, 총 6,996 frame 위치에서 106개 token을 전부 유지했다.
목표 30%와 달리 관측 유지율은 100%다. 새 offline 추론에서도 전체 유지가 재현됐다.

현재 구현은 버린 token의 복원 오차만 reconstruction loss로 합산한다.
전부 유지하면 버린 token이 없으므로 reconstruction loss가 0이 된다.
이때 pruning penalty는 `(1-0.3)^2=0.49`, 가중치 10을 곱하면 전체 loss에 4.9가 남는다.
이는 복원을 완벽하게 배웠다는 증거가 아니라 복원 대상 자체가 사라진 결과다.

첫 update의 reconstruction loss는 31,317.5였고, 전체 9,540 update 중 8,724회에서
reconstruction 0 / pruning 0.49가 관측됐다. 처음 관측된 것은 update 647이며 마지막 1,000회는 모두 이 상태였다.
초기 큰 reconstruction sum과 상대적으로 작은 pruning 비용 때문에 전체 유지가 유리해졌다는 설명이
코드와 로그에 부합한다. 다만 손실 가중치 변경 실험으로 원인을 분리 검증하지는 않았다.

구현 근거: [선택 비율 penalty](lavis/models/vldrive/ccdp.py:258),
[버린 token에만 적용하는 reconstruction](lavis/models/vldrive/qformer_mefa.py:1085).

중간 보고에서 loss 감소를 근거로 정상 진행이라고 표현했지만,
완료 후 token 선택을 직접 확인한 결과 압축 목표는 달성하지 못했다.
궤적 학습은 개선됐고 CCDP 압축 학습은 실패한 상태로 구분해야 한다.

![Train loss와 지점별 오차, pruning penalty](outputs/morai_paper_last_encoder_train_replay_20260911/train_summary.png)

## 다음 판단에 필요한 사항

현재 checkpoint는 정지·일부 직진 모방이 개선된 실험 결과로 보존할 가치가 있다.
다음 학습에서는 정지 편중과 회전/출발 구간의 학습 비중, CCDP 손실 규모 및 선택 확률 포화를 우선 조사해야 한다.
논문과의 동일성을 유지할 설정과 별도 실험으로 바꿀 설정을 구분해 검증해야 하며,
이번 분석에서는 모델 구조·loss·데이터·학습 가중치를 변경하지 않았다.
실제 MORAI 연속 제어 주행은 이번 offline 비교에 포함되지 않는다.

재실행 도구: [offline_train_replay.py](tools/morai/offline_train_replay.py).
상세 수치: [train_analysis.json](outputs/morai_paper_last_encoder_train_replay_20260911/train_analysis.json).
새 예측: [fresh_replay.json](outputs/morai_paper_last_encoder_train_replay_20260911/fresh_replay.json).
