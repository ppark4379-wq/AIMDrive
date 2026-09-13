# 반복량·학습 안정성·신호 입력 후속 실험 — 2026-09-11

## 범위와 현재 상태

사용자가 승인한 후속 검증은 회전의 구간당 반복 횟수 맞추기, 학습률에 따른 안정성 비교,
신호색을 보존하는 입력 처리, 희귀 정지-녹색 표본 비중과 빨간불 오인의 동시 확인이다.
기존 데이터, encoder/LLM 가중치, token cache는 보존했다. 모델 구조는 변경하지 않았다.
새 진단 모델은 실제 주행 모델에 설치하지 않는다.

회전 R1/R2, CPU 입력 감사, 신호의 전체 3,170개 paired-feature 비교 및
2×2 분류기 실험을 모두 완료했다. 모든 성능은 원래 train 자료의
오프라인 진단이며 validation 주행 성능이나 closed-loop 안전성 결과가 아니다.

사전 계획: [EXPERIMENT_PLAN.md](outputs/diagnosis_matched_exposure_20260911/EXPERIMENT_PLAN.md).
새로 실증된 hue 문제에 대한 좁은 구현 예외는 원본 계획을 덮어쓰지 않고
[별도 변경 기록](outputs/diagnosis_matched_exposure_20260911/COLOR_SAFE_IMPLEMENTATION_AMENDMENT.md)에 남겼다.

## 1. 회전: 반복 횟수를 맞춰도 부족함이 남았다

동일한 32개 window를 조건마다 **각 500회**, 총 16,000 update 학습했다.
두 조건 합계는 32,000 update다. 기존 BF16 출력 MLP, 원래 frozen LLM 특징,
FP32-master AdamW, L1, batch 1-window, 11-frame 감독, weight decay .06,
gradient clipping 10, 초기값·sample 순서는 동일하다.

- R1: LR 1e-4 고정.
- R2: LR 1e-4 → 1e-6 cosine 감소. R1과 다른 것은 이 일정과 그에 따른 누적 AdamW decay다.
- 선택 자료는 352개 감독 위치/345개 고유 frame이며, 원본 GT 그대로다.
- R1의 첫 1,000 update는 이전 실험의 BF16 가중치뿐 아니라 FP32 master와 Adam 상태까지
  완전히 일치했다. 기존 실험과 무관한 다른 시작점에서 비교한 것이 아니다.
- 아래는 중간 최고치가 아닌 사전 고정한 마지막 update의 결과다.

FDE는 +2.5초 마지막 위치 오차다. 회전 방향 판정은 마지막 lateral 부호가 맞고
예측 lateral 절댓값이 .5m 이상인지를 뜻한다.

| 선택한 32개 구간의 조건 | 구간당 노출 | 좌표 L1 | 회전 FDE | 회전 방향 일치 |
| --- | ---: | ---: | ---: | ---: |
| 이전 1,000-update 조건 | 31~32회 | 0.6855m | 4.0907m | 31.5% |
| R1 고정 학습률 | 500회 | 0.4590m | 3.0136m | 68.5% |
| R2 감쇠 학습률 | 500회 | 0.4594m | 3.2788m | 63.6% |
| 사전 암기 기준 | — | <0.05m | <0.5m | ≥95% |

두 조건 모두 정지 FDE <0.1m만 통과했다. 반복량 증가는 일부 도움이 됐지만,
**“구간당 반복이 31회여서 실패했을 뿐, 500회면 충분하다”는 설명은 지지되지 않았다.**

마지막 5개 고정 평가 시점의 L1 표준편차는 R1 0.0290m/R2 0.00723m였다.
감쇠 학습률은 결과를 덜 흔들리게 했지만, 그 결과를 충분히 정확하게 만들지는 못했다.
이는 독립 실험 5회의 통계나 여러 seed에서 입증한 보편적 결론이 아니다.

### 원래 경로 전체에 적용할 수 있는 개선 모델인가?

아니다. 선택하지 않은 구간까지 포함한 원래 학습 위치 6,996개에서는 손해도 있었다.

| 원래 train 전체 평가 | 원래 모델 | R1 | R2 |
| --- | ---: | ---: | ---: |
| 전체 FDE | 0.9381m | 1.1570m | 1.1452m |
| 직진 FDE | 1.8426m | 3.1053m | 2.9294m |
| 좌회전 FDE | 3.9574m | 4.0706m | 4.3387m |
| 우회전 FDE | 4.3534m | 3.3806m | 3.5855m |

선택한 자료를 더 맞추는 동안 다른 원래 주행 구간을 보존하지 못했다. 이 때문에
진단용 가중치를 production에 적용하지 않았다. 이전 두 구간 실험과 노출 횟수는
같지만 구간 다양성·총 optimizer update·누적 decay가 달라, 32개 구간이 원천적으로
표현 불가능하다고 결론 내릴 수도 없다.

### Gradient 및 표현에서 좁혀진 범위

- head L1 gradient clipping은 R1 68.54%/R2 65.74%의 update에서 발생했다.
  정지 anchor window는 0%, 회전 anchor window는 약 94~99.7%였다.
- 이것은 Adam의 실제 parameter update도 같은 비율로 줄었다는 뜻이 아니다.
  clipping을 바꾼 대조 실험이 없으므로 clipping 제거를 해결책으로 확정하지 않는다.
- 최종에도 FP32 master와 모델 가중치가 실제로 갱신됐다. optimizer가 아예 작동하지
  않았다는 설명은 맞지 않는다.
- 선택한 회전 143개에서 종점 오차 제곱합의 약 97%가 횡방향이었다. 반복 후에도
  주된 문제는 이동 거리보다 좌우로 얼마나 굽히는지를 맞추는 부분이다.
- 전체 352개 특징의 변화량은 공통 평균 대비 작지 않았지만, 분산 대부분이 소수
  방향에 집중돼 있었다. 두 회전 22개의 특징 차이는 공통 평균 대비 약 3.9%였으나
  이전 두 구간 학습은 좌우를 구분했으므로 작은 차이를 정보 부재로 해석하면 안 된다.
- 이번에는 upstream을 동결했다. 따라서 encoder/connector/LLM 중 하나를 유일한
  고장 부위로 특정하거나 auxiliary loss 경쟁이 주원인이라고 입증한 것은 아니다.

근거: [회전 상세 보고서](outputs/diagnosis_matched_exposure_20260911/model_matched_report.md),
[독립 결과 감사](outputs/diagnosis_matched_exposure_20260911/audit_results.md),
[특징 spectrum 검사](outputs/diagnosis_matched_exposure_20260911/audit_conditioning.md).

## 2. 신호 입력: 색조 증강이 정답 의미를 바꾸는 문제를 확인했다

현재 MORAI 학습 runner는 원래 `color_jitter=.4`를 전달했고, 공통 transform은 이를
밝기·대비·채도뿐 아니라 **hue .4**에도 사용했다. 빨간 신호를 다른 색으로 바꿔도
traffic target은 원래 빨간불로 남는 경로다.

저장소의 범용 `vision_encoder/train_pretrain.py`에서 `--color-jitter`의 실제
argparse 기본값은 **0.1**이다(도움말 문자열의 0.4는 기본값과 불일치한다).
따라서 MORAI의 수정 전 0.4가 모든 원본 학습 경로의 공통 기본값이었다고 말하면
안 된다. 이것은 저장소의 설정 차이에 대한 사실이며, 논문 실험이 실제로 어떤
명령행 값을 사용했는지까지 확정한 것은 아니다. 범용 trainer 기본값은 수정하지 않았다.

실제 신호 이미지 16개에 고정된 32개 난수 조건씩, 총 512개 변환을 검사했다.
현재 legacy production transform과 감사 코드 출력은 **512/512 tensor bit 일치**했다.
동일한 resize, 밝기·대비·채도 계수, 적용 순서에서 hue 연산만 제외해 비교했다.

| 실제 등불 영역의 변환 후 색 | 기존 hue .4 | hue만 제외 |
| --- | ---: | ---: |
| 원래 빨강/초록 유지 | 106 | 494 |
| 빨강 ↔ 초록으로 직접 바뀜 | **110** | **0** |
| 다른 색으로 바뀜 | 278 | 0 |
| crop/미세한 크기로 관측 기준 미달 | 18 | 18 |
| 합계 | 512 | 512 |

이것은 **모델 오분류율이 아니라 입력 등불색의 변조율**이다. 원본 이미지가 512개라는
뜻도 아니다. 또한 epoch25 원본 checkpoint에는 정확한 augmentation/source 기록이
없어, 과거 전체 학습이 반드시 같은 설정이었다고 확정할 수는 없다. 반면 현재 코드의
이 동작과 다음 학습에서 그대로 재현될 수 있는 문제는 직접 확인됐다.

### 구현한 수정

- MORAI 학습에 `--rgb-hue-jitter` 옵션을 추가하고 기본값을 **0**으로 설정했다.
- 기존 증강 재현은 `--rgb-hue-jitter 0.4`로 명시할 수 있다.
- 공통 CARLA transform/loader 기본값은 legacy 그대로다. MORAI 기본값만 바뀐다.
- 밝기·대비·채도·geometry의 설정/분포, 입력 크기, 모델 구조는 유지했다.
  hue 난수 소비가 사라지므로 이후 모든 augmentation 난수의 bit 단위 재현은 주장하지 않는다.
- deterministic 추론 tensor는 변하지 않음을 검사했다. 원본 가중치·라벨·cache는 그대로다.
- 새 checkpoint에 hue 설정이 기록된다. 기록된 설정과 다른 `--resume`은 막고 별도
  `--warm-start`를 요구한다. 과거 기록이 없는 checkpoint는 검증 불가 경고를 낸다.
- 신규 10개와 기존 관련 34개, **총 44개 테스트 통과**.

이 변경은 **기존 학습 recipe의 입력 증강 수정**이며, 논문 encoder 구조 교체가 아니다.
논문 동일 재현에서 증강 recipe까지 동일한지와 별개로 구분해 기록해야 한다.
기존 모델이 이 수정만으로 자동으로 신호를 잘 읽게 되는 것은 아니며, 효과를 학습된
표현에 반영하려면 이후 encoder 학습과 cache 갱신이 필요하다.

근거: [512회 색 감사](outputs/diagnosis_matched_exposure_20260911/traffic_hue_results.json),
[역사적 provenance의 한계](outputs/diagnosis_matched_exposure_20260911/audit_augmentation_provenance.md),
[학습 옵션 설명](../vision_encoder/README_MORAI.md),
[테스트 결과](outputs/diagnosis_matched_exposure_20260911/color_safe_tests.xml).

## 3. 전체 시야 보존과 희귀 표본 비중 비교

### 입력을 넓게 남긴다고 신호가 무조건 잘 보이지는 않는다

같은 입력 크기에서 front/center를 원본 전체 시야를 보존하는 letterbox로 바꾸면,
원래 crop에서 빠진 영역은 들어오지만 개별 물체는 작아진다. 실제 첫 녹색 frame의
front strict-green pixel 수는 다음처럼 줄었다.

| 첫 녹색 frame | 기존 front | full-FOV front |
| --- | ---: | ---: |
| 991 | 6 | 2 |
| 1591 | 22 | 7 |
| 2491 | 7 | 2 |
| 2791 | 25 | 7 |

기존 center에서는 네 경우 모두 해당 등불이 빠졌고, letterbox center에서는 작게
일부가 들어왔다. 이 비교는 FOV·크기·padding·배치가 함께 달라지는 입력 대조이며,
색 정보만 순수하게 바꾼 단일 요인 실험이 아니다. production FOV는 교체하지 않았다.

### 전체 유효 자료에서의 2×2 대조 — 완료

유효 train 3,170개 모두를 같은 원본 encoder로 FP32 재추론했다. 각 frame의
LiDAR·현재 속도·target point·다른 카메라는 동일한 한 번의 dataset 로드에서 공유하고,
front/center만 legacy 또는 letterbox로 바꾼다. 예전 FP16 cache와 직접 비교하지 않는다.

이후 원래 신호 분류기 초기값에서 다음 네 조건을 각 seed 607/608/609로 학습했다.

1. Legacy 입력 + 무가중.
2. Legacy 입력 + 완만한 상태별 가중치.
3. Full-FOV 입력 + 무가중.
4. Full-FOV 입력 + 같은 가중치.

가중치는 현재 신호 class × 정지 여부 네 그룹의 역빈도 제곱근을 사용하고 전체 유효
표본 평균을 1로 정규화한다. 정지-녹색은 약 11.879배, 정지-적색은 .7435배다.
기존의 극단적인 99배 가중치와 다르다. 현재 속도는 가중치 구분에만 쓰고 분류기 입력에
새로 추가하지 않는다. 원래 분류기 초기값·sample 순서·최종 epoch100·threshold .5를 고정한다.

각 조건은 2,500 optimizer update이며 12조건 합계 30,000이다. 이전 실험의
3,352 raw frame/조건당 2,700 update와 분모가 달라, 이전 결과와의 차이를 입력 변경
하나의 효과라고 해석하지 않는다. 첫 녹색 4개, 정지-녹색 8개와 정지-적색 2,042개의
오분류를 함께 평가한다. 반복은 새 독립 신호 사건을 추가한 것이 아니다.

| 조건, 각 3 seeds의 최종 epoch100 | 유효 자료 정확도 | 첫 녹색 | 정지-녹색 | 정지-적색 오인 | 이동-적색 오인 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy + 무가중 | 98.96~99.15% | 모두 0/4 | 모두 0/8 | 모두 0/2,042 | 5~12/336 |
| Legacy + 완만한 가중 | 98.42~98.52% | 모두 0/4 | 모두 0/8 | 모두 0/2,042 | 11~19/336 |
| Full-FOV + 무가중 | 99.05~99.21% | 모두 0/4 | 모두 0/8 | 모두 0/2,042 | 5~10/336 |
| Full-FOV + 완만한 가중 | 98.61~98.64% | 모두 0/4 | 모두 0/8 | 모두 0/2,042 | 9~12/336 |

12조건 모두 첫 녹색 네 장면과 정지-녹색 여덟 장면의 분류에 실패했다. 완만한
가중치는 첫 녹색의 비적색 확률을 무가중 약 0.2~1.1%에서 약 3.8~12.0%로 올렸지만,
고정 판정 기준 50%를 넘지 못했다. 정지-적색 오인은 늘지 않았지만 이동-적색 오인은
존재하므로, 빨간불 전체를 완벽하게 분류했다고 말하면 안 된다.

Full-FOV는 3,170개 모두에서 traffic feature를 실제로 바꿨다. 변화가 적용되지 않아
똑같은 결과가 나온 것은 아니다. 원래 VE waypoint head도 정지-녹색 8개에서
FDE가 legacy **9.565m**, Full-FOV **9.557m**로, 출발 궤적을 맞히지 못하는 양상이
유지됐다. 이 수치는 downstream LLM이 아닌 **VE의 보조 waypoint head** 결과다.
VE 전체 유효 자료 FDE도 0.8561→0.8622m로 개선되지 않았다.

### 실행 및 독립 검증

- 전체 6,340회 paired forward: 452.8초. 동일 checkpoint strict-load, 실행 전후
  source/checkpoint hash 불변, 모든 동일 frame의 비전방 입력 공유 확인.
- 12개 head 조건은 모두 실제 2,500 optimizer update 완료. 초기값·seed별 sample
  순서 일치, 최종 epoch100과 101개 평가 기록(초기값 포함) 확인.
- 새 traffic token에서 CPU 원래 head를 적용한 결과와 GPU 저장 logits의 최대 차이는
  legacy 1.91e-6/FOV 1.43e-6이며, class 판정 불일치는 0개.
- 독립 NumPy 계산으로 저장된 12개 head와 모든 최종 확률을 재현했다.
  최대 확률 차이는 3.33e-7이며 GT·mask·속도·3,170개 분모도 일치했다.
- 측정된 train 내 오인율은 closed-loop 안전 합격 기준이 아니다. 이 비교도
  frozen encoder의 분류기 학습이며, 색조 증강을 고친 encoder 재학습이나
  downstream 실제 출발 행동을 검증하지 않는다.

근거: [신호 모델 비교 보고서](outputs/diagnosis_matched_exposure_20260911/traffic_model_report.md),
[2×2 전체 결과](outputs/diagnosis_matched_exposure_20260911/traffic_2x2_results.json),
[독립 2×2 감사](outputs/diagnosis_matched_exposure_20260911/audit_traffic_2x2_results.json),
[추출 provenance](outputs/diagnosis_matched_exposure_20260911/traffic_extraction_all3170.json).

## 4. 최종 결정

1. **적용한 것은 신호색을 뒤바꾸는 학습 증강의 예방 수정이다.** 원본 학습 모델이나
   inference FOV, dataset 라벨은 교체하지 않았다. 44개 회귀 테스트를 통과했다.
2. **반복량 증가·LR 감소·full-FOV·완만한 가중치만으로 현재 모델 문제를 해결하지 못했다.**
   일부 수치는 개선됐지만 목표 회전 정밀도와 첫 녹색 인식에 도달하지 못해
   진단 모델을 production으로 승격하지 않았다.
3. **다음 encoder 학습에는 색조 안전 설정을 반영해야 한다.** 이 변경 효과를 기존
   token에 소급할 수는 없으므로, encoder를 실제로 학습한 뒤 token cache를 갱신하고
   후속 connector/LLM 학습과 구분해 검증해야 한다. 이것이 회전까지 해결한다는
   보장은 없으며, 이번에 전체 encoder/LLM 재학습을 시작한 것은 아니다.
4. **새 독립 정지-녹색 사례는 아직 늘지 않았다.** 이번 데이터는 여전히 네 사건/여덟
   정지-녹색 frame이다. 새로운 시나리오에서의 실제 수집은 별도로 필요하다.
   반복이나 전처리 변형을 새 사건으로 세지 않았다.

정리하면, 자료가 아예 학습에 들어가지 않아서 생긴 문제라는 설명은 지지되지 않았다.
회전은 현재 frozen 특징/출력층 조건에서 여러 구간을 정밀하게 맞추지 못했고,
신호에서는 실제 입력의 의미를 훼손할 수 있는 증강 설정을 찾아 수정했다.
증강 수정의 실제 주행 효과와 회전의 단일 근본 원인은 아직 별도 검증이 필요하다.
이번 승인된 진단 학습은 모두 종료했고 GPU를 반환했다.
