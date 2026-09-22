# VLDrive 실제 학습 시간 분석 및 부하를 늘리지 않는 최적화

측정일: 2026-09-19, Windows / WSL Ubuntu-20.04.

## 1. 결론

현재 학습의 주된 시간 소모는 GPU 신경망 계산이 아니라 CPU AdamW 계산과 CPU↔GPU 전송이었다.
CPU 스레드와 메모리 배치를 유지하고 **CPU AdamW의 fused 실행만 활성화**했다.
실제 `utils.trainer.train()`에서 update 평균은 **4.2253초 → 2.4754초**, 약 **41.4% 단축**됐다.
처리량 기준 약 1.71배다. 구간을 나눈 별도 측정에서도 **4.0333초 → 2.4663초**, 약 38.9% 단축됐다.

같은 seed와 같은 8개 학습 window로 비교한 모든 실행에서 loss, gradient norm, retention 값과
최종 전체 학습 파라미터 SHA-256이 정확히 일치했다. GPU 메모리 증가는 관찰되지 않았으며,
실제 학습 경로의 프로세스 RAM 최대치는 19.48 → 19.50 GiB로 유사했다. 모든 실행에서 swap 사용은 0이었다.

이는 짧은 구간의 처리시간 검증이다. 전체 15 epoch의 수렴, 주행 품질, 장시간 최고 온도를 검증한 결과는 아니다.

## 2. 실험 환경과 고정 조건

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 총 16311 MiB |
| CPU | Intel Core Ultra 7 265K, 20코어/20 logical processors |
| RAM | Windows 약 64 GB, WSL 약 31.0 GiB |
| PyTorch / CUDA | 2.8.0+cu128 / 12.8 |
| 실행 환경 | `/opt/miniforge3/envs/lmdrive/bin/python`, Ubuntu-20.04 |
| Python 경로 | `PYTHONPATH=.runtime/stage2_wsl_deps:.` |
| 데이터 | `data/raw/basic_drive`에 대응하는 기존 navigation/vision token cache |
| Vision cache | `data/processed/vision_tokens/basic_drive_morai` |
| Instructions | `data/processed/instructions/basic_drive_lmdrive_v2` |
| Navigation token IDs | `data/processed/navigation_tokens/basic_drive_morai_interval2` |
| 모델 초기화 | 기존 TinyLlama/BERT 로컬 사전학습 자산에서 매 실행 새로 초기화 |
| seed | 607 |
| 1 update 입력 | batch size 1, 11프레임, frame sampling interval 2 |
| window 선택 | 현재 train stride 5, 동일 shuffle 순서 |
| 학습 window index | 115, 140, 1310, 250, 1090, 760, 1655, 1270 |
| 정밀도 | 기존 BF16 autocast / TinyLlama BF16, CPU master와 Adam state FP32 |
| optimizer | AdamW, lr 1e-5, weight_decay 0.06, 기존 cosine schedule 5595 updates |
| loss | 현재 `morai_repair_v1`, reconstruction reduction `mean`, 기존 가중치 유지 |
| gradient clip | 기존 10.0 |
| gradient checkpointing | 기존 true 유지 |
| 학습 파라미터 | 1,197,549,580개 |

실제 모델 파라미터 장치도 집계했다. CPU에 놓인 신경망 파라미터는 없었다.

| 장치 / dtype / requires_grad | 파라미터 수 |
|---|---:|
| cuda:0 / BF16 / true | 1,038,729,226 |
| cuda:0 / FP32 / true | 158,820,354 |
| cuda:0 / BF16 / false | 69,736,450 |
| cuda:0 / FP32 / false | 23,858,688 |

CPU에 있는 것은 별도의 optimizer FP32 master와 moment, 입력 준비용 데이터다.
cached 입력이므로 stage 2에서 vision encoder를 다시 추론하지 않았다.

## 3. 측정 방법

총 5회 실행 × 8 updates = **실제 VLDrive optimizer update 40회**를 수행했다.
실행 순서는 기존 4 threads → 기존 2 threads → fused 4 threads → 실제 train fused → 실제 train 기존 방식이었다.
각 실행은 별도 프로세스였고, 앞 실험의 가중치를 다음 실험에 이어 쓰지 않았다.

처음 2 updates는 초기 CUDA/optimizer 할당과 준비 비용을 포함하므로 주 비교에서는 제외했다.
주 결과는 update 3~8의 산술평균이다. 원래 1~8의 시간도 아래에 보존했다.

- 구간 측정: 단계 전후 `torch.cuda.synchronize()` + `time.perf_counter()`를 사용했다.
- `CPUAdamW.step()`과 동일한 순서/계산을 임시 실험 코드에서 전송·CPU AdamW·역방향 전송으로 나눴다.
- 실제 학습 경로 재검증: 추가 구간별 synchronization 없이 기존 `train()`과 수정된 `make_optimizer()`를 호출했다.
  최종 비교는 이 경로의 기존 `update_time_s`를 사용했다.
- 일반 Dashboard telemetry는 켜 두었다. 브라우저와 Dashboard 서버는 측정에 띄우지 않았다.
- 정상 학습 중 100 update마다 발생하는 gradient telemetry는 별도 1회 측정에서 약 0.0199초였다.
- 리소스는 별도 저빈도 모니터가 약 1초 대기 간격으로 RSS, WSL available RAM, swap,
  프로세스 CPU 사용량, nvidia-smi의 온도/메모리/전력을 표본 수집했다. 외부 명령 시간만큼 실제 간격은 더 길 수 있다.
- 온도·전력은 표본 최댓값으로, 순간 최고치나 전체 컴퓨터 전력 측정값은 아니다.
- 원래 데이터·cache·모델 checkpoint에는 쓰지 않았으며 실험용 모델 checkpoint도 저장하지 않았다.

초기 `preflight`, 전체 train/validation baseline 평가, epoch 평가와 checkpoint 저장은 이 8-update 비교에 포함하지 않았다.
별도 확인한 모델 초기화는 약 31.4~34.0초, optimizer 초기화는 약 2.1~2.7초였다.
따라서 이 보고서의 update 시간은 `train.py` 실행부터 종료까지의 전체 벽시계 시간이 아니다.

## 4. 구간별 시간: 어디서 느렸는가

단위: 초/update, update 3~8 평균.

| 구간 | 기존 4 threads | fused 4 threads | 실행 위치 |
|---|---:|---:|---|
| cache/label 데이터 읽기 | 0.2974 | 0.2973 | CPU / 파일시스템 |
| batch 구성 및 입력 전송 | 0.0009 | 0.0009 | CPU → GPU |
| 순전파 | 0.1625 | 0.1413 | GPU 중심 |
| 역전파 | 0.2843 | 0.2834 | GPU |
| gradient clipping | 0.0319 | 0.0319 | GPU |
| gradient 복사 | 0.6766 | 0.7509 | GPU → CPU |
| AdamW 계산 | **2.1287** | **0.5332** | CPU |
| 갱신한 가중치 복사 | 0.4492 | 0.4259 | CPU → GPU |
| metric 수집 | 0.0009 | 0.0008 | CPU / GPU 동기화 |
| 전체 update | **4.0333** | **2.4663** | 위 구간과 제어 비용 포함 |

기존 CPU optimizer + 왕복 복사는 **3.2545초, 전체의 약 80.7%**였다.
GPU 순전파+역전파는 약 0.4469초, 약 11.1%였다.
fused 적용의 주된 효과는 CPU AdamW 계산 시간이 약 75% 줄어든 것이다.
GPU/파일 읽기 구간의 작은 차이를 별도 최적화 성과로 해석하지 않는다.

쉽게 말하면, GPU가 답과 수정량을 빠르게 계산해도 CPU가 12억 개의 숫자를 여러 번 읽고
고치는 동안 다음 문제로 넘어가지 못했다. fused는 CPU에서 이어지는 AdamW 연산을 합쳐
실행하므로 같은 수정 작업을 더 적은 중간 처리로 마친다. GPU에 optimizer를 옮기는 기능이 아니다.

## 5. 실제 학습 함수 재검증 및 채택 여부

| 실험 | 3~8 update 평균 | 판단 |
|---|---:|---|
| 구간 측정, 기존 4 threads | 4.0333초 | 기준 |
| 구간 측정, 기존 2 threads | 5.4546초 | 35.2% 느려져 미채택 |
| 구간 측정, fused 4 threads | 2.4663초 | 38.9% 단축 |
| 실제 train(), 기존 4 threads | 4.2253초 | 실제 경로 기준 |
| 실제 train(), fused 4 threads | 2.4754초 | **41.4% 단축, 채택** |

구간 측정의 추가 동기화 때문에만 생긴 효과가 아니라 실제 학습 함수에서도 확인한 개선이다.
단, 반복 횟수가 작고 동일 PC의 background load/파일 cache 영향이 있으므로 항상 41.4%를 보장하지 않는다.

### 각 update 시간 원자료

| Update | 기존 구간 측정 | 2 threads | fused 구간 측정 | 실제 train 기존 | 실제 train fused |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.2555 | 10.7971 | 5.6161 | 7.1827 | 5.7128 |
| 2 | 3.8391 | 7.0301 | 3.2450 | 5.1074 | 2.8585 |
| 3 | 3.7072 | 6.1341 | 2.9442 | 5.0557 | 2.7665 |
| 4 | 4.5723 | 5.1998 | 2.4729 | 4.2038 | 2.6118 |
| 5 | 4.3899 | 5.1753 | 2.3135 | 4.2145 | 2.4668 |
| 6 | 3.8391 | 5.0942 | 2.3011 | 4.0265 | 2.3985 |
| 7 | 3.8065 | 5.2521 | 2.4246 | 4.0094 | 2.3312 |
| 8 | 3.8847 | 5.8721 | 2.3417 | 3.8418 | 2.2779 |

## 6. 컴퓨터 부하 확인

표는 각 프로세스 생애 중 수집된 표본의 최댓값/최솟값이다.
CPU 100%는 코어 1개 기준이다. 400%는 약 4개 코어의 연산량이며 CPU 전체 400%가 아니다.

| 측정값 | 기존 구간 측정 | 2 threads | fused 구간 측정 | 실제 train 기존 | 실제 train fused |
|---|---:|---:|---:|---:|---:|
| 프로세스 RSS 최대 GiB | 19.6559 | 19.5539 | 19.5669 | 19.4812 | 19.5035 |
| WSL available RAM 최소 GiB | 10.7482 | 10.7668 | 10.8385 | 10.8065 | 10.9072 |
| swap 최대 GiB | 0 | 0 | 0 | 0 | 0 |
| 프로세스 CPU 최대 % | 400.4 | 201.8 | 398.3 | 401.3 | 399.6 |
| GPU 전체 사용 메모리 표본 최대 MiB | 7857 | 7854 | 7854 | 7841 | 7850 |
| GPU 온도 표본 최대 °C | 41 | 42 | 43 | 42 | 44 |
| GPU 전력 표본 최대 W | 53.15 | 48.82 | 54.20 | 53.93 | 53.65 |

PyTorch peak allocated는 모든 실행에서 **5.4764 GiB**, peak reserved는 **5.78125 GiB**로 동일했다.
nvidia-smi 값은 Windows 그래픽 프로그램 등의 사용량도 포함하므로 PyTorch 수치와 다르다.
Windows 전체 가용 RAM도 실험 중 별도 확인했으며 한 측정 시점에 약 14.9 GiB가 남아 있었다.

실험 시작 시 WSL 가용 RAM 25 GiB 이상을 요구했고, 실행 중 available RAM <3 GiB,
swap >0.25 GiB, GPU 사용 메모리 >13500 MiB, GPU 온도 ≥80°C를 발견하면 다음 update 진입 전에 중단하도록 했다.
모든 실험에서 중단 조건은 발생하지 않았다. 이는 실험용 표본 감시였으며 즉시 차단하는 하드웨어 보호 기능은 아니다.
실험 감시 코드도 요청대로 제거했다.

CPU 스레드 수, batch 크기, GPU 메모리 배치, WSL 메모리 제한을 올리지 않았다.
이 측정 범위에서는 컴퓨터에 추가적인 큰 자원 부담을 주는 변경이 아니라고 판단했다.
다른 대형 프로그램과의 동시 사용이나 장시간 학습에서도 부하가 절대 늘지 않는다는 보장은 아니다.

## 7. 수치 및 저장·복원 검증

다섯 실행의 동일 update끼리 비교한 아래 값의 최대 절대 차이는 모두 **0**이었다.

- total loss, trajectory loss, pruning loss, reconstruction loss.
- sampled retention, soft retention, gradient norm.
- 최종 전체 학습 파라미터 SHA-256.

공통 최종 학습 파라미터 SHA-256:

```text
198e9ac3eec9bb17eceb574f95580a249f90f3316ab58765e4c5a7739184724b
```

작은 optimizer 검증도 별도로 수행했다.

- CPU/CUDA 모델 × FP32/BF16 = 4개 조합, 각 6 optimizer steps.
- 기존/fused의 모델 가중치, CPU FP32 masters, `step`, `exp_avg`, `exp_avg_sq`를 매 step bitwise 비교: 모두 일치.
- 3번째 step에서 fused optimizer state를 복원하고 이후 3 steps 진행: 연속 실행과 일치.
- parameter group별 LR multiplier와 cosine schedule 포함.
- 잘못된 `cpu_fused` 타입 및 `state_device=model`과의 부적절한 조합 거부 확인.
- 별도 262144-element FP32 tensor의 10-step 실험에서도 가중치/moment 최대 차이 0.
- 기존 전체 회귀 테스트: **275 passed, 3 skipped, 30 subtests passed** (20.95초).
  skip은 선택적인 TinyLlama checkpoint 테스트와 Windows 전용 2개 테스트였다.

위 bitwise 일치는 이번 PyTorch/하드웨어/입력에서 관찰한 결과다. 다른 버전이나 모든 미래 학습 입력에 대한 수학적 보장은 아니다.

## 8. 실제 적용 내용

생산 코드 변경은 다음 두 파일에 한정했다.

1. [utils/trainer.py](../../utils/trainer.py): `CPUAdamW(..., fused=False)` 및
   `make_optimizer(..., cpu_fused=False)` 옵션 연결과 유효성 검사.
   기존 CPU FP32 masters/state, gradient/weight 전송 순서, optimizer 구현 식별자와 나머지 학습 경로는 유지.
2. [configs/vldrive_train.json](../../configs/vldrive_train.json): 현재 환경에서 검증한 옵션 활성화.

```json
"cpu_threads": 4,
"optimizer": {
  "lr": 1e-05,
  "weight_decay": 0.06,
  "state_device": "cpu",
  "cpu_fused": true
}
```

일반 Python API의 기본값은 false이므로 옵션이 없는 다른 설정은 기존 방식으로 동작한다.
현재 설정에서 기존 실행 방식으로 되돌릴 때는 `optimizer.cpu_fused`를 false로 바꾸면 된다.
CPU fused 지원은 현재 설치된 PyTorch 2.8.0에서 소스와 실제 실행으로 확인했다.
이전 checkpoint의 strict source/settings 검사는 우회하지 않았으므로 이전 코드/설정 checkpoint를 억지로 이어 학습하지 않는다.

## 9. 이번에 적용하지 않은 방향과 다음 우선순위

| 후보 | 판단 및 이유 |
|---|---|
| CPU 스레드 2개 | 실제로 35.2% 느려졌다. 기존 4개 유지. |
| CPU 스레드 대폭 증가 | 추가 CPU 점유를 요구하므로 이번에는 실험/적용하지 않았다. |
| optimizer 전체 GPU 이전 | 현재 model-state optimizer는 FP32 모델을 요구한다. 약 12억 파라미터의 FP32 가중치+gradient+2 moments만 약 17.85 GiB이므로 activation 제외로도 현재 GPU 용량을 넘는다. 적용하지 않았다. |
| batch 확대 / gradient checkpointing 해제 | GPU 메모리 및 계산 부하 증가 가능성이 있다. 현재 GPU 순전파·역전파 비중도 작아 적용하지 않았다. |
| 전체 dataset RAM preload / 대형 pinned-memory buffer | RAM 증가를 동반한다. 데이터 읽기는 약 0.30초로 우선순위가 낮아 적용하지 않았다. |
| GPU↔CPU 복사 개선 | fused 이후 약 1.18초/update가 남는다. 다음 최우선 조사 후보지만, 대형 pinned buffer를 먼저 할당하지 말고 작은 재사용 버퍼와 메모리 상한을 정한 비교가 필요하다. 이번에는 미적용. |
| 소량 prefetch 또는 bounded cache | 데이터 읽기를 줄일 다음 후보. 전체 캐시를 복제하지 않고 작게 제한한 실험으로 효과부터 확인해야 한다. 이번에는 미적용. |
| loss/selector/precision/모델 변경 | 처리시간 최적화 범위를 벗어나므로 변경하지 않았다. |

현재 5595 updates에 이번 실제 train 평균을 단순 곱하면 update 구간만 기존 약 6.57시간,
fused 약 3.85시간이다. 모델 로딩·preflight·평가·저장 시간을 제외했고 이후 token 수/시스템 상태가 변하므로
**전체 학습 완료 시간의 예측값으로 사용하면 안 된다.** 실제 Dashboard ETA로 다시 확인해야 한다.

## 10. 실험 자료 정리

사용자가 요청한 대로 임시 실험 스크립트 `utils/_training_time_experiment.py`, 해당 임시 bytecode가 있다면 그것,
임시 `.runtime/training_time_20260919/`의 JSON/JSONL 측정 파일은 결과를 이 문서에 옮긴 뒤 삭제했다.
실험 가중치 파일은 만들지 않았다. 기존 학습 로그·checkpoint·데이터와 기존 회귀 테스트는 삭제하지 않았다.
최종 산출물은 생산 코드/설정의 작은 수정과 이 Markdown 보고서다.
