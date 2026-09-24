# 실시간 11프레임 추론 개선

## 범위와 원리

현재 checkpoint `outputs/logs/vldrive_20260921_172411_715080/model-best.pt`를 유지한다.
학습 신경망 소스, 가중치, Gumbel 선택, 11프레임 입력과 약 2초의 시간 범위를 유지한다.
이 변경은 `inference/` 실행 경로에만 적용한다.

### 추론 계산

`inference/fast_window.py`는 DDIA의 mask와 RoPE sin/cos를 매 층에서 다시 계산하지 않고,
같은 forward 안의 층들이 공유한다. 가중치와 GEMM 크기, softmax, 출력 head 계산은 기존과 같다.
forward가 끝나거나 예외가 발생하면 attention 실행 함수를 복원한다.
동일 모델의 동시 forward는 지원하지 않는다. 현재 실시간 코드는 Stage-2 소비 스레드가 하나다.

Vision Encoder 토큰은 이전과 같이 보관한다. MEFA·Q-Former·LLM의 시각 토큰 계산은
전체 11프레임에 대해 수행한다. 창이 이동하면 과거 프레임의 MEFA 문맥도 바뀌고,
Gumbel 선택도 다시 이루어지므로 이전 시각 KV를 그대로 재사용할 수 없다.
`use_cache=True`만 켜는 방식은 적용하지 않았다.

더 공격적인 빈 토큰 제거와 DDIA 부분 행렬 계산도 시험했지만 기본 구현에 남기지 않았다.
48쌍에서 평균 절대 좌표 차이는 0.00232m, 최대 차이는 0.125m였다.
BF16의 계산 형태를 바꾸지 않는 방식을 최종 비교 대상으로 삼았다.

### 시간에 따른 이력 선택

`inference/timed_history.py`는 제한된 크기의 실제 프레임 버퍼에서 최신 시각 기준
`t-2.0, t-1.8, ..., t-0.2, t`에 가까운 11프레임을 선택한다.
각 목표 시각에서 허용 오차는 기존 0.06초이며, 동일 프레임 중복이나 보간은 하지 않는다.
유효한 11개가 없으면 해당 창의 추론을 보류한다.

수집 Hz와 모델 입력의 시간 간격을 분리했다. 예를 들어 원본 10Hz 수집이라면:

```text
첫 입력: [0, 2, 4, ..., 18, 20]  → 약 2초
다음 입력: [1, 3, 5, ..., 19, 21] → 약 2초
```

0.26초를 조금 넘는 간격 때문에 버퍼 전체를 즉시 비우지 않는다.
기본 1초 초과의 단절이나 시뮬레이터 시간 역행에는 초기화한다.
센서/자세/예측의 유효시간 검사는 기존대로 유지하며, 버퍼 보존은 오래된 결과 사용 허가가 아니다.
최초 준비에는 여전히 약 2초의 실제 입력이 필요하다.

## 실제 checkpoint 출력 비교

RTX 5060 Ti, PyTorch 2.8.0+cu128, CPU 연산 스레드 2개, Stage-2 BF16.
전체 데이터 범위에서 고른 12개 창 × 4개 RNG seed 조건, 총 48쌍을 비교했다.
각 쌍에는 동일 입력과 동일 RNG를 적용하고, 전후 실행 순서를 교대로 바꿨다.
실시간 실행에서는 매 프레임 재시드하지 않는다.

| 항목 | 기존 | DDIA 공통 계산 재사용 |
|---|---:|---:|
| Stage-2 평균 | 72.09ms | 65.44ms |
| 중앙값 | 72.62ms | 65.66ms |
| p95 | 79.09ms | 72.21ms |
| tensor 최고 GPU 메모리 | 3.085GiB | 3.112GiB |

같은 Gumbel 선택 개수, waypoint 최대/평균 차이 **0m**.
추가로 실제 navigation 문장이 서로 다른 12개 창에서도 최대 차이 0m를 확인했다.
단위 테스트에서는 개수뿐 아니라 Gumbel mask 자체가 같은지도 검사한다.
이는 측정한 입력/장비의 결과이며, 모든 환경에서 bitwise 동일함을 보장한다는 뜻은 아니다.

## 전체 파이프라인 실험 방법

저장된 6개 장면의 카메라 JPEG 4개와 LiDAR를 메모리에 미리 읽고 10Hz 수신을 모사했다.
실제 `run_live`의 CPU JPEG decoding/전처리, Vision Encoder, 이력 선택,
Stage-2, JSON 출력 코드를 실행했다. 같은 프로세스에서 모델은 한 번만 로딩했다.
구간당 18초이며 초기 이력 준비를 포함한 Hz와 준비 후 anchor 진행 기준 Hz를 구분했다.
프레임은 장면 사이를 순환하므로 성능 실험이며 주행 정확도 실험은 아니다.

UDP 네트워크·MORAI 렌더링·브라우저·route navigation·제어기는 이 실험에 포함하지 않았다.
실차나 시뮬레이터 제어값은 송신하지 않았다. CPU 사용은 process CPU time / wall time의 코어 환산이다.

초기 후보인 `capture_hz=10`, CPU 2개, 큐 3개는 출력 빈도가 늘어도
Encoder/Stage-2 경쟁으로 지연이 늘어 채택하지 않았다.

| 설정 | 준비 후 출력 Hz | 센서→출력 중앙값 | CPU 코어 환산 | 최고 GPU tensor 메모리 |
|---|---:|---:|---:|---:|
| 기존 5Hz / CPU 2 / 큐 3, 2회 | 5.00 | 181~183ms | 0.89~0.92 | 3.31~3.55GiB |
| **채택: 개선 5Hz / CPU 2 / 큐 1** | **5.00** | **173ms** | **0.85** | **3.30GiB** |
| 개선 10Hz / CPU 2 / 큐 3, 2회 | 약 6Hz | 503~513ms | 약 2.00 | 약 3.64GiB |
| 개선 10Hz / CPU 1 / 큐 1, 2회 | 6.12~6.52 | 311~335ms | 약 1.7 | 약 3.62GiB |
| 개선 10Hz / CPU 1 / 큐 1 / GPU 순차 실행 | 6.77 | 312ms | 1.40 | 3.30GiB |

채택 설정에서는 18초에 80개 출력(초기 이력 준비 포함 4.44Hz), 준비 후 5Hz였다.
지연 p95는 189ms, history reset/queue drop/추론 폐기는 모두 0이었다.
출력별 이력 길이는 모두 2초였고 목표 시각 오차는 실질적으로 0(부동소수점 오차 수준)이었다.
기존 2회 중 초기 1회에 reset 1개/drop 2개가 발생했고 다른 1회는 둘 다 0이었다.
이는 cold start 영향을 포함하므로 실제 네트워크에서 초기화가 없어졌다는 증거로 해석하지 않는다.

10Hz 후보는 현재 장비에서 처리율보다 **입력의 노후화와 CPU 부하 증가**가 컸다.
따라서 기본 수집값은 5Hz로 유지했다. GPU 순차 실행 후보도 최종 코드에서 제거했다.
5Hz 설정에서 남은 실행 시간에는 JPEG decoding/전처리 약 50ms,
encoder 약 46ms, Stage-2 약 64ms 및 polling/스케줄링 시간이 있다.
이번 수정만으로 현재 PC의 실제 MORAI 손주행이 10Hz가 된다고 주장하지 않는다.

## 채택 설정과 회귀 검증

```json
{
  "target_hz": 5.0,
  "capture_hz": 5.0,
  "resample_history": true,
  "history_reset_gap_s": 1.0,
  "optimized_window": true,
  "prediction_hz": null,
  "encoder_queue_size": 1,
  "cpu_threads": 2
}
```

이력 단위 테스트는 10Hz 수집 시에도 2초 간격 조건 유지, 순간적인 0.27초 공백 후
버퍼를 버리지 않고 다음 유효 시각에 복구, 중복 프레임 거절, 장시간 단절/명시적 reset과
generation 무효화를 검사한다. 시각 오차가 큰 입력을 허용해 정상 출력처럼 만드는 방식은 아니다.
원래의 stale/제어 유효시간과 시뮬레이터 시간 역행 초기화도 유지한다.

실험 전용 Python 실행 파일은 결과 정리 후 삭제하고, 기능 회귀 테스트는 `tests/`에 남긴다.

최종 회귀: **165 passed (20.98s)**. 실시간 입력, DDIA, offline 입력,
navigation/dashboard anchor 연결, waypoint/expert/lateral 제어기 호환성,
`--no-control` 우선 적용 및 상충 옵션 거절을 포함한다.
모델·학습 소스의 Git diff는 없으며, checkpoint source/provenance 검증도 통과했다.
이 테스트 결과는 실제 MORAI 주행 성공이나 실제 네트워크 지연의 검증을 뜻하지 않는다.

## 실행 및 설정

설정 파일: `configs/vldrive_realtime.json`의 `runtime`.

- `target_hz`: checkpoint 학습 이력 간격. 현재 5Hz, 즉 0.2초. 수집 속도를 바꾸려고 이것을 10으로 바꾸면 안 된다.
- `capture_hz`: 새 센서 프레임의 수집 목표. 실제 센서 공급과 처리 능력에 따라 달라진다.
- `resample_history`: 실제 수신 시각에 맞춰 11프레임을 선택한다.
- `history_reset_gap_s`: 버퍼 전체를 초기화할 장시간 단절 기준.
- `optimized_window`: 학습 코드를 유지하며 DDIA 공통 계산을 재사용한다.
- `prediction_hz`: null이면 별도 출력 주기 제한 없음. 새로운 유효 창이 있어야 출력한다.

손주행 시각화:

```bash
python inference/realtime_vldrive.py --no-control
```

로그의 `history_span_s`, `history_max_offset_s`, `history_resets`,
`sensor_to_output_ms`, `window_execution`으로 선택 시간과 실제 지연을 확인할 수 있다.
원래 수집 방식과 비교하려면 `resample_history=false`, `capture_hz=5`,
`optimized_window=false`로 설정한다.
