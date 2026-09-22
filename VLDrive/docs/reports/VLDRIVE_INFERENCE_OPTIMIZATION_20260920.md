# VLDrive 무제한 추론 및 CPU/GPU 실행 최적화

`configs/vldrive_realtime.json`의 `runtime.prediction_hz`를 **null**로 설정했다. 인위적인 추론 빈도 제한이 없으며, 새로운 유효 입력 이력이 준비되면 다음 waypoint를 계산한다. 학습과 같은 5Hz 목표의 입력 이력, 11프레임, 허용 시간 간격과 stale 검사는 유지한다.

## 실측 결과

동일 프로세스에서 동일 best 모델/encoder를 한 번 로딩한 후 **A1 → B1 → A2 → B2** 순서로 실제 MORAI UDP를 각각 40초 측정했다. A도 상한을 제거한 기존 구현이므로, 아래 개선은 0.5Hz 제한 제거 효과와 구분된다. 각 구간은 첫 11프레임 준비 시간을 포함하고 모델 로딩은 제외한다.

| 항목 | 기존 A1 | 최적화 B1 | 기존 A2 | 최적화 B2 |
|---|---:|---:|---:|---:|
| 유효 출력 수 | 75 | 127 | 82 | 124 |
| 실제 처리율 | 1.88Hz | **3.16Hz** | 2.05Hz | **3.10Hz** |
| 센서 수신→출력 중앙값 | 812ms | **683ms** | 800ms | **641ms** |
| 이력 초기화 | 10 | **0** | 10 | **0** |
| 입력 큐 drop | 10 | **0** | 10 | **0** |
| 계산 후 폐기된 예측 | 7 | **0** | 6 | **0** |
| CPU 전처리 호출 | 384 | **195** | 387 | **194** |
| CPU 전처리 누적 경과 시간 | 10.31s | **5.49s** | 10.27s | **5.36s** |
| 평균 CPU 사용량, 코어 환산 | 1.73 | 1.78 | 1.76 | 1.76 |
| GPU 실제 tensor 최고 점유 | 3.580GiB | 3.593GiB | 3.595GiB | 3.614GiB |

평균 처리율은 약 **1.96Hz → 3.13Hz**, 약 **1.6배**였다. B2는 화면 HTTP 서버도 켰으며 후반 약 20초 동안 상태 API를 91회 조회했다. 브라우저의 전체 렌더링 부하를 정밀 비교한 실험은 아니다.

MORAI와 GPU를 공유하는 실제 입력의 순차 비교이므로 각 구간의 센서 내용, instruction 문장 변형과 외부 GPU 부하는 완전히 동일하지 않다. 결과는 현재 장면·장비에서 관측한 범위이며 모든 주행 상황의 고정 3.1Hz를 보장하지 않는다.

## 적용한 변경

1. **상한 제거**: `runtime.prediction_hz=null`. 숫자를 설정하면 선택적으로 해당 Hz 상한을 다시 적용할 수 있다. 0이나 음수는 오류로 처리한다.
2. **CPU 중복 작업 제거**: LiDAR 수신 시각으로 이력에 사용할 수 없는 중간 프레임을 먼저 거른다. 그 뒤 필요한 프레임만 JPEG 4개 decoding과 LiDAR 전처리를 수행한다. 유효 샘플의 전처리 자체는 기존 adapter를 그대로 사용한다.
3. **GPU 작업 대기 분리**: `runtime.separate_encoder_stream=true`로 encoder를 별도 CUDA stream에서 실행한다. 매번 GPU 전체 작업을 기다리던 동기화를 현재 stream 범위로 한정했다. Stage-2는 필요한 feature의 준비 완료 event만 기다린다.
4. **이력 조회 경량화**: 대기/상태 확인 루프에서는 metadata만 읽는다. 실제 추론 직전에 한 번만 `[1,11,106,256]` tensor를 구성한다.
5. **화면 비활성화 처리**: `visualization.enabled=false`일 때 Dashboard 객체와 HTTP 서버를 생성하지 않는다.
6. 실행 보고서에 실제 추론 구간 시간과 `output_hz`를 기록하고, 출력별 `preprocessing_ms`도 기록한다.

CUDA stream을 분리할 때는 모델 초기화 완료를 기다리고, feature clone 뒤 준비 event를 기록한다. 소비 stream은 이 event를 기다리며 `record_stream()`으로 메모리가 소비 작업 완료 전에 재사용되지 않도록 한다. 이는 [PyTorch CUDA stream 문서](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams)와 [record_stream 문서](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html)의 의존성/메모리 수명 규칙에 따른다.

## CPU/GPU 관점의 해석

CPU는 UDP parsing, 이미지 decoding, 전처리, navigation, 모델 실행 요청 및 로그를 담당한다. 중간 프레임의 중복 전처리를 제거해 전처리 호출을 약 절반으로 줄였다. 실제 실행 프로세스의 평균 CPU 사용은 전후 약 1.7–1.8코어 수준으로 비슷했다. PyTorch CPU 연산 스레드는 기존 2개를 유지했다.

GPU는 vision encoder와 Stage-2 모델을 모두 실행한다. 두 작업이 자신의 준비된 입력으로 진행하도록 하여 encoder가 Stage-2 때문에 오래 기다리는 구간과 이력 재수집을 줄였다. 공유 실행에서 encoder 경과 시간 중앙값은 기존 102–113ms에서 169–172ms로 증가했고 Stage-2는 291–299ms에서 약 285ms가 되었다. **각 계산 시간이 모두 짧아진 것이 아니라 전체 실행의 대기와 끊김이 줄어 처리율이 좋아졌다.** 겹쳐 실행되는 두 구간의 시간을 합산하거나 그 역수만으로 전체 Hz를 계산하면 안 된다.

GPU 실제 tensor 점유의 최대 차이는 약 35MiB였다. PyTorch 예약 메모리는 A1 3.78GiB에서 B2 4.68GiB까지 늘었다. 이 값에는 동일 프로세스에서 반복 생성한 stream과 앞선 실행의 allocator cache가 포함된다. 실제 지속 실행은 encoder stream 1개를 사용한다. 실험 중 시스템 전체 GPU 사용 메모리 순간값은 약 9GiB/16GiB, GPU 온도는 약 49–52°C였다. 프로세스 RSS 최고값은 모델 로딩을 포함해 약 6.85GiB였다.

입력 큐 3개와 실제 특징 이력 11개는 그대로다. CPU 스레드 확대, 다중 모델 프로세스, 대형 batch, compile/대형 캐시 추가는 적용하지 않았다. 모델 파라미터, 학습 config, loss, encoder 정밀도, Gumbel 선택, reconstruction off, waypoint 의미 및 제어 미연결을 유지했다.

## 변경 파일

- `configs/vldrive_realtime.json`
- `inference/realtime_vldrive.py`
- `inference/live_pipeline.py`
- `inference/vldrive_runtime.py`
- `tests/test_realtime_vldrive.py`
- `tests/test_live_navigation_dashboard.py`
- `utils/inference_dashboard/README.md`

## 실행과 원본 측정

WSL lmdrive 환경의 VLDrive 폴더에서 기존 명령을 그대로 사용한다:

```bash
python inference/realtime_vldrive.py --config configs/vldrive_realtime.json
```

- `outputs/inference/optimization_20260920_005138/comparison.json`
- 같은 폴더의 A1/B1/A2/B2_dashboard별 `timings.json`, `inference_report.json`, `waypoints.jsonl`

## 수치 및 회귀 검증

처음 실제 live 입력으로 old/new encoder를 bitwise 비교했을 때 최대 차이 0.001953125가 있어 완전 일치 검사는 실패했다. 이 실패를 숨기거나 허용 오차를 새로 늘리지 않고, 기록 센서의 동일 입력으로 기존 코드 반복과 stream 변경을 분리해 다시 검증했다.

| 비교 | 최대 token 절댓값 차이 | 평균 절댓값 차이 |
|---|---:|---:|
| 기존 encoder → 기존 encoder 반복 | 0.0009765625 | 0.00001567 |
| 기존 encoder → 별도 stream encoder | 0.001953125 | 0.00001494 |
| 별도 stream 특징 → 소비 stream 이력 | **0** | **0** |

기존 encoder도 bitwise 반복 일치를 보장하지 않았다. old/new의 차이는 작업 전부터 사용하던 token 검사 `atol=0.002, rtol=0.002` 안이다. 이를 모든 입력에서 완전 동일하다고 표현하지 않는다. 동일한 cached token window와 Gumbel RNG 상태를 넣었을 때 기존/수정 Stage-2의 **waypoint 및 선택 token 수는 정확히 일치**했다.

기존 `run_offline`의 A–D 전체 회귀도 그대로 실행했다. 실제 기록 센서 11프레임 → encoder → Stage-2 검사를 통과했고, 캐시 대비 token 최대 차이는 0.001953125, waypoint 좌표 최대 차이는 **0.0078125m**였다. 기존 waypoint tolerance `atol=0.05m, rtol=0.01`을 유지했다. 동일 raw feature/RNG 반복 출력은 정확히 일치했고 reconstruction 호출은 0회였다.

- `outputs/inference/optimization_20260920_005138/numerical_checks.json`
- `outputs/inference/optimization_20260920_005138/offline_regression/inference_report.json`

추가 회귀 테스트는 무제한/유한 Hz 설정 검증, metadata 조회 중 tensor stack 금지, CUDA producer→consumer 이력 정합성 및 이력 삭제 시 메모리 수명, 샘플링 전 전처리 생략과 anchor instruction/카메라 정합성을 포함한다.

최종 전체 테스트는 **319 passed, 1 skipped, 30 subtests passed**(34.77초)였다. CUDA stream handoff 테스트도 GPU에서 통과했다. 임시 비교/수치 검증 스크립트와 원본 코드 임시 복사본은 삭제했다. 모델 파일과 기록 데이터는 수정하지 않았고, 실험 프로세스는 모두 종료했다.
