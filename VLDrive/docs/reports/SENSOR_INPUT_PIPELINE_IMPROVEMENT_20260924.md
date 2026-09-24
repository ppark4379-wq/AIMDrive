# MORAI 입력 조립 개선 결과 — 2026-09-24

이 문서의 window 단위 VE 실행은 실제 GPU 지연에서 성능 회귀가 확인되어
[연속 VE 입력 처리](STREAMING_VISION_INPUT_FIX_20260924.md)로 교체했다.
아래는 당시 구현과 진단 이력을 보존한 기록이다.

## 적용 범위와 결과

실시간 센서 입력을 보관하고 11프레임으로 조립하는 경로를 개선했다.
기본 설정 `resample_history=true`에 적용된다. 모델, checkpoint, 학습 loss,
제어 알고리즘은 이번 작업에서 변경하지 않았다. 현재 checkpoint의
`velocity=0`, 카메라/LiDAR 전처리, float16 캐시 경계도 유지한다.

CPU 회귀 테스트 **136개 통과**. 실제 저장 센서 9장면의 입력 tensor는
기존 전처리와 모든 값이 정확하게 같았다. 시간 간격 모형에서는 기존
선택 방식의 입력 고갈을 재현하고 새 방식의 지속적인 입력 생성을 확인했다.
실제 MORAI와 GPU 모델을 함께 돌린 변경 후 Hz는 이번에 측정하지 않았다.

## A1Challenge에서 확인하고 반영한 점

다음 구현들을 비교했다. 아래 경로는 `Z:/A1Challenge/` 기준이다.

| 참고 위치 | 확인한 동작 | VLDrive에 반영 |
|---|---|---|
| `Code/A1Challenge/StreamPETR/tools/MORAI_Ver2_UDP/Camera.py:145` | 이전 카메라 구현은 queue에서 한 장을 가져와 decode | 오래된 입력 작업이 밀리는 문제를 비교하는 기준 |
| `A1Challenge/StreamPETR/tools/MORAI_Ver2_UDP/Camera.py:174` | queue를 비워 최신 JPEG만 decode | 선택 전에는 압축 JPEG 참조만 보관하고 필요한 입력만 decode |
| `A1Challenge/StreamPETR/tools/morai_ver2_pretrained_inference_main.py:124` | 처음 전체 카메라를 채운 뒤 마지막 정상 영상 유지 | 모든 카메라의 번호가 매번 증가해야 한다는 조건 제거 |
| 같은 파일 `:325`, `:360`, `:417` | 짧은 lock의 최신 snapshot 슬롯, CPU 전처리와 GPU 작업 분리 | raw 수집, 입력 준비, LLM 소비를 분리하고 최신 완성 window 한 개 발행 |
| `AIM2026/MORAI_UDP/Camera.py:189` | 최신 JPEG 선택, 별도 UDP 수신 | 기존 수신 스레드를 유지하고 수집 단계의 무거운 계산 제거 |
| `AIM2026/main.py:213`, `:299` | 마지막 카메라 영상 유지, Event로 추론 대기 해제 | 실제 새 anchor 기반 처리와 Event 알림 |

참고 구현은 새 영상이 없어도 snapshot 번호를 증가시킬 수 있고, 영상의
최대 나이와 센서 간 시각 차이 검사가 충분하지 않다. 이 부분은 가져오지 않았다.
VLDrive는 실제 LiDAR 번호가 증가해야 새 프레임으로 인정하고, 재사용하는
카메라의 원래 번호와 시각을 유지한다. 기존 freshness/skew 검사는 그대로 적용한다.
참고 프로젝트의 변경 전후 실측 성능을 입증하는 자료는 확인하지 못했다.

## 달라진 입력 흐름

```text
카메라 / LiDAR / 차량 상태 / GPS의 기존 UDP 수신 스레드
    ↓ 최신 완성 센서 값과 원래 수신 시각 조회
가벼운 raw snapshot 수집 + 해당 시점 navigation 계산
    ↓ JPEG bytes / LiDAR 배열 참조 / 센서 번호·시각 보관
제한된 raw 이력에서 약 0.2초 간격의 실제 11프레임 선택
    ↓ 필요한 새 특징 수에 맞춰 Encoder 작업 예산 확인
선택된 입력만 JPEG decode·전처리 → Vision Encoder
    ↓ 이미 계산한 동일 LiDAR anchor 특징은 재사용
가장 최근에 완성된 11프레임 window 한 개 발행
    ↓ Event 알림
기존 CCDP / MEFA / Q-Former / LLM → waypoint
```

- 수집 스레드는 JPEG decode와 LiDAR 전처리를 수행하지 않는다.
- raw 이력을 먼저 보관하므로, 처리 주기에 들지 않았던 중간 센서 프레임도
  이후 약 2초 이력을 구성할 때 선택할 수 있다.
- 최신 LiDAR에 이전 카메라/차량 상태를 결합할 수 있지만, 수집 시 설정된
  최대 나이 0.3초와 센서 수신 시각 차이 0.15초를 만족해야 한다.
- 늦은 카메라 때문에 거부된 LiDAR도 카메라가 도착하면 다시 검사한다.
- 같은 프레임을 복제하거나 시각을 새로 붙여 11개를 채우지 않는다.
- GPU가 처리하는 중에도 raw 수집은 계속된다. 완성 window가 여러 개 밀리면
  소비자는 최신 완성본을 가져온다.
- 시뮬레이터 시간 역행과 긴 입력 단절에서는 generation을 바꿔 이전 결과를 폐기한다.
  추론 도중 navigation/센서가 무효해진 결과도 출력하지 않는다.
- 여기서 시각은 기존 수신기의 monotonic 수신 완료 시각이다. 센서의 물리적인
  동시 촬영이나 시뮬레이터 공통 시계 동기화를 새로 구현한 것은 아니다.

변경 위치:

| 파일 | 역할 |
|---|---|
| `inference/live_frame.py` | 가벼운 `snapshot_raw()`와 선택 후 `prepare_snapshot()` 분리, 동일 카메라 decode 재사용 |
| `inference/input_windows.py` | raw 이력, 시간 기반 11프레임 선택, 특징 캐시, Encoder 작업 예산, 최신 window 발행 |
| `inference/live_pipeline.py` | 새 입력 producer 연결, 추론 전후 유효성 검사, 진단·소요 시간 기록 |
| `inference/realtime_vldrive.py` | 새 설정 검증 |
| `configs/vldrive_realtime.json` | 실제 실행 기본값 |

## 수정한 입력 고갈 문제

센서가 0.1125초마다 들어올 때 기존 조기 선택 조건은 중간 프레임을 버려
0.225초 간격의 특징만 남겼다. 그 이력에서 anchor 기준 0.2초 간격을 다시
찾으면 오차가 0.025 → 0.050 → 0.075초로 커진다. 세 번째 과거 프레임부터
허용오차 0.06초를 넘으므로, 센서가 계속 들어와도 11개를 완성하지 못했다.

새 방식은 이 단계에서 원본을 버리지 않는다. 먼저 raw를 보관하고
anchor - 0.2 × k 시각에 가까운 서로 다른 실제 프레임 11개를 선택한다.
기존 허용오차 0.06초를 늘리지 않고 해결했다.

## 설정과 연산량

`configs/vldrive_realtime.json`의 `runtime`:

| 설정 | 현재 값 | 의미 |
|---|---:|---|
| `target_hz` | 5 | 학습 이력의 목표 간격: 0.2초, 11개로 약 2초 |
| `capture_hz` | 5 | 새 경로에서 window 준비를 시작하는 최대 빈도. raw 수신을 5Hz로 자르지 않음 |
| `encoder_max_hz` | 10 | 새 VE 특징 계산의 지속 작업 예산. 초기/이력 reset에는 11개까지 burst 허용 |
| `input_buffer_frames` | 64 | raw 이력과 특징 캐시의 프레임 개수 한도 |
| `prediction_hz` | null | 별도 LLM 출력 빈도 제한 없음. 새로 준비된 입력이 필요함 |
| `cpu_threads` | 2 | 기존 CPU 계산 스레드 설정 유지 |

초기에 검토한 VE 예산 5Hz는 채택하지 않았다. 0.1125초 raw의 경우 anchor가
옮겨질 때 과거 11개 중 선택되는 원본도 달라진다. 매 window마다 항상 새 특징
한 개만 계산하면 되는 구조가 아니다. 예산 5Hz에서는 필요한 특징이 쌓일 때까지
기다리다가 window 준비가 약 0.56Hz로 떨어졌다. 10Hz 예산으로 이 조건을 처리한다.

**실제 VE 계산량이 이전보다 늘어날 수 있다.** 아래 모형에서는 약 8.94회/초였다.
예산 설정은 실행 속도 보장이 아니다. 실제 GPU가 느리거나 MORAI와 경쟁하면
window 준비나 출력이 느려질 수 있으며, 준비가 출력 유효시간 3초를 넘으면
결과를 폐기한다. 이번 작업으로 실제 추론 10Hz를 달성했다고 판단할 수 없다.

메모리는 무한히 쌓이지 않는다. raw는 최대 64개와 작업 중 선택본 참조를
보관하고 시간 범위도 제한한다. 특징 64개의 tensor 본체는 약 6.625MiB다.
실제 센서 9장면의 JPEG+LiDAR 크기는 프레임당 약 2.83~3.16MiB였다.
raw 전체 RAM은 payload 크기 및 작업 중 참조에 따라 달라진다.

## 검증 결과

### 시간 간격 재현 — CPU 가상시간, 모델 계산 시간 제외

| 입력 | 방식 | 완성 window | 준비 빈도 | 새 VE 작업 빈도 |
|---|---|---:|---:|---:|
| raw 201개, 0.1125초 간격 | 기존 조기 선택 | 0 | 0 | 해당 없음 |
| 같은 raw | 새 방식, VE 예산 10Hz | 90 | 4.347Hz | 8.938Hz |
| 같은 raw | 새 방식, VE 예산 5Hz | 12 | 0.556Hz | 5.000Hz |
| raw 221개, 0.1초 간격 | 새 방식, VE 예산 10Hz | 101 | 5.000Hz | 5.000Hz |

빈도는 최초 완성 이후의 anchor 간격 기준이다. 0.1125초/10Hz 예산 조건에서는
새 특징 194개, 특징 재사용 796회, 예산 대기 4회였다. 초기 11개 이후 window당
새 특징은 평균 2.056개, 최대 5개였다. 완성 window 최대 간격은 0.5625초였다.
실제 GPU 추론이나 UDP 네트워크 성능 수치와 구분해야 한다.

### 실제 저장 센서 전처리 비교 — CPU 2스레드

대상: `basic_drive/new_structure_full_route_20260917_004823/` 아래
첫 번째 lap의 `episode_000`, 전체 3,009장면 중 인덱스
0, 376, 752, 1128, 1504, 1880, 2256, 2632, 3008.

- 동일 JPEG, LiDAR, 난수 seed에서 새 입력과 기존 전처리 입력 **모든 tensor 정확히 일치**.
- 전면 `[1,3,224,224]`, 나머지 이미지 `[1,3,128,128]`,
  LiDAR `[1,40000,4]`, velocity `[1,1]` 등 기존 규격 유지.
- raw snapshot 180회: 중앙값 **0.0051ms**, p95 **0.0361ms**.
- 새 경로 JPEG decode+전처리 중앙값 **55.5ms**,
  기존 직접 전처리 중앙값 **51.8ms**. 소수 순차 측정이므로 작은 차이에
  성능 개선 의미를 부여하지 않는다. 전처리 연산 자체를 빠르게 만든 결과는 아니다.
- 개선의 초점은 무거운 계산을 수집 단계에서 분리하고, 거부되거나 이미
  계산한 입력에 반복 작업을 하지 않도록 하는 데 있다.

### 회귀 검사

**136 passed in 21.25s**, 실제 실행 config 검증도 통과.

대상: `test_raw_window_pipeline`, `test_live_sensor_snapshots`,
`test_live_navigation_dashboard`, `test_inference_window_optimization`,
`test_realtime_vldrive`, `test_morai_realtime_contract`, `test_waypoint_control`.

원본 시각/번호 보존, 캐시 재사용, 늦은 카메라 재시도, 이력 reset,
연산 예산, 메모리 개수 한도, 잘못된 센서 입력 거부와 추론 도중 입력이
무효해진 경우의 출력 폐기를 검사했다. 실제 GPU나 제어 UDP는 실행하지 않았다.
일회성 측정은 stdin으로 실행하여 임시 실험 스크립트 파일은 남기지 않았다.
회귀 테스트는 `tests/`에 유지한다.

## 실시간 확인 방법

AIMDrive 폴더에서 손주행 시각화:

```bash
python VLDrive/inference/realtime_vldrive.py --no-control
```

`outputs/inference/<실행>/inference_report.json`에 출력이 0이어도
`input_preparation` 및 `history_selection`을 주기적으로 남긴다.

- `raw_frames`: 실제 새 LiDAR anchor로 수집한 입력 수.
- `prepared_windows`: 준비하여 발행한 완성 window 수.
- `encoded_frames` / `feature_cache_hits`: 새 VE 계산 / 기존 특징 재사용 수.
- `encoder_budget_deferrals`: Encoder 작업 예산 부족으로 미룬 횟수.
- `stale_windows`: 준비 완료 시 anchor가 출력 유효시간을 넘은 window 수.
- `history_selection.reason`: warmup, 시간 정렬 실패, 예산 대기, 인코딩 상태.
- `raw_retries`: 늦은 센서 때문에 거부되었다가 같은 LiDAR로 수집에 성공한 수.

`waypoints.jsonl`에는 원래 센서 번호·이력 시각과 `new_vision_frames`,
`window_preparation_ms`, `window_vision_ms`, `window_preprocessing_ms`를 기록한다.
`total_inference_ms`는 해당 window 준비에서 실제 새로 계산한 VE 시간 합과
Stage2 시간 합이며 CPU 전처리·대기를 포함하지 않는다.
실제 입력 지연은 `sensor_to_output_ms`로 확인한다.

후속 MORAI 실행에서는 출력 Hz뿐 아니라 이 지연과 `stale_windows`, GPU 부하를
함께 비교해야 한다. 현재 검증은 입력 조립의 기능과 시간 선택 결함 수정에 대한
검증이며, 실제 주행 품질 또는 변경 후 GPU 성능 측정은 포함하지 않는다.

## 실행 사전 검증 보완

`vldrive_20260924_231017_631646` 실행은 센서 처리 이전의 `load_contract()`에서
실패했다. 최근 병합된 `9b60831`이 학습 provenance에 기록된 다음 파일들을
변경했기 때문이다. 단순 줄바꿈 차이는 아니며, 해당 커밋 부모의 CRLF 파일이
학습 당시 SHA256과 일치하는 것을 확인하고 실제 변경 내용을 검토했다.

- `datasets/morai_sensor_record_dataset.py`: admission/geometry 증거 검증 강화,
  동일 수식의 좌표 변환 함수 추출. 현재 캐시 추론에서 사용하는 `child` 및
  `verify_seal` 함수는 변경되지 않았다.
- `utils/morai_transfer_training.py`: VE 학습 checkpoint의 입력 계약 메타데이터
  저장 추가. 현재 캐시 기반 Stage2 추론이 실행하는 학습 함수가 아니다.
- `utils/vision_encoder_visualization.py`: SensorRecord용 축 선택 추가.
  기존 VLDrive 시각화의 기본 축은 유지된다.

`verify_sources()`에 검토된 세 파일의 **정확한 이전/현재 SHA256 조합만**
호환 대상으로 추가했다. 학습 provenance와 checkpoint를 수정하지 않았고,
소스 검증이나 모델·캐시 계약 검증을 비활성화하지 않았다.

동일 실행 config와 `vldrive_20260921_172411_715080/model-best.pt`로 실제
`load_contract()` 전체 통과를 확인했다. 관련 CPU 검사 **23개 통과**.
허용된 파일도 추가 수정되거나 이전 버전 해시가 다르면 계속 거부되는지 검사했다.
이 검증에서는 GPU 모델 로딩·UDP 주행은 실행하지 않았다.

## 실제 실행에서 확인된 성능 회귀 — 23:17 관측

실행 `vldrive_20260924_231402_717620`을 중단하지 않고 JSONL 및 report를 읽었다.
아래는 2026-09-24 23:17:40 KST의 관측이며 실행 중이므로 이후 수치는 달라진다.
진단 중 실행 코드·설정은 변경하지 않았고 추가 GPU 모델도 실행하지 않았다.

**새 방식은 현재 GPU 처리 지연에서 큰 성능 회귀를 보였다.** raw 수신은
정상이지만, window 전체를 고른 뒤 순차 인코딩하는 동안 약 2초가 지나서
다음 window와 이전 window의 겹침이 거의 사라진다. 특징 재사용에 실패하고
매번 11개를 다시 인코딩한다. 계산 시간이 제외된 이전 CPU 시간 모형으로는
이 문제를 검출하지 못했다.

| 관측 항목 | 값 |
|---|---:|
| raw 수신 속도, 직전 911개 로그 | 약 8.81Hz, 최대 수신 간격 0.156초 |
| 카메라/상태/LiDAR 수신 나이, 직전 snapshot | 약 8~60ms |
| 새로 인코딩한 프레임/window, 성공 출력 중앙값 | 11개 |
| window 준비 시간, 성공 출력 중앙값 | 2.215초 |
| 그중 VE 시간 합 / 전처리 합, 각각 중앙값 | 1.786초 / 0.273초 |
| LLM 포함 Stage2 시간, 중앙값 | 0.274초 |
| 센서 anchor부터 결과까지, 중앙값 | 2.556초 |
| 성공 출력 40개의 첫 결과~마지막 결과 | 128.883초, 약 0.303Hz |
| 출력 간격 중앙값 / 최대 | 2.214초 / 27.084초 |
| 누적 새 특징 / 특징 캐시 재사용 | 639 / 2 |
| 완성 window / 준비 완료 시 stale 폐기 | 56 / 2 |
| Stage2 실행 후 결과 폐기 | 16 |
| history reset / Encoder 예산 대기 | 0 / 0 |
| GPU 조회 한 번의 사용률 / VRAM | 96% / 8195MiB(전체 16311MiB) |

카운터는 worker가 계속 변경하므로 서로 정확히 원자적으로 찍힌 값은 아니다.
성공 출력의 이력 길이는 중앙값 1.987초, 최대 시각 오차는 0.05944초로
선택 조건 자체는 충족했다. 문제는 이력을 **선택한 이후 계산하는 지연**이다.
현재 결과 유효시간은 3초라서 성공 결과도 만료 직전이다. Stage2 후 폐기 카운터는
만료·입력 무효·reset을 합산하므로 16건 각각의 사유를 기존 로그만으로
확정할 수는 없다. 수신 지속, reset 0, 예산 대기 0, 실제 준비 지연과 거의 없는
캐시 재사용은 window 준비 구조가 주 병목이라는 직접 근거다.

`live_pipeline.py`의 stale 문구는 완성된 window의 anchor가 오래되면
"새로운 센서 프레임이 없습니다"를 출력한다. 실제 raw 수신 유무를 판별한
문구가 아니므로 이번 상황에서는 잘못된 설명이다. `History 0/11`도 raw가
없다는 뜻이 아니라 아직 인코딩 완료되어 발행된 window가 없다는 뜻이다.

수정 방향은 raw 수집 분리를 유지하면서 VE 특징을 지속적으로 쌓고 재사용하는
방식으로 바꾸는 것이다. 현재처럼 묶음을 정한 뒤 과거 11개를 순차 재계산하는
구조는 유지하기 어렵다. 이전 조기 선택의 시간 정렬 실패를 재도입하지 않도록
선택 규칙도 함께 검증하고, 특히 VE 한 장 약 0.16~0.20초와 Stage2 약 0.27초의
실제 지연을 포함한 재현 검사가 필요하다. 단순히 만료 3초를 늘리면 오래된
결과를 더 많이 표시할 뿐 이 병목을 해결하지 못한다.
