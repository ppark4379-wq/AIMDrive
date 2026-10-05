# 로컬 connector 학습과 MEFA 비교

LLM 없이 CCDP, temporal MLP, Q-Former, projection, 복원 모듈을 함께 학습한다.
Vision encoder는 지정된 체크포인트를 strict load한 뒤 고정한다.
원본 F: 데이터는 읽기만 하며 특징 캐시와 결과는 `outputs/logs/`에 저장한다.

```powershell
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/run.py --stage cache
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/run.py --stage train
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/run.py --stage evaluate
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/diagnose.py --output VLDrive/outputs/logs/connector_mefa_local_pilot --selection-mode topk
```

기본 설정은 `local_mefa.json`이다. 다른 실험은 설정을 복사하고 `output`을
새 디렉터리로 지정한다. 같은 설정의 중단된 학습은 `--stage train --resume`으로
마지막 완료 epoch부터 재개한다. 기존 체크포인트는 명시적 resume 없이 덮어쓰지 않는다.
필요한 추가 Python 패키지는 로컬 `.runtime/mefa_deps/`에 설치되어 있다.

## 입력과 분할

- 체크포인트에 기록된 `causal_ego_topic_speed_mps_50ms` 계약을 사용한다.
  미래 Ego 신호, 50ms보다 오래된 속도, GPS나 0으로의 대체는 허용하지 않는다.
- 기존 sealed 센서 reader의 파일 해시·카메라 전처리·LiDAR calibration 검사를 재사용한다.
- 동적/정적 장애물 각각 앞 2개 주행은 train, 마지막 1개 주행은 validation이다.
  각 주행에서 시간상 고르게 분포한 8개 시퀀스를 사용한다. 총 train 32, validation 16.
- 시퀀스는 원본 프레임 간격 2로 뽑은 11프레임이다. 센서 gap 150ms 초과 구간을 넘지 않는다.
  각 시퀀스 마지막 프레임만 loss와 평가에 포함한다. 따라서 항상 과거 10프레임이 있다.
- connector 학습/검증 주행은 분리되지만 encoder 사전학습 데이터와의 중복은 미확인이다.
  작은 pilot이며 독립 환경 일반화나 최종 성능으로 해석하지 않는다.

## 학습과 평가

기존 MORAI 설정처럼 encoder 특징 공간에서 `reconstruction_reduction="mean"`을 사용한다.
이는 제거된 토큰·채널 평균 MSE이며, 논문의 비정규화 합과 숫자 크기가 다르다.
목적함수는 `10 * Lprun + Lrec`, 유지 비율 목표는 0.3이다.
Q-Former는 기본 768차원/12층이며 이번 실험에서는 무작위 초기화한다.
CCDP 선택은 학습에서 Gumbel, 평가에서 argmax를 사용한다.
학습 후 가중치가 존재하는 것만으로 CCDP의 중요 토큰 선택 능력이 검증된 것은 아니다.

평가에서는 정상 과거로 만든 마스크를 과거 0 조건에 그대로 전달한다.
선택 인덱스, 유효 마스크, 원본 복원 정답이 같은지 assertion으로 확인한다.
`delta_lrec = zero_lrec - normal_lrec`를 전체와 주행별로 기록한다.
제거된 토큰이 없는 프레임은 오차 0으로 평가하지 않고 별도로 제외·집계한다.
학습 유지 비율과 argmax 평가 유지 비율을 모두 보고 선택 붕괴를 확인해야 한다.

## 산출물

- `cache_source.json`: encoder hash, 입력 계약, 데이터 경로, 캐시 생성 코드 hash
- `manifest.json`, `features/`: 주행 분할, 프레임 시각, 검증 가능한 특징 캐시
- `model-last.pt`: connector 전체 가중치, optimizer, RNG, 설정 및 데이터 hash
- `metrics.jsonl`: epoch별 train loss, 유지 비율, 각 모듈 gradient 및 validation 결과
- `evaluation-initial.json`, `evaluation-last.json`: 과거 유무의 프레임별 paired 결과
- `evaluation-topk-diagnostic.json`: 동일 예산 top-k와 무작위 마스크의 추가 진단.
  두 선택에서 공통으로 제거된 토큰만 복원 오차 비교에 사용한다. 이 결과는
  argmax 유지 비율이 학습됐다는 근거로 사용할 수 없다.

테스트는 `tests/test_connector_only_experiment.py`와 기존
`tests/test_qformer_mefa_connector.py`에 있다. 합성 입력 테스트 결과는 실제 성능 수치로 사용하지 않는다.

## 동적 선택 노이즈 감소 실험

`dynamic_anneal.json`은 같은 초기값과 같은 특징 캐시에서 10 epoch를 학습한다.
학습 update 0~256에서 Gumbel noise scale을 1에서 0으로 선형 감소시키고,
남은 64 update는 noise=0으로 학습한다. temperature와 top-k 개수는 바꾸지 않는다.
노이즈 0에서 학습 forward 마스크는 argmax와 같으며 straight-through gradient는 유지된다.

이 실행 폴더의 캐시는 원래 pilot에서 그대로 복사했으므로 `cache_source.json`은
원본 추출 설정을 보존한다. 캐시 생성 단계를 다시 실행하지 않고 다음을 사용한다.

```powershell
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/run.py --stage train --config VLDrive/tests/connector_only/dynamic_anneal.json
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/diagnose.py --output VLDrive/outputs/logs/connector_mefa_dynamic_anneal
```

진단 기본 정책은 argmax다. 각 프레임에서 실제 선택된 개수만큼 무작위 토큰을 골라
20개 random seed로 비교한다. 개수는 프레임마다 다를 수 있으며 고정 예산으로 바꾸지 않는다.
공통으로 제거된 토큰만 평가하고, frame/episode별 결과와 선택 개수 범위를 기록한다.

## MEFA 과거 정보만 평가

가중치와 CCDP 마스크를 고정한 `correct`/`zero` history 비교만 실행한다.
추가 학습, 토큰 순위/무작위 선택 비교, LLM 평가는 수행하지 않는다.

```powershell
# 기존 validation cache 재확인 (원본 F: 드라이브 없이 가능)
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/validate_mefa.py --stage evaluate --cached-only --output VLDrive/outputs/logs/mefa_history_cached_recheck
# 기존 train/validation 주행을 모두 제외한 새 20개 주행으로 확대
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/validate_mefa.py
# 현재 PC의 encoder 경로와 데이터 경로를 명시
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/validate_mefa.py --visual-checkpoint 'C:\Users\장원정\Desktop\checkpoint-best-object.pth.tar' --dataset-root 'F:\dataset_label'
# 완료된 확대 검증 결과를 표/그림/보고서로 출력
& 'C:\conda-envs\botsort\python.exe' VLDrive/tests/connector_only/summarize_mefa.py --output VLDrive/outputs/logs/mefa_history_validation_20_drives
```

확대 검증은 F: 원본 데이터와 encoder 가중치가 연결되어 있어야 한다.
결과는 `mefa_result.json`, 평가 전 고정한 조건은 `evaluation_plan.json`에 저장된다.
신뢰구간은 주행 단위 paired bootstrap으로 계산하며, 카테고리당 한 주행뿐인
기존 캐시 재확인에서는 신뢰구간을 제시하지 않는다.
