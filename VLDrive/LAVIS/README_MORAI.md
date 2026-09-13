# MORAI 학습

논문 수식 기준의 동적 CCDP–MEFA–TinyLlama/DDIA 학습은
[VLDrive 통합 학습 안내](README_VLDRIVE_TRAINING.md)를 참고한다. 아래 기존 route/speed 모델은 별도 MORAI 변형이다.

MORAI 학습 구현의 기준 경로는 `VLDrive/LAVIS`다. 이전 `experiments/morai_stage6`
학습 모듈은 호환 import만 제공한다. ROS bridge·수집·후처리는 기존 수집 폴더에서 실행한다.
기존 데이터·가중치·실험 출력·private Python 환경은 이동하거나 삭제하지 않았다.

## 구현 위치

| 역할 | 실제 구현 |
|---|---|
| 전체 목표 모델 | `lavis/models/vldrive/morai_model.py`: `MoraiDrivingModel`, `MoraiDrivingConfig` |
| RGB/LiDAR 인코더와 frozen/fusion/all_features 정책 | `lavis/models/vldrive/visual_encoder.py` |
| CCDP 및 고정 토큰 수 선택 | `lavis/models/vldrive/ccdp.py`, `token_selection.py` |
| causal MEFA와 실제 Q-Former | `lavis/models/vldrive/qformer_mefa.py` |
| 실제 TinyLlama와 DDIA | 기존 `tinyllama_backbone.py`, `ddia_attention.py`, `multimodal_sequence.py` |
| route 조건과 metric XY/speed decoder | `route_encoder.py`, `trajectory_decoder.py` |
| visual encoder 단독 학습 모델 | `lavis/models/vldrive/visual_trajectory.py` |
| 센서 전처리·시퀀스·teacher·파일 무결성 | `lavis/datasets/datasets/morai/` |
| 학습·평가·checkpoint·preflight | `lavis/runners/morai_trainer.py`, `morai_visual_trainer.py`, `morai_metrics.py` |
| 설정 | `lavis/projects/vldrive/morai_*.json` |

전체 모델: 4 Camera + LiDAR → LMDrive encoder → route-conditioned CCDP → causal
MEFA/Q-Former → pretrained TinyLlama/DDIA → 미래 XY와 speed.
CCDP/MEFA는 논문을 바탕으로 한 독립 구현이며 저자 코드와 동일하다는 의미는 아니다.
`vldrive_model.py`는 기존 언어 모델 인터페이스다. MORAI 전체 학습 진입점은
`morai_model.py`이며, raw 센서에서 시작하는 별도의 완성된 모델이다.

- 모델 입력: raw 센서(또는 명시적으로 frozen cached feature), route, receive elapsed time,
  실제 기록된 instruction. `/Ego_topic`·미래 trajectory·GPS metadata는 모델 입력에서 거부한다.
- LiDAR anchor와 cross-sensor 인과성은 receive monotonic time으로 검사한다. teacher 미래
  보간은 Ego source clock 내부에서만 수행한다. 서로 다른 sensor header clock을 빼지 않는다.
- route는 FLU 41×2, 시간 이력은 기본 4 frame, teacher horizon은 0.5–2.5초다.
- 단독 visual 모델은 encoder + 정규화한 causal GRU + XY/speed다. 전체 모델과 별개로
  시각 인코더 학습 가능성을 확인하는 보조 학습 경로이며 CCDP/MEFA/LLM을 포함하지 않는다.

## 실행 환경

```bash
cd VLDrive/LAVIS
conda env create -f environment_morai.yml
conda activate vldrive_morai
```

현재 로컬 환경은 새로 만들 필요 없이
`../experiments/morai_stage6/.runtime/stage6h_env/bin/python`을 사용할 수 있다.
새 clone에는 이 환경·dataset·pretrained weight가 포함되지 않는다.

MORAI 진입점은 `LAVIS_IMPORT_PROFILE=morai`를 설정하고 실제 LAVIS 모듈을 import한다.
기존의 임시 `sys.modules` 패키지 생성은 사용하지 않는다. 기본 profile은 기존 LAVIS의
전체 registry import를 유지한다. MORAI는 검증한 optimizer·strict checkpoint 방식을
유지하는 **전용 runner**를 사용한다. CARLA용 `train.py --cfg-path ...`와 혼용하지 않는다.
직접 Python에서 사용할 때는 import 전에 같은 profile을 설정하고
`lavis.common.morai_runtime.bootstrap(VLDrive 경로)`를 호출해 팀 custom timm을 선택한다.
일반 PyPI timm으로 대체하지 않는다.

## 전체 목표 모델 학습

다음 명령은 현재 보관된 dataset과 가중치를 원래 위치에서 읽는다.

```bash
python train_morai.py \
  --config lavis/projects/vldrive/morai_fusion.json \
  --dataset-root ../experiments/morai_stage6/diagnostics/stage6f_calibration_20260906/dataset_axis_v2 \
  --asset-root ../experiments/morai_stage6/assets/pretrained \
  --output outputs/morai_fusion_run01
```

항상 새로운 output 디렉터리를 사용한다. 기본값은 240 updates, train 8 / val 16 windows다.
검사만 하려면 `--mode preflight`; 작은 연결 확인은
`--updates 2 --train-windows 2 --val-windows 2`를 사용한다.
`morai_frozen.json`은 visual encoder 고정, `morai_all_features.json`은 RGB/LiDAR
feature backbone도 학습한다. 세 설정 모두 실제 pretrained TinyLlama를 사용한다.

`--dataset-root`/`--asset-root`를 생략하면 JSON의 경로를 LAVIS 기준으로 해석한다.
기본 dataset 위치는 `LAVIS/data/morai`다. `split_file`은 dataset 기준이다.
raw 학습도 teacher·route·feature provenance와 run별 독립 split을 갖춘 완성된
Stage6E 데이터셋이 필요하다. 팀 `MoraiDrive`의 다른 episode 포맷을 바로 읽는 loader는 아니다.

평가/재개는 같은 명령에 `--checkpoint outputs/.../checkpoint.pt`를 지정한다.
평가는 `--mode evaluate`, 재개는 총 목표 update 수를 `--updates`로 지정한다.
정확한 재개에는 같은 설정·선택·데이터·코드·backend가 필요하다.

```bash
python tools/morai/verify_checkpoint.py \
  --experiment outputs/morai_fusion_run01 --output outputs/morai_fusion_restore01
```

이 검사는 예측 동일성과 같은 저장 상태에서의 다음 optimizer 업데이트를 확인한다.

## Visual encoder 단독 학습

```bash
python tools/morai/select_visual_windows.py \
  --config lavis/projects/vldrive/morai_visual.json \
  --dataset-root ../experiments/morai_stage6/diagnostics/stage6f_calibration_20260906/dataset_axis_v2 \
  --output outputs/visual_selection01

python train_morai_visual.py \
  --config lavis/projects/vldrive/morai_visual.json \
  --dataset-root ../experiments/morai_stage6/diagnostics/stage6f_calibration_20260906/dataset_axis_v2 \
  --asset-root ../experiments/morai_stage6/assets/pretrained \
  --selection outputs/visual_selection01/selection.json \
  --policy all_features --output outputs/visual_all_features01
```

`--policy frozen|fusion|all_features`를 같은 selection으로 비교한다. 기본 600 updates이며
`--updates`로 조정할 수 있다. 60km/h 이하·route 시작점 거리 조건은 작은 학습 가능성
실험의 표본 선택 기준이다. 대회 구간 분류나 실제 주행 안전 판정으로 사용하지 않는다.
복원 검사는 동일 인자에 `--mode verify --experiment outputs/visual_all_features01`과
새 output 경로를 지정한다. 학습에 `--updates`를 덮어썼다면 검증에도 같은 값을 사용한다.

## 데이터와 과거 실행의 보존

새 학습 결과는 `LAVIS/outputs`, cache는 `LAVIS/.runtime`에 저장하고 Git에서 제외한다.
원본 데이터·가중치·tests·옛 실험 보고서는 유지한다. 실제 `test_*`는 회귀 테스트이며
제품 코드로 이름만 바꾸지 않았다. 기존 visual probe v2의 검증된 정규화/readout 구현은
`visual_trajectory.py`로 통합했다.

기존 Stage6H/I checkpoint는 코드 위치와 내용 hash가 달라 새 코드에서 strict resume이
거부된다. 무결성 검사를 완화하지 않았다. 과거 보고서 비교는 가능하지만 정확한 과거
재개에는 해당 실행의 원래 source snapshot이 필요하다. 새 위치에서는 새 실행을 시작한다.
이번 정리는 학습 연결의 정리이며 최종 아키텍처·일반화 성능의 확정은 아니다.
