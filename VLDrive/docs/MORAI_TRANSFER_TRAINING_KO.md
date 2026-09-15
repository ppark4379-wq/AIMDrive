# MORAI 맞춤 visual encoder 학습

기존 `train.py vision-encoder`와 논문 기반 학습 파일은 유지합니다.
추가 경로는 `train.py vision-encoder-morai`입니다. CCDP·MEFA·DDIA·LLM을 연결하지 않습니다.

## pull 이후 두 학습 경로

| 구분 | 팀의 `vision-encoder` | MORAI 맞춤 `vision-encoder-morai` |
|---|---|---|
| 역할 | 논문 recipe를 목표로 유지하는 비교 경로 | 현재 수집 데이터에 맞춘 visual encoder 후보 |
| 데이터 | 가공 dataset과 split JSON | 검증·봉인된 Sensor Record publication 복수 개 |
| 초기화 | ImageNet RGB, 나머지 모듈 신규 학습 | 원본 LMDrive visual checkpoint 전체 strict load |
| 입력 속도 | 기존 가공 dataset 계약 | 직전·현재 GPS fix로 계산한 인과적 속도 |
| 학습 과제 | waypoint, 검증된 신호등 | 객체 격자·객체 속도·waypoint·신호등 |
| 설정 | 단일 frozen training config | PC별 경로 config와 실행 시 학습 인자 |

공유되는 것은 `models/vision_encoder.py`의 ResNet·PointPillars·fusion·query 구조뿐입니다.
한 경로의 checkpoint나 manifest를 다른 경로가 자동으로 선택하지 않으며, 결과 폴더와
provenance도 별도로 기록합니다. 현재 MORAI 데이터 실험은 두 번째 경로를 사용하고,
첫 번째 경로는 구조·학습 recipe 비교를 위해 그대로 보존합니다. 이 구분은 논문 전체
재현 완료를 뜻하지 않습니다.

## 모델과 학습 계약

- `models/vision_encoder.py`의 기존 Memfuser R50/PointPillars/Transformer를 사용합니다. 원본 LMDrive 전체 checkpoint의 SHA256과 888개 state key를 strict 검사합니다.
- RGB·LiDAR backbone, encoder, decoder 및 detection/waypoint/traffic-light head를 모두 학습합니다. Stop-sign head는 GT가 없어 고정합니다.
- batch 1에서 BatchNorm의 running mean/variance만 유지하고 affine 가중치는 학습합니다.
- 입력은 Camera 4대(+front focus crop), 단일 LiDAR sweep, 직전 GPS 위치와 현재 GPS 위치 사이의 속도(m/s)입니다. GPS 속도는 simulation timestamp 간격 0.25초 이하일 때만 사용하며, 첫 프레임·시간 단절·비정상 속도는 제외합니다. Ego 속도를 대체 입력으로 사용하지 않습니다.
- global path의 25m 앞 목표점은 waypoint GRU 조건입니다. 학습의 로컬 목표점 변환과 미래 waypoint GT는 수집 시점의 Ego pose를 사용합니다. 향후 추론에서는 GPS/IMU로 별도 동일 좌표 변환을 검증해야 합니다.
- 정답: 50×50 객체 격자, 0.5/1.0/1.5/2.0/2.5초 미래 실제 Ego 경로, 객체 속도, 검증된 red/green 신호. 미검증 객체·신호 loss는 mask합니다.
- loss: `0.5 detection(LAV) + 0.5 waypoint L1 + 0.05 object speed L1 + 0.1 traffic-light CE`.
- 기본은 전체 유효 프레임, batch 1, AdamW LR 1e-5 / weight decay 0.01, gradient clip 1, augmentation 없음, 1 epoch입니다.

이는 MORAI의 기존 v3_lav 학습 방식에 기반한 전이학습입니다. 논문 학습 스케줄 재현이라는 뜻은 아닙니다. 이전 v3_lav 실험은 MORAI에 이미 적응한 초기 checkpoint와 zero speed를 사용했지만, 이 경로는 요청에 따라 **원본 LMDrive checkpoint + GPS 속도**로 시작합니다. 과거 결과와의 차이에 데이터량·초기화·속도 입력 변경이 함께 포함됩니다. `--velocity-source zero`는 과거 입력과 비교할 때만 명시적으로 사용합니다.

## 폴더 구분

| 위치 | 역할 |
|---|---|
| `models/` | 기존 visual encoder 모델, 변경 없음 |
| `datasets/morai_sensor_record_dataset.py` | sealed manifest 결합, 원본 hash, 분할 검사 |
| `datasets/morai_transfer_adapter.py` | MORAI 센서 전처리 |
| `datasets/morai_transfer_targets.py` | 객체 정답 렌더링 |
| `datasets/morai_gps_speed.py` | GPS 기반 인과적 속도 산출 |
| `utils/morai_transfer_*` | 별도 MORAI 학습·loss·평가 |
| `configs/morai_transfer.local.json` | Git 제외 PC별 데이터·초기 가중치 경로 |
| `tools/train_visual_morai.sh` | 단일 실행 명령 |

## 실행

`configs/morai_transfer.example.json`을 `configs/morai_transfer.local.json`으로 복사하고 경로를 지정합니다. `manifest_indices`에 여러 campaign의 최종 `manifest_index.json`을 넣습니다. 업로드 중인 raw 폴더를 넣지 않습니다.

검증 중인 추가 campaign을 반드시 포함하려면 `manifest_campaigns`에 campaign 폴더를 지정할 수 있습니다. 해당 폴더의 최신 training manifest를 선택하고 seal 검사를 수행합니다. 아직 준비되지 않았으면 이전 데이터만으로 조용히 시작하지 않고 중단합니다. 재현 실행은 `config.json`의 확정된 publication 경로·hash를 사용합니다. 명령행의 `--manifest-index`를 명시하면 PC 설정의 전체 목록을 대신합니다.

```bash
conda activate lmdrive
cd ~/AIMDrive/VLDrive
bash tools/train_visual_morai.sh --epochs 1
```

데이터 검증과 실제 forward/backward만 실행하려면:

```bash
bash tools/train_visual_morai.sh --preflight
```

GPU 없는 검사 환경에서는 `--device cpu --preflight`를 사용합니다. 이 옵션은 optimizer update가 0입니다. 실제 GPU 학습 성능 검증을 대신하지 않습니다.

객체 head만 균형 미세조정하려면 PC별 config의 `object_initial_checkpoint`와
`object_initial_checkpoint_sha256`에 시작할 MORAI checkpoint를 지정하고 실행합니다.
데이터는 같은 config의 `manifest_indices`를 사용하며, 추가 publication은 목록에
명시적으로 넣습니다. 스크립트에는 특정 PC의 드라이브나 campaign 경로가 없습니다.

```bash
bash tools/train_visual_morai_object_head.sh
```

동일 validation baseline과 매 epoch의 객체 cell AP/F1을 비교하고
`checkpoint-best-object.pth.tar`와 `checkpoint-last.pth.tar`를 별도로 저장합니다.

## 여러 컴퓨터의 데이터

각 publication의 기존 train/validation lap을 그대로 유지합니다. 전체 데이터를 프레임 단위 무작위로 다시 나누지 않습니다. 동일 episode의 같은 manifest는 한 번만 사용하고, 상이한 revision 또는 다른 split이 겹치면 중단합니다. 다른 episode가 동일한 Camera4+LiDAR 원본을 참조해도 중단합니다.

`path_mappings`는 `[["기존 경로 접두사", "새 경로 접두사"]]`입니다. 여러 PC의 절대경로가 바뀐 경우 원본 manifest를 수정하지 않고 해석 경로만 교체합니다. 원본 파일·검증 seal의 hash는 계속 검사합니다. 검증 seal·원본·campaign 기록 없이 일부 파일만 복사한 수집본은 먼저 후처리를 완료해야 합니다.

## 산출물과 판단

`config.json`(데이터·코드·checkpoint hash와 학습 계약), `baseline.json`, `train.jsonl`, `epoch_*.json`, `checkpoint-last.pth.tar`, `report.json`을 저장합니다. 실제 gradient 전달과 `[1,106,256]` 출력도 검사합니다. 셀 AP/F1은 객체 AP가 아닙니다. 같은 경로의 held-out lap 평가는 독립 지도 일반화를 뜻하지 않습니다. 카메라의 물리적 동시 촬영 공식 보증도 주장하지 않습니다.
