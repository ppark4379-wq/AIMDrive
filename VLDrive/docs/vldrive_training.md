# VLDrive 논문 기준 MORAI 학습

`train_vldrive.py`가 frozen MORAI-trained LMDrive 구조 Vision Encoder token → 동적 CCDP/MEFA → 실제 TinyLlama/DDIA →
frame별 waypoint L1 + cycle loss를 연결한다. 기존 `train_morai.py`의
route/speed 조건 모델은 별도 MORAI 실험 경로로 보존한다.

| 경로 | 용도 |
|---|---|
| `train_vldrive.py`, `morai_paper.json` | 논문 수식에 맞춘 connector–LM–waypoint 학습 |
| `train_morai.py`, `morai_dynamic.json` | 동적 CCDP와 논문 reconstruction을 적용한 MORAI route/speed 변형 |
| 기존 `morai_frozen/fusion/all_features.json` | 고정 K 및 기존 reconstruction을 명시적으로 유지한 비교 실험 |

## 구현과 선택 사항

- `qformer_mefa.py`: 학습 기본값은 per-token hard Gumbel, 추론은 동적 argmax다.
  Reconstruction은 학습에서 기본 활성화되고 추론에서는 실행하지 않는다.
- 식 (8): 원래 위치에 enhanced token을 넣고 명시적으로 `filled * M + e * (1-M)`를 계산한다.
  Discrete placement만 detach하고 선택 mask의 straight-through 경로는 유지한다.
- 식 (13): dropped token/channel squared error 합을 frame별로 계산하고 batch/frame 평균을 낸다.
  기존 dropped-element mean은 `reconstruction_reduction="mean"` 비교 옵션이다.
- 논문 경로는 `R=0.3`, `Z=10`, reconstruction 4 layers, λ=10/1을 사용한다.
  `sequence_length=11` 이상으로 실제 10개 과거 frame을 검사한다. Frame padding은 받지 않는다.
- `VLDriveModel`은 실제 valid 위치를 이용한다. `predict_per_frame=True`이면 `[B,T,10]`을 반환한다.
  기본 언어 API는 기존 `[B,10]`을 유지한다. Padding/범위 밖 readout은 거부한다.
- `VLDriveTrainingModel.forward(samples)`는 task가 사용할 scalar loss dictionary를 반환한다.
  `predict(model_inputs)`에는 visual input과 instruction만 허용한다. Ego/GPS/IMU/route는 입력하지 않는다.
- TinyLlama decoder와 text embeddings는 모두 학습한다. 사용하지 않는 vocabulary/end head와
  visual encoder는 고정한다. BF16 계산 + CPU의 FP32 AdamW master/state로 작은 갱신을 누적한다.
  CPU offload는 16GB GPU용 메모리 선택이며 모델 계층을 생략하거나 LLM을 부분 학습하지 않는다.
- AdamW lr=1e-5, weight decay=.06, 15 epochs cosine schedule. `--updates`는 조기 중단 지점만
  정하고 원래 schedule은 유지하므로 정확한 재시작이 가능하다.

Argmax 추론, 빈 선택에서 최고 확률 token 하나를 유지하는 처리, 원위치 replacement의
gradient surrogate, LayerNorm, Q-Former의 정확한 수치, TinyLlama checkpoint,
readout 및 MORAI 전처리는 독립 구현 선택이다. 저자 코드와 동일함을 주장하지 않는다.
특히 Q-Former의 정확한 width/layer 수는 논문·부록에 명시되지 않았다.

## 데이터와 지시문

현재 competition 경로는 `MoraiDrive/dataset_local/competition_train`을 읽기 전용으로 사용한다.
Vision Encoder 추론 결과 `[106,256]`은 `LAVIS/.runtime`에 별도 저장한다. 각 cache frame에는
Vision Encoder 학습 때와 같은 **현재 ego speed, 현재 LiDAR 한 sweep, FLU→FRU 변환**을 강제한다.

Waypoint GT는 `global path`의 앞점이 아니라 동기화된 실제 ego 위치의 0.5/1.0/1.5/2.0/2.5초
미래 좌표다. 따라서 빨간불에 멈춘 frame의 GT도 정지 차량 주변에 남는다. NPC/object가 없는
시나리오도 정상이며 object label을 loader의 필수값이나 모델 입력으로 요구하지 않는다.

Global path의 `target_point`(`x_command`, `y_command`)는 Vision Encoder 사전학습의
waypoint prediction head에서 사용한다. `memfuser`의 `return_feature`는 그 head 이전에 반환하므로
connector용 `[106,256]` token 추출에는 `target_point`가 사용되지 않는다. Cache metadata의
`target_point_source`는 공통 dataset adapter가 읽는 값의 출처이며 token에 주입된다는 의미가 아니다.
Connector 이후에는 같은 경로의 `gt_command`를 LLM navigation instruction으로 바꿔 사용한다.
반대로 global-path 앞점은 waypoint GT로 절대 쓰지 않는다. `gt_command`와
`route.json/nav_road_options`의 일치도 먼저 검증한다. 생성한 문장은 사람이 작성한 자유 문장
annotation은 아니며, 현재 대회 경로의
expert 정책을 모방하기 위한 결정적 command instruction이다. Instruction JSONL 형식은 다음과 같다.

```json
{"schema_version":"morai-global-path-instructions-v1","route_id":"full_loop_...","start_frame":80,"stop_frame":124,"command":3,"text":"Please proceed straight at the next intersection.","source":"recorded_global_path","route_sha256":"..."}
```

명령이 바뀌는 경계를 가로지르는 11-frame window는 제외한다. train/validation 사이에는 미래 GT가
새지 않도록 25 frame을 버린다. 선언값과 실제 구간 간격을 모두 검사한다. 분리·제외 수와
asset·cache·split·instruction·route·구현 해시를 기록한다. 원본 센서 전체를 해시했다는 의미는 아니다.
Future ego GT는 학습 loss에만 들어가며 실제 예측 API는 visual token과 instruction만 받는다.

현재 waypoint 좌표 규약은 LMDrive head와 같은 `[left, -forward]`다. 실제 차량 좌표
`[forward, left]`로 사용할 때는 `[-y, x]`로 바꿔야 한다. 평가기의 종·횡방향 MAE도 이 규약을 따른다.

## 실행

현재 작업 공간에는 asset, instruction 및 `full_loop_tokens_best` cache 준비가 완료되어 있다.
이 경우 아래 준비 명령은 건너뛰고 `train_vldrive.py`부터 실행한다.
완성된 cache 경로에 생성 명령을 다시 실행하면 덮어쓰기를 막기 위해 오류가 발생한다.
새 encoder checkpoint로 token을 다시 만들 때는 새 cache 경로를 지정한다.

[MORAI 환경 안내](README_MORAI.md)의 Python 환경을 사용한다. 순서는 Vision Encoder 완료 → asset
검증 → instruction 생성 → token cache → preflight → 본 학습이다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive/LAVIS
PY=/opt/miniforge3/envs/vldrive-ve/bin/python
DATA=/mnt/c/Users/autonav009/Desktop/AIMDrive/MoraiDrive/dataset_local/competition_train
SPLIT=../vision_encoder/configs/morai_full_loop_80_20.json
VISION=../vision_encoder/output_morai/full_loop_expert/model-best.pth.tar

$PY tools/morai/prepare_assets.py --vision-checkpoint "$VISION" \
  --vision-name morai-full-loop-v1 \
  --language-source ../weights/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --download-qformer

$PY tools/morai/build_global_path_instructions.py --dataset-root "$DATA" \
  --output .runtime/morai_vldrive/full_loop_instructions.jsonl

$PY tools/morai/cache_expert_vision_tokens.py --dataset-root "$DATA" \
  --split-file "$SPLIT" --checkpoint "$VISION" \
  --output .runtime/morai_vldrive/full_loop_tokens_best --batch-size 1 --workers 2

$PY train_vldrive.py --config lavis/projects/vldrive/morai_paper.json \
  --mode preflight --output outputs/morai_paper_preflight

$PY train_vldrive.py --config lavis/projects/vldrive/morai_paper.json \
  --output outputs/morai_paper_run01
```

`morai_paper.json`은 논문/LMDrive 계열에 가까운 BERT-base 크기 Q-Former(768/12/12) 설정이다.
논문이 정확한 Q-Former 수치를 공개하지 않았으므로 이것은 명시적인 상속 선택이다. 16GB에서 먼저
배관을 확인할 때만 `morai_compact_16gb.json`(256/2/8, random init)을 사용하고 최종 논문 비교와
혼동하지 않는다. pretrained LMDrive Vision Encoder는 사용하지 않는다. 현재 MORAI 데이터로 학습한
encoder를 사용하며 RGB backbone은 ImageNet, LiDAR backbone과 fusion/queries/heads는 random으로 초기화했다.

`--mode preflight`는 optimizer 갱신 없이 gradient, 과거/미래 인과성, history 의존성,
teacher 입력 거부 및 실제 history 길이를 검사한다. 학습에도 preflight가 선행된다.
`--checkpoint <기존 run>/checkpoint.pt`와 새 output으로 재개한다.
`--mode evaluate --checkpoint ...`로 독립 validation run을 평가한다.
평가에는 `model-best.pt`, `model-last.pt`, 전체 `checkpoint.pt`를 모두 사용할 수 있다.
Preflight와 평가 전후 난수 상태를 보존해 이어 학습의 Gumbel/dropout 순서를 유지한다.
학습에는 모든 frame label을 쓰고 window 평가에는 마지막 frame만 쓴다.

각 epoch에 `model-last.pt`와 validation FDE가 가장 낮은 `model-best.pt`를 저장한다. 마지막
`checkpoint.pt`만 optimizer master/state/RNG를 포함하는 재개용 파일이라 매우 클 수 있다.
중간의 거대한 optimizer checkpoint는 기본적으로 만들지 않는다.
`model-best.pt`는 해당 실행의 시작 가중치까지 포함한 최저 FDE다. 재개 시 새 output의 best는
재개 시점 이후의 best이며, 이전 output의 best 파일과 보고서는 보존된다.

체크포인트는 모델/optimizer master/state/RNG와 source·asset·dataset hash를 검증한다.
수정 전 checkpoint는 source/config 계약이 달라 strict resume 대상이 아니다.
기존 파일을 덮어쓰거나 검증을 끄지 말고 새 run으로 시작한다.
전체 TinyLlama optimizer checkpoint는 약 14GB일 수 있어 저장 주기를 고려한다.

기존 `tinyllama_ddia.yaml`은 visual token을 직접 넣는 언어 API 예시다.
그 파일로 CARLA `train.py` 학습이 연결됐다는 뜻은 아니다. MORAI 통합 진입점은 위 명령이다.
CARLA leaderboard, 실제 language-grounding 성능, 장시간 token retention 수렴 및 대회 주행 성능은
별도 데이터/closed-loop 검증이 필요하다. 짧은 optimizer 검사로 구조나 성능을 확정하지 않는다.

식 (13)의 sum은 feature scale에 따라 reconstruction loss가 waypoint/pruning보다 클 수 있다.
`training.jsonl`의 세 loss와 gradient norm, 평가의 frame별 retained count를 확인한다.
평균 loss나 고정 token budget으로 바꾸면 비교 실험 설정으로 명시해야 한다.

근거: [VLDrive 논문](https://arxiv.org/abs/2511.06256),
[공식 부록](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_VLDrive_Vision-Augmented_Lightweight_ICCV_2025_supplemental.pdf).
