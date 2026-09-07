# VLDrive 논문 기준 MORAI 학습

`train_vldrive.py`가 frozen LMDrive → 동적 CCDP/MEFA → 실제 TinyLlama/DDIA →
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
gradient surrogate, LayerNorm, Q-Former의 256 width/2 layers/8 heads, TinyLlama checkpoint,
readout 및 MORAI 전처리는 독립 구현 선택이다. 저자 코드와 동일함을 주장하지 않는다.
특히 Q-Former의 정확한 width/layer 수는 논문·부록에 명시되지 않았다.

## 데이터와 지시문

기존 Stage6E artifact/teacher/split 검증을 재사용하며 원본 데이터는 수정하지 않는다.
Cached mode는 저장된 frozen encoder feature를 사용한다. Raw mode는 같은 센서 adapter와
고정 encoder를 실행한다. 설정의 `model.input_mode`로 선택한다.

논문 경로에는 실제로 주어진 navigation instruction annotation이 필요하다.
현재 annotation이 없는 MORAI 데이터에 임의 문장을 생성하거나 teacher route를 문장으로
변환하지 않는다. 파일이 없으면 모델 로딩 전에 실패한다.

Instruction JSONL의 각 행은 다음 네 필드만 가진다.

```json
{"run_id":"run_0003","text":"실제로 제공한 주행 지시문","source":"실제 지시 기록의 출처","available_receive_monotonic_ns":123456789000}
```

위 숫자/문장은 **형식 예시**이며 실제 annotation으로 사용하지 않는다.
시간은 recorder의 receive monotonic domain이다. 첫 frame 이전에 사용 가능한 지시만 적용하고,
window 안에 새 instruction event가 있으면 그 window를 제외한다. 제외 수를 provenance에 남긴다.
이미 사용이 끝난 지시의 유효 종료 시점은 새 event로 기록해야 하며, 이 형식은 별도 종료 필드가 없다.
GT는 학습 loss에만 들어가며 실제 예측 API와 분리된다.

## 실행

[MORAI 환경 안내](README_MORAI.md)의 Python 환경/asset 준비를 사용한다.
아래 경로의 dataset, pretrained assets, 실제 instruction 파일이 준비돼 있어야 한다.

```bash
cd /home/dydwns/AIMDrive/VLDrive/LAVIS
../experiments/morai_stage6/.runtime/stage6h_env/bin/python train_vldrive.py \
  --config lavis/projects/vldrive/morai_paper.json \
  --dataset-root ../experiments/morai_stage6/diagnostics/stage6f_calibration_20260906/dataset_axis_v2 \
  --asset-root ../experiments/morai_stage6/assets/pretrained \
  --instructions /path/to/actual_instruction_events.jsonl \
  --output outputs/vldrive_small_run --updates 20
```

`--mode preflight`는 optimizer 갱신 없이 gradient, 과거/미래 인과성, history 의존성,
teacher 입력 거부 및 실제 history 길이를 검사한다. 학습에도 preflight가 선행된다.
`--checkpoint <기존 run>/checkpoint.pt`와 새 output으로 재개한다.
`--mode evaluate --checkpoint ...`로 독립 validation run을 평가한다.
학습에는 모든 frame label을 쓰고 window 평가에는 마지막 frame만 쓴다.

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
