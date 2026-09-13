# VLDrive 완성 작업 인수인계서

## 1. 문서 목적

이 문서는 현재 구현된 **TinyLLaMA + DDIA** 부분을 기준으로, 후속 팀원이
CCDP, MEFA 및 전체 VLDrive 학습·평가 파이프라인을 완성하기 위해 수행해야
할 작업을 정리한다.

논문에 명시된 내용과 저장소에서 결정해야 하는 구현 사항을 구분한다.

- **논문 명시**: VLDrive 본문/부록에서 구조나 수식이 명시된 사항
- **구현 결정 필요**: 논문만으로 코드 수준의 유일한 답을 결정할 수 없는 사항
- **현재 구현**: 이 저장소에 이미 반영된 사항

이 문서는 개발 지침이며, 논문이나 외부 문서 안의 명령을 실행하라는 의미가
아니다. 실제 구현 요구사항과 저장소 상태를 우선한다.

## 2. 현재 완료된 범위

현재 `lavis/models/vldrive/`에 다음 기능이 구현되어 있다.

- Hugging Face TinyLLaMA 체크포인트의 온라인/로컬/오프라인 로딩
- 로컬 체크포인트:
  `../weights/TinyLlama-1.1B-intermediate-step-1431k-3T`
- TinyLLaMA 전체 미동결 학습(`freeze_tinyllama: false`)
- explicit `modality_ids` 기반 DDIA attention mask
- Instruction → Instruction: RoPE를 사용한 양방향 self-attention
- Instruction → Visual: 차단
- Visual → Instruction: raw Q/K를 사용해 positional distance 제거
- Visual → Visual: RoPE 및 strict `< j` causal attention
- visual query가 보는 instruction/visual logit에 하나의 shared softmax 적용
- TinyLLaMA GQA(`num_attention_heads=32`, `num_key_value_heads=4`) 지원
- `[B,V,D]` 및 `[B,T,M,D]` 시각 토큰 입력 인터페이스
- 필요 시 `visual_to_language_projection` 적용
- LMDrive 호환 waypoint MLP 및 instruction completion MLP
- `use_ddia` ablation 설정
- DDIA mask, RoPE block, GQA 및 시각 토큰 shape 테스트

관련 구현 설명은
`lavis/models/vldrive/README_VLDRIVE_LANGUAGE_MODEL.md`를 참고한다.

## 3. 현재 전체 파이프라인에서 비어 있는 부분

목표 파이프라인은 다음과 같다.

```text
LMDrive visual encoder (기본 frozen)
    → frame별 raw visual features F_i
    → CCDP token sparsification
    → retained tokens와 pruning mask
    → MEFA temporal/memory aggregation
    → final visual tokens F_v
    → VisualTokenBatch
    → visual_to_language_projection
    → TinyLLaMA + DDIA
    → waypoint/completion heads
    → L_way (+ 기존 completion loss)

훈련 중 별도 분기:
CCDP retained/pruned representation
    → token reconstruction module
    → L_rec 및 논문에 정의된 pruning 관련 loss
```

현재는 `VisualTokenBatch` 이후만 구현되어 있다. 따라서 비전 인코더 출력부터
`VisualTokenBatch`를 생성하는 connector와 실제 CARLA task 연결이 필요하다.

## 4. CCDP 담당 작업

### 4.1 먼저 확인할 자료

- VLDrive 본문 CCDP 절의 모든 식과 기호
- VLDrive supplementary의 token retention ratio, sampling 방식, layer 수
- 공개 VLDrive 코드가 있다면 해당 구현과 체크포인트 키
- LMDrive visual encoder가 실제 반환하는 tensor shape 및 token 의미
- `drive_models/drive.py`에서 Q-Former 입력 전후 shape

확인 결과와 근거를 CCDP 모듈 README에 기록한다. 논문에 없는 값을 관행만으로
정한 경우 반드시 **구현 결정**으로 표시한다.

### 4.2 권장 파일 구조

```text
lavis/models/vldrive/connector/
├── __init__.py
├── ccdp.py
├── token_selector.py
├── token_reconstruction.py
├── mefa.py
├── vldrive_visual_connector.py
└── README_VLDRIVE_CONNECTOR.md
```

파일명은 변경 가능하지만 CCDP, reconstruction, MEFA, top-level connector의
책임을 서로 분리한다.

### 4.3 CCDP token sparsification

- [ ] visual encoder의 frame별 출력 계약을 확정한다.
- [ ] global/local feature가 논문의 정의와 같은 방식으로 계산되는지 구현한다.
- [ ] pruning score 또는 retention decision을 논문 식과 동일하게 구현한다.
- [ ] 고정 token 개수 대신 설정 가능한 retention ratio를 사용한다.
- [ ] batch마다 retained token 수가 달라질 수 있으면 padding mask를 생성한다.
- [ ] hard selection이 학습 신호를 막지 않도록 논문에서 제시한 gradient 경로를
      그대로 구현한다.
- [ ] selection 결과에 retained indices, pruning mask, scores를 포함한다.
- [ ] 시간 순서와 원래 spatial token 순서를 보존하거나, 변경 시 metadata를 남긴다.
- [ ] empty selection이 생기지 않도록 논문 규칙 또는 명시적 validation을 둔다.

권장 출력 예시는 다음과 같다.

```python
@dataclass
class CCDPOutput:
    retained_tokens: Tensor          # [B,T,M_max,C]
    retained_token_mask: Tensor      # [B,T,M_max]
    retained_indices: Tensor         # [B,T,M_max]
    pruning_scores: Tensor           # 원본 token 위치에 대응
    reconstruction_context: object   # reconstruction 분기에 필요한 정보
```

`M_max`를 32로 하드코딩하지 않는다.

### 4.4 token reconstruction 훈련 분기

- [ ] 논문의 learnable reconstruction embedding을 정확한 shape로 구현한다.
- [ ] retained token과 mask로 reconstruction input을 구성한다.
- [ ] 논문에 명시된 Transformer block 수와 hidden dimension을 확인한다.
- [ ] 원본 visual feature를 target으로 reconstruction loss를 계산한다.
- [ ] reconstruction loss가 token selection decision까지 역전파되는지 검증한다.
- [ ] reconstruction module은 inference에서 생략 가능하게 만든다.
- [ ] `L_rec`의 weight를 YAML에서 조절 가능하게 만든다.
- [ ] 논문에 별도 pruning loss가 있으면 이름과 식을 그대로 대응해 구현한다.

## 5. MEFA 담당 작업

### 5.1 MEFA 입력과 출력

입력은 CCDP의 retained tokens와 유효 mask이어야 한다. 출력은 TinyLLaMA 쪽에서
바로 받을 수 있는 최종 visual tokens이어야 한다.

```python
VisualTokenBatch(
    visual_tokens=final_visual_tokens,          # [B,T,M,D] 또는 [B,V,D]
    visual_attention_mask=final_visual_mask,    # [B,T,M] 또는 [B,V]
)
```

### 5.2 구현 체크리스트

- [ ] memory bank가 보관하는 frame 범위 `Z`를 논문/부록에서 확인한다.
- [ ] current frame과 adjacent/history frame의 포함 방향을 명확히 한다.
- [ ] 첫 프레임처럼 memory가 부족한 경우의 padding/initialization을 정의한다.
- [ ] temporal encoding의 정확한 종류와 삽입 위치를 확인한다.
- [ ] memory update 시 detach 여부를 논문 또는 실험 목적에 맞춰 명시한다.
- [ ] sequence 간 memory leakage가 없도록 batch/sample 단위 상태를 분리한다.
- [ ] training shuffle, distributed training 및 inference reset 시점을 처리한다.
- [ ] aggregation 이후 token mask와 chronological order를 보존한다.
- [ ] MEFA 출력 dimension을 `visual_feature_dim`과 일치시키거나 설정을 갱신한다.
- [ ] variable `T`, variable `M`, padded frame에 대한 테스트를 작성한다.

### 5.3 stateful memory 사용 시 필수 API

MEFA가 실행 간 상태를 유지한다면 다음과 같은 명시적 API가 필요하다.

```python
connector.reset_memory(batch_ids=None)
connector.forward_frame(frame_features, batch_ids, is_sequence_start)
connector.forward_sequence(frame_features, frame_mask)
```

전역 singleton memory나 sample ID를 구분하지 않는 queue는 사용하지 않는다.

## 6. visual encoder와 connector 통합

- [ ] 기존 LMDrive visual encoder를 별도 삭제·변경하지 않고 재사용한다.
- [ ] `freeze_visual_encoder` 설정을 추가하며 논문 기본값을 확인한다.
- [ ] LMDrive pretrained visual encoder checkpoint를 로딩한다.
- [ ] encoder 출력에서 CCDP가 요구하는 raw token을 정확히 선택한다.
- [ ] 기존 `ln_vision` 또는 대응 normalization의 사용 여부를 논문과 대조한다.
- [ ] 기존 Q-Former를 VLDrive에서도 사용하는지 논문을 근거로 확정한다.
- [ ] Q-Former를 사용하지 않는다면 제거 범위와 checkpoint incompatibility를 문서화한다.
- [ ] mixed precision에서 encoder/connector/LLM dtype 경계를 처리한다.
- [ ] connector 출력에 `.detach()`를 적용하지 않는다.

특히 Q-Former 유지 여부와 connector 입력 위치는 임의로 결정하지 말고 논문 그림,
supplementary 및 공개 코드로 확인해야 한다.

## 7. TinyLLaMA + DDIA 쪽에서 후속 보완할 사항

### 7.1 이번에 확정한 호환성 구현 선택

VLDrive 논문과 supplementary는 TinyLLaMA 및 DDIA의 알고리즘은 명시하지만,
정확한 Hugging Face 체크포인트 ID, Transformers 버전, FP16/BF16 여부는
명시하지 않는다. 공식 VLDrive 저장소도 아직 코드를 공개하지 않았다. 따라서
다음 항목을 **논문 명시가 아닌 우리 팀의 호환성 구현 선택**으로 확정했다.

- TinyLLaMA 체크포인트:
  `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Transformers: `4.31.0`
- 기본 precision: `fp16`
- 로컬/오프라인 체크포인트 경로:
  `../weights/TinyLlama-1.1B-intermediate-step-1431k-3T`

TinyLLaMA 가중치는 단일 safetensors 파일이 GitHub LFS의 객체 크기 제한을
초과하므로 `VLDrive/weights/`를 Git에서 제외한다. 저장소를 새로 받은 팀원은
다음 명령으로 로컬 체크포인트를 복원한다. 계정 비밀번호나 토큰은 저장소에
기록하지 않는다.

```bash
cd VLDrive/LAVIS
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', local_dir='../weights/TinyLlama-1.1B-intermediate-step-1431k-3T', local_dir_use_symlinks=False)"
```

Transformers 4.28의 공식 LLaMA 구현은 별도의 KV head를 만들지 않아
TinyLLaMA의 GQA 가중치와 shape가 맞지 않는다. 4.31은 GQA의
`num_key_value_heads`와 `repeat_kv`를 지원하면서 현재 DDIA가 사용하는 3-value
attention API도 유지한다. 반면 현재 5.x API는 attention 반환 형식이 달라 직접
사용할 수 없다. 이에 따라 dependency 파일을 4.31.0으로 고정하고 DDIA loader가
지원 범위 밖 버전에서 설명 가능한 오류를 내도록 수정했다.

YAML의 `precision`은 이제 실제 모델 dtype으로 전달된다. visual token은 언어
embedding dtype으로 변환되고 visual projection 및 prediction head도 같은 dtype으로
생성된다. `fp32`, `fp16`, `bf16`, `auto`를 지원하지만 기본 FP16은 논문 값이 아닌
메모리 사용량과 재현성을 위한 구현 선택이다.

DDIA 수식 자체는 변경하지 않았다. 논문 식 (9)에 따라 Instruction→Instruction은
RoPE self-attention, Instruction→Visual은 차단, Visual→Instruction은 raw Q/K,
Visual→Visual은 RoPE와 strict `< j`를 사용하며 하나의 shared softmax로 정규화한다.

### 7.2 후속 통합 시 확인할 사항

현재 언어 모델 코드는 connector 연결 전에 독립적으로 작성되었다. 전체 통합 시
다음 항목을 반드시 검증·보완한다.

- [ ] 대상 Linux/CUDA 환경에서 로컬 TinyLLaMA 실제 로딩 smoke test
- [ ] 새 환경이 `transformers==4.31.0`과 `huggingface-hub==0.16.4`를 실제 사용함을 확인
- [ ] 기존 LMDrive 주요 모듈이 Transformers 4.31.0에서 import/forward 되는지 확인
- [ ] gradient checkpointing 사용 시 `modality_ids` 전달이 유지되는지 확인
- [ ] DDP/FSDP에서 attention layer에 저장한 modality state가 안전한지 확인
- [ ] AMP fp16/bf16에서 all-masked row 및 softmax 안정성 확인
- [ ] strict `< j`로 인해 첫 visual query가 instruction만 보는 동작 확인
- [ ] `use_ddia: false`가 Hugging Face 표준 causal attention과 수치적으로 대응하는지 확인
- [ ] 논문의 최대 temporal context가 TinyLLaMA 기본 2048 position을 넘는지 계산
- [ ] 2048을 넘는 경우 논문에 근거한 RoPE scaling/길이 정책을 결정
- [ ] cached decoding은 현재 지원하지 않으므로 필요한 경우 별도 구현
- [ ] `torch.compile`, FlashAttention과 DDIA의 호환성은 별도 최적화 단계에서 검증

현재 `modality_ids`는 attention module 속성으로 전달한다. 장기적으로는 사용하는
Transformers 버전을 확정한 뒤 decoder layer 호출 경로를 명시적으로 확장하는 편이
DDP, checkpointing 및 동시 forward에 더 안전하다.

### 7.3 2026-09-04 수정 및 검증 결과

- `LAVIS/requirements.txt`: `transformers==4.31.0`으로 고정
- `environment.yml`: Transformers 4.31.0 및 호환되는
  `huggingface-hub==0.16.4`로 변경
- `tinyllama_ddia.yaml`: 기본 precision을 명시적인 `fp16`으로 변경
- `tinyllama_backbone.py`: precision 변환 함수와 DDIA 버전 guard 추가
- `vldrive_model.py`: 설정 dtype 전달 및 visual/projection/head dtype 정렬
- `test_tinyllama_loading.py`: FP16 로딩과 실제 Q/K/V gradient 검사 강화

별도로 구성한 Transformers 4.31.0 환경에서 로컬 체크포인트를 사용해 다음을
확인했다.

- TinyLLaMA+DDIA 전체 forward 통과
- hidden-state shape `[1,6,2048]`
- attention shape `[1,32,6,6]`
- FP16 dtype 적용
- Instruction→Visual attention 차단
- 첫 visual query의 Visual→Visual attention 차단(strict `< j`)
- attention row의 shared normalization 합이 1
- 실제 TinyLLaMA 첫 layer Q/K/V gradient가 모두 0보다 큼
- top-level language model의 visual projection과 waypoint head FP16 forward 통과

4.31 로딩 시 표시되는 `rotary_emb.inv_freq` 경고는 checkpoint에 저장되지 않은
결정적 RoPE frequency buffer를 config로 다시 만드는 알림이다. 학습된 파라미터
누락으로 취급하지 않는다.

## 8. waypoint prediction 연결

현재 기본값은 마지막 유효 visual token 하나에서 10차원 waypoint를 예측한다.
LMDrive는 각 유효 프레임의 마지막 visual token을 골라 프레임별 waypoint를 만든다.
전체 VLDrive에서는 다음을 확정해야 한다.

- [ ] VLDrive 논문/공개 코드의 prediction token 선택 규칙 확인
- [ ] frame별 마지막 retained token이 항상 의미 있는지 확인
- [ ] MEFA가 frame 경계를 유지한다면 frame별 prediction index 생성
- [ ] frame 경계가 사라지면 dedicated prediction token 필요 여부 확인
- [ ] `waypoint_token_indices`를 `[N,2]`의 `(batch, sequence_index)`로 전달
- [ ] 예측 shape를 LMDrive loss target과 동일한 `[N,10]`으로 맞춘다.
- [ ] inference 출력은 leaderboard가 기대하는 `[T,5,2]` 경로로 변환한다.
- [ ] instruction completion loss를 유지할지 VLDrive 논문과 대조한다.

prediction token 전략은 현재 **LMDrive 호환 구현 선택**이며 논문 일치로 단정하면
안 된다.

## 9. 데이터셋 및 task 수정

현재 `CarlaVoiceDataset`은 `rgb_*`, LiDAR, `text_input`, `valid_frames`,
`local_future_waypoints` 등을 반환한다. 후속 작업은 다음과 같다.

- [ ] 기존 dataset 반환 형식을 가능한 한 보존한다.
- [ ] frame padding mask를 명시적으로 생성한다.
- [ ] `valid_frames`로 CCDP/MEFA 및 visual attention mask를 생성한다.
- [ ] 새 VLDrive top-level model이 `samples` dict를 직접 받을 adapter를 만든다.
- [ ] `carla_drive` task가 `VLDriveOutput`으로 loss를 계산하도록 분기한다.
- [ ] `L_way`, completion loss, pruning/reconstruction loss를 이름별로 반환한다.
- [ ] 총 loss 식과 각 weight를 YAML에서 설정한다.
- [ ] evaluation metric과 기존 LMDrive task 동작을 깨지 않는다.

권장 총 loss 인터페이스:

```python
{
    "loss": total_loss,
    "waypoints_loss": waypoint_loss,
    "end_loss": end_loss,
    "pruning_loss": pruning_loss,
    "reconstruction_loss": reconstruction_loss,
}
```

논문에서 정의하지 않은 loss를 기본값으로 추가하지 않는다.

## 10. 설정 파일 확장

`lavis/projects/vldrive/tinyllama_ddia.yaml`에 최소 다음 설정을 추가한다.

```yaml
model:
  visual_encoder_name: ...
  visual_encoder_checkpoint: ...
  freeze_visual_encoder: true

  connector:
    use_ccdp: true
    retention_ratio: ...
    minimum_retained_tokens: ...
    use_reconstruction: true
    reconstruction_loss_weight: ...

    use_mefa: true
    memory_frames: ...
    temporal_encoding: ...
    output_dim: ...

  waypoint:
    prediction_token_strategy: ...

run:
  init_lr: ...
  tinyllama_lr: ...
  connector_lr: ...
  waypoint_head_lr: ...
  gradient_accumulation_steps: ...
```

`...` 값은 논문/부록/공개 코드에서 확인한 뒤 채운다. 확인되지 않은 값은 주석에
`IMPLEMENTATION CHOICE`라고 표시한다.

## 11. optimizer 및 checkpoint

- [ ] optimizer에 TinyLLaMA, CCDP, MEFA, projection, waypoint head가 포함되는지 검사
- [ ] frozen visual encoder가 optimizer에서 제외되는지 검사
- [ ] 모듈별 learning rate가 필요한지 논문에서 확인
- [ ] `runner_base.py`가 `requires_grad=True` 파라미터만 저장한다는 점을 고려
- [ ] fine-tuned TinyLLaMA와 connector 가중치가 checkpoint에 실제 포함되는지 검사
- [ ] frozen base TinyLLaMA는 로컬 base checkpoint에서 먼저 로딩한 뒤 fine-tuned
      checkpoint를 `strict=False`로 덮어쓰는 경로 검증
- [ ] resume 시 optimizer, scaler, epoch, memory state 처리 확인
- [ ] checkpoint key 목록과 trainable parameter 목록을 테스트 로그로 저장

## 12. leaderboard 및 CARLA inference 통합

- [ ] `leaderboard/team_code/lmdriver_agent.py`에 별도 VLDrive 선택 경로 추가
- [ ] 기존 LMDrive agent를 기본적으로 변경하지 않는다.
- [ ] TinyLLaMA/connector/checkpoint 경로를 config에서 읽는다.
- [ ] route 시작과 종료 시 MEFA memory를 reset한다.
- [ ] frame sampling rate가 MEFA 학습 조건과 같은지 확인한다.
- [ ] predicted waypoint의 좌표계, 부호, 단위를 기존 PID 입력과 일치시킨다.
- [ ] 센서 전처리와 visual encoder 입력이 학습과 완전히 동일한지 검사한다.
- [ ] 실시간 latency, GPU memory, token 수를 기록한다.
- [ ] DDIA on/off 및 CCDP/MEFA ablation을 agent config에서 선택 가능하게 한다.

## 13. 필수 단위·통합 테스트

### CCDP

- [ ] retention ratio와 실제 retained token 수
- [ ] variable token count 및 padding mask
- [ ] retained index가 원본 token과 정확히 대응
- [ ] reconstruction loss가 selector까지 gradient 전달
- [ ] inference에서 reconstruction branch가 실행되지 않음

### MEFA

- [ ] sequence 시작 시 memory가 비어 있음
- [ ] 서로 다른 batch/sample 사이 memory leakage 없음
- [ ] `T < Z`, `T = Z`, `T > Z` 처리
- [ ] padded frame/token 무시
- [ ] chronological flatten 순서 검증

### 전체 모델

- [ ] 작은 synthetic batch의 forward/backward
- [ ] TinyLLaMA Q/K/V projection gradient가 0이 아님
- [ ] CCDP/MEFA/projection/waypoint head gradient가 0이 아님
- [ ] frozen visual encoder gradient가 없음
- [ ] DDIA on/off 모두 유효한 output shape
- [ ] fp32, fp16 또는 bf16 smoke test
- [ ] DDP 2-GPU smoke test
- [ ] checkpoint save → 새 process load → 동일 입력 출력 비교
- [ ] 실제 CARLA batch 1개 overfit 테스트

## 14. 성능 및 메모리 검증

- [ ] 원본 LMDrive와 VLDrive의 parameter 수 비교
- [ ] trainable/frozen parameter 수 기록
- [ ] pruning 전후 token 수와 retention ratio 기록
- [ ] Tiny/Short/Long route별 평균 sequence length 기록
- [ ] DDIA attention memory 사용량 측정
- [ ] end-to-end latency와 peak GPU memory 측정
- [ ] 논문의 parameter 감소율이나 driving score는 같은 조건으로 재현하기 전까지
      달성했다고 보고하지 않는다.

## 15. 권장 구현 순서

1. 논문·supplementary·공개 코드에서 CCDP/MEFA 세부 사양 확정
2. visual encoder 출력 shape를 고정하는 fixture 작성
3. CCDP selection과 단위 테스트 구현
4. reconstruction branch와 gradient 테스트 구현
5. MEFA sequence mode와 memory reset 테스트 구현
6. top-level visual connector에서 `VisualTokenBatch` 반환
7. `VLDriveModel`에 connector를 연결하는 samples adapter 구현
8. frame별 `waypoint_token_indices` 생성
9. task의 전체 loss 및 optimizer parameter group 연결
10. save/resume 및 local TinyLLaMA loading 검증
11. 한 batch overfit 후 짧은 학습 실행
12. leaderboard agent와 CARLA closed-loop 평가 연결
13. ablation 및 성능/메모리 프로파일 실행

## 16. 완료 기준

다음 조건을 모두 만족해야 VLDrive 통합이 완료된 것으로 간주한다.

- [ ] CCDP와 MEFA가 논문 식에 대응하며 근거가 문서화됨
- [ ] 임의의 고정 token 수 없이 variable-length visual token 지원
- [ ] visual encoder부터 waypoint loss까지 end-to-end forward/backward 성공
- [ ] TinyLLaMA, CCDP, MEFA, projection, waypoint head에 gradient 확인
- [ ] reconstruction loss가 selection 경로에 gradient 전달
- [ ] DDIA의 raw/RoPE block과 shared softmax 테스트 통과
- [ ] checkpoint 저장 및 재로딩 성공
- [ ] 기존 LMDrive 모델과 config가 계속 동작
- [ ] local/offline TinyLLaMA 로딩 성공
- [ ] CARLA leaderboard agent에서 route inference 성공
- [ ] 재현 가능한 실행 명령, 환경, checkpoint, 결과 로그 문서화

## 17. 실행 명령 기준안

정적/단위 테스트:

```bash
cd VLDrive/LAVIS
pytest -q tests/vldrive
```

로컬 TinyLLaMA 통합 테스트:

```bash
TINYLLAMA_TEST_CHECKPOINT=../weights/TinyLlama-1.1B-intermediate-step-1431k-3T \
pytest -q tests/vldrive/test_tinyllama_loading.py
```

connector 통합 완료 후 학습:

```bash
python train.py --cfg-path lavis/projects/vldrive/tinyllama_ddia.yaml
```

실제 데이터 경로, distributed launcher, GPU 수와 checkpoint 경로는 팀 환경에 맞게
설정하되 최종 README에 정확한 명령을 남긴다.

## 18. 작업 시 주의사항

- 기존 `drive_models/drive.py`와 `blip2_models/modeling_llama.py`를 삭제하거나
  전역적으로 VLDrive 방식으로 바꾸지 않는다.
- CCDP/MEFA의 가짜 placeholder 출력을 실제 구현으로 보고하지 않는다.
- 논문에 없는 설계를 논문 명시 사항처럼 문서화하지 않는다.
- visual token을 detach하지 않는다.
- padding token/frame이 DDIA나 reconstruction loss에 참여하지 않도록 한다.
- 사용자 계정, 비밀번호, Hugging Face token을 코드/YAML/로그에 저장하지 않는다.
- 공개 체크포인트는 현재 로컬 경로를 사용하며 비공개 자원에는 read-only token을
  환경 변수로만 전달한다.
