> 논문 기준 connector 통합은 `training_model.py`와 `train_vldrive.py`에 있다.
> [VLDrive 학습 안내](../../../README_VLDRIVE_TRAINING.md)를 참고한다.

> MORAI의 raw 센서부터 시작하는 전체 모델은 `morai_model.py`를 사용한다.
> [MORAI 학습 안내](../../../README_MORAI.md)를 참고한다. 아래는 기존 언어 모델 API 문서다.

# VLDrive TinyLLaMA + DDIA implementation

## 1. Source files inspected

The implementation was checked against `drive_models/drive.py`, the legacy
`blip2_models/modeling_llama.py`, `Qformer.py`, `blip2.py`, `tasks/drive.py`,
`runners/runner_base.py`, the leaderboard agent/config, the LMDrive training
YAML and dependency files. The supplied VLDrive ICCV 2025 paper, especially
Section 3.5, Figure 4 and Eq. (9), was treated as the algorithmic source.

## 2. LMDrive original pipeline

LMDrive loads its LLaMA/Vicuna checkpoint in `drive.py`, freezes the LLM unless
LoRA is enabled, runs four learned Q-Former queries per frame, and maps Q-Former
features to the LLM hidden size with `llm_proj`. Its ordinary layout is valid
instruction tokens, flattened chronological frame tokens, then text padding.
The standard LLaMA position IDs are implicit. A 2-D mask hides padding.

For each valid frame, LMDrive records the last token of that frame. The selected
hidden state is passed to either a hidden-size/ReLU/10 MLP or the optional GRU
decoder. The ten outputs represent five `(x,y)` waypoints. A second MLP predicts
instruction completion. `runner_base.py` saves only parameters whose
`requires_grad` is true; consequently trainable TinyLLaMA weights are included
in VLDrive checkpoints and frozen base weights are reloaded from the configured
base checkpoint.

## 3. VLDrive modification

The new `vldrive` model consumes final visual tokens rather than implementing a
vision encoder, CCDP, or MEFA. It projects them when necessary, constructs
explicit modalities, applies TinyLLaMA with DDIA, and retains the LMDrive MLP
waypoint head. Legacy LMDrive files remain unchanged.

## 4. TinyLLaMA checkpoint

Default: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`. This identifier
is an **implementation choice**: the paper names TinyLLaMA but does not establish
this exact Hugging Face checkpoint identifier. A model name or local path,
cache directory, and offline (`local_files_only`) load are supported. The loader
uses `AutoTokenizer` and `AutoModelForCausalLM`.

The checkpoint has been downloaded to
`VLDrive/weights/TinyLlama-1.1B-intermediate-step-1431k-3T`. The project YAML
uses this local path with `local_files_only: true`.

Model checkpoints under `VLDrive/weights/` are intentionally excluded from Git
because the TinyLLaMA safetensors file exceeds GitHub's per-object LFS limit.
Restore the local checkpoint after cloning with:

```bash
cd VLDrive/LAVIS
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', local_dir='../weights/TinyLlama-1.1B-intermediate-step-1431k-3T', local_dir_use_symlinks=False)"
```

## 5. DDIA Eq. (9) mapping to code

`build_ddia_attention_mask` implements these allowed pairs from explicit IDs:

- instruction query -> every valid instruction key;
- instruction query -> visual key: blocked;
- visual query -> every valid instruction key;
- visual query -> strictly earlier visual keys (`key_index < query_index`).

`compose_ddia_logits` builds raw and RoPE logits, selects the appropriate block,
then performs one softmax over the combined allowed keys. Padding never acts as
a query or key. A safe diagonal used internally for all-masked padding queries
is zeroed from the output and therefore carries no information.

## 6. Instruction/visual token layout

The implementation uses `[right-padded instruction block, chronological visual
block]`. The fixed padded instruction block is an **implementation choice** for
simple batching; explicit modality IDs ensure padding does not affect DDIA.
The paper defines instruction and temporal visual sets but does not prescribe
this batch-padding layout.

## 7. RoPE behavior

- Instruction -> instruction: rotated Q and K.
- Instruction -> visual: blocked.
- Visual -> instruction: raw/non-RoPE Q and K (distance-decoupled term).
- Visual -> visual: rotated Q and K, with the strict earlier-token mask.

Raw and rotated tensors coexist until block composition.

## 8. GQA support

Head counts are read from checkpoint configuration. K/V are first projected to
`num_key_value_heads`, and raw K, rotated K, and V are expanded by `repeat_kv`
to `num_attention_heads`. No equality between query and KV head counts is
assumed. The old LMDrive LLaMA already has GQA code, but it cannot preserve raw
and rotated Q/K simultaneously, so it was not modified.

## 9. Trainable/frozen parameters

`freeze_tinyllama` defaults to false. TinyLLaMA, its pretrained Q/K/V/O
projections used by DDIA, `visual_to_language_projection`, the waypoint MLP,
and completion MLP are trainable. No LoRA/QLoRA is introduced. The reporting
utility counts total, trainable, frozen and attention-related parameters.

## 10. CCDP/MEFA interface contract

Provide `visual_tokens` as `[B,V,D_visual]` or `[B,T,M,D_visual]`, plus optional
mask `[B,V]` or `[B,T,M]`. Temporal input is flattened in time-major,
within-frame token order. `M` is not fixed. Inputs are not detached, so future
connector losses and waypoint gradients can coexist.

## 11. Paper-specified choices

TinyLLaMA is tuned; DDIA uses instruction self-attention, raw visual-to-text
similarity, RoPE within each modality, a shared normalization denominator, and
causal/earlier visual context. Hidden representations feed an MLP trajectory
predictor.

## 12. Our implementation choices

The exact checkpoint, Transformers 4.31.x compatibility target, FP16 default,
fixed padded instruction-first batch layout, legacy hidden-size/ReLU/10 waypoint
MLP, last-valid-visual-token default extraction, and lack of cached DDIA decoding
are implementation choices. The VLDrive paper and supplement do not identify an
exact Hugging Face checkpoint, Transformers version, or numeric precision.
Callers can supply explicit waypoint token indices for every frame.

## 13. Known uncertainties

The paper's exact tokenizer checkpoint, connector output dimension, and precise
per-frame prediction-token selection are not specified sufficiently for a unique
implementation. `visual_feature_dim` and token indices therefore remain
configurable. DDIA uses strict `< j` as required by Eq. (9); it does not add a
visual self edge. Empty-instruction samples are unsupported because a first
visual token would otherwise have no valid Eq. (9) key.

## 14. Dependency compatibility

Legacy LMDrive pinned Transformers 4.28.0. The official 4.28 LLaMA implementation
does not create separate KV heads and cannot load TinyLLaMA's GQA projections.
Transformers 4.31 added `num_key_value_heads`, `num_key_value_groups`, and
`repeat_kv` while retaining the three-value attention return API used here.
Therefore `environment.yml` and `LAVIS/requirements.txt` pin 4.31.0 and the loader
rejects other versions when DDIA is enabled. This pin is an **implementation
choice for compatibility**, not a version stated by VLDrive. The legacy LMDrive
imports must still be smoke-tested in the final Linux/CUDA environment.

The YAML explicitly selects FP16 and `VLDriveModel.from_config` maps it to
`torch.float16`. Visual tokens, the visual projection, and prediction heads are
aligned with the TinyLLaMA embedding dtype. FP16 is likewise an implementation
choice; `fp32`, `bf16`, and automatic BF16/FP16 selection remain supported.

Validation with Transformers 4.31.0 and the downloaded local checkpoint passed
an end-to-end DDIA backbone forward at FP16: hidden states `[1,6,2048]` and
attention weights `[1,32,6,6]`. The modality mask, row normalization, and
instruction-to-visual blocking were checked on the returned attention weights.
An actual TinyLLaMA backward pass produced non-zero first-layer Q/K/V gradients.
The top-level VLDrive language model also passed an FP16 forward with a trainable
visual projection and produced waypoint output `[1,10]`.

## 15. How to run tests

From `VLDrive/LAVIS`:

```bash
pytest -q tests/vldrive/test_ddia_attention.py tests/vldrive/test_vldrive_language_forward.py
TINYLLAMA_TEST_CHECKPOINT=/path/to/local/checkpoint pytest -q tests/vldrive/test_tinyllama_loading.py
```

The second command is deliberately offline and does not download a 1.1B model.

## 16. How to start fine-tuning

The existing CARLA task does not yet emit the new CCDP/MEFA visual-token
contract, so the training command is ready only after that teammate integration:

```bash
python train.py --cfg-path lavis/projects/vldrive/tinyllama_ddia.yaml
```

## 17. Files created/modified

Created: all Python and documentation files in `models/vldrive`, the VLDrive
YAML, and `tests/vldrive`. Modified: `lavis/models/__init__.py` to register the
separately selectable architecture, plus the dependency files to select the
validated Transformers 4.31 API family.
