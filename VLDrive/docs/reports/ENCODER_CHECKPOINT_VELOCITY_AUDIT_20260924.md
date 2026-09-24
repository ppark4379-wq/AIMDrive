# Z: encoder checkpoint velocity audit

## Scope

Read-only inspection of `Z:\DCM\weights\checkpoint-best-object.pth.tar` on 2026-09-24. No live inference, vehicle control, training, or active configuration changes were performed. Metrics below were read from the checkpoint; they were not remeasured in this audit.

- File size: 376,619,011 bytes.
- Copied-file SHA256: `3b792386602413aa1fc3bb3a7bd0ff1fbe97a2aa9fe6653d5e4f2fc4b0a895bc`.
- Audit copy and extracted metadata: `VLDrive/.runtime/checkpoint_audit/`.
- This differs from the existing local `outputs/checkpoints/checkpoint-best-object.pth.tar` (125,930,546 bytes).

## Checkpoint evidence

- Epoch: 2; optimizer steps: 84,118.
- Initial checkpoint SHA256: `d0cd815c5ca5303b4844532b18c1420d508bd408f244bad08b9d9d48eee25be4`, matching the current Stage2 encoder source.
- `pretraining.velocity_source`: `gps`.
- `pretraining.velocity_contract`: `causal consecutive GPS fixes, dt<=0.25s; no Ego fallback`.
- `pretraining.trainable_scope`: `full`.
- Learning rate: 2e-6; batch size: 1.
- Training input publications: basic-driving and static-obstacle campaigns.
- Available training frames: 45,084; selected after thinning long stops: 42,059 per epoch.
- Validation frames: 5,384; held-out laps on the same map/route according to provenance.
- `target_point`: `official global path 25m lookahead; prediction head only`.
- Future waypoint horizons: 0.5, 1, 1.5, 2, 2.5 seconds.
- Feature model: `memfuser_baseline_e1d3_return_feature`; token shape `[106,256]`.

Checkpoint-recorded evaluation:

| Metric | Value |
|---|---:|
| Encoder waypoint ADE | 0.203176 m |
| Encoder waypoint FDE | 0.400024 m |
| Encoder waypoint coordinate L1 | 0.112894 |
| Traffic-light accuracy, 1,950 labeled frames | 0.995897 |

These are encoder/head evaluation metrics, not current VLDrive Stage2 or closed-loop driving metrics. Filename `best-object` does not establish waypoint-optimal model selection.

## Architecture and compatibility

CPU `torch.load(weights_only=True, mmap=True)` inspection found 888 state entries. Relative to the current encoder there are no missing/extra keys or tensor shape mismatches. Both velocity_fc tensors and all eight waypoint-generator tensors changed; all encoder/decoder transformer entries also changed. Therefore the new checkpoint is not merely the old encoder with speed enabled.

The recorded source digests for `models/vision_encoder.py` and `utils/morai_transfer_metrics.py` match the current files (allowing CRLF/LF normalization). Dataset and trainer source digests differ, so current dataset/trainer code alone cannot establish every historical preprocessing detail.

In `models/vision_encoder.py`, velocity passes through `velocity_fc` and is added to the six signal/waypoint query embeddings before the transformer decoder. The direct GRU waypoint head also consumes target_point. With `return_feature=True`, that head is bypassed and the encoder returns feature tokens for Stage2. Thus its recorded direct-head ADE cannot be transferred to the LLM output.

Current Stage2 was trained with the original encoder's constant-zero-velocity cache. `inference/live_frame.py::prepare_encoder_frame` still supplies zero velocity. `inference/vldrive_runtime.py::load_encoder` checks checkpoint hash and the recorded preprocessing contract. The new file does not include `morai_encoder_contract`, so merely replacing its config path will not produce a supported full inference run.

## Interpretation and next comparison

Actual speed can reduce ambiguity in distance traveled at fixed future times. It does not identify the desired route, guarantee correct turns, or prove improved closed-loop tracking. GPS-derived speed is an interval estimate; a later switch to Vehicle Status speed requires matching units, timing, and reverse-speed conventions. GPS outages also require an explicit speed-source policy.

Use identical recorded scenes to compare this encoder with real speed versus zero speed, holding target_point and preprocessing fixed. Evaluate direct-head ADE/FDE and feature changes. Separately compare the old full pipeline with a Stage2 adapted to newly generated tokens from this encoder. A direct encoder swap into the old Stage2 is useful only as a compatibility diagnostic, not a fair measure of the new encoder's eventual quality. Preserve the current pipeline as baseline.

Conclusion: the metadata confirms nonzero GPS-derived velocity was used in training, and this is a plausible improvement candidate. Whether it improves the current full VLDrive waypoint output remains unmeasured; matching token generation and Stage2 adaptation are required for a controlled assessment.
