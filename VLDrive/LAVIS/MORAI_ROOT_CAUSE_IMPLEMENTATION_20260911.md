# Root-cause corrections and gated restart — 2026-09-11

## Scope

Only VLDrive source/artifacts were changed. MORAI raw data, expert and collector
were not edited. The obsolete paused `morai_repair_epoch25_20260910T192642Z`
training child was deliberately terminated to release GPU memory. Its output,
logs and previous checkpoints remain available; exit -9 is this cancellation,
not a newly discovered training crash.

## Implemented corrections

1. **RoPE arithmetic**: compute angles, cosine and sine in FP32, then cast the
   trigonometric values to the activation dtype. DDIA masking and its raw vs
   rotated attention blocks are unchanged.
2. **Versioned traffic-loss mask**: `vision_encoder/morai_traffic_labels.py`
   verifies the sidecar against the original aggregate, individual frames,
   route, split, implementation and deterministic policy. Train keeps all 3,352
   frames. Unknown affecting lights (70) and ambiguous red/green permissions
   (112) are excluded **only from the traffic auxiliary loss**. No class is
   relabelled; no new distance cutoff or future-waypoint change is introduced.
3. **Encoder weights-only warm start**: `--warm-start` plus mandatory expected
   SHA256, strict model/shape/dtype checks, fresh AdamW/scaler/epoch/schedule and
   new-output guard. `--resume` cannot silently substitute for corrected-label
   fine-tuning. `--feature-checkpoint last` explicitly hands off the final model.
4. **Controlled CCDP backward candidate**: opt-in `retained_output_st` with
   `training_variant=paper_objective_st_probe`. The hard forward, dynamic argmax
   inference, Eq8/12/13, R=.3, lambda10/1, summed reconstruction and original
   auxiliary gradients are unchanged. A zero-forward task-gradient bypass
   reaches CCDP and its upstream visual input normalization. It is an
   independent, unproven surrogate, NOT claimed identical to author code.
   The default remains `legacy`. No top-k budget, BCE or changed reconstruction
   reduction is used in this experiment. BERT query-FFN initialization is not
   changed concurrently.

## Ordered training gates

The first stage is a **bounded train-only A/B**, not the full downstream run:

- `lavis/projects/vldrive/morai_root_cause_pilot.json` defines both arms.
- Same original epoch25 frozen cache, original instructions and boundary
  exclusion, seed607, local pretrained TinyLlama/BERT initialization, 32 unique
  train evaluation windows, matched balanced draw list, lr1e-5, 320 updates.
- The balanced subset is a shared diagnostic selection, not a new dataset or
  a claim of paper-prescribed sampling. Original train/purge/validation split
  and actual future ego labels stay intact. No validation score selects an arm.
- Preflight checks actual full-model backward, temporal causality and forbidden
  teacher inputs. A separate real eleven-frame waypoint loss measures selector
  gradients apart from pruning/reconstruction losses.
- At160 and320 updates, evaluate turning magnitude/lateral error/FDE, both turn
  signs, stationary/straight errors, sampled/soft/dynamic inference retention
  and saturation. Merely outputting32 tokens cannot pass.
- Comparison rejects different initial weights, initial forward, auxiliary
  gradients, settings, sampling, source/data provenance or completed updates.
- The candidate must pass all gates, recover task gradient and improve curved
  FDE by at least5% over baseline to be selected. If only baseline passes, keep
  baseline. If neither passes, STOP; do not loosen thresholds automatically.

Launcher: `tools/morai/launch_root_cause_pilot.py`. It uses the common training
lock, detached sequential child processes, explicit logs and comparison report.
It **does not automatically start long full training** after a pilot. Each arm
has two rotating optimizer/RNG checkpoints and a model-only last checkpoint.
For interrupted-arm strict resume, run `train_morai_root_pilot.py` with identical
config/arm, a NEW output and `--checkpoint <arm>/checkpoint_slotN.pt`. Source,
data, frozen asset, config and optimizer integrity checks remain enabled.

After a satisfactory A/B review:

1. Fine-tune our epoch25 encoder with corrected-label sidecar, fresh optimizer,
   low fusion/query LR, RGB initially frozen. Proposed bounded run: 3 epochs,
   batch1/accum32/reference-lr8e-5/warmup1 => actual fusion LR
   [5e-6,5e-6,2.5e-6]. No public LMDrive encoder is introduced.
2. Check first-known-green events991/1591/2491/2791, stationary false-GO,
   actual waypoint error and token changes. A traffic-head-only gain is
   insufficient because the connector receives pre-head tokens. Zero-velocity
   intervention is sensitivity analysis, not a new GT or driving-safety proof.
3. Generate a complete NEW cache using the selected new encoder SHA. Preserve
   the old cache; never mix old/new token files.
4. Start joint CCDP–MEFA–TinyLlama/DDIA–waypoint-head training from pretrained
   assets/new trainable modules and fresh optimizer against the new cache.
   The original80%/purge25/20% partition stays intact. Select/report train
   behavior as requested, and finally verify simulator closed-loop driving.

## Evidence locations

- Traffic sidecar and immutable-data audit:
  `../vision_encoder/.runtime/traffic_labels_v2/full_loop_20260911*.json`
- Model/CCDP/RoPE tests: `tests/vldrive/`
- Dataset and weights-only tests: `../vision_encoder/tests/`
- Actual GPU verification: `outputs/root_verify_20260911_*`
- Detached paired run: `.runtime/training_runs/morai_root_pilot_*/status.json`
- Runtime stage, checkpoints and actual metrics are authoritative; a unit-test
  pass is not a claim that MORAI planning/driving performance is fixed.

## Verification and launched first stage

- LAVIS CPU regression:113 passed,1 skipped (the optional local-asset test;
  the actual full TinyLlama model was separately exercised on the GPU).
- VE CPU regression:36 passed, including corrected traffic masks, weights-only
  warm start and the train-only departure evaluator.
- All3,352 train raw per-frame hashes matched. Actual future ego targets and
  every other input/target except `traffic_available` were preserved in adapter
  tests; no validation samples were introduced into train.
- Both real-model GPU preflights passed with10 history frames; future-to-past
  prediction delta0. Initial weights, predictions, source/data and sampling
  were equal. Selector waypoint gradient L2 changed from1.4413e-10 to0.0147523,
  while weighted pruning/reconstruction selector gradients were unchanged.
  See `outputs/root_restart_verification_20260911.json`.
- Same old-final weights + corrected RoPE alone did **not** repair turning in
  the3 selected train clips: mean FDE5.58524m→5.58621m (essentially unchanged).
  Historical forward reproduced the saved predictions exactly; original and
  transferred weight hashes stayed unchanged. This is why a numerical fix is
  not being advertised as an already demonstrated driving fix.
- The paired pilot was launched at2026-09-11 06:40:44 KST:
  `.runtime/training_runs/morai_root_pilot_20260910T214044Z/`.
  It runs the legacy arm first, then the opt-in ST arm,320 updates each.
  Encoder fine-tuning, token recaching and full downstream training remain
  later gated stages; they have not been started by this launcher.
- The next encoder-stage event evaluator is
  `../vision_encoder/tools/evaluate_traffic_departures.py`. CPU metadata dry-run
  verified64 train frames from the4 known-red→green events. GPU old/new-encoder
  comparison awaits the corrected-label encoder trial; no result is assumed.
