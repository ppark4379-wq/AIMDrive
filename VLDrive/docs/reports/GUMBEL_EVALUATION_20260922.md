# Gumbel evaluation alignment (2026-09-22)

Request: always use Gumbel for VLDrive stage-2 evaluation, matching live token selection, and explain why decreasing training waypoint loss has not produced better driving.

## Implementation

`utils.trainer.evaluate` explicitly calls `model.predict(..., compute_auxiliary=False, token_selection_mode='gumbel')`. This covers before/after training, every epoch, best-FDE checkpoint selection, and the standalone stage-2 evaluator. Live inference already explicitly uses Gumbel and is unchanged. Model/selector architecture, weights, losses, temperature, retention target, optimizer, and control parameters are unchanged.

Evaluation seeds Python, NumPy and PyTorch once per pass, not per frame. `evaluation_seed` is 607 in `configs/vldrive_train.json`, defaulting to the run seed in the training entry point. The existing RNG-preservation decorator restores caller RNG states afterward. Reports record `selection_policy=gumbel`, `evaluation_seed`, and `stochastic_passes=1`. A fixed seed controls comparisons; a single pass does not measure variability across seeds.

Historical model configs keep their serialized default selection policy for checkpoint compatibility. The production evaluation entry point overrides it unconditionally. Existing recorded argmax metrics and best-checkpoint labels are not rewritten: the old epoch-93 best was selected using argmax. Future runs select best using Gumbel evaluation.

Checkpoint compatibility permits only the exact reviewed old/new source hashes for the changed evaluator and training entry point. Assets, model config, data provenance, weight digests and unrelated source changes remain checked. Exact training resume remains strict. Model-only evaluation can load the existing checkpoint despite the reviewed evaluation-source changes.

## Interpretation

Training waypoint loss was already computed with Gumbel in train mode. Therefore argmax-versus-Gumbel evaluation cannot by itself explain decreasing training loss without better live driving. Training supervises all valid history frames with a mean coordinate L1 loss; ADE/FDE evaluation scores one last-frame anchor per window. Both use recorded expert observations rather than the observations caused by the model's own driving.

Latest live run `vldrive_20260922_114021_759574` loaded epoch 93 / update 10416 correctly. It delivered 93 predictions at about 3.61 Hz over the first-to-last output interval; median sensor-to-output age was about 550 ms. At route progress 83.13 m, route projection error reached 9.52 m while the model still produced an almost straight/rightward trajectory under a left-turn instruction. This documents a live failure, not its unique cause.

The live speed cap was 3 m/s. Recorded frames at route progress 22..83 m had median speed 4.499 m/s, range 4.019..11.161 m/s (97 frames). The current sensor inputs and recent motion history are thus not guaranteed to match training even at the same route location. Velocity input to the frozen encoder is constant zero, but the model still receives visual history; this does not establish that it lacks all speed information. Do not change speed limits on this evidence alone.

Training window eligibility also requires an instruction candidate valid across the full history; live inference uses the current instruction with recent history. This is a concrete data-policy difference, not a proven root cause of the observed missed turn. Aggregate error reduction is insufficient to rule out poor accuracy on the first turn or on off-path states.

## Verification

51 trainer-utility and realtime-inference tests passed. Tests cover mandatory Gumbel override despite an argmax model default, repeatability, preservation of caller RNG, live sampling behavior, rejection of unreviewed source edits, and rejection of dataset-provenance changes. The configured live checkpoint passed `load_contract` after the patch.

The saved epoch-93 model was re-evaluated over the original 447 training anchors with seed 607. The standalone evaluator completed successfully; no simulator control was launched by it. A new user-run live inference process was observed during evaluation, and an attempted interruption of our offline job found it had already completed. No other process was stopped. Do not use this concurrently run evaluation to benchmark inference throughput.

## Completed same-checkpoint comparison

| Selection | ADE (m) | FDE (m) | Waypoint L1 |
|---|---:|---:|---:|
| Recorded epoch-93 argmax evaluation | 0.219257 | 0.436791 | 0.123919 |
| New Gumbel evaluation, seed 607 | 0.214206 | 0.435296 | 0.120908 |

Gumbel retained 32.398% of tokens, mean 34.342 / 106; peak allocated GPU memory 3.0899 GiB. Result: `outputs/logs/vldrive_gumbel_eval_20260922_best93_verified/report.json`, with corresponding `train_predictions.json`. An earlier attempt stopped at provenance verification before scoring; the verified result is the completed evaluation.

The selection-policy difference did not cause an offline accuracy collapse in this measured pass: Gumbel ADE was about 5.05 mm lower and FDE about 1.49 mm lower. This is not a multi-seed superiority claim. It provides no evidence that changing argmax evaluation alone resolves the missed live turn. The dominant unresolved question is why accuracy on recorded expert inputs does not transfer to the current live sensor/history/instruction and vehicle-state conditions.

Additional read-only subset check joined prediction sample IDs to verified cache GT references and recorded navigation traces. The 15 evaluated anchors within route progress 40..100 m had Gumbel ADE 0.160921 m and FDE 0.320075 m. Their actual progress ranged 57.76..96.13 m with gaps, so this is not full temporal coverage of the turn; training may also supervise earlier frames inside these windows. Even recorded-input accuracy near the first turn is substantially better than the observed live behavior. Its ADE/FDE must not be numerically equated to the live route-projection error: the references and metric definitions differ.
