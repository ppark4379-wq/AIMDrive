# Full-route 100-epoch continuation result (2026-09-22)

Run: `vldrive_20260921_172411_715080`. Source weights: `vldrive_20260920_172744_065238/model-last.pt`. Batch 4, initial LR 3e-6, cosine schedule, fresh optimizer. Completed 100 epochs / 11,200 updates. Final update: 2026-09-22 08:10:08 KST. Logged loop duration: 52,885.73 seconds (14h 41m 26s); final evaluation/save additional.

Evaluation is on 447 training windows, last-frame anchors, argmax selection; no held-out validation or new closed-loop test. These are fitting metrics, not measured real-time vehicle path errors.

| Epoch | ADE (m) | FDE (m) |
|---|---:|---:|
| Starting weights | 0.527450 | 0.969320 |
| 30 | 0.398291 | 0.753216 |
| 58 | 0.274500 | 0.523649 |
| 70 | 0.232450 | 0.464361 |
| 80 | 0.227906 | 0.454831 |
| 90 | 0.218691 | 0.437611 |
| 93 (best FDE) | 0.219257 | 0.436791 |
| 100 (last) | 0.219229 | 0.438024 |

Final ADE improved 58.44%, FDE 54.81%, evaluation waypoint L1 58.70% (0.300138 -> 0.123945) against starting weights. Epoch-average training waypoint loss: 0.374744 -> 0.134129 (unweighted mean of update logs, including the short final batch). Last single-batch waypoint loss 0.095130 is not the dataset mean.

| Horizon (s) | L2 (m) | Lateral MAE (m) | Longitudinal MAE (m) |
|---|---:|---:|---:|
| 0.5 | 0.067055 | 0.007756 | 0.065004 |
| 1.0 | 0.119519 | 0.020679 | 0.112175 |
| 1.5 | 0.191223 | 0.047676 | 0.169686 |
| 2.0 | 0.280325 | 0.076573 | 0.241273 |
| 2.5 | 0.438024 | 0.119055 | 0.379575 |

Final epoch mean Gumbel retention 32.314%, mean p_keep 32.316%, p_keep std 0.388274. Argmax evaluation retention 31.601%, mean selected tokens 33.496 / 106, range 21..46. No aggregate all-KEEP/all-DROP collapse; semantic token importance was not evaluated. Reconstruction loss epoch mean 0.110041 -> 0.088664. No nonfinite top-level numeric training log entries. All monitored module gradients nonzero at final snapshot. Report confirms trainable parameters changed, frozen parameters unchanged. Peak allocated GPU memory 6.9955 GiB.

Best and last model files exist and SHA-256 sidecars verified. Best is epoch 93 / update 10416. Its FDE advantage over last is only 0.001233 m; the two are effectively tied at this aggregate scale. Recommend best as the initial new-model comparison candidate, keeping the previous driving baseline. No inference configuration changed.

The run did not reach ADE 0.01 m. Near-term lateral MAE is below 1 cm, but that is not full-trajectory ADE. Improvement flattens near epoch 90 while LR approaches zero (last update 5.901e-14), so this does not establish an irreducible model/data error floor. Additional epochs at the exhausted schedule do not constitute a useful continuation experiment. Large-error scene attribution and actual closed-loop driving still require separate evaluation.

Sources: run report.json, epoch_metrics.jsonl, training.jsonl, model-best.pt/model-last.pt checksum sidecars. No training/model/control settings changed during this audit.
