# Full-route warm-start training analysis
Inspection time (local): 2026-09-20T23:44:40.250000

Finalization confirmed: report.json, checkpoint.pt with checksum sidecar, train_predictions.json, and epoch 30 metrics are present. All 6720 updates completed. Trainable parameters changed and frozen parameters remained unchanged. Peak CUDA allocation: 6.05019 GiB.

Training uses 447 sampled windows from the existing combined train/validation cache, batch 2 (6690 full batches and 30 singleton tail batches). There is no held-out validation.

| Epoch | Mean training waypoint L1 | Evaluated training ADE (m) | FDE (m) |
|---|---:|---:|---:|
| 1 | 1.3166 | 2.0735060309142703 | 3.590172960008799 |
| 5 | 1.0356 | 1.7831753181489813 | 3.131907531261657 |
| 10 | 0.7503 | 1.3124003143661713 | 2.3190628183968274 |
| 15 | 0.6106 | 0.8997587188169824 | 1.609291043078023 |
| 20 | 0.4762 | 0.6900739527392676 | 1.2807797707041413 |
| 25 | 0.3678 | 0.5471855926874885 | 0.9937102938416922 |
| 29 | 0.3379 | 0.5274137339292758 | 0.9691449455533944 |
| 30 | 0.3322 | 0.5274495277892702 | 0.9693202712173832 |

Best completed epoch: 29; training-FDE-selected model-best.pt.
Both model-best.pt and model-last.pt SHA256 checks passed during inspection.

Best checkpoint horizon metrics:
| Horizon (s) | Lateral MAE (m) | Longitudinal MAE (m) |
|---|---:|---:|
| 0.5 | 0.0165 | 0.1756 |
| 1.0 | 0.0529 | 0.3206 |
| 1.5 | 0.1116 | 0.4373 |
| 2.0 | 0.1796 | 0.5938 |
| 2.5 | 0.2693 | 0.8446 |

Argmax keep fraction: 0.29635; mean selected tokens: 31.413; min/max 14/46. This is not all-drop/all-keep collapse.
Epoch 30 mean stochastic training keep ratio 0.320837, p_keep mean 0.320857, p_keep std 0.339800. Token importance/task causality is not proven by these statistics.

Interpretation: repeated training substantially improved fitting on recorded scenes. This does not establish a data-shortage-only root cause: training duration, inclusion of the formerly held-out portion, batch size, and optimizer schedule changed together. Last-epoch L1 is not lateral error or closed-loop path tracking error. The last individual update (L1 0.11548) is not the epoch mean (0.33219).
Evaluation uses argmax token selection and recorded instruction/history combinations; live uses Gumbel and currently different instruction timing/history. Offline scores do not directly guarantee live control results.
The next useful step is replay/live visualization of this best checkpoint, with scenario-level lateral and longitudinal errors, rather than blindly extending a nearly zero-learning-rate schedule.

Training loop elapsed to last update: 22243.4 seconds (~6h10m43s), including intervening evaluations and saves. Finalization time is additional.

## Final checkpoint recommendation

Use model-best.pt, epoch 29 / update 6496. Its FDE is 0.96914495m versus epoch 30 model-last.pt 0.96932027m, a negligible 0.000175m difference. ADE 0.52741373 vs 0.52744953m. Epoch 30 evaluation waypoint L1 is 0.30013783, final 2.5s lateral MAE 0.26906955m and longitudinal MAE 0.84498325m. Both model-only files passed size and SHA256 verification after finalization. The two epochs are practically tied; the recommendation follows the configured training FDE criterion, not evidence of significant real-world superiority. Realtime config still points to the previous 20260919 run; no inference config was changed during this analysis.
