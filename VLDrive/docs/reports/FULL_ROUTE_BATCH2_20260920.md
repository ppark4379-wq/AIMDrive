# Full-route Stage 2 training with batch size 2

Configuration: configs/vldrive_train.json

- epochs: 30 (user setting preserved)
- batch_size: 2
- validation_enabled: false
- include_validation_in_train: true
- learning rate unchanged: 1e-5

NavigationCacheDataset combines existing train and validation rows for training without modifying original cache files, split metadata, instruction labels, or previous checkpoints. Both source splits appear in the resulting training windows. Existing continuity and common-instruction eligibility checks remain active.

Coverage: 2941 cached rows (2370 former train + 571 former validation). There are 2228 eligible windows; stride 5 including the endpoint selects 447. Batch size 2 gives 224 optimizer updates/epoch, including a final singleton, and 6720 updates over 30 epochs. The old cache's 25 purged boundary frames are still absent; missing navigation/future-target rows and windows with no common instruction also remain excluded. Full-route means both temporal portions participate, not every recorded frame is supervised.

No validation dataset is instantiated or evaluated. Before/after and epoch checkpoint scoring use the training samples, explicitly labeled training; model-best.pt uses training FDE. These scores measure fitting and are not held-out generalization estimates.

The trainer collates multiple samples, averages the existing model loss, adjusts cosine scheduling and epoch callbacks to minibatch updates, logs all batch indices and actual batch size, and preserves the incomplete final batch. Exact resume still requires identical settings and provenance. This changed configuration starts a new run initialized from outputs/logs/vldrive_20260919_182054_301289/model-last.pt using warm_start_checkpoint. All trainable weights are transferred without resetting modules; optimizer, random state and schedule are fresh. The 30 epochs are additional epochs. It is not an exact resume of the old 15-epoch batch-1 checkpoint. Explicit --checkpoint takes precedence for exact resume/evaluation, so the configured warm start is not reapplied on resume. Transfer provenance is written to warm_start.json.

Validation:
- 62 tests passed across trainer utilities, navigation cache, training model and language forward.
- Added regression coverage for two epochs with an incomplete batch, per-epoch sample coverage, exact stochastic resume mid-epoch, and combining validation rows into training.
- Actual CUDA smoke test on real cached data: one batch-2 forward/backward/CPU-optimizer update succeeded, all parameters finite. Loss 3.8819375, waypoint L1 2.1143897, gradient norm before clipping 21.5159187.
- Peak PyTorch allocated CUDA memory: 6.153787 GiB. This is a one-update measurement, not a worst-case memory bound or throughput benchmark.
- No operational checkpoint was changed by the smoke test. Temporary smoke script/log deleted after verification.

Run from VLDrive:

```bash
python train.py --config vldrive_train.json
```

Warm-start follow-up: 37 checkpoint/trainer tests passed. Real model-last.pt weight transfer and gradient/causality preflight passed; source update 5595, new schedule 6720 updates, optimizer not restored. Evidence: outputs/analysis/full_route_warm_start_preflight_20260920/report.json and warm_start.json. No full training started.
