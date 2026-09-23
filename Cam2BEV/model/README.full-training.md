# Full 1_FRLR training

This job trains Cam2BEV on the official synthetic four-camera dataset. It does not
train the RGB segmentation frontend or establish MORAI deployment accuracy.

## Configuration and artifacts

- Configuration: `model/config.1_FRLR.full.yml`
- Training/validation: 33,199 / 3,731 aligned samples.
- Input order: front, rear, left, right; target: BEV including occlusion.
- Shape: 256x512, batch size 2, learning rate 0.0001.
- Maximum 100 epochs; stop after 20 epochs without improved validation mean IoU.
- GPU environment: `.venv-gpu`; TensorFlow 2.21 with `TF_USE_LEGACY_KERAS=1`.
- CPU smoke environment `.venv-training` is preserved.
- Subsequent supervisor starts cap TensorFlow GPU memory at 6144 MB and cuDNN workspace at
  512 MB, leaving headroom for the simulator/display. Training/evaluation respect
  `CAM2BEV_GPU_MEMORY_MB`.
- CUDA library paths inside the isolated environment are configured by the supervisor.
- The RTX 5070 requires initial PTX compilation with this TensorFlow build; the first
  GPU execution can be much slower than subsequent runs.
- Seed: 42 for NumPy and TensorFlow; GPU execution is not guaranteed deterministic.

Each run writes to `model/output/full/<timestamp>/`:
- `metrics.csv`: train/validation losses, mean IoU, road IoU, road precision/recall.
- `Checkpoints/best_weights.hdf5`: best validation mean IoU.
- `Checkpoints/best_road_weights.hdf5`: best validation road IoU.
- `Checkpoints/last_weights.hdf5`: weights at the most recent epoch.
- Periodic epoch weights and TensorBoard events.

Road metrics use class 0 and argmax predictions. The optional 0.7 confidence threshold
in the UDP inference script is NOT calibrated by these metrics. Validate its precision/
recall and tune it on a separate validation set before using it.

## Monitoring the background job

From the repository root:
```bash
cat model/output/full-training-status.json
tail -f model/output/full-training-supervisor.log
tail -f model/output/full-data-download.log
tail -f model/output/full-training.log
```

Status reports the current stage, PID and any failure. Logs are created when each
stage starts. Downloads can take hours at the official server's transfer rate.
The machine/WSL must remain running; there is no reboot/startup service.

The supervisor runs:
1. Complete official downloads and check file alignment/counts.
2. Verify GPU forward/backward/optimizer execution.
3. Run full training; refuse silent CPU fallback.
4. Evaluate the best-road checkpoint on the full validation split.

Dataset preparation is in `scripts/download_1_frlr.py`. Complete archives are retained.
On servers without HTTP range support an interrupted incomplete archive restarts.
No existing unrelated datasets or Python environments are removed.

To start a new supervised run once no other run is active:
```bash
python3 -u scripts/run_full_training.py
```
The supervisor has a file lock against duplicate full jobs. Do not start independent
`train.py` processes while the supervised run is active.

## Use limitations

The checkpoint from the earlier 4-sample smoke test is not used as initialization;
full training starts from fresh model weights. This job's outputs should not be
confused with `model/output/smoke/`.

The supplied dataset geometry differs from the MORAI rig. Even a high validation
road IoU does not establish MORAI accuracy. MORAI calibration, RGB semantic segmentation,
domain validation/fine-tuning and evaluation of obstacles/occluded areas remain necessary.


## Active run (2026-09-23)

Full dataset download/alignment verification completed: 33,199 train and 3,731 validation.
The active run is `model/output/full/2026-09-23-17-28-10`, supervisor PID 3166.
Its training process started before the 6 GB environment update and uses memory growth.
An attempted restart to apply the cap was rejected by automatic approval review due to
possible unsaved training progress loss; the active run was preserved.
The new cap applies on subsequent supervisor starts, not retroactively.

GPU pilot verification passed for actual uNetXST training, validation, road metrics and
best/best-road/last checkpoints with batch 2 and the 6144 MB cap. Pilot log:
`model/output/gpu-model-smoke-limited.log`.
