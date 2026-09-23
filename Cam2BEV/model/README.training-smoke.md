# Provided-data training verification (2026-09-23)

The original `train.py` and four-camera `uNetXST.py` ran without source changes.
This is a pipeline smoke test, not a trained production model or a MORAI accuracy test.

- Environment: WSL Ubuntu 20.04, Python 3.8.10, tensorflow-cpu 2.5.0.
- Isolated environment: repository-local `.venv-training`; other Python environments unchanged.
- Official data source: https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data
- Dataset: 1_FRLR, real training split 4 samples and validation split 2 samples.
- Each sample includes front/rear/left/right plus bev+occlusion.
- Local subset: `data/1_FRLR_smoke`; provenance and filenames: `manifest.json`.
- All images are 604x964; camera/label filenames match; input/output palettes cover 100% of pixels.
- Network input/output: 256x512, batch 1, 1 epoch, 4 optimizer steps.
- Training: exit 0; loss 3.8210, validation loss 2.4717, validation mIoU 0.0387.
- Epoch duration reported by Keras: 62 seconds (includes initial graph tracing).
- Checkpoints: `model/output/smoke/2026-09-23-16-50-28/Checkpoints/`.
- `best_weights.hdf5` and `e001_weights.hdf5` were generated.
- Training log: `model/output/smoke-training.log`.
- Package lock snapshot: `model/output/smoke-environment.txt` (`pip check` passed).

## Repeat in the existing environment

From the repository root:

```bash
cd model
OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 \
  ../.venv-training/bin/python train.py -c config.1_FRLR.smoke.yml
```

Each run creates a new timestamp directory. To use the weights from the recorded run:

```bash
OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 \
  ../.venv-training/bin/python predict.py -c config.1_FRLR.smoke.yml \
  --model-weights output/smoke/2026-09-23-16-50-28/Checkpoints/best_weights.hdf5 \
  --prediction-dir output/smoke/2026-09-23-16-50-28/Predictions
```

## Limits

The full official dataset has NOT been downloaded. Original full-training configuration
still points at `data/1_FRLR`; the smoke configuration points at `data/1_FRLR_smoke`.
Do not interpret the saved smoke checkpoint as a usable road segmentation model.
Four samples and one epoch establish only that preprocessing, optimization,
validation and checkpoint generation work.

GPU training on the RTX 5070 was not tested. This environment deliberately installs
the CPU build of TensorFlow. GPU dependencies/compatibility must be prepared separately
before a full GPU run. MORAI camera calibration, segmentation and training-data
adaptation remain separate work.

Reload verification: original predict.py exited 0, reloaded best_weights.hdf5 and saved two valid 256x512 PNGs. All 358 checkpoint weight arrays were finite. Prediction log: model/output/smoke-prediction.log.
