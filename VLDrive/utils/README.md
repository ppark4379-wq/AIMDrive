# Utility layout

- `common/`: runtime configuration, numerical adapters and exact source relocation compatibility.
- `training/`: Stage-1 training, MORAI transfer training and shared `training_helper.py` functions.
- `evaluation/`: trajectory/token metrics, MORAI transfer objectives and evaluation.
- `config/`: Python configuration loaders and validators; JSON/YAML values remain in `../configs/`.
- `visualization/`: encoder figures and the inference/training dashboards.

The Stage-2 training loop is `../training/training.py`. The root `train.py`
remains the command-line entry point. Inference also uses the checkpoint loader
in `training.training` and the precision helpers in `utils.training.training_helper`.

`common/source_relocations.json` records exact source hashes before and after
this reorganization. Existing checkpoint/cache provenance can use the old paths
only for these recorded versions; subsequent code edits still fail verification.
Dataset, checkpoint and model identity checks are unchanged.
