# Utility layout

- `common/`: runtime configuration, numerical adapters and exact source relocation compatibility.
- `training/`: shared batching/precision/checkpoint helpers and MORAI transfer loss functions.
- `evaluation/`: trajectory/token metrics and MORAI transfer evaluation.
- `config/`: Python configuration loaders and validators; JSON/YAML values remain in `../configs/`.
- `visualization/`: offline token extraction/rendering, encoder figures and the inference/training dashboards.

Training loops, reusable training helpers and losses live in `../training/`.
The root `train.py` remains the command-line entry point. Inference also uses the checkpoint loader
in `training.connector_llm.training` and the precision helpers in `training.common.runtime.precision`.

`common/source_relocations.json` records exact source hashes before and after
this reorganization. Existing checkpoint/cache provenance can use the old paths
only for these recorded versions; subsequent code edits still fail verification.
Dataset, checkpoint and model identity checks are unchanged.
