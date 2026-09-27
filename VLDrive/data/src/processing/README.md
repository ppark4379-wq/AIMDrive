# Data processing

Run from `VLDrive` with the training Python environment:

```bash
python data/src/processing/process_data.py
```

The command reads `training/vldrive_train.json`, allocates a unique timestamp ID,
and writes all new artifacts inside `data/processed/<id>/`:

```text
<id>/
  instructions/
  navigation_tokens/
  traffic_labels/
  vision_tokens/
  processing_config.json
  status.json
  manifest.json
```

Order: navigation instructions, optional traffic labels, GPS-conditioned frozen
vision tokens, navigation text tokens, validated version manifest. Only after
success is `data_version` in the training config updated. Training is not started.
`--id NAME` chooses a new ID; existing directories are never overwritten.
`--no-activate` leaves the training config unchanged. Failed versions retain
`status: failed` for diagnosis and are not selected for training.

Set raw inputs in `preprocessing.datasets`. Each source needs the supported
Sensor Record format and a namespace; registered raw sources are verified before
processing. The runner checks that input paths and models exist before allocating
an output directory. It does not automatically repair missing raw datasets.

Traffic auxiliary labels are optional for Stage2. To generate them, configure
`preprocessing.traffic_labels.jobs` with entries containing `name`, `dataset_root`
and `split_file` for the existing measurement-based traffic-label builder.
Outputs are `<name>.json` and `<name>_report.json` inside the version. These jobs
require the builder's legacy measurement/split format, not arbitrary Sensor Record
episodes. With no jobs, `traffic_labels/status.json` explicitly records
`not_requested`; no labels are fabricated and no Stage2 loss is changed.

Existing active caches live physically in `stage2_v001`. Older artifacts are
preserved in `legacy_v001`; they are an archive, not a newly validated training
bundle. Old category paths are compatibility junctions for historical checkpoints
and saved configs. They contain no duplicated cache payload. Do not delete these
links while using historical runs.
