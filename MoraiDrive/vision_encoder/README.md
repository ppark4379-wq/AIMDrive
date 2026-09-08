# MORAI vision encoder

This directory combines the local MORAI loader with upstream collection/index
changes through `cad3e74`. It uses four camera files and one 360-degree LiDAR scan,
keeps all valid frames and preserves original trajectory frame IDs.

## Prepare and train

Use Python 3.10 with `requirements.txt`. Install the PyTorch/torchvision and
`torch_scatter` builds matching your CUDA runtime on the training machine.
The audit used PyTorch 2.0.1 CPU, torchvision 0.15.2, NumPy 1.26.4 and Pillow 9.5.

From `MoraiDrive/vision_encoder`, prepare a newly collected job with one command:

```bash
bash scripts/prepare_dataset.sh /path/to/job
bash scripts/train.sh /path/to/job
```

The preparation command selects completed, navigation-valid routes, creates a
deterministic route-level train/validation split, audits every indexed frame and
runs real train/validation loader batches. Pass split options after the path, for
example `--val-ratio 0.1 --min-frames 100`.

The split builder requires at least two eligible routes for a nonempty train/val
split. When an episode summary exists, only completed episodes are eligible.
Older captures without a summary use the route layout checks.

From `MoraiDrive/vision_encoder` on a CUDA training machine:

```bash
bash scripts/train.sh /path/to/job
# Multiple GPUs:
GPU_NUM=2 bash scripts/train.sh /path/to/job
```

Training derives disjoint route lists from `dataset_index.txt` and
`dataset_index_test.txt`. Optional `--train-routes` and `--val-routes` select
explicit subsets. Both must be supplied together. To use a shared index:

```bash
bash scripts/train.sh /path/to/job --val-index dataset_index.txt \
  --train-routes route_a route_b --val-routes route_c
```

Missing files, unknown routes, overlapping selections and empty datasets raise
errors. `navigation.route_valid == false` frames are excluded; older measurements
without that field remain eligible. No fallback substitutes a missing index.

`--waypoint-interval` defaults to five saved frames between each of five target
waypoints. At a measured 10 saved frames/second this corresponds to 0.5 through
2.5 seconds. Real-time sensor collection is not guaranteed to have a fixed rate;
check `timestamp_ns`. Targets near route ends retain last-frame clamping.

Memfuser requires `--multi-view --with-lidar`; the launch script supplies both.
The center image is a crop of the front image, not a fifth physical camera.
Segmentation, depth and `waypoints.npy` are unsupported. The collector backfills
`future_waypoints` at finalize; waypoint regression uses `measurements_all.json`.

LiDAR axes are preserved; the loader clips XYZ only and samples/pads to 40,000
points. Detection/raster labels use the same yaw-plus-pi/2 BEV convention as
waypoints (image lateral +x, forward -y). Actual sensor axes and extrinsics still
need overlay validation on the installed MORAI sensors.

The inherited loss combines detection, waypoints, velocity, traffic light and stop
sign. This is not waypoint-only training. Placeholder labels and limited scenarios
are not sufficient evidence of useful supervision or driving quality.

Detection targets are generated from `measurements_full[*].actors_data` at load
time. The separately stored `3d_bbs/*.npy` files are not model inputs; preparation
reports count mismatches as warnings. To enforce the collection contract separately,
run `python tools/audit_dataset.py /path/to/job --strict-3d-bbs`.

## Checks

```bash
python -m pytest tests -q
python tools/validate_training.py ../data_collection/output
```

`validate_training.py` runs the e1d3 architecture on a real batch using random
weights, checks output shapes, the actual training loss implementation, gradients
and one optimizer step. It does not download or save a checkpoint. Use `--device
cuda` to repeat on a CUDA machine. CPU support here is for this validation tool;
the full `train_pretrain.py` loop remains CUDA-oriented.

See [PIPELINE_REVIEW.md](PIPELINE_REVIEW.md) for the audit results and open issues.
