# MORAI vision encoder

The local MORAI dataset implementation is retained. The MORAI loader,
transforms and renderer were imported from origin/main at `d4aca07`.
Memfuser and backbone implementations already match that revision.

Run from this directory in an environment with `requirements.txt` installed:

```bash
python train_pretrain.py /path/to/output --dataset morai \
  --train-routes train_route_a train_route_b --val-routes validation_route \
  --waypoint-interval 5 --model memfuser_baseline_e1d3 \
  --multi-view --with-lidar --multi-view-input-size 3 128 128 \
  --no-prefetcher --eval-metric l1_error --saver-decreasing \
  --batch-size 4 --epochs 25
```

Replace route names with entries from `dataset_index.txt`. Training and
validation routes must be explicit and disjoint; unknown names and empty
datasets are errors. MORAI does not use town/weather arguments.

- Index entries accept `route/` or `data/route/`; files reside in `root/data/route/`.
- Frames with `navigation.route_valid == false` are excluded. Older measurements
  without this field remain eligible. Missing measurement files raise an error.
- Every valid frame is used, keeping its original frame ID for future targets.
- `--waypoint-interval` is the number of saved frames between each of five target
  waypoints. At 10 saved frames/second, 5 means targets at 0.5, 1, 1.5, 2, 2.5
  seconds. Adjust it to the actual collection frequency. Route-end targets retain
  the existing last-frame clamping behavior.
- LiDAR uses one scan, preserves MORAI axes, and pads/subsamples to 40,000 points.
- RGB is read per camera. Segmentation, depth and `waypoints.npy` are unsupported.

The inherited training loop includes detection, waypoint, velocity, traffic-light
and stop-sign losses. Older datasets with placeholder labels need a separate
decision about which supervision to train; these changes do not make those labels
valid or switch training to waypoint-only loss.

Regression checks (NumPy required for LiDAR checks):

```bash
python -m unittest discover -s tests -v
```

Before a full run, validate one real batch and a forward/backward pass in the
training environment. The local Windows environment used for integration has no
PyTorch installation.
