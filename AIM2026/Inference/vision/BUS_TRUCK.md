# Bus/truck perception supplement

`VisionRuntime` runs `BusTruckSupplement` after the ordinary VE box decoder,
before publishing the existing `morai_ve_udp_objects_v2` packets to Expert.
The supplement returns normal `Box` objects; Expert receives no new lane veto,
uncertainty field, or special object type.

The detector reuses LiDAR clustering and temporal confirmation. Its geometric
gate accepts long, tall bodies (5.5–18 m observed length, height at least 2.2 m).
It can accept a narrow visible side without inventing a hidden vehicle width.
This is **not a trained bus/truck classifier**: large structures can satisfy the
gate, and partial/sparse returns can fail it. No simulator NPC identity or GT
position/speed is used. Live precision and recall still need MORAI validation.

Only confirmed large-body candidates enter missing-detection memory. Ordinary
VE boxes are passed through unchanged. Overlapping supplements are suppressed
against current VE detections. Missing candidates expire rather than persisting
indefinitely. Existing Expert tracking still processes the published boxes.

Configuration under `Inference.vision`:

```yaml
bus_truck_supplement: true  # false returns the ordinary VE decoder output
large_vehicle_hold_s: 2.0  # zero disables constant-velocity missing-box hold
```

`--detection-hold-s` forwards the hold duration. `--lane-motion-tracking` supplies
map geometry and selects the moved lane-history tracker for large candidates
only. That tracker uses its own bounded confirmation/lifetime rules instead of
the constant-velocity hold duration. Bounded positional uncertainty becomes
ordinary box dimensions.

Configure LiDAR refinements in `AIM2026/config_local.yaml` under `Inference.vision`:
`lidar_heading` controls heading/length refinement, `lidar_centers` controls
cluster-center output. Each accepts `true` (on) or `false` (off); set both to `false`
to disable the same features as `--no-lidar-rules`. The legacy CLI flag remains
an explicit off override, but repeat scripts use the config by default.
To disable the supplement independently, set `bus_truck_supplement: false`.
The two moved helpers are `detection_memory.py` and `lane_tracking.py` here;
neither is invoked by Expert's detection receiver anymore.


## Independent highway comparison after the 2026-10-29 merge

The remote implementation from commit `2fb1a7d6` is kept in
`postprocess.py` (`apply_big_vehicles`, `big_vehicle`). It grows detected
large-vehicle outlines and rescues low-score detector candidates using LiDAR.
The local implementation remains in `bus_truck.py`, `lidar_obstacles.py`,
`detection_memory.py`, and `lane_tracking.py`; it confirms large-body candidates
and retains missing detections over time.

| Profile | `big_vehicles` (remote) | `bus_truck_supplement` (local) |
| --- | --- | --- |
| `AIM2026/config_highway_remote.yaml` | true | false |
| `AIM2026/config_highway_local.yaml` | false | true |
| `AIM2026/config_highway_baseline.yaml` | false | false |

Both true is rejected by config validation and the runtime constructor, before
weights load. These profiles share the same sensor, model, LiDAR heading/center,
and control settings from `config_local.yaml`; only the two switches and status
output directory differ. Baseline disables these two additions, not the shared
LiDAR heading/center refinements. Existing local configurations retain their
local supplement default; the remote option remains off by default.

Run sequentially in WSL from the repository root, with MORAI reset to the same
initial conditions before each run:

```sh
bash AIM2026/Expert/run_highway.sh --config AIM2026/config_highway_remote.yaml --seed 6520247091589786966 --highway-density dense --highway-bus-heavy --validation-events
bash AIM2026/Expert/run_highway.sh --config AIM2026/config_highway_local.yaml --seed 6520247091589786966 --highway-density dense --highway-bus-heavy --validation-events
bash AIM2026/Expert/run_highway.sh --config AIM2026/config_highway_baseline.yaml --seed 6520247091589786966 --highway-density dense --highway-bus-heavy --validation-events
```

Use the same CLI options for every profile. To compare the local lane-history
variant, add `--lane-motion-tracking` to all comparison commands; only the local
supplement consumes its lane geometry. Do not use the existing fixed replay
scripts for this comparison: they select their own configuration. A fixed seed
controls scenario randomness but does not guarantee identical live sensor timing.
These are isolated correction paths on the merged codebase, not complete
snapshots of the pre-merge remote and local applications.

The original 293 local files are backed up in
`.runtime/merge-backup-20261029/` with a SHA-256 manifest, and the original local
Git stash is retained as `preserve-local-before-origin-main-20261029`.
