# Immutable traffic auxiliary supervision, version 2

`full_loop_20260911.json` is the strictly checked sidecar for the existing full-loop dataset and `configs/morai_full_loop_80_20.json`. The source MORAI files were not edited. See `full_loop_20260911_report.json` for train counts/source hashes and `full_loop_20260911_validation.json` for completed checks.

Pass `traffic_label_sidecar=<path>` to `create_carla_dataset("morai", ...)`. The adapter exposes `traffic_label_provenance`; it is `None` when the optional sidecar is absent, preserving old behavior. The training CLI uses `--traffic-label-sidecar` when connected by the runner.

## Policy

- Preserve every original `traffic_light` class. This is a mask-only correction, not a new GO/STOP controller.
- Exclude 70 train observations with an affecting MGeo signal ID but unknown state from the traffic auxiliary loss.
- Exclude 112 train observations with red plus green-left permission (state33). Do not relabel them all GO. The same conservative composite rule also handles state49, which does not occur in this train set.
- Keep all 3,352 train frames and their actual future ego waypoint supervision. Only 3,170 contribute traffic classification loss.
- Keep first known green frames991,1591,2491,2791 available and not-red.
- Do not apply a new 45m cutoff or change route association. Raw `observed_signals` contain ID/state, not signal distance; the raw measurement canonical digest, route index/current link, route JSON digest and complete aggregate digest bind the original context. The source does not store a stop-line-distance field, so this version does not invent one.

At initialization the loader deterministically rebuilds the policy decisions and compares the complete artifact, including source, split and implementation hashes. Each actual `measurements_full` record is separately checked against its canonical digest before its label is used. This deliberately fails closed if a sidecar is reused with altered sources or policy. Change the version/output file and rebuild explicitly when those inputs change; do not overwrite a provenance file to imitate compatibility.

The sidecar covers both declared anchor splits to permit ordinary loader construction; no validation performance was evaluated or used to set this policy. Reported counts and the3352 actual per-frame checks are train only.

Completed CPU checks:16 new tests plus2 existing split tests, and all3352 original train per-frame measurement hashes. The new real-adapter test verifies exact equality of RGB/LiDAR/velocity/target-point inputs and all targets except `traffic_available`, including actual future ego waypoint tensors. No GPU training was performed by the data-correction task.
