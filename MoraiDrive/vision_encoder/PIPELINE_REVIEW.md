# MORAI pipeline review - 2026-09-07

Reviewed local merge `37fcb82` against the collected upstream changes through
`cad3e74`. No branch, commit or push was created during this review.

## Verdict

The collector file format, route/index selection, camera/LiDAR loader, Memfuser
input/output tensors and training loss computation now connect in executable
CPU checks. This is not yet a complete closed-loop autonomous driving system.

## Problems fixed during the review

| Problem | Result |
|---|---|
| Upstream train script supplied CARLA town/weather instead of MORAI routes | Script uses MORAI split indexes; CLI derives route lists and rejects overlap before initializing the model |
| Split builder scanned failed episodes back into training | When the summary exists, only episodes marked completed are eligible |
| Single-frame validation called the old three-argument extractor and omitted waypoint interval | Updated to the local two-argument API and explicit interval |
| Upstream LiDAR test used the removed rotation keyword | Tests the MORAI preprocessing utility directly |
| Dataset samples were described as /5 subsampled | Validation output/docs now describe all valid frames |
| Detection BEV used yaw while waypoint BEV used yaw + pi/2 | MORAI detection/heatmap now use the same forward direction as waypoint targets; synthetic forward-object regression added |
| Single-GPU LR scaling called an uninitialized distributed process group | Uses args.world_size, initialized to 1 for single-process training |
| Launcher rank ignored LOCAL_RANK / legacy spelling | Accepts both local-rank spellings and the torchrun environment |
| HybridEmbed forced CUDA while inferring feature dimensions | Uses the backbone's device and dtype; added an offline random-backbone constructor option for validation |
| Requirements omitted tensorboardX and allowed incompatible NumPy/torchvision versions | Added dependency and constrained versions compatible with torch 2.0.1 |

## Evidence

Temporary environment outside the repository: Python 3.10.20, torch 2.0.1 CPU,
torchvision 0.15.2 CPU, torch_scatter 2.1.2 CPU, NumPy 1.26.4, Pillow 9.5.0.
The system Python and CUDA environment were not replaced.

- Collector + vision encoder tests: **38 passed**.
- Scenario runner tests: **17 passed** after installing its gRPC dependency in the temporary environment.
- Expert tests: **79 passed; 4 setup errors** because neither
  `expert/.runtime/path.csv` nor `expert/.runtime/scenario_runner/path.csv` exists.
  These four tests did not exercise their traffic-light behavior.
- Real repository capture `data_collection/output/data/sync_test`:
  single-frame and two-frame batch adapter checks passed; 81 eligible frames.
- Actual writer -> images/NPY/JSON -> successful-route split -> loader exercised
  with synthetic episodes. Checks cover exclusion of failed episodes and invalid
  navigation inputs, original waypoint timeline, >40,000 points, intensity,
  positive detection labels, BEV direction and backfilled future waypoints.
- Real sample -> e1d3 Memfuser -> existing LAVLoss/classification/waypoint losses ->
  backward -> SGD step passed on CPU with random weights. Camera, LiDAR and
  waypoint branches received nonzero finite gradients; all updated parameters
  remained finite. No checkpoint downloaded or written.

Batch contract:

| Tensor | Shape |
|---|---|
| front RGB | B x 3 x 224 x 224 |
| left/right/rear/center RGB | B x 3 x 128 x 128 |
| LiDAR | B x 40000 x 4 |
| target point | B x 2 |
| waypoint target/prediction | B x 5 x 2 |
| detection target/prediction | B x 2500 x 8 |
| traffic-light / stop-sign prediction | B x 2 |

## Remaining integration and validation work

1. **Scenario-to-collection orchestration is missing.** The scenario runner
   configures the simulator and expert; it does not start/stop collect.py for
   each episode. collect.py handles one recording per invocation. A continuous
   recording across reset/respawn would corrupt trajectory targets across that
   boundary. Connect episode start/end and writer finalization before unattended
   multi-scenario collection.
2. **The MORAI inference/control agent is missing.** Checkpoint loading, identical
   online camera/LiDAR/navigation preprocessing, waypoint-to-control conversion and
   simulator actuation must still be implemented. VLDrive's separate language
   model tree is not part of this training path.
3. **Physical coordinate alignment is not established by shape tests.** Synthetic
   world-yaw labels now agree across detection and waypoints, but actual LiDAR
   frame axes, handedness, sensor mounting offsets, GNSS/ego/object coordinate
   origins and camera calibration need an overlay check in MORAI. The sensor
   config still labels extrinsics/origin as unconfirmed. No speculative raw LiDAR
   rotation was introduced.
4. **Training targets are spaced by frame count.** A waypoint interval of 5 only
   means 0.5 seconds at an actual saved 10 Hz. Real-time collection can vary;
   inspect timestamp_ns or introduce timestamp-based target selection. Existing
   last-frame clamping near episode ends also remains.
5. **GPU training and driving quality are unverified.** The full training loop is
   CUDA-oriented. CUDA/DDP, AMP, complete epochs, checkpoint resume and live control
   were not exercised. The real sample is a small sync capture with limited object
   labels; finite losses do not establish dataset quality or driving performance.
6. **Supervision remains multi-task.** Detection, waypoint, velocity, traffic-light
   and stop-sign losses are active. Empty/placeholder labels in old captures do
   not become reliable supervision merely because the pipeline runs.

## Reproduce

See README.md for split creation and CUDA training commands. In a compatible
Python environment, from the repository root:

```bash
PYTHONPATH=MoraiDrive/data_collection python -m pytest MoraiDrive/data_collection/tests MoraiDrive/vision_encoder/tests -q
python MoraiDrive/vision_encoder/tools/validate_training.py MoraiDrive/data_collection/output
python -m pytest MoraiDrive/scenario_runner/tests -q
PYTHONPATH=MoraiDrive/expert/src python -m pytest MoraiDrive/expert/src/expert/tests -q
```

The last command needs the runtime path fixtures described above. The CPU smoke
validator creates no trained output; it is safe to run before a full training job.
