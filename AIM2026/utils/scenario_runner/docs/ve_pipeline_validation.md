# VE + GPS drive incident evidence

From `MoraiDrive`, run the existing GPS/VE scenario with validation enabled:

```bash
bash scenario_runner/tools/run_ve_full_route.sh --mission full_loop_tunnel_vehicle --seed 123 --validation-events
```

This command is for a later MORAI session. The implementation and unit tests do not start MORAI.

## Actual execution paths

| Command | Ego controller | Validation before this change | New connection |
| --- | --- | --- | --- |
| `python3 -m scenario_runner.main_runner --ve-objects --validation-events ...` | Separate ROS Expert process, `AutonomousDriving` and `PathController`; VE object ROS input | Event Monitor and Incident, when option given | Unchanged; no new cycle journal |
| `bash scenario_runner/tools/run_ve_scenario.sh ... --validation-events` | Wrapper for the preceding Runner path | Same | Unchanged |
| `bash scenario_runner/tools/run_ve_full_route.sh ... --validation-events` | `run_ve_gps_full_route.py` starts `run_ve_gps_expert.py`; driver launches `VLDrive/inference/ve_objects_udp.py`, calls `AutonomousDriving.execute()`, sends MORAI UDP control | No Event Monitor or Incident | Event Monitor, Incident, and cycle journal enabled |

The GPS runner uses gRPC for scene setup, NPC activity and collision observation. The driving process uses GPS, IMU, UDP status and VE boxes. `run_ve_gps_expert.py` already writes `inputs.jsonl`, `control.jsonl`, `expert.log`; the VE subprocess writes its publisher audit. `[PLAN]` and `[CTRL]` originate in `autonomous_driving.py`; `[PP]` originates in `pure_pursuit.py`. These console strings are not used as diagnosis evidence.

## Journal and incident schema

The runner supplies its `run_instance_id` to the GPS driver as `--run-id` only with `--validation-events`. The driver writes `<ve_gps_full_route output>/driver/pipeline.jsonl`. Each line has `schema=aimdrive_pipeline_v1`, `run_id`, driver-local `sequence`, `cycle_id`, `stage`, `timestamp_ns`, `clock_domain=host_monotonic_ns`, controller name, model hash, and source file hashes. `cycle_id` joins stages within the driver; VE sequence and pose sequence are distinct source identities. Sensor source timestamps are kept inside `sensor_samples`, separate from host receipt and inference completion times.

Stages:

- `perception`: tracked boxes with track IDs, dimensions and projected-grid pose; VE sequence, sensor receipts, inference completion time, validity.
- `planning`: Ego input, actual `ObjectInfo` values supplied to Planner, VE/pose lineage, path, target speed, mode, reason, collision metrics and debug details when JSON-safe. Missing outputs are `null` or `unavailable`.
- `control`: linked planning cycle, input path and speed, calculated command in radians, wire command with normalized steering, command creation and successful UDP send timestamps, observed Ego pose. UDP send success means a successful socket call, not an actuator acknowledgement.

The driver flushes after each control line. On `collision_started`, the existing recorder saves at most the preceding 30 host-monotonic seconds to `perception_trace.jsonl`, `planning_trace.jsonl`, `control_trace.jsonl` alongside the existing Ego/NPC/event traces. If the journal cannot be read or flushed, `manifest.json` records the error and the original collision failure policy remains in effect.

The VE publisher also receives the runner run ID in its audit rows. A matching 30-second slice is saved as `ve_audit_trace.jsonl` when available.

Run diagnosis again without MORAI:

```bash
python3 -m scenario_runner.validation.diagnose scenario_runner/.runtime/incidents/incident_<id>
```

The GPS route saves `validation_runs/<run_id>/summary.json` and `validation_events.jsonl` with the same option. It retains the existing Ego observation replay file.

## Interpretation limits

- `perception.delivery` compares VE track IDs with the exact Planner input in one driver cycle. It does not equate VE IDs with MORAI NPC actor IDs.
- `planning.footprint` tests a path against the *current* footprint of a VE object explicitly marked stationary, only in the same coordinate frame. It does not predict moving-object occupancy.
- `control.stop_target` uses the existing `PathController` stop threshold of 0.05 m/s and its full-brake rule. Other target-speed to command mappings have no diagnostic threshold.
- `control.ego_response` needs an applied-command acknowledgement and an explicit response contract. Current MORAI status supplies neither; this check returns `INSUFFICIENT_EVIDENCE` for real runs.
- Source sensor clocks are not equated to host monotonic time. NPC GT observations lack an explicit frame contract and VE-to-NPC geometry matching is not asserted.
- The normal Runner `--incident-rerun` command is for its ROS Expert path. For a GPS incident, `rerun.json` contains the original GPS route options plus its seed. It is a restart from seed, not a snapshot restore. GPS replay comparison of NPC placement is not automated here.
