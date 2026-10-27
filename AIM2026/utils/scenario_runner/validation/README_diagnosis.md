# Failure diagnosis MVP

## Highway, roundabout, and static launchers

The common launcher `AIM2026/Expert/run_highway.py` records the run when `--validation-events` is given
(off by default; without it nothing is written and the run is unchanged):

```bash
bash AIM2026/Expert/run_highway.sh --seed 124 --validation-events
bash AIM2026/Expert/run_roundabout.sh --variant normal       --seed 124 --validation-events
bash AIM2026/Expert/run_roundabout.sh --variant nonuniform   --seed 124 --validation-events
bash AIM2026/Expert/run_roundabout.sh --variant speed_random --seed 124 --validation-events
bash AIM2026/Expert/run_roundabout.sh --variant brake        --seed 124 --validation-events
bash AIM2026/Expert/run_static.sh --seed 124 --static-obstacle-count 2
```

`run_static.sh` runs one `full_loop_static` segment with the same integrated UDP
Expert as the highway and roundabout launchers. It enables validation events
automatically. Use `--static-obstacle-count 1` or `2` to select the number of
random obstacles; omitting it uses the scenario's existing default of one.
Add `--check` to validate the route and Expert setup without connecting to
MORAI. The actual drive requires MORAI and the same network/sensor setup as
the other launchers.
The static launcher writes a private sensor config and leaves the shared
configuration unchanged. It defaults to a 229-byte local Vehicle Status packet
and left-positive steering when the selected config omits those values.
Set up MORAI sensor UDP addresses for your host and pass a matching AIM2026
config with `--config` when the default `AIM2026/config_local.yaml` does not
match your network. WSL IP addresses and MORAI sensor sets are host-specific.

The run writes `AIM2026/outputs/scenarios/static-*/report.json` and, when a
pipeline journal exists, `failure_report/report.html`. Collision or watchdog
failure Incidents are diagnosed automatically and written under
`MoraiDrive/scenario_runner/.runtime/incidents/incident_*/diagnosis.json` and
`diagnosis.txt`. A clean run has no Incident; its observations and summary are
under `.runtime/validation_runs/<run_id>/`.

- Ego observations, Incidents (collision, Timeout, progress-watchdog stall) go under
  `MoraiDrive/scenario_runner/.runtime/validation_runs` and `.../incidents`.
- The integrated runtime writes the Perception → Signal Gate → Planning → Control journal
  (`pipeline.jsonl`) next to its other driver logs in `AIM2026/outputs/expert/<run>/`, and the
  launcher builds `AIM2026/outputs/scenarios/<run>/failure_report/report.html` when the run ends.
- The same seed repeats the scenario conditions; it does not guarantee the same steering commands or trajectory.
- The report separates OBSERVED / SUSPECTED / INSUFFICIENT_EVIDENCE. VERIFIED needs a recorded counterfactual rerun.

## Incident diagnosis from the command line

Run from `MoraiDrive` without MORAI:

```bash
python3 -m scenario_runner.validation.diagnose scenario_runner/.runtime/incidents/incident_<id>
```

The command writes `diagnosis.json` and `diagnosis.txt` inside the Incident.
New Incidents are diagnosed at collision or tunnel-vehicle Timeout capture and again after the Runner's
`summary.json` is copied into the bundle. A diagnosis exception is reported but
does not change the existing failure handling.

With `--validation-events`, the exact `Tunnel vehicle timed out` failure also
creates a separate `scenario_timeout` Incident. It keeps the same 30-second
Ego/NPC/pipeline window and an empty `validation_events.jsonl` when there was
no collision. Existing collision Incidents retain `collision_started`.

## Evidence and clocks

| Source | Fields used | Clock / join key | Current connection |
|---|---|---|---|
| `ego_trace.jsonl` | position, speed, collision IDs, route progress | run_id, sequence, lap elapsed seconds, host monotonic capture | Connected |
| `actors_trace.jsonl` | NPC position, speed, actor ID | Ego observation sequence and host monotonic capture | Connected; samples about 5 Hz |
| `validation_events.jsonl` | collision started/cleared, actor ID | run_id and lap elapsed seconds | Connected |
| `scenario_result.json` | tunnel NPC spawn, motion, phase | Copied only from matching run filename; SHA256 in manifest | Tunnel vehicle only; collision-time snapshot may precede cleanup |
| `summary.json` | Runner return/exception | source_run_id | Copied into new Incidents after Runner stops |
| `rerun_result.json` | seed/config/weather/NPC placement comparison | source_run_id | Available after rerun |
| `perception_trace.jsonl`, `planning_trace.jsonl`, `control_trace.jsonl` | VE input, path, target speed, computed and sent command | GPS+VE driver run_id, host monotonic timestamp, cycle_id | Connected for GPS+VE runs; other controllers need an adapter |
| Expert `[PLAN]`, `[CTRL]`, `[PP]` | console debug only | No common run_id/frame timestamp in Incident | Not joined or parsed |
| Expert VE object input audit | receipt/source timestamps and ROS sequence | Separate file; no Incident run_id | Not joined |

`autonomous_driving.py` computes local/planned paths, target velocity and
ControlInput. `path_controller.py` consumes the path and target speed and
computes acceleration and front-wheel steering. Expert publishes `/local_path`
and `/ctrl_cmd`. The ordinary Runner can substitute VE object input for Expert,
but it does not run a separate VE + Planning controller. The Incident recorder's
`record_planner_sample()` hook accepts producer-supplied structured samples;
the GPS+VE driver instead supplies its structured pipeline journal. Other
controllers do not call the adapter today.

## Rules and limits

| Check | Current rule | Required evidence |
|---|---|---|
| `runtime.exit` | Report stored status/error; do not infer a root cause | Matching run_id summary |
| `runtime.setup_match` | Report explicit false comparisons as reproduction issues | Matching source_run_id rerun result |
| `scenario.npc` | Report existing tunnel NPC phase/spawn/motion/failure flags | Same-seed scenario result, stored checksum when present |
| `ego.trace_quality` | Validate run_id, contiguous sequence, monotonic capture, finite XYZ and nonnegative speed | Entire Ego trace |
| `ego.stall` | Use only Runner `progress_watchdog` grace, timeout and minimum progress; disabled means not applicable | Same-lap route progress covering the configured interval |
| `collision.event` | Match event actor ID/time/run_id with manifest, then locate Ego/NPC samples by sequence | Complete event trace and matching Ego/NPC rows |
| `timeout.event` | Match the exact tunnel timeout in manifest, same-seed scenario result, and same-run exception summary | All three matching files |
| `timeout.ego_progress` | Report first/last recorded route progress, duration, and final speed; no new stall threshold | Valid same-run Ego trace |
| `timeout.npc_state` | Report the stored NPC gap, planned/commanded speed and configured encounter gap | Same-seed NPC result and scenario config |
| `timeout.plan_control` | Report target speed and transmitted brake from the latest linked Planning/Control cycle | Same run_id, host monotonic clock, cycle_id, sent_ns |
| `timeout.yield_npc_coupling` | Suspect an interaction only when the NPC remains beyond the configured encounter gap, its command speed is below its plan, and repeated Planner yield reasons and transmitted control are logged | Same-run NPC result, config, and linked pipeline records |
| `timeout.ve_uncertainty` | Leave direct VE attribution unresolved even if object motion is marked `uncertain` | Matched VE/NPC state and causal Planning evidence |
| `perception.ve` | Only explicit bad `ready`, `valid`, `stale` at the matching frame is observed | Same run_id, observation_sequence and clock_domain; otherwise insufficient |
| `planning.path` | Missing/invalid 2-point XY path is suspected only at the exact matching frame | Same run_id, observation_sequence and clock_domain |
| `control.response` | No command-to-response judgment yet | Published command, target speed, matched Ego frame, established response criterion |
| `planning.occupancy` | No path/object intersection judgment yet | Same-frame path, obstacle footprint, coordinate frame and time |

No new thresholds are applied to perception freshness, command response, or
obstacle occupancy. A collision alone is evidence of contact, not a Planning
or Control diagnosis. `OBSERVED` denotes stored facts or data defects;
`SUSPECTED` is reserved for rule-based candidates. Missing data is
`INSUFFICIENT_EVIDENCE`, and disabled/nonparticipating modules are
`NOT_APPLICABLE`.

## Evidence report (report.html)

```bash
python3 -m scenario_runner.validation.failure_report <run_dir_or_incident_dir> [--out DIR] [--window-sec 30]
python3 -m scenario_runner.validation.diagnose <incident_dir> --html     # same report for an Incident
```

`<run_dir>` is a Runner output folder containing `driver/pipeline.jsonl` (VE+GPS driver) and, for the
tunnel-vehicle scenario, `tunnel_vehicle_runs/<run_id>.json` and `report.json`. Output goes to
`<source>/failure_report/report.html` and `report_data.json`. The 87 MB journal is read once on the
host; the page embeds only reduced series, BEV frames, findings and the cited log rows (about 0.4 MB).

* Failure types: `collision` (Incident manifest), `timeout` (NPC `failure_reason` / Incident), `stall`
  (analysis rule: trailing 30 s path length / time < 0.5 m/s for >= 30 s; not a Runner policy).
* Perception/Planning/Control are joined by `run_id` + `cycle_id` on `host_monotonic_ns`; every
  citation is `file:line` and opens the row excerpt plus a command to find it.
* Statuses: `OBSERVED` stored fact/summary statistic, `SUSPECTED` rule-based candidate,
  `INSUFFICIENT_EVIDENCE`, and `VERIFIED` only when a verification record
  (`kind=counterfactual_rerun`, `same_seed_and_setup`, `outcome_changed`, `evidence_file`) names the finding.
  None is produced automatically.
* GT NPC series are drawn only if the frame/clock alignment check passes (pipeline Ego vs Runner
  `route_gap_m` at shifted clocks). VE `track_id` is never equated with GT `actor_id`.
* Existing `diagnose_pipeline`/`diagnose_timeout` checks are included unchanged (they judge only the
  final cycle); thresholds used by new hypotheses are listed in section 6 of the report.

## Stage diagnosis (signal path) and the summary-first report

`validation/stage_diagnosis.py` joins `driver/control.jsonl` (Gate output + `signal_gate` Audit), `driver/inputs.jsonl`
(VE head output), `driver/pipeline.jsonl` and, for validation only, `signal_truth.jsonl` (clock fitted on Ego motion).
Rules are independent `@rule` functions: GREEN_BUT_STOPPED, PERCEPTION_UNAVAILABLE, SIGNAL_HEAD_MISMATCH,
SIGNAL_PROPAGATION_LOSS, SIGNAL_GATE_BLOCK_REASON, SIGNAL_DEBOUNCE_INSUFFICIENT, SIGNAL_STALE_REJECTED,
SIGNAL_PERMISSION_RESET, PLANNER_GATE_CONSISTENCY, CONTROL_PLAN_CONSISTENCY, CONTROL_RESPONSE_MISMATCH and the
composite SIGNAL_GATE_CAUSE_CHAIN. Gate rejection reasons come from the Audit when the run has it (OBSERVED); older
runs fall back to replaying the real `SignalPolicy` over the logged packets (reconstruction, shown as SUSPECTED).
Code 0 means "unconfirmed" and is never counted as red. VERIFIED still needs a counterfactual rerun record.

`report.html` order: Executive Summary (Korean headline, termination / behavior / cause candidates, INSUFFICIENT folded)
-> Stage Diagnosis (Perception -> Signal Gate -> Planning -> Control -> Ego, ok / problem / unverified per stage,
click for evidence + chain graph, drag the graph for a window summary) -> timeline, time series, BEV -> details.
