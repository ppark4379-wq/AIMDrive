# Tunnel dynamic obstacle trial

`full_loop_tunnel_dynamic` drives only checkpoint 14 to checkpoint 15. It
enables one fixed pedestrian path in the tunnel: A -> B follows the side of
the road, then B -> C crosses Ego's route. The path is stored under
`zones.tunnel.dynamic_paths` in `config/full_loop_links.yaml`. The scenario
settings in `config/full_loop.yaml` fix the model, speed, trigger distance,
arrival tolerance, and timeout. No model, speed, route, or actor count is
randomized in this first trial.

`TunnelDynamicHandler.validate()` checks the route, path, and numeric settings
before MORAI starts. `arm()` checks model availability after the world starts.
`update()` waits until Ego reaches the fixed trigger, spawns one pedestrian,
commands the B waypoint, switches to the C waypoint at B, and stops and removes the actor at
C. The run fails if the pedestrian times out or Ego reaches the goal first.
The gRPC adapter receives the chosen spawn and waypoint commands. The tunnel
passes `speed_mps` unchanged to the waypoint command; the existing general
`control_pedestrian()` API is only used to stop the actor at C.

The global `full_loop` mode keeps `tunnel_dynamic.enabled: false` and retains
the existing tunnel sudden-brake event. Other segment modes are gated out even
if this setting is enabled in a local config.

From the `MoraiDrive` directory, a manual trial uses:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --no-sync-runtime
```

To drive Ego yourself while the scenario still spawns and tracks the
pedestrian, run:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --manual-ego
```

`--manual-ego` disables Sync Runtime and does not launch the expert. MORAI
switches Ego to Keyboard control; focus the simulator window, check that its
Ego Controller shows Manual-Keyboard, and drive with W/A/S/D. Gear D is set
at startup. The pedestrian still spawns when Ego reaches the configured
route trigger. Edit A/B/C in `config/full_loop_links.yaml` and restart the
trial to compare placements. Use Ctrl+C to stop the runner and remove its
spawned actor.

## Route-relative straight-path candidates

`zones.tunnel.candidate_catalog` in `config/full_loop_links.yaml` defines
three distances from checkpoint 14, a left/right offset, and path lengths.
The offline generator combines each position with five straight behaviors:
same-direction, opposite-direction, left-to-right crossing, right-to-left
crossing, and diagonal crossing. Positive `d` means left of the official
route heading. It interpolates route XYZ at `s` and applies the lateral
offset to XY; Z remains the route height estimate.

From `MoraiDrive`, generate a JSON file for review without starting MORAI:

```bash
python -m scenario_runner.tools.generate_tunnel_dynamic_candidates --output .runtime/tunnel_dynamic_candidates.json
```

Each generated path has an ID, behavior, start/end `(s,d)`, start/end XYZ,
and a review status. All 15 current paths have `candidate` status. After a
live check, add IDs to `approved_ids` or `disabled_ids` in the YAML catalog.
The default runner still executes the fixed A -> B -> C path. A candidate
runs only when its ID is explicitly selected in a manual trial. Random
selection is not enabled. `walk_then_cross` remains the separate multi-stage
behavior.

Start with a same-direction trial to check placement and walking speed:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --manual-ego --tunnel-candidate-id straight_same_direction_01
```

Then test a crossing with another explicit ID, for example
`cross_left_to_right_01`. The runner starts Ego at checkpoint 14 and spawns
the pedestrian at the candidate start with the start-to-end yaw and native
walking velocity when Ego reaches the trigger. This follows the existing
general dynamic-obstacle movement: no waypoint RPC is used for straight
candidates. The runner tracks the end position and removes the actor there. With
`trial_trigger_before_start_m: 10.0`, the first same-direction candidate
spawns about 15 m after checkpoint 14. Check the `trial candidate=` log for
the selected ID and world coordinates, `method=native_straight` for the spawn
method, periodic `actor=(x,y) remaining=...` logs for actual motion, and
`reached final waypoint` for completion. If MORAI reports an impossible
position jump, the runner logs `implausible jump` and keeps the actor until
shutdown for inspection; that path is not marked complete. Observe in
MORAI whether the pedestrian appears on the road surface, walks in the
expected direction and speed, and clears walls and traffic lanes. A timeout
or failed waypoint command is a failed trial. Mark a reviewed ID in
`approved_ids` or `disabled_ids`, then regenerate the JSON to see its status.
Disabled IDs cannot be selected for a trial.

Live trials of both `SetPedestrianWaypoint` and waypoint-inclusive spawning
reported the end coordinate immediately (15.11 m in 0.07 s) and left the actor
standing there. Straight candidates therefore use native spawn movement.
The fixed A -> B -> C trial still uses waypoint commands for its turns and
needs separate live validation.

The A/B/C coordinates are calculated from the official route and are **trial
candidates**. Their surface height, the crossing clearance, and the gRPC
pedestrian control speed unit still need a live MORAI check. Offline tests
verify path validation and state transitions; they cannot establish that the
simulator follows the commanded path or that Ego responds safely. Keep this
mode for controlled trial runs until those observations are recorded.
