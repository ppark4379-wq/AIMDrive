# Tunnel dynamic obstacle trial

`full_loop_tunnel_dynamic` drives only checkpoint 14 to checkpoint 15. It
enables one fixed pedestrian path in the tunnel: A -> B follows the side of
the road, then B -> C crosses Ego's route. The path is stored under
`zones.tunnel.dynamic_paths` in `config/full_loop_links.yaml`. The scenario
settings in `config/full_loop.yaml` fix the model, speed, interaction timing,
arrival tolerance, and timeout. No model, speed, route, or actor count is
randomized in this first trial.

`TunnelDynamicHandler.validate()` checks the route, path, and numeric settings
before MORAI starts. `arm()` checks model availability after the world starts.
`update()` waits until Ego/NPC ETA and braking-distance conditions pass, spawns one pedestrian,
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
at startup. The pedestrian still spawns when Ego's approach satisfies the
interaction timing conditions. Edit A/B/C in `config/full_loop_links.yaml` and restart the
trial to compare placements. Use Ctrl+C to stop the runner and remove its
spawned actor.

## Expert driving

The default controller is `aimdrive_expert`. The runner imports the current
`expert.src.autonomous_driving.AutonomousDriving` from the configured external
Expert, including its Jamming planner. Omit `--manual-ego` to let it drive Ego:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --no-sync-runtime --tunnel-random-mix
```

For a single path with Expert driving:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --no-sync-runtime --tunnel-candidate-id straight_opposite_02
```

Both explicit trial options set `candidate_policy: trial`, so existing
`candidate` paths can be tested with Expert as well as keyboard driving.
Disabled paths stay excluded, and testing does not change any approval status.
Add `--seed 12345` to replay a mix. `--no-sync-runtime` runs the same asynchronous
trial setup used previously, while the Expert sends Ego controls through ROS.
The runner switches Ego to Auto control and launches the Expert process.
`[KCityFullLoop] using external AIMDrive expert process` confirms this path;
the tunnel controller/policy and schedule are logged and saved to the run report.
Offline tests check process launch and route export through a mocked MORAI
boundary; actual obstacle avoidance still needs a live trial.

## Route-relative straight-path candidates

`zones.tunnel.candidate_catalog` in `config/full_loop_links.yaml` defines
`division_count: 4`, an entry/exit margin, a left/right offset, and path lengths.
The generator excludes `endpoint_margin_m` from each end, divides the usable
route length equally, and uses each section midpoint as the behavior center.
Start/end distances lie symmetrically around that center; opposite-direction
paths reverse their start/end order. Every path must fit inside its section.
Straight paths span 20 m along the route. Crossing paths span 5 m along
the route and 5 m laterally (about 7.1 m total); diagonals span 10 m along
the route and 5 m laterally (about 11.2 m total). World-space lengths vary
slightly with route curvature.
The current 137.1 m tunnel with 10 m margins has centers about 24.6, 53.9,
83.2, and 112.5 m from checkpoint 14. These update automatically with the
official tunnel route length. Each center has five straight behaviors:
same-direction, opposite-direction, left-to-right crossing, right-to-left
crossing, and diagonal crossing. Positive `d` means left of the official
route heading. It interpolates route XYZ at `s` and applies the lateral
offset to XY; Z remains the route height estimate.

Opposite-direction pedestrians alternate lanes by section: sections 1/3 use
`opposite_lane_1_offset_m: 2.5`, and sections 2/4 use
`opposite_lane_2_offset_m: 0.0`. Their start `s` stays greater than their end
`s`, so both lanes move against Ego's route direction. These lane offsets
are separate from `side_offset_m`. The +2.5 m placement was observed inside
lane 1 in the manual MORAI trial. The 0 m placement follows Ego's global path
and is a lane 2 trial candidate; confirm the lane position in MORAI with
`--tunnel-candidate-id straight_opposite_02`. These values are offsets from
the Ego path, not surveyed lane centerlines.

From `MoraiDrive`, generate a JSON file for review without starting MORAI:

```bash
python -m scenario_runner.tools.generate_tunnel_dynamic_candidates --output .runtime/tunnel_dynamic_candidates.json
```

Each generated path has an ID, behavior, section index/bounds, center distance,
start/end `(s,d)`, start/end XYZ, and a review status. All 20 current paths have
`candidate` status. IDs end in `_01` through `_04` for the four sections. After a
live check, add IDs to `approved_ids` or `disabled_ids` in the YAML catalog.
The default runner still executes the fixed A -> B -> C path. A candidate
runs only when its ID is explicitly selected in a manual trial. Random
selection is enabled with the random-mix option below. `walk_then_cross` remains the separate multi-stage
behavior.

Start with a same-direction trial to check placement and walking speed:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --manual-ego --tunnel-candidate-id straight_same_direction_01
```

Then test a crossing with another explicit ID, for example
`cross_left_to_right_01`. The runner starts Ego at checkpoint 14 and spawns
the pedestrian at the candidate start with the start-to-end yaw and native
walking velocity when Ego reaches the interaction window. This follows the existing
general dynamic-obstacle movement: no waypoint RPC is used for straight
candidates. The runner tracks the end position and removes the actor there. The first
same-direction candidate starts about 17.1 m after checkpoint 14; its
interaction center is about 24.6 m. Its spawn position depends on Ego speed
and ETA. Check the `trial candidate=` log for
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

## Seeded random behavior and model mix

From `MoraiDrive`, run four sequential events with keyboard-controlled Ego:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --manual-ego --tunnel-random-mix
```

The default `tunnel_dynamic.random_mix` policy picks four different behaviors
from the five configured behaviors, assigning one to each section in order.
Each event independently picks an available model from `dynamic_obstacle_models`.
This is the same pool as the general dynamic scenario, including its animal
models. Set `random_mix.models` to a nonempty list to restrict the model pool.
Model repeats are allowed. `allow_behavior_repeat: false` prevents a behavior
from repeating anywhere in the run; true permits repeats. Currently only
`events_per_section: 1` is supported. All events use `speed_mps: 2.5`.

Replay a logged seed with the same candidate configuration and model availability:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_dynamic --once --manual-ego --tunnel-random-mix --seed 12345
```

`random mix seed=... schedule=[...]` logs every selected candidate ID and model
before the first event. Reusing the seed reproduces those choices if the catalog,
policy, and available model pool are unchanged. The report at
`scenario_runner/.runtime/tunnel_dynamic_runs/<run_instance_id>.json` records
the seed, model pool, selected coordinates/models, trigger distances, and each
event's completion or failure. A Ctrl+C stop records an incomplete run.

Only one actor is active at a time: its end is checked and the actor is removed
before the next section is armed. Slow down or stop Ego to observe an event;
missing the next interaction window before its actor can spawn fails the run. A timeout,
spawn/control failure, or impossible position jump stops a mix without skipping
to a later event. The opposite-direction paths retain their section-based lane
alternation. `--tunnel-candidate-id` and `--tunnel-random-mix` are mutually exclusive.

Explicit CLI trials may use unapproved `candidate` paths with either controller.
Disabled paths are excluded. YAML-only Expert mixes default to the `approved`
candidate policy; set `tunnel_dynamic.candidate_policy: trial` to use candidates
there as well. A valid mix must cover every section and meet the configured
behavior-repeat rule. Random selection does not approve any path or model/path
combination. Enable the policy in YAML with `random_mix.enabled: true`.

## Interaction timing

If Expert stays near zero speed before any pedestrian spawns, check the
`[Expert] morai_msgs=...` startup line. The runner prefers the configured Expert's
`devel/lib/python3/dist-packages` message build over inherited ROS overlays.
An older `~/morai_ws` `EgoVehicleStatus` definition has a different MD5 from the
current bridge and blocks `/Ego_topic` reception, so Expert sends no controls.
Restart the runner after updating the launcher; an existing Expert process keeps
its already imported message definitions.

Every tunnel path uses the same timing calculation as general dynamic obstacles.
For crossings and diagonals, the interaction point is the actual intersection
between the commanded pedestrian segment and Ego's route polyline. For straight
same/opposite paths, the section anchor is the Ego interaction position and the
pedestrian segment midpoint supplies its approach distance. This is an encounter
reference; a parallel pedestrian in another lane need not physically intersect Ego.
The A -> B -> C trial includes A -> B travel in the NPC ETA to the B -> C crossing.

On every waiting tick:

```text
npc_eta = start_to_interaction_m / pedestrian_speed_mps
safe_stop = ego_speed * reaction_time + ego_speed^2 / (2 * deceleration) + margin
trigger_distance = min(maximum, max(minimum, ego_speed * (npc_eta + lead), safe_stop + 1))
eta_delta = remaining_route_distance / ego_speed - npc_eta
```

The defaults inherit `dynamic_obstacle` settings: lead 1.5 s, distance range
12..60 m, Ego speed at least 2 m/s (7.2 km/h), ETA delta 0..3 s, reaction
1 s, deceleration 3.5 m/s², and stopping margin 3 m. Individual overrides go
under `tunnel_dynamic.interaction_trigger`; an empty mapping inherits those
general settings. The old `trial_trigger_before_start_m` and
`spawn_trigger_before_crossing_m` settings are no longer used.

A stopped or slow Ego waits without spawning. Once moving, Ego must be inside
the computed trigger distance, with enough braking distance and a matching ETA.
An early ETA waits; an unsafe braking distance, late NPC arrival, or a passed
interaction fails the run without spawning. These conditions cannot guarantee
that four sequential events fit a continuously fast drive: slow down or stop
to let the active pedestrian finish before approaching the next section.

`waiting` and `TRIGGER` logs show remaining distance, Ego/NPC ETA and stopping
distance. The run JSON records the interaction geometry, effective timing settings,
actual trigger Ego position, and timing snapshot. The candidate catalog JSON
continues to hold path geometry; it does not encode a fixed spawn threshold.

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
