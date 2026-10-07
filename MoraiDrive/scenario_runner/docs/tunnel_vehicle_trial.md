# Whole-tunnel vehicle behaviors

Both tunnel modes start at CP14 (global path index 3768, link
`A2256W000448`) and end at CP15 (index 4045). Vehicle mode requires `--once`.
It excludes pedestrian missions, highway SpawnPoints and the legacy tunnel
brake event. The general `full_loop` retains its original brake event.

## Plans and actor lifetime

One run selects one behavior and spawns one vehicle during setup, before Ego
starts driving. It uses the highway helper's existing route, waypoint, speed
limit and velocity APIs, retaining MORAI's default NPC AI. Live trials of the
previous slow-vehicle implementation reported 4 m/s motion despite a rejected
`SetAI` command, so this mode does not toggle AI merely to confirm startup.
Actual motion remains a required observation, independently of command ACKs.

The four pedestrian sections are reused as **event anchors**, not short
vehicle routes. Their centers are 24.636, 53.908, 83.181 and 112.453 m from
entry. No section boundary destroys or replaces the actor.

| Behavior | Spawn and continuous interaction | Completion |
| --- | --- | --- |
| `vehicle_slow` | Same lane, initially 40 m ahead; 4 m/s to exit | Ego goal and observed NPC exit/travel |
| `vehicle_sudden_brake` | Same lane, initially 40 m ahead; 9 m/s, slow approach if needed; brake in a safe section and at a 20-35 m gap | Observed stop, Ego reaction, hold, resume motion and NPC exit |
| `vehicle_stopped` | Pre-positioned in section 2 or 3; zero speed and position verified | Ego passes with at least 15 m route and physical clearance |
| `vehicle_oncoming` | Actual opposite lane near exit, 6 m/s toward entry | Observed approach/travel and pass with at least 15 m clearance |
| `vehicle_cut_in` | Legal adjacent same-direction route, then an allowed MGeo lane-change edge to Ego's route | Same ID, observed target link and lateral movement, then exit |

The default initial gap is 40 m and its minimum is 30 m. Both intended and
actual gaps use the actual Ego position at CP14. Spawn positions and yaw come
from real MGeo waypoints near the official route's planned s position. No new
world XYZ coordinates are hardcoded. An invalid Ego start fails before spawn.

Sudden braking reuses `tunnel.py`'s hard-brake/resume helpers and the extracted
`observe_tunnel_braking()` helper. Stop confirmation, forced-stop fallback,
Ego speed-drop observation, emergency **resume**, hold and resume settings are
inherited from the legacy configuration. A missed safe brake window fails;
it does not force an unsafe brake or replace the vehicle. With the default
40 m gap and 20 m minimum run-in, only section 3 qualifies for moving events;
stopped vehicles can use section 2 or 3.

Moving NPCs remain until the exit. If Ego finishes earlier after the behavior
was verified, Ego stops at CP15 and the runner continues observing that same
NPC until its exit or timeout. Stopped/oncoming actors may be removed earlier
only after a verified pass and clearance. Failure and Ctrl+C save a failed
report and attempt to remove the owned actor. A rejected destroy is reported
and leaves it tracked for the runner's teardown retry.

## Actual map evidence and cut-in limitation

The opposite route is:

`A2256W000131 -> A2256W000129 -> A2256W000028`

MGeo nodes connect `840 -> 843 -> 004 -> 008`. The core tunnel links are 131
and 129; 028 continues beyond the entrance. Samples of link tangents projected
onto the official route have dot products -0.99999 to -0.99931, and lateral
separations 3.445-3.597 m. Route connectivity, lane separation and direction
are checked offline; observed heading/progress and lane are checked live.
The same-direction tunnel core remains `448 -> 126` with its exit continuation.

**The current map cannot perform vehicle_cut_in in this tunnel.** Inspection
of nearby MGeo links found no same-direction adjacent lane or permitted lateral
edge to 448/126. The separated lane at approximately +3.5 m is the opposite
lane, not a same-direction cut-in lane. The adapter/SDK has no targeted native
lane-change command; a SpawnPoint's lane-change flag does not provide one.
Explicit cut-in selection fails offline before MORAI starts. Random selection
logs the exclusion. No reversed forward route or transform teleport is used.

For a map with a legal same-direction adjacent edge, the implemented cut-in
handler first assigns only the adjacent route, then assigns the permitted
adjacent-to-target route to the **same actor** at the safe event region/gap.
It checks actual target link, lateral displacement and continuous movement.
A synthetic MGeo fixture tests this successful path and failed transitions.
Actual K-city cut-in needs a suitable map topology or a supported targeted
native command before it can be enabled.

## Commands

From `MoraiDrive`, inspect the full behavior plans without MORAI:

```bash
python -m scenario_runner.tools.generate_tunnel_vehicle_candidates --output .runtime/tunnel_vehicle_candidates.json
python -m scenario_runner.main_runner --scenario full_loop_tunnel_vehicle --once --tunnel-vehicle-behavior vehicle_slow --seed 123 --check
```

Change the behavior argument to `vehicle_sudden_brake`, `vehicle_stopped`,
`vehicle_oncoming` or `vehicle_cut_in`. The last one must reject this map with
the explicit topology reason above. `--check` invokes the vehicle validator;
model availability and observed motion still require live MORAI.

Expert trial (the default controller):

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --no-sync-runtime --tunnel-vehicle-behavior vehicle_slow --seed 123
```

Manual Ego and seeded random behavior/model:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --manual-ego --tunnel-vehicle-behavior vehicle_oncoming --seed 123
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --manual-ego --tunnel-vehicle-random --seed 123
```

One random behavior is selected per run; sections do not select different
vehicles. Behavior/event selection uses the scenario RNG. Its state is frozen
before weather selection for the later model draw, so weather history does not
alter the vehicle choices. The same seed, configuration, map and available
model pool reproduce behavior, event section and model. The obsolete
`--tunnel-vehicle-candidate-id` option is removed. Both tunnel modes permit
`--manual-ego` only with `--once`.

## Reports and verification

Reports: `scenario_runner/.runtime/tunnel_vehicle_runs/<run_instance_id>.json`.
They include seed, behavior, event section, model, actor ID, route/waypoint
plan, actual/intended spawn, gap, phase transitions, observations, travel,
brake/pass/lateral metrics, continuity, cleanup result and failure reason.
Finite state, expected link/lane, direction, backwards progress, implausible
steps, timeout and actual completion are validated. ACKs cannot establish
braking, arrival, lane-change completion or safe Ego response.

```bash
python -m unittest scenario_runner.tests.test_tunnel_vehicle_dynamic scenario_runner.tests.test_tunnel_dynamic
python -m unittest discover -s scenario_runner/tests -t .
```

MORAI live trials must still verify available model placement, height, actual
speed, continuous route following, brake/hold/resume, stationary vehicle
avoidance, oncoming lane clearance, Expert response and actor cleanup. Offline
passing tests do not approve simulator behavior or certify a safe encounter.
