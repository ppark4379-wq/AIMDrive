# Whole-tunnel vehicle behaviors

Both tunnel modes start at CP14 (global path index 3768, link
`A2256W000448`) and end at CP15 (index 4045). Vehicle mode requires `--once`.
It excludes pedestrian missions, highway SpawnPoints and the legacy tunnel
brake event. The general `full_loop` retains its original brake event.

## Plans and actor lifetime

Setup selects the primary behavior/model and arms its scheduler without
creating a vehicle. Expert or keyboard Ego driving starts first. Native
highway traffic similarly activates when Ego approaches its traffic region;
tunnel traffic keeps direct vehicle handles for behavior control.

Every runtime tick checks Ego's measured km/h speed (converted to m/s) and
route progress. The default gate is **Ego speed >= 2 m/s OR progress >= 3 m
from CP14**, with actual speed additionally above the 0.6 m/s NPC motion floor.
A stationary Ego never spawns a moving NPC, even after passing the progress
gate. Waiting at CP14 does not consume the actor's movement/run timeout.

For forward traffic, the spawn position is recalculated as Ego's current
route s plus a gap of at least 40 m, 30 m minimum, and its reaction/braking
stopping distance. The actual world pose and yaw come from the nearest MGeo
waypoint, rather than fixed XYZ. A spawn must leave space for the encounter,
event run-in and exit; a missed window fails instead of creating a vehicle
past its event. The offline catalog's spawn pose is only a preview, marked
`spawn_preview_only`; live reports contain the runtime pose and timing.

The helper issues **spawn -> route/waypoint -> unpause -> speed limit ->
velocity**. `configure_ai=False` and `unpause=True` are independent, retaining
MORAI's default NPC AI without the unsupported `SetAI` command. Legacy helper
callers keep their defaults. Subsequent runtime ticks verify actual speed
above 0.5 m/s **or** at least 0.5 m displacement within 2 seconds. Tick handling
never sleeps while awaiting startup, including during native frame stepping.
ACKs do not satisfy this check. Failed motion immediately cleans up the actor.
Resume retains the existing unpause/speed restore and actual motion check.

Slow vehicles spawn at no more than 60% of Ego's measured speed, capped at
4 m/s and further reduced to allow Ego to catch them before the exit margin.
They slow further if Ego slows before encountering them and remain in the
route. An observed route and physical gap <= 20 m records the encounter;
exit without an encounter fails. Sudden-brake approach speed follows Ego
(up to 9 m/s) before the event, then reduces to close the safe braking gap.
After the verified interaction, vehicles continue to outside cleanup.

Oncoming vehicles use actual opposite-lane route distance from their exit
spawn to a section 2/3 interaction anchor to calculate NPC ETA. They spawn
only when Ego ETA and NPC ETA differ by at most 1.5 seconds and the initial
gap is sufficient. A slow Ego therefore meets the oncoming vehicle inside
the tunnel rather than arriving after it has already driven away.

The four pedestrian sections are reused as **event anchors**, not short
vehicle routes. Their centers are 24.636, 53.908, 83.181 and 112.453 m from
entry. No section boundary destroys or replaces the actor.

| Behavior | Spawn and continuous interaction | Completion |
| --- | --- | --- |
| `vehicle_slow` | Runtime same-lane safe gap; Ego-relative speed capped at 4 m/s | Observed travel past CP15, outside continuation and Ego clearance |
| `vehicle_sudden_brake` | Runtime same-lane safe gap; Ego-relative cruise capped at 9 m/s, slow approach if needed; brake in a safe section and at a 20-35 m gap | Observed stop, Ego reaction, hold, resume motion and outside continuation |
| `vehicle_stopped` | Runtime safe gap at Ego-relative speed capped at 9 m/s; stop at section 2 or 3 anchor and verify stationary state | Ego passes with at least 15 m clearance; actor remains stationary until teardown |
| `vehicle_oncoming` | ETA-gated actual opposite lane near exit, 6 m/s toward entry | Observed approach/travel and pass; continue past CP14 to entrance-side cleanup |
| `vehicle_cut_in` | Legal adjacent same-direction route, then an allowed MGeo lane-change edge to Ego's route | Same ID, observed target link and lateral movement, then exit continuation |

The default initial gap is 40 m and its minimum is 30 m. Both intended and
actual gaps use Ego's current position at runtime spawn. Spawn positions and yaw come
from real MGeo waypoints near the official route's planned s position. No new
world XYZ coordinates are hardcoded. Invalid runtime Ego position/progress fails before spawn.

Sudden braking reuses `tunnel.py`'s hard-brake/resume helpers and the extracted
`observe_tunnel_braking()` helper. Stop confirmation, forced-stop fallback,
Ego speed-drop observation, emergency **resume**, hold and resume settings are
inherited from the legacy configuration. A missed safe brake window fails;
it does not force an unsafe brake or replace the vehicle. With the default
40 m gap and 20 m minimum run-in, only section 3 qualifies for moving events;
stopped vehicles can use section 2 or 3.

Interaction completion and actor lifetime are separate report fields:
`interaction_complete` and `actor_active`. Crossing a section boundary or
passing Ego never removes the actor. Moving vehicles continue on their real
MGeo routes to a cleanup point **40 m of NPC route distance beyond CP15**
(forward behaviors), or **40 m beyond CP14** (oncoming). Removal also requires
at least **30 m physical separation from Ego**. Cleanup uses the NPC's own
route geometry; the Ego segment projection cannot measure reverse movement
past CP14. The actual map has approximately 188 m of forward continuation
and 53 m of entrance-side opposite continuation.

If Ego finishes earlier, it stops at CP15 and the runner observes that same
moving NPC until outside cleanup or timeout. A stopped vehicle first cruises
to its selected section anchor, then receives `stop_vehicle`; actual speed
and position verify its stationary state. After Ego passes, it remains
stationary, including after interaction completion, until scenario teardown.
Teardown then removes it and updates the report. Failure and Ctrl+C attempt
immediate cleanup. A rejected destroy is recorded and the actor stays tracked
for the runner's cleanup retry. Removal uses the adapter's acknowledged
`destroy_vehicle_by_id`, because the SDK's `Actor.destroy()` returns `None`.

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

Random selection chooses the primary behavior; sections do not replace its
vehicle. `tunnel_vehicle_dynamic.additional_behaviors` can schedule separate
traffic actors (for example `[vehicle_oncoming]` with a primary slow vehicle).
Each actor has its own plan, model draw, state, ID, timing and report, shares
the world/ownership list, and remains until its own lifecycle ends. Occupied
spawn locations defer additional traffic. There is no global one-vehicle
limit. Failure/teardown attempts cleanup of every owned traffic actor.

Behavior/event selection uses the scenario RNG. Its state is frozen before
weather selection for the later model draw, so weather history does not alter
vehicle choices. The same seed, configuration, map and available model pool
reproduce the selected behaviors, events and models. Runtime spawn positions,
speeds and timing additionally depend on the observed Ego trajectory. The
obsolete `--tunnel-vehicle-candidate-id` option is removed. Both tunnel modes
permit `--manual-ego` only with `--once`.

## Reports and verification

Reports: `scenario_runner/.runtime/tunnel_vehicle_runs/<run_instance_id>.json`.
Additional actors use `<run_instance_id>_traffic_<index>.json`.
They include seed, behavior, event section, model, actor ID, route/waypoint
plan, actual/intended spawn, gap, phase transitions, observations, travel,
spawn gate/ETA/actual Ego speed, commanded NPC speed, encounter evidence,
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
