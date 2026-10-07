# Whole-tunnel vehicle behaviors

Both tunnel modes start at CP14 (global path index 3768, link
`A2256W000448`) and end at CP15 (index 4045). Vehicle mode requires `--once`.
It excludes pedestrian missions, highway SpawnPoints and the legacy tunnel
brake event. The general `full_loop` retains its original brake event.

## Expert policy

Vehicle trials export `planning.jamming.enabled: true` to the Expert runtime
configuration. Expert uses the same Jamming avoidance planner as a standalone
launch: pass a slow lead when an adjacent corridor and the return are safe,
and keep a following gap when a safe pass is unavailable. The planner checks
moving obstacles during the pass and return; emergency braking is reevaluated
as traffic changes. Static avoidance retains its existing policy.

The private Expert route includes 35 m of the original downstream path after
CP15, so planning can validate the Jamming rejoin and steering tail. The
scenario's CP14-to-CP15 mission boundaries and actor cleanup remain unchanged.
Other missions retain their existing Jamming configuration.

## Plans and actor lifetime

Setup selects the primary behavior/model and arms its scheduler without
creating a vehicle. Expert or keyboard Ego driving starts first. Native
highway traffic similarly activates when Ego approaches its traffic region;
tunnel traffic keeps direct vehicle handles for behavior control.

Every runtime tick checks Ego's measured km/h speed (converted to m/s) and
route progress. NPC `ActorState.velocity` from the live K-city SDK is km/h:
`npc_state_velocity_unit: kmh` converts its vector magnitude to m/s exactly
once in the tunnel traffic reader. Native NPC velocity/limit arguments remain m/s;
cut-in external control converts m/s to control km/h;
`mps` is available for an SDK supplying that telemetry unit. Samples preserve
raw magnitude, declared unit and displacement-derived speed for comparison.
The legacy tunnel/roundabout readers are unchanged. The default gate is **Ego speed >= 2 m/s OR progress >= 3 m
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

For immediate traffic, the helper issues **spawn -> route/waypoint -> unpause
-> speed limit -> velocity**. Sudden-brake departure waiting instead uses an
atomically paused spawn, a verified actor pause, route/waypoint and zero speed
commands; unpause and cruise commands occur only on the departure trigger. `configure_ai=False` and `unpause=True` are independent, retaining
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
exit without an encounter fails. Sudden-brake vehicles cruise at a fixed
4 m/s; their speed is not adjusted to manufacture a closing gap. They wait
for Ego to reach at least cruise + 0.5 m/s before runtime spawn so Ego can
naturally catch them; the paused departure stage below keeps them in place
until Ego approaches. After the verified interaction, vehicles continue
to outside cleanup.

Oncoming vehicles use actual opposite-lane route distance from their exit
spawn to a section 2/3 interaction anchor to calculate NPC ETA. They spawn
only when Ego ETA and NPC ETA differ by at most 1.5 seconds and the initial
gap is sufficient. A slow Ego therefore meets the oncoming vehicle inside
the tunnel rather than arriving after it has already driven away.

The four pedestrian sections are reused as **event anchors**, not short
vehicle routes. Their centers are 24.636, 53.908, 83.181 and 112.453 m from
entry. No section boundary destroys or replaces the actor.

Spawn locations now receive a seed-based offset in each NPC's travel direction:
`tunnel_vehicle_dynamic.spawn_forward_offset_range_m: [0.0, 10.0]`.
The offset is drawn once per plan before weather selection and stays fixed while
waiting for Ego. Slow/braking traffic starts from the live safe Ego gap and adds
0-10 m forward along Ego's route. Oncoming traffic shifts from its exit-side
baseline toward the tunnel entrance by 0-10 m, retaining the opposite lane;
its NPC ETA is recalculated from that shifted position before spawn.

Offsets use MGeo waypoints and can differ slightly after waypoint snapping.
Near the exit or an event boundary, the forward shift is reduced to retain
startup/run-in and the braking, hold, resume and encounter distance budgets.
`[0.0, 0.0]` disables the shift. Reports/logs include the requested offset,
applied offset and nominal/actual route position. Seed replay also requires the
same Ego trajectory for forward traffic's live baseline.

| Behavior | Spawn and continuous interaction | Completion |
| --- | --- | --- |
| `vehicle_slow` | Runtime same-lane safe gap; Ego-relative speed capped at 4 m/s | Observed travel past CP15, outside continuation and Ego clearance |
| `vehicle_sudden_brake` | Runtime safe gap; paused spawn, Ego-approach departure, 4 m/s cruise, stable following in section 2/3, immediate 0.5 m/s command | Actual deceleration, 8-second observed trough hold, 4 m/s resume and outside continuation |
| `vehicle_oncoming` | ETA-gated actual opposite lane near exit, 6 m/s toward entry | Observed approach/travel and pass; continue past CP14 to entrance-side cleanup |
| `vehicle_cut_in` | Opposite-lane geometry, Ego-direction spawn, direct steering across the centerline, then forward route | Same ID, settled target-lane entry and observed native forward-route motion, then exit continuation |

The default initial gap is 40 m and its minimum is 30 m. Both intended and
actual gaps use Ego's current position at runtime spawn. Spawn positions and yaw come
from real MGeo waypoints near the official route's planned s position. No new
world XYZ coordinates are hardcoded. Invalid runtime Ego position/progress fails before spawn.

### Stable following and sudden braking

`tunnel_vehicle_brake.py` implements the new event separately from the legacy
`tunnel.py` brake event. It follows roundabout's trough-hold/resume pattern;
roundabout's NPC-to-NPC following gap values are not used here.
`tunnel_vehicle_dynamic.sudden_brake` contains all event thresholds.

Before cruising, sudden-brake traffic now starts with `pause: true` and zero
spawn/command speed. The same actor already has its forward route and waits in
`waiting_for_departure`; no startup-motion or stable-following deadline runs
while it is intentionally paused. Actual speed <= 0.2 m/s and displacement <=
0.5 m are required during the wait. Departure requires at least 5 m of observed
Ego approach, then one of: route gap <= 35 m, Ego arrival time <= 3 seconds,
or observed Ego slowing in response after approaching. The last condition
lets an Expert that stops short of the distance threshold release traffic.
Approach/response observations persist into the later following detector,
which still requires actual moving/stable following before sudden braking.

Departure unpauses the same actor and commands 4 m/s. Motion startup deadlines
restart at departure, and actual movement must be observed before following
and braking. An unchanged far-away Ego cannot release it. A 45-second wait
limit, rejected pause/release or unexpected pre-departure motion fails with
owned-actor cleanup. `sudden_brake.departure_*` controls the thresholds;
`departure_wait_enabled: false` restores immediate startup for comparisons.
Logs include `waiting_departure` and `DEPART` with the observed release reason.

Each tick records route gap, measured Ego/NPC speeds, relative speed and gap
rate in a 2-second rolling window. The newest 1 second provides median relative
speed and a least-squares slope of gap against time, instead of gating on
noisy per-tick derivatives. Filtering requires >= 3 samples spanning >= 0.5 s;
gap spread must remain <= 1 m. The detector first requires observed
catch-up (positive relative speed and shrinking gap), and an observed Ego
speed decrease of at least 0.5 m/s. Following then needs both speeds >= 1 m/s,
gap > 8 m, relative speed magnitude <= 0.5 m/s and gap-rate magnitude <=
0.5 m/s continuously for 1.5 seconds. Samples more than 1 second apart reset
settling. Observed catch-up and Ego deceleration are historical evidence;
a small subsequent speed recovery does not erase that evidence. Reports store the stable-window gap median and an upper estimate
of max(observed maximum, median + 2 m). There is no fixed 20-35 m trigger.

Sections 2 and 3 are allowed spatial regions, without a preselected event
anchor. Stable following inside either region immediately commands the
0.5 m/s trough if sufficient distance remains for deceleration, the hold,
resume verification and 20 m exit clearance. The default budget at 4 m/s is
55.6 m. A section boundary alone neither triggers braking nor removes an actor.

Command ACKs are insufficient: within 1.5 seconds actual NPC speed must fall
by at least 2 m/s, reach <= 0.7 m/s and show measured deceleration >= 2 m/s^2.
Only then does the **8-second** hold clock start. The observed speed must stay
<= 0.7 m/s throughout; missing samples or a speed escape fail the event.
After 8 seconds it commands normal 4 m/s cruise and unpauses the same actor.
Within 5 seconds actual speed >= 3.5 m/s and displacement >= 0.5 m verify
resume. The vehicle continues to its outside-tunnel cleanup point.

A gap <= 8 m, a missed brake window, or failed deceleration/hold/resume marks
the report failed and restores cruise on the same actor. Successful recovery
keeps that actor until normal outside cleanup, then reports the run failure.
Rejected recovery commands or unobserved recovery motion fail immediately
and attempt cleanup. There is no emergency destroy/re-spawn to erase an event.

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
moving NPC until outside cleanup or timeout. Unexpected failures and Ctrl+C
attempt immediate cleanup; sudden-brake safety recovery follows the lifecycle above. A rejected destroy is recorded and the actor stays tracked
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

### Cut-in from opposite-lane geometry

`vehicle_cut_in` deliberately starts in the opposite lane but travels in
Ego's direction. The opposite MGeo links supply **drivable coordinates**;
their route direction and lane-change-edge permissions do not constrain
this maneuver. The spawn pose uses a real opposite-lane waypoint with yaw
from the Ego global path. No opposite route, reversed route or transform
teleport is assigned to the actor.

The scheduler uses the existing runtime safe forward gap and event run-in.
Initially it spawns with zero velocity and **`multi_ego=True`**, verifies external AUTO control and D gear,
unpauses, sets the speed limit and commands movement directly on every tick,
including startup. Ordinary NPC (`multi_ego=False`) AUTO mode was rejected in
the live trial and must not be used for this behavior. The official
[MORAI gRPC example](https://help-morai-sim.scrollhelp.site/ko/sim-api-guide/Working-version/grpc-api-1-26.R1)
also creates directly controlled vehicles with `multi_ego=True`. Its 7 m/s
speed cap also respects Ego's speed at spawn. A pure-pursuit controller follows
the source centerline until the chosen safe region and a 20-35 m gap. It then
tracks a smooth lateral transition over 18 m, crossing the centerline into
Ego's lane. `tunnel_vehicle_dynamic.cut_in` configures steering normalization,
wheelbase, maximum angle, lookahead, transition length, gap and entry criteria.
The configured 2.7 m wheelbase and 35-degree maximum angle are trial vehicle
parameters; matching them to the chosen model remains a live calibration.

Observed target-corridor geometry, lateral error <= 0.6 m, heading error <=
8 degrees, displacement >= 1.5 m and actual motion must persist for 0.5 seconds
before the forward route is assigned. Only then does the same NPC switch to
native CRUISE control, explicitly enable its constant-speed cruise driver,
and restore route speed. Another observed >= 0.5 m
of forward travel confirms handoff. Entry/route ACKs alone cannot complete it.
Before route assignment, blank/stale current-link metadata does not prevent
geometry-based target entry. Native route handoff then verifies actual forward
link membership. Source/crossing phases validate geometry, Ego-direction heading, continuity
and positive route progress; native phases return to strict forward-route
validation. Crossing a section boundary never creates a replacement actor.

Only cut-in uses multi-Ego; other behaviors continue to spawn native NPCs.
The direct adapter converts its physical m/s argument to control km/h;
steering remains normalized to [-1, 1]. The gRPC proto omits the velocity
unit; km/h follows MORAI
[vehicle control documentation](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/26.R1/ros-2)
for the corresponding speed command and must be checked against live motion.
Multi-Ego control-mode acceptance/readback, direct steering execution and the
same actor's route/CRUISE handoff must work
in live MORAI. If external mode or native handoff is rejected, the run fails
with the explicit command reason and cleans up its owned actor. Other vehicle
behaviors retain their existing NPC AI/control flow. No unsupported SetAI
command is required by this cut-in control-mode path.

`vehicle_oncoming` and `vehicle_cut_in` cannot run simultaneously. Explicit
conflicting primary/additional selections fail before MORAI; seeded random
primary selection excludes the conflict with configured additional traffic.

## Commands

From `MoraiDrive`, inspect the full behavior plans without MORAI:

```bash
python -m scenario_runner.tools.generate_tunnel_vehicle_candidates --output .runtime/tunnel_vehicle_candidates.json
python -m scenario_runner.main_runner --scenario full_loop_tunnel_vehicle --once --tunnel-vehicle-behavior vehicle_slow --seed 123 --check
```

Change the behavior argument to `vehicle_sudden_brake`,
`vehicle_oncoming` or `vehicle_cut_in`. `--check` invokes the geometry/plan validator;
model availability and observed motion still require live MORAI.

Expert sudden-brake trial (the default controller):

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --no-sync-runtime --tunnel-vehicle-behavior vehicle_sudden_brake --seed 123
```

Expert with one random vehicle behavior and one random available vehicle model:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --no-sync-runtime --tunnel-vehicle-random
```

The default random pool is `vehicle_slow`, `vehicle_sudden_brake` and
`vehicle_oncoming`. It selects one primary behavior per run and randomizes the
vehicle model (`random_model: true`); the default additional traffic list is
empty. `excluded_model_patterns` removes aerial model names from the tunnel
pool before selection (case insensitive), including the observed drone
`AR_DJMini3pro_2022` and AR_DJ/drone/quadcopter/quadrotor/helicopter/aircraft/
eVTOL/UAM names. Exclusions also apply to explicit model overrides; an empty
filtered pool fails before spawn. The report records candidates/exclusions.
The name filter does not classify unknown models whose names match no pattern. Cut-in remains selectable explicitly for later trials and is excluded
from this random pool. Omitting `--seed` uses a new seed each run. Add
`--seed 123` to reproduce a choice, and inspect `[TunnelVehicle] SELECT` for
the selected behavior/model and seed.

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
The brake report includes `expert_follow_gap_m`, `expert_follow_gap_upper_m`,
`follow_window_samples`, `follow_stable_duration_sec`, `pre_brake_npc_speed_mps`,
`brake_command_speed_mps`, `brake_start_elapsed`, `minimum_npc_speed_mps`,
`measured_deceleration_mps2`, `hold_start`, `hold_end`, `actual_hold_duration_sec`,
`resume_command_speed_mps` and `resume_motion_verified`.
Finite state, expected link/lane, direction, backwards progress, implausible
steps, timeout and actual completion are validated. ACKs cannot establish
braking, arrival, lane-change completion or safe Ego response. While waiting,
`[TunnelVehicleBrake] waiting reason=...` and `following_blockers` report window
collection, speed matching, gap trend/spread, historical catch-up/deceleration,
settling, allowed region or remaining-space failures. Cut-in reports include source-corridor semantics,
normalized steering/guide commands, measured lateral movement, target-entry
observation and native-route handoff travel.

```bash
python -m unittest scenario_runner.tests.test_tunnel_vehicle_dynamic scenario_runner.tests.test_tunnel_dynamic
python -m unittest discover -s scenario_runner/tests -t .
```

MORAI live trials must still verify available model placement, height, actual
speed, continuous route following, brake/hold/resume, oncoming lane clearance, Expert response and actor cleanup. Offline
passing tests do not approve simulator behavior or certify a safe encounter.
