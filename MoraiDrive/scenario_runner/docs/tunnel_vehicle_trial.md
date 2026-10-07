# Tunnel slow vehicle trial

`full_loop_tunnel_vehicle` starts at checkpoint 13 and ends at checkpoint 15.
The approach before checkpoint 14 gives the first NPC enough spawn distance.
Expert drives Ego. One selected `vehicle_slow_01` through `vehicle_slow_04`
is tested per run. Pedestrian events, highway SpawnPoints and the legacy
`tunnel_sudden_brake` event are gated out of this mode.

The catalog reuses the pedestrian catalog's four divisions and 10 m endpoint
margins. Each 20 m vehicle path is centered on its section anchor (about
24.6, 53.9, 83.2, 112.5 m from checkpoint 14). Vehicle positions follow Ego's
official path (`d=0`), with route height and forward yaw. The model is inherited
from `highway_npc.model`; the first trial does not randomize it.

From `MoraiDrive`, inspect candidates without MORAI:

```bash
python -m scenario_runner.tools.generate_tunnel_vehicle_candidates --output .runtime/tunnel_vehicle_candidates.json
```

Run the first candidate with Expert:

```bash
python -m scenario_runner.main_runner --zone full_loop --scenario full_loop_tunnel_vehicle --once --no-sync-runtime --tunnel-vehicle-candidate-id vehicle_slow_01
```

Change the suffix to `_02`, `_03`, or `_04` for the other sections. The
default candidate for this dedicated trial mode is `vehicle_slow_01`.

The trigger is 40 m of route distance before the planned spawn position.
Both the planned and observed initial NPC-to-Ego route gaps must be at least
30 m; missing that window fails the trial before spawning or removes a
misplaced actor. It does not shorten the required gap to force a spawn.

The handler reuses the existing tunnel MGeo link/waypoint lookup and
`_spawn_highway_vehicle` route/AI/speed operations. It checks the actual
spawn position/link and command acknowledgements, then records position,
speed and route gap while the vehicle drives at 4 m/s (14.4 km/h). The
remaining same-direction NPC links are assigned to AI; the handler removes
the NPC when it reaches the selected path's end, after observing motion.
It rejects an immediate jump to the end, a wrong route/lane, unavailable
state, or a 20 s timeout. Ctrl+C and failures record an incomplete trial and
clean up the runner-owned actor.

Reports are saved in `scenario_runner/.runtime/tunnel_vehicle_runs/`.
Completion proves only the checks observed in that run; it does not mark
the path approved or guarantee an Ego interaction. In MORAI, verify lane
placement, continuous forward motion, actual speed, Ego's following response,
and removal at the intended end. Four candidates are independent trials,
not a four-actor sequence. Sudden-brake, stopped-vehicle and random-mix
behaviors can be added after these slow-vehicle trials are checked.
