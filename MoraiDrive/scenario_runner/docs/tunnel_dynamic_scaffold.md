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
commands A -> B, switches direction at B, and stops and removes the actor at
C. The run fails if the pedestrian times out or Ego reaches the goal first.
The gRPC adapter only receives the chosen spawn and control commands.

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

The A/B/C coordinates are calculated from the official route and are **trial
candidates**. Their surface height, the crossing clearance, and the gRPC
pedestrian control speed unit still need a live MORAI check. Offline tests
verify path validation and state transitions; they cannot establish that the
simulator follows the commanded path or that Ego responds safely. Keep this
mode for controlled trial runs until those observations are recorded.
