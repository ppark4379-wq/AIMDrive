# LMDrive PID source

Upstream: https://github.com/opendilab/LMDrive

Commit: `308d8da0b4ff3463b1649fe483a9b2c56b108189`

`pid_controller.py` is copied unchanged from `leaderboard/team_code/pid_controller.py`.
The Apache-2.0 LICENSE is included. The adapted waypoint controller in
`control/waypoint_pid.py` follows `LmdriverAgent.control_pid` and the gains in
`leaderboard/team_code/lmdriver_config.py` at this commit. The final steering
scale 0.8 comes from `lmdriver_agent.py`'s `run_step`.

The active lateral-only adapter is now `control/lmdrive_lateral.py`; it reuses
the unchanged PID above, normalized angle,
stationary error rule, clipping and final 0.8 scale. Its nominal 10 Hz scheduler
adapts the upstream every-other-20-Hz-tick execution to the MORAI worker.

The active longitudinal output comes from MoraiDrive/expert, not LMDrive.
`control/steering_target.py` is a local extension: the active remaining-path
mode replaces upstream's first-two-waypoint midpoint with an unpassed, forward
target on the predicted polyline. The original midpoint remains a comparison
option. Path-insufficiency braking and held-target validity checks are local.
MORAI coordinate conversion, held-target display reprojection, prediction-age
checks, reset on invalid input and UDP transport are local additions. The
historical `WaypointPID` class's speed ceiling is not used by the live worker.
