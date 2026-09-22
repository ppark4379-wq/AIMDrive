# Waypoint controller audit (2026-09-22)

Read-only audit requested after the user observed plausible predicted waypoints but insufficient steering. No model, controller, speed-limit, or navigation settings were changed. No simulator commands were sent by this audit.

## Scope and observed configuration

Primary run: `outputs/inference/vldrive_20260922_115732_664187` (38.199 seconds of control logs). Epoch-93 model-best, normalized steering scale 1.0, sign +1, speed cap 1.0 m/s, PID 10 Hz, UDP send 20 Hz. Source: inference_report.json, waypoints.jsonl, control.jsonl, bridge_report.json. The user's current speed cap is 1 m/s, not the earlier 3 m/s.

353 distinct PID updates, 705 tracking sends, 60 hold sends (1 warming_up, 50 history_or_navigation_unavailable, 9 prediction_stale). One transition from tracking to hold reset the PID. Hold fraction 7.84%; these include startup and are not all mid-drive failures. Bridge rejected 0 messages, watchdog brakes 0, forwarded 769 packets including shutdown stops. Acknowledgements are from the bridge, not wheel-angle feedback.

## Actual algorithm

1. Transform the prediction's five points from the prediction-time ego pose to the current ego pose. Raw [left, -forward] becomes [left, forward].
2. Remove every point with `horizon_s - prediction_age_s <= 0.05`, based on wall-clock sensor receipt age. No spatial check that the vehicle has passed that point is used for this removal.
3. Select the first two remaining points and take their midpoint. No path curvature, vehicle wheelbase, or explicit cross-track error enters steering.
4. `error = atan2(aim_left, aim_forward) / (pi/2)`. At speed < 0.01 m/s this error is forced to zero (existing I/D history can still yield a transient command).
5. `steer = clip(1.25*error + 0.75*mean(last_40_errors) + 0.3*(error-previous_error), -1, 1) * 1.0 * 1.0`.
6. Target speed is the distance between the two selected points divided by their 0.5 s horizon spacing, capped at 1 m/s. The speed error is clipped to 0..0.35; throttle is capped at 0.75. Overspeed beyond target*1.1 sets full brake. `clip_delta` limits speed error, not steering.
7. Repeat the latest command between 10 Hz PID updates at 20 Hz transmission. Invalid/stale inputs cause accel=0, brake=1, steer=0 and reset PID history. The bridge forwards the command without a steering attenuation filter.

The display plots all five waypoints in prediction-time coordinates. The controller consumes a reprojected, shortened subset in current coordinates, so the displayed full curve is not the actual steering aim.

## Replay of logged decisions

Recomputed current coordinates, midpoint and P/I/D terms for all 353 actual PID updates (not the duplicated 20 Hz sends). Maximum absolute midpoint discrepancy: 8.88e-16 m; steering discrepancy: 2.78e-17. Thus the small logged commands are produced by the documented arithmetic, not an unexplained downstream multiplier.

| Quantity across distinct PID updates | Median | Range |
|---|---:|---:|
| Steer | 0.119164 | -0.056980..0.272879 |
| Target bearing | 5.282 deg | -3.698..13.701 deg |
| Prediction age | 0.733763 s | 0.488887..1.244674 s |
| Target midpoint distance | 4.896117 m | 4.443605..7.677700 m |
| Distance actually moved since prediction pose | 0.574325 m | 0..0.924926 m |
| Predicted speed from selected point spacing | 4.312755 m/s | 3.432463..4.764322 m/s |
| Actual speed | 0.774230 m/s | 0..1.479382 m/s |

309 updates used original zero-based point indices [1,2], 44 used [2,3]; none used [0,1]. The original first point was still physically ahead at all 353 updates, with forward distance 1.185941..2.265671 m (median 1.579815 m). It was discarded solely by the temporal horizon condition. Without temporal removal, the original first-two midpoint had median distance 2.658663 m; that alone does not prove it would be a better aim or yield larger steering.

## Concrete understeering example

Frame 908, route progress 83.129 m, PID-update age 0.770489 s:

- Aim: left 0.816217 m, forward 4.474870 m; distance 4.548700 m.
- Bearing: 10.337122 deg; normalized angular error 0.114857.
- P=0.143571, I=0.068175, D=0.000331 -> **steer=0.212077**. No clipping occurred.
- With the user's max wheel angle 40 deg and the documented UDP normalized-angle convention, this requests about **8.48 deg**.
- As a geometric comparison only, a simple rear-axle bicycle/pure-pursuit calculation for that same aim and wheelbase 3 m gives `atan(2*3*left / distance^2) = 13.316 deg`, normalized steer about **0.332909**. This is 57% larger than the current command, not proof of an optimal controller setting. Rear-axle/GPS reference offset, simulator steering response and discrete path interpolation still require validation.

Maximum actual command was 0.272879 at frame 854: aim [1.519099, 6.230939] m, bearing 13.701391 deg, P=0.190297, I=0.069432, D=0.013149. Its requested wheel angle is about 10.92 deg at max 40 deg; it never approached the full normalized clamp.

Official UDP convention: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/ros-1 (UDP Ego Ctrl Cmd section): normalized steering equals requested degrees divided by maximum steering angle. This is distinct from the ROS CtrlCmd radians interface.

## Findings and limits

1. The old 0.8 attenuation is absent. Steer remains well below +/-1. No extra rate limiter, deadband, or smoothing was found in our PID-to-bridge path. This does not inspect MORAI's internal actuator implementation.
2. Angular-error PID was transferred without vehicle-geometry calibration. Two aims on the same ray produce identical commands despite requiring different curvature at different distances. The I term is a rolling mean, not a time integral; at constant error it remains bounded instead of accumulating to eliminate a persistent tracking bias.
3. Time-only pruning conflicts with slow spatial path following: the model predicts about 4.3 m/s while the actual vehicle is capped at 1 m/s, yet nearby unpassed points are removed. The resulting aim is typically about 4.9 m away. Simply choosing earlier points can reduce the bearing and steering further, so target selection and steering geometry must be addressed together.
4. Control holds zero steering and clear PID memory; this can interrupt tracking. The latest run's bridge had no watchdog failure; 9 prediction-stale sends were observed. Holds are a secondary contributor, not established as the principal cause of the small tracking commands.
5. Measured route projection error grew from 0.050 to 4.893 m over this run. This is not a pure controller-only tracking test: the model replans during motion. A plausible plotted curve is not proof that every predicted segment is feasible or correct.
6. Actual wheel-angle feedback is parsed by the receiver (`steer`) but not retained in current control logs. The audit establishes commanded steering and bridge forwarding, not exact physical wheel response. Estimation from yaw/speed is not a substitute for this feedback.

Recommended next change: preserve prediction-age freshness checks and pose reprojection; select a spatial lookahead from the predicted polyline using actual speed; convert curvature to wheel angle with wheelbase and calibrated max-angle/protocol mapping (e.g. pure pursuit), including ego/GPS-to-rear-axle reference. Log and display the actual aim, commanded angle and returned steer with its documented units. Keep the global route out of the control target; use predicted waypoints. Compare controller tracking at the same speed and on the same predicted path before retuning the model. No implementation changes applied in this audit.
