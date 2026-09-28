# BEV object position and motion output (schema 5)

Read each matched object's `tracks[].position.xyz_m` for the BEV OBB center
(x forward, y left, metres, in `coordinate_frame`). XY is the minimum-area
rectangle center of the full associated cluster. Z is the midpoint of observed
min/max Z, not a BEV measurement. `position.bev_obb` includes center, length,
width, corners and unoriented yaw modulo pi. Yaw is not vehicle heading.
This encloses observed returns; partial visibility and bad clustering can bias
the center relative to the physical vehicle. No dimensions prior is applied.

`clusters[].centroid_xyz_m` remains the arithmetic point centroid for audit.
Do not use it for the requested BEV center. The method is now
`cluster_bev_obb_center`, replacing the bbox-supported median position.
When multiple clusters attach to one ID, the nearest optical-depth candidate
is retained as an ambiguous position; velocity is suppressed.

`tracks[].motion.velocity_xy_mps` is linear regression over recent OBB centers
using LiDAR timestamps. `speed_mps` is its magnitude. This is the coordinate
derivative in moving base_link, including ego translation/rotation effects.
`absolute_velocity_xy_mps` is null and `ego_compensated` is false. Do not feed
this into planning as an object's world velocity. Ego pose synchronization and
motion compensation remain required for that output.

The estimator requires 3 samples, uses up to 0.6 seconds, resets after a 0.5 s
gap, clock epoch changes, ambiguity, or a jump above the configured speed gate.
Missing associations do not create predicted positions. The display remains
ID/status/Cluster/XY range; detailed OBB and motion live in the JSONL. HOLD does
not add samples to the motion history. This is an initial observed-box tracker,
not a validated full-body or ego-compensated dynamic-object estimator.
