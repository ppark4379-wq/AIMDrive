# Situation instruction features v0.3

Run 'python -m MoraiDrive.data_collection.situation_labeling.build_labels' with
'--episode', '--route-definition', '--global-path', and a new '--output'
directory. The command writes frame_features.jsonl, object_features.jsonl,
situation_rules.jsonl, and report.json. Existing output directories are not
overwritten.

The output uses instruction_state values rather than S/R numbers. A matched
state describes current evidence and supplies sentence variables. It is not
an Expert action label or an entry/trajectory safety certificate. The original
frame valid and invalid_reasons fields remain visible; object geometry
quality is checked internally and rule_geometry_usable is persisted. Rules check
nearby actors after spatial filtering. A full, usable object inventory is still required.

Internal feature calculation provenance:
- Ego direction is derived from adjacent admitted GT positions; a held
  direction is marked ego_heading_source=held_prior_motion. Acceleration is a
  backward speed difference.
- MGeo link IDs come from the local official map's link_set.json when present.
  A match farther than 3 m has no link ID.
- The roundabout merge point is explicit in the official route YAML and
  matches Expert ROUNDABOUT_ENTRY_MERGE_POINT.
- A lead bumper gap is a current-frame longitudinal projection of oriented GT
  boxes. It requires a fresh Ego motion heading, a matched circulating link,
  compatible actor heading, and lateral proximity.
- Pedestrian path clearance uses an orientation-independent enclosing circle.
  It describes present path overlap, not a future collision prediction.
- Static passage gap is a conservative geometric candidate; connected
  drivable corridor and road-boundary safety are not inferred.

Current sealed GT has no LiDAR-frame-aligned Expert future OBB collision result.
Roundabout collision_risk is a current merge ETA overlap heuristic; it is
null when key traffic evidence cannot be resolved. future_collision_predicted
remains null internally and is not persisted. The same fail-closed behavior applies to avoidance and
between-obstacle passage when a safe trajectory is not verified.
The internal resume_candidate describes a possible transition, not an Expert resume event.

Highway Rule rows also emit `label_candidate` (`KEEP_LANE`, `ACC_FOLLOW`,
`CHANGE_LEFT`, `CHANGE_RIGHT`) and a separate `behavior_state`. A lane change
is inferred from consecutive Ego GT positions relative to adjacent MGeo lane
centers; aborted changes use the actual return direction. Same-lane lead
bumper gaps use oriented GT boxes. Target-lane front/rear `gap_m` uses MGeo
center distance, as in Expert; separate bumper gap fields use GT boxes. The
initial target-lane front/rear thresholds match Expert (20 m / 12 m plus
1.5 s of closing speed). `target_lane_collision_safe` remains null because
frame-aligned Expert trajectory validation is unavailable. Unresolved nearby
vehicle geometry or lane motion produces an unknown label.

## Feature definitions

config/feature_definitions.yaml lists only fields persisted for rule decisions,
sentence rendering, and alignment to existing Expert observations. Shared
inputs live under common.frame and common.object; zone-specific inputs live
under zones.<zone>.frame and zones.<zone>.object. Dynamic obstacle uses a
feature Rule without an Expert label. Other zones can attach aligned Expert
labels when available.

build_labels computes its intermediate geometry and motion features in memory,
evaluates the Rule, then writes only the fields in this YAML to
frame_features.jsonl and object_features.jsonl. Removing an intermediate from
the saved rows does not change the current Rule calculation. Zone fields are
written only for that zone. The YAML states type, unit, source, meaning, and
evidence level; editing it changes output selection, not the formulas.

The build validates the saved rows for missing common fields, unexpected zone
fields, unknown fields, and wrong types. Existing full-feature outputs are not
rewritten. To inspect one with the reduced contract, use:

    python -m MoraiDrive.data_collection.situation_labeling.feature_definitions --features-dir situation_features_dynamic_lap060 --allow-legacy-extra

Use --allow-legacy-missing as well for older outputs that predate current
common fields. Extra fields are reported, not silently treated as Rule inputs.

## Instruction mapping and optional Expert postprocessing

`build_labels` now maps a matched current-frame `instruction_state` through
`config/label_instruction_map.yaml` and renders a sentence with
`config/instruction_catalog.yaml`. `mapper.py` selects the ID and `renderer.py`
checks and fills its variables. The rule state, `instruction_source`,
`instruction_id`, and rendered `instruction` remain separate fields. A
highway `label_candidate` is used only as a candidate and never becomes an
Expert `label`. Dynamic obstacle has no Expert-sourced label (no Expert state
contract exists for it): `dynamic_obstacle_rule`'s `instruction_state` values
(`approach`, `yield_start`, `yielding`, `waiting`, `resume`, `normal`,
`behavior_unverified`) are the only label dimension and map through
`label: null` rows like static obstacle's rule-only states.

With `--expert-log PATH`, existing Expert observations are attached first.
Repeat the option to combine Roundabout JSONL and Expert text logs. An
observed Expert label plus a matched rule state selects an exact mapping; a
conflicting pair leaves the sentence null. Collection and Expert runtime are
unchanged.

Existing `*_roundabout.jsonl` plan records and `[HighwayDecision]` lines in
`driving_debug_*.log` have Ego XY but are sparse. Postprocessing assigns only
a unique Ego position match within 0.5 m; it never carries a state forward
between log entries. An independently supplied journal containing actual ROS
`ego_header_seq` plus `ego_source_ns` can also be joined to
`quality/active_signals.json` within a 25 ms source-time gap. The older
`ego_seq` timing field is an internal callback count and is never a join key.
Both methods report `physical_simultaneity_verified: false`.
The parser currently reads Roundabout plan JSONL and `[HighwayDecision]` text
lines. `[STATIC]` lines in `supervisor.log` are not frame aligned because they
have no sensor source timestamp or header sequence; they are not converted
to Expert labels.

As of this change, Expert's own `roundabout_entry` planner emits one
`[EXPERT_STATE] {...}` JSON line per tick (not deduped) directly into
`supervisor.log` during real collection, carrying true `ego_header_seq`/
`ego_source_ns` plus `roundabout_phase` (`cruise`/`follow`/`yield`/
`collision_slowdown`/`emergency`/`rear_pressure`) and the other `RAW_FIELDS`
`expert_state_mapping.yaml` matches on. This is the phase Expert itself
decided internally (`roundabout_entry.py`'s `phase` variable) -- a finer label
than the printed `[ROUNDABOUT] {reason}` text, which collapses `yield`/
`collision_slowdown`/`emergency` into just two reason strings. For episodes
collected after this change, pass that episode's `supervisor.log` as
`--expert-log` (with `--episode`) and the exact-match path above resolves
every roundabout frame with no separately supplied journal needed. Episodes
collected before this change have no `phase` in their log and cannot be
backfilled.

Static, tunnel, and much of Highway may remain without Expert labels with
current sparse logs. Missing Expert matches do not prevent a supported Rule
instruction from rendering; it is marked `instruction_source: rule_state` or
`rule_label_candidate`. Missing sentence variables leave the text null. No
GT rule candidate is promoted to an Expert label.

To attach instructions to an existing feature output without rebuilding any
features or touching the collection, run:

```text
python -m MoraiDrive.data_collection.situation_labeling.attach_expert_instructions --features-dir SITUATION_FEATURES --output NEW_OUTPUT_DIR
python -m MoraiDrive.data_collection.situation_labeling.attach_expert_instructions --features-dir SITUATION_FEATURES --expert-log EXISTING_DRIVING_DEBUG.log --expert-log EXISTING_ROUNDABOUT_DEBUG.jsonl --output NEW_OUTPUT_DIR_WITH_EXPERT
```

The new directory contains `situation_instructions.jsonl` and `report.json`;
the input feature directory is read only. `--episode EPISODE_DIR` is needed
only for a separately supplied journal with true ROS header seq and source
time. Existing sparse logs do not require it.

## Camera frame viewer

After `build_labels` has produced a feature directory, create a standalone
HTML viewer with front camera images, current-frame rule states, evidence,
unknown reasons, and a frame slider:

```text
python -m MoraiDrive.data_collection.situation_labeling.visualize_labels --features-dir situation_features_static_lap060 --episode-root G:\static_shapes_100_retry_20260926_112627_lap_060 --output situation_viewer_static_lap060
```

Open `index.html` from the output directory. The viewer checks that the
camera and LiDAR native filenames match index by index, then copies compressed
images beside the page so it works without the source drive. The displayed
`instruction_state` is a current-frame GT rule state. It is not an Expert
behavior label. Rule instructions are marked by their `instruction_source`;
an Expert label appears only with an aligned Expert observation. Pass
`--instructions PATH` to
view a separate `situation_instructions.jsonl` produced from existing features.