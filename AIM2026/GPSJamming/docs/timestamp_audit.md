# Camera/LiDAR timestamp comparison — 2026-09-28

Current port configuration: MORAI LiDAR Host Sensor Port **2368**, Destination
Port **2369**. The measurements below retain the original capture's port numbers.

UDP implementations and timing config are unchanged. A standalone header-only
diagnostic captured 15 seconds without YOLO, point processing or assembly delay.
The capture does not check JPEG completeness or establish the LiDAR scan reference
(start/middle/end); it compares the original timestamps of observed packet groups.

Evidence: `artifacts/timestamp_audit/20260928T062205Z_0debb9af/` contains raw observations
in `timestamps.jsonl` and candidate statistics in `summary.json`.

| Stream | Observations | Median positive timestamp increment |
|---|---:|---:|
| Front camera, sender 127.0.0.1:5000, destination 5050 | 300 | 48 ms |
| LiDAR, sender 127.0.0.1:2369, destination 2368 | 144 | 102,677 raw ticks |

The camera header supplies seconds plus nanoseconds. The strongest observed
LiDAR interpretation is **microseconds elapsed since the current hour began**:

```
hour_ns = floor(camera_timestamp_ns / 3_600_000_000_000) * 3_600_000_000_000
lidar_candidate_ns = hour_ns + lidar_raw * 1000
```

For front-camera pairs nearest in packet receipt time (within 200 ms),
LiDAR candidate minus camera timestamp: median +1.393 ms, min -32.770 ms,
max +40.479 ms (144 pairs). This is diagnostic evidence, NOT calibrated latency,
an offset to subtract, or guaranteed acquisition alignment. Normal frame rate
differences and network jitter also contribute to the residual.

No hour rollover/reset was captured. Future synchronization must choose among
the preceding/current/following hour using a verified common clock, handle reset
and rollover explicitly, and reject excessive timestamp differences. A static
source/Unix anchor alone cannot safely handle a per-hour timestamp reset.

Four camera sender ports (5000–5003) were observed on destination 5050; all were
analyzed separately. Camera packets rejected by the header parser may include
non-JPEG traffic; they are not evidence of packet loss or corruption. The initial
capture's `camera_invalid` counter represents these ignored-or-invalid packets.

Reproduce from workspace root after stopping any viewer using these ports:

```powershell
.\AIM2026\GPSJamming\compare_sensor_timestamps.ps1
```

The command captures 30 seconds into a new directory and releases the sockets.
No clock configuration is edited. The launcher uses the existing botsort Python.

## Timestamp selection implementation

The viewer now uses `timing/synchronization.py`. LiDAR timestamps are reconstructed
using the nearest of the previous/current/next camera hour. The standalone LiDAR
TimestampMapper and original packet values remain unchanged. This observed clock
model assumes the same normal-mode Unix clock used in the audit; a changed sensor
configuration or simulator timing mode requires another audit.

`timing.synchronization` in perception.yaml controls `max_delta_ms: 50`,
`max_age_ms: 800`, and `camera_buffer_size: 30`. Already tracked camera images and
their own bbox records are buffered together. Each selected completed LiDAR frame
uses the nearest timestamp in that available buffer, not the newest camera image.
This does not wait for future frames or infer intermediate MOT results.

Incomplete LiDAR, invalid raw timestamps, stale receipts and differences exceeding
50 ms are rejected. Backward/duplicate LiDAR time clears the camera buffer; genuine
hour rollover is reconstructed continuously. Camera time reversal also clears the
buffer when delivered by the receiver. The current BoTSORT receiver drops backward
camera timestamps upstream, so simulator clock reset may require a viewer restart.

The displayed image, boxes and cloud belong to the selected pair and are retained
between updates (only until the freshness limit). Log schema 2 emits one row per
selected LiDAR attempt, not per camera frame; under load the viewer processes the
newest queued LiDAR frame. A camera frame may appear in multiple rows. The fields
`synchronization.status`, `acquisition_delta_ns` (LiDAR minus camera), and
`synchronized` distinguish accepted/rejected selections. `synchronized=true` means
the frame timestamps passed this configured model/tolerance, not exact simultaneous
exposure or validated moving-scene accuracy. Ego data is still not time aligned.

Frame-level selection does not deskew points within a rotating LiDAR scan. The
scan timestamp's start/middle/end meaning and moving-scene accuracy remain unverified.

### Display retention on failed selection

#### Gap measurements (schema 4)

The top-left HUD now shows only `delta_ms` and `lidar clusters`, describing the
displayed pair (including the retained pair during HOLD). Attempt results remain in
the terminal: `[SYNC]` reports consecutive failures, skipped assembled frames and
time since last success; `[CLOCK]` shows both absolute timestamp values and raw LiDAR;
`[SUCCESS_INTERVAL]` reports wall/sensor intervals plus failures/skips between successes;
`[DISPLAY]` reports completed intervals of a display state, including HOLD. Consecutive
states for the same pair can be summed to obtain total dwell. Logs use English field
names to avoid console encoding issues. Existing JSONL persistence is retained.

Observation JSONL now includes `gap_metrics` on every LiDAR selection attempt,
including attempts before any camera arrives. Here success means a timestamp-valid
pair was processed, even if zero object clusters matched. It is not a per-object
tracking outage metric. An epoch reset starts a new interval.

- `consecutive_failed_attempts`: consecutive rejected selection attempts.
- `skipped_lidar_frames_this_attempt`: assembled frame IDs skipped since the last
  attempt (queue drops or deliberate newest-frame selection, not UDP packet loss).
- `elapsed_since_success_ms`: monotonic time since last successful processing.
- `recovery.failed_attempts_between`: rejected attempts between two successes.
- `recovery.skipped_lidar_frames_between`: unprocessed assembled frames between successes.
- `recovery.success_interval_wall_ms`: processing completion interval between successes.
- `recovery.success_interval_sensor_ms`: LiDAR measurement timestamp interval.
- `recovery.failure_reasons`: reason counts for the interval.

An independent `artifacts/association/display/*.jsonl` file records intervals
between changes of displayed pair or mode (`matched`, `hold`, `unmatched`). Repeated
renders do not restart an interval. `duration_ms` includes time the rendering thread
was blocked in inference. The last interval is closed on normal shutdown. These
are OpenCV submission intervals, not physical monitor scanout measurements. A pair
may span successive matched and hold intervals; sum them for total pair dwell time.
No input during an outage is reflected in display dwell, not invented frame failures.

HOLD is bounded in code, but its timeout is checked by the rendering thread. If
inference blocks that thread, a submitted image remains on screen beyond that timeout
until the next submission. This is why display intervals are logged separately.

Summarize either log with `scripts/summarize_gaps.py <log-file>` using the botsort
Python. It reports median/p95/max success intervals, failure/skip counts, stage costs,
or display durations depending on the input. Use this baseline to decide whether
to optimize inference/queueing or move rendering to an independent loop; do not
increase timestamp tolerance simply to conceal misses.

`visualization.hold_last_pair_ms` (default 100 ms) retains the previous complete
annotated image when a selection fails. The image, boxes and projected cloud are
held together, never old points on a new image. `HOLD` identifies the displayed
pair and duration; the other HUD statistics describe the latest selection attempt.
Failed attempts remain failed in JSONL and do not become motion measurements.
Repeating a render does not extend retention, sensor freshness limits still apply,
and clock discontinuities clear the cache. This reduces blinking, not timestamp
rejection rates or sensor latency. Set the duration to 0 to disable retention.

Press H to toggle retention during a run. The HUD reports the displayed pair's
camera/LiDAR receipt ages (including held frames), plus wall times for detector,
tracker, LiDAR processing and association executed in the current loop. JSONL
`performance` records those stage times, selected-pair receipt ages and queue drop
counts. These are not physical exposure-to-screen latency. The latest detector
measurement may refer to a newer camera frame than the selected historical pair.

Run the moved project with:

```powershell
.\AIM2026\GPSJamming\run_front_camera_lidar.ps1
```
# Camera selection diagnostics

The association viewer now uses a bounded JPEG buffer (`camera.frame_buffer_size: 8`,
`camera.frame_max_age_ms: 400`). It selects the newest pending LiDAR frame first,
then the nearest camera within the existing 50 ms tolerance before inference.
No future camera wait is added. Tracker input timestamps strictly increase;
older skipped images are never replayed through BoT-SORT. Previously processed
camera results may still be reused by the synchronizer if within tolerance.
The LiDAR target stays fixed during inference; newly arriving LiDAR frames are
considered at the next iteration. Buffer capacity is not an intentional delay.
Receipt age is not acquisition-to-display latency; the existing sensor freshness
check also runs after processing. LiDAR processing and HOLD settings are unchanged.
`buffer_dropped_frames` counts capacity evictions; metadata states `expired` and
`skipped_for_newer_pair` distinguish age expiry and deliberate forward selection.
BoTSORT's standalone receiver retains its default latest-only behavior.

Each observation now records `camera_selection_diagnostic`. On failures the
console prints `[CAMERA_SELECTION]`: the nearest complete accepted JPEG frame
available at selection time, its signed LiDAR-minus-camera `delta_ms`, tolerance
check, and whether it is in the synchronizer buffer. States are `overwritten`
(replaced before consumption), `pending` (waiting for consumption), `processing`
(consumed but not yet marked ready), `processed`, or `decode_rejected`.
This is bounded metadata for 240 received frames, not a JPEG queue. It excludes
partial JPEGs, rejected senders, and future arrivals. Absence does not prove that
the simulator never sent a frame. A close overwritten frame is evidence of a
lost synchronization opportunity, not proof that inference on it would succeed.

`[EGO]` distinguishes no datagrams from invalid packets and stale valid states.
For an isolated check, close the viewer and run `scripts/check_ego_udp.py` with
the BoTSORT Python environment; it listens on port 9094 for ten seconds. A UDP
listener does not initiate a sender connection; the simulator must publish to it.

