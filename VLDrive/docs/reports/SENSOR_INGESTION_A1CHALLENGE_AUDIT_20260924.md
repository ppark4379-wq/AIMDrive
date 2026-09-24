# MORAI input ingestion audit and A1Challenge comparison

Date: 2026-09-24. Scope: current VLDrive input code, saved live runs, reference source under `Z:\A1Challenge`, and CPU-only temporal-window reproduction. No production source/config edits, GPU benchmark, live simulator run, control transmission, or new model training were performed.

## Main findings

1. Reception is already asynchronous: a thread per UDP receiver continuously publishes the latest complete sample; capture, Vision Encoder, and Stage2 inference run separately. It is inaccurate to describe the system as stopping UDP reception until every inference finishes.
2. Input admission nevertheless requires every camera, LiDAR, and Vehicle Status sample ID to advance. One lagging stream can prevent a new multimodal frame even when its previous sample is still within allowed age/skew.
3. The recent timed-history implementation combined with capture_hz=5 can starve inference at slightly slower-than-nominal simulator delivery rates. This failure mode was reproduced with the actual class on CPU.
4. A1Challenge/AIM2026 implements independent reception, newest-JPEG-only decoding, last-image reuse, a single latest preprocessed snapshot, and Event-based inference. Several of these mechanisms already exist in VLDrive; the important remaining differences are admission, timestamp selection, and unnecessary preprocessing.

## Current pipeline and code evidence

| Location relative to VLDrive | Behavior |
|---|---|
| `inference/morai_udp.py:173` | Per-stream receiver thread; SO_RCVBUF request 4 MiB; latest complete sample slot |
| `inference/morai_udp.py:197` | Receive and consume each datagram; publish timestamp after parsing/assembly |
| `inference/morai_udp.py:227` | Snapshot atomic for one sensor, not a jointly synchronized snapshot of all sensors |
| `inference/live_pipeline.py:60` | Startup waits for fresh required streams |
| `inference/live_pipeline.py:115` | New complete LiDAR sweep triggers a capture attempt |
| `inference/live_pipeline.py:121` | LiDAR sequence marked seen before subsequent acceptance tests; rejected sweep is not retried when another sensor arrives |
| `inference/live_pipeline.py:125` | Early interval gate skips frames before preprocessing |
| `inference/live_frame.py:227` | Sequentially snapshot/decode cameras before reading LiDAR/status and checking all missing inputs |
| `inference/live_pipeline.py:137` | Cross-stream receive-time skew rejection after CPU preprocessing |
| `inference/live_pipeline.py:164` | All sensor IDs must advance relative to last captured frame |
| `inference/live_pipeline.py:173` | Bounded encoder queue; drop oldest on full (current capacity 1) |
| `inference/vldrive_runtime.py:216` | History.accepts repeats all-new sample-ID requirement |
| `inference/timed_history.py:49` | Latest anchor requires 11 distinct frames near t-2.0,...,t, each within 60 ms |
| `inference/live_pipeline.py:225` | Stage2 waits until that complete history exists; no new anchor means no new inference |

There are already separate capture and VE workers. Merely adding threads or a queue does not address the primary remaining gates. Camera receivers retain compressed JPEG bytes; they do not decode every queued image in the UDP receiver. LiDAR packet geometry conversion still occurs on the receive thread. Socket backlog is a possible latency contributor, not a measured root cause in this audit. Enlarging socket buffers alone is not a fix.

## Recent saved run evidence

Run: `outputs/inference/vldrive_20260924_155639_090691`.

- `outputs=0`, stopped by user.
- 311 navigation log entries, all valid, spanning 69.644 seconds of simulation time.
- `discarded_skew=0`, `rejected_frames=0`, `history_resets=1`, `capture_queue_drops=5`.
- Navigation log LiDAR frame IDs advance by exactly 2 at every adjacent entry.
- Status receive-time intervals associated with those entries: median about 0.223 s, mean about 0.2247 s, p95 about 0.2485 s. These are status receipt proxies, not recorded VE history timestamps.
- Frame-ID advancement relative to receipt time implies roughly 8.9 LiDAR sweeps/s and about 4.45 navigation/capture attempts/s after alternate-sweep selection.
- The run received camera/LiDAR/status data; lack of all sensor traffic or navigation validity does not explain its zero outputs.

The actual accepted VE-history timestamps and reasons for every unsuccessful history snapshot were not persisted. Therefore the exact cause of every live rejection cannot be proved retrospectively. The timing mismatch is a strongly supported candidate and an independently reproduced code defect.

For context, the earlier run `vldrive_20260923_194100_300863`, with the older history path and encoder queue size 3, recorded 12,773 outputs over approximately 7,420 s (about 1.72 outputs/s). Median VE/Stage2/preprocessing times were approximately 199/358/22 ms; median sensor-to-output latency was about 970 ms. These are historical measurements under different runtime conditions, not a controlled before/after comparison. They show that input waiting is not the only possible performance limit.

## CPU-only reproduction of timed-history starvation

Actual `TimedTokenHistory`, actual production append/accepts/snapshot logic, 201 synthetic real-ID samples per case, finite zero tensors of shape [106,256], no GPU/model forward. Capture rate affects admission exactly as in the class. Frames were not duplicated to fill history.

| Synthetic source spacing | capture_hz | Accepted frames | Complete windows | First complete window |
|---|---:|---:|---:|---:|
| 0.200 s | 5 | 201 | 191 | 2.000 s |
| 0.225 s | 5 | 201 | 0 | never |
| 1/8.9 s | 5 | 101 | 0 | never |
| 1/8.9 s | 10 | 201 | 183 | about 2.0225 s |

At capture_hz=5, the minimum accepted spacing is 0.2-0.06=0.14 s. An approximately 0.112 s source therefore loses alternate sweeps, producing approximately 0.225 s spacing. At offsets of 1, 2, and 3 frames, deviation from a 0.2 s target grows to approximately 0.025, 0.050, and 0.075 s. The third lag exceeds 0.060 s, so selection stops at three frames rather than reaching eleven. No reset is required for this starvation.

The capture_hz=10 case is evidence about temporal sampling only. It is NOT evidence that this PC can run VE and LLM at 10 Hz. Prior actual compute replay found contention and latency growth with higher VE rates; see `REALTIME_WINDOW_OPTIMIZATION_20260923.md`.

## A1Challenge reference

Examined both `Z:\A1Challenge\AIM2026` and `Z:\A1Challenge\A1Challenge\AIM2026`; Camera.py and EgoHandler.py hashes match between these copies. Also inspected older `Code\A1Challenge` and the MORAI UDP example. The most directly useful input pipeline is AIM2026:

| Reference location under AIM2026 | Mechanism |
|---|---|
| `MORAI_UDP/Camera.py:169` | Dedicated camera receive thread |
| `MORAI_UDP/Camera.py:189` | Drain complete JPEG queue; decode only newest available JPEG per camera |
| `MORAI_UDP/Camera.py:227` | Nonblocking sockets, 8 MiB requested receive buffer |
| `MORAI_UDP/Camera.py:284` | Bounded per-camera queue (default 5), drop oldest when full |
| `MORAI_UDP/Camera.py:298` | select/read until socket temporarily empty; assemble fragments |
| `MORAI_UDP/EgoHandler.py:94` | Independent state reception, latest parsed state getter |
| `main.py:213` | Keep last valid image for each camera; wait for all cameras only on initial fill |
| `main.py:256` | CPU preprocessing then latest snapshot overwrite |
| `main.py:274` | Signal new snapshot with Event |
| `main.py:302` | Inference waits on Event, takes newest snapshot |
| `README.md:88` | Explicit CPU preprocessing/GPU inference parallelization rationale |

No checked document established a quantitative before/after speedup or that this was exactly the user's historical bug. It is an implementation reference, not an independently benchmarked solution.

### Behavior not to copy

After initial fill, the reference increments snapshot sequence and preprocesses even if no camera has received a new image. Its camera timestamps are recorded but no maximum image age or cross-camera skew gate is enforced in that parser. An inference-rate counter can therefore rise without new sensor observations. Old images can persist indefinitely, and a new parser wall timestamp can conceal their age. VLDrive must retain original sensor IDs/times and reject stale data; repeated complete observations must not be relabeled as eleven different temporal frames.

Its custom camera packet formats also differ from our current MOR/EI receiver protocol. Reuse the design, not its parser wholesale. UDP fragmentation and full LiDAR sweep assembly still require all necessary constituent packets; dropping arbitrary fragments to 'keep latest' can destroy complete samples.

## Recommended implementation order

1. Correct sampling starvation first. Separate receipt rate, VE input scheduling, model-history spacing, and Stage2 output rate. Keep a bounded timestamped history of actual sensor samples and plan usable two-second windows; do not discard every other arriving sweep solely based on time since the last accepted frame. Account for processing capacity explicitly. Do not merely pad/duplicate history or widen all time limits.
2. Snapshot small per-sensor metadata and immutable sample references first; choose a bounded-time-aligned input around an actual anchor; check age/skew/novelty/navigation before JPEG decoding and heavy transforms. Preserve the selected sample identities through preprocessing. Retain a small sensor history when needed to select the closest suitable sample.
3. Remove the requirement that every individual sensor ID advance on every anchor. Permit a recently received auxiliary view or status sample to be reused only within explicit age/skew limits, while requiring actual new anchor/visual information and preserving true timestamps. Coordinate this change with both capture and TokenHistory.accepts. Evaluate the resulting multimodal input distribution on recorded data.
4. Keep latest pending work slots and Event/Condition notification. Current queue=1 already avoids a long VE FIFO. Avoid repeatedly decoding an unchanged JPEG (cache by sensor ID). Events reduce polling overhead, but are not a major GPU compute speedup by themselves.
5. Separate packet receiving from expensive LiDAR conversion if profiling shows socket drain delays. Consume/reassemble packets correctly and discard obsolete complete frames, not arbitrary packet fragments. Prefer common simulator source timestamps only where the protocol provides a verified common clock; current post-parse receive timestamps do not prove physical synchronization.
6. Measure UDP complete-frame rate, accepted novel anchors, VE completion rate, complete-history rate, Stage2 output rate, and sensor-to-output p50/p95 independently. Include all rejection reasons (especially incomplete_history/time_alignment/all_new_gate) and resource usage. Verify actual new-input rate rather than repeated-forward count.

First expected benefit is preventing inference starvation and reducing stale-input latency. An absolute Hz improvement cannot be claimed without the controlled replay and live reception measurement after implementation.
