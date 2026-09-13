# Native collector: save failures and frame completion

The `sync_native_files` backend publishes each camera, LiDAR, JSON and NPY file
through a temporary file in the destination directory, followed by `fsync` and
`os.replace`. A false return from OpenCV image saving is a collection error.

The background writer saves sensors first, then labels. Only after all required
files exist does it write `.frame_commits/NNNN.json`, containing their relative
paths, sizes and SHA-256 digests. This receipt is the frame completion point.
The frame count and `capture_provenance.jsonl` advance after that point. Logs
show `queued samples` separately from `saved`; queued frames are not complete.

If a writer fails, the collector raises an error and reports the number actually
committed. The checkpoint records the last committed native frame. Uncommitted
files are retained under `.incomplete_frames/`, outside per-frame training
folders. The collector does not stop Sync Runtime.

On restart with the same manifest, completion receipts and actual payload hashes
are checked before appending. Stale checkpoints and missing/truncated provenance
are reconstructed from receipts. Crash leftovers are quarantined, and the first
uncommitted sample index is reused. A missing or modified committed payload
stops resume with its path instead of silently counting it as valid. Retrying
requires the original native sensor files and, when enabled, matching label
spool records to remain available.

Pre-change datasets do not have completion receipts. Automatic append to those
folders is refused, because their labels alone cannot prove successful sensor
saving. Preserve the old dataset and start a new Sync Runtime capture session
for collection with the new format. Existing folders are never overwritten by a
new collection run. Run one collector per output/manifest.

This change covers save failure detection and completion/resume accounting.
It does not strengthen original PNG CRC validation or repair previously damaged
images. Stored digests detect later byte changes, not image correctness at the
source. Atomic replacement protects against partial publication during process
interruption; it is not a guarantee against hardware faults or power loss across
WSL/Windows storage.

Validation includes injected failures in image/LiDAR/label/receipt writes,
stale and torn checkpoints, interrupted provenance, missing/corrupted payloads,
and collector failure followed by a retry without frame gaps or duplication.
