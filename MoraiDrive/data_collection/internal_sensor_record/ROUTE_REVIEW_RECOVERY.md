# Sensor Record route review recovery

Route auditing and manifest generation resolve the map and route from the run's
sensor_contract.json and episode_000/episode.json. They check the recorded link,
node, route-definition and global-path hashes before using those files. No fallback
to the current checkout's default map is allowed. The legacy runtime layout locates
config/full_loop_links.yaml relative to the recorded map root and verifies its hash.

Normal new collection/review commands are unchanged. An existing sealed route audit
that used another map is rejected rather than silently skipped.

For an existing campaign:
```bash
cd ~/AIMDrive/MoraiDrive
./data_collection/tools/03_validate_internal_sensor_record.sh scenario_campaign_20260915_010309 --rebuild-route-review
```

This explicitly archives route_signal_audit_v1, review_closeout and admission_v1
under each episode's route_review_history/<unique timestamp>/, then regenerates
those dependent stages and runs normal campaign validation and manifest creation.
Raw ROS chunks, native sensor files, timestamps and unrelated sealed stages are
not changed. Previous campaign validation directories remain historical evidence;
their hashes refer to the old archived outputs, not the replacement directories.
Use the NEW successful campaign report and manifests for subsequent processing.

If interrupted, run ordinary 03 again without --rebuild-route-review to reuse newly
sealed stages. An incomplete stage is preserved and reported instead of overwritten.
A run with ADMISSION_RECOVERY.json requires separate reconciliation and is rejected
before anything is moved. Concurrent capture/training/review locks prevent rebuilding
while those cooperating processes own the dataset.

This does not add NAS transfer or claim that diagnostic collision laps are approved
expert demonstrations. Existing validation, admission and task coverage checks remain.
