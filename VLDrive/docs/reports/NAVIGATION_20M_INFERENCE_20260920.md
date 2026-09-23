# Navigation instruction activation at 20 m

- Active observer policy: `configs/navigation_lmdrive_20m.json`, `lookahead_m=20` (previously 30).
- Offline labels: `data/processed/instructions/basic_drive_lmdrive_20m`; 1 episode, 2,990 valid frames, 487 instruction instances, 32 observed instruction types.
- Inference text IDs: `data/processed/navigation_tokens/basic_drive_live_20m`; 1,192 unique texts, maximum 32 tokens. All eight variants of recorded instances are cached. New live strings, including changing distance arguments, use the same local tokenizer on demand.
- Generation settings: `configs/navigation_inference_tokens.json`.
- Runtime hook: `inference/navigation_tokens.py`; checks tokenizer, labels, visual manifest and live policy. Live sensor timeout is intentionally allowed to differ from offline frame spacing.
- Existing training labels, training text cache, encoder cache, model weights, PID settings, speed and compound lookahead are unchanged. This new text cache is an inference cache, not a rebuilt epoch/window training cache.

## Verification

Replayed identical poses from `outputs/inference/vldrive_20260920_024427_703568/navigation.jsonl`:

| Policy | First left-turn compound instruction, remaining route distance |
|---|---:|
| 30 m | 29.5051 m |
| 20 m | 19.4561 m |

The threshold is checked at sensor updates, so activation occurs on the first available frame within 20 m. Distance is measured along the route to the annotated event start, not to the visual center of an intersection. This shared threshold also applies to roundabouts and other upcoming maneuver events. Existing compound instructions can describe a second maneuver farther ahead.

Best checkpoint `outputs/logs/vldrive_20260919_182054_301289/model-best.pt` restored successfully. A recorded 11-frame feature-window smoke test produced five finite waypoints with the new cached instruction. Cached IDs matched direct tokenizer IDs exactly (2 cache hits including the comparison, 0 misses). The smoke test intentionally verifies wiring, not scene-specific driving accuracy. Details: `outputs/inference/navigation_20m_verification.json`.

## Run from VLDrive

Live UDP verification completed without control transmission: `outputs/inference/vldrive_20260920_035405_585476/inference_report.json`. During 20.10 seconds, the best model produced 71 waypoint predictions (3.53 outputs/s), with 71 text-cache hits and zero runtime tokenizations. This stationary/current-scene run confirmed sensor-to-model wiring; the intersection threshold comparison above used replay, not a new driven turn.

Regression checks: realtime model tests 11 passed. The full suite initially had 355 passed, 1 skipped, 30 passing subtests and three failures because mocked model fixtures inherited the newly enabled real tokenizer cache. Those fixtures now explicitly disable the optional cache; both affected test modules pass (67 tests). Runtime model source/provenance checks passed. Temporary verification code was removed; JSON evidence and this report remain.

```bash
python inference/realtime_vldrive.py
```

To use the previously connected PID/UDP controller:

```bash
python inference/realtime_vldrive.py --control
```

Restart a running inference process to load the new configuration. No retraining is required to run this experiment. Whether steering itself now starts at the correct place still needs a driving comparison; delaying the instruction does not guarantee a matching steering delay.
