# MORAI offline inference repair

The saved capture_007 has provisional logical-camera -> capture-folder mapping:
front->front, rear->right, left->rear, right->left. Do not change live UDP ports until
the simulator configuration is checked. Folder names from old captures never change.

Changes:
- Default RGB segmenter: NVIDIA SegFormer B2 Cityscapes, replacing B0 after a four-camera comparison.
- Gamma correction is an optional experiment; gamma=1 remains default. Gamma=0.5 did not fix the errors.
- Optional per-camera BEV visibility masks exclude negative camera depth and out-of-image/crop samples in uNetXST.
  Existing training calls omit the option and retain the original architecture behavior.
- Ground plane z is configurable in the same coordinate frame as camera position. Default z=0 is unverified.
- Original BEV road predictions and geometric support remain separate from road_supported,
  which intersects the BEV road with selected-camera ground-projected semantic road.
  None of these implements complete occlusion reasoning or vehicle clearance.
- Results record exact checkpoint hash, segmenter model/revision, mapping, calibration and masking option.
- HTML uses UTF-8-compatible numeric entities to survive Windows PowerShell pipeline encoding.

Run from Linux repository directory, using the existing environments:

```bash
HF_HOME=$PWD/data/models/huggingface .venv-segmentation/bin/python scripts/segment_morai_capture.py --capture data/morai/capture_windows_007 --output data/morai/segmentation_new
OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 .venv-gpu/bin/python scripts/infer_morai_bev.py --capture data/morai/capture_windows_007 --segmentation data/morai/segmentation_new --run data/morai/inference_new --source-map data/morai/inference_007/source_mapping.json --weights data/morai/inference_007/frozen_best_road_weights.hdf5 --ground-z 0
python3 scripts/inspect_morai_geometry.py --capture data/morai/capture_windows_007 --source-map data/morai/inference_007/source_mapping.json --output data/morai/calibration_new --ground-z 0
```

These commands process Windows UDP captures offline in WSL; they do not establish Windows-to-WSL UDP transport.
The configured mapping is a visual estimate until simulator settings are confirmed.
Use --legacy-unmasked for a controlled comparison with visibility masking disabled.
Use a new output directory per experiment. Original results are not overwritten.

Validation: 19 geometry/UDP tests passed; B2 produced 40 segmentation images;
both masked B0 and masked B2 produced 10 finite BEV predictions using identical frozen weights.
Qualitative improvement is visible on this capture, but small objects and near-vehicle omissions remain.
No MORAI ground truth was available, so no IoU or accuracy improvement is claimed.
