# Refign-HRDA ACDC trial

Status: upstream source and inference adapter prepared. No pretrained MORAI inference
has been run: the official ETH checkpoint download returns HTTP 429 (rate limited).
No Refign prediction images or accuracy claims have been generated.

Upstream source: https://github.com/brdav/refign
Pinned checkout: 4efc6a8db31a8624068612e85b21fd841ba8bfd0
Checkpoint: refign_hrda_star_acdc.ckpt (1,770,404,093 bytes)
Official new URL:
https://www.research-collection.ethz.ch/server/api/core/bitstreams/3c0a4730-74c2-4ce5-ae51-9b5b6c2f4b58/content
Original ETH record: https://hdl.handle.net/20.500.11850/626140

The upstream README's old /bitstream/handle URL returned HTML instead of weights.
The replacement URL and published MD5 were found through the public ETH DSpace API.
Metadata is saved as data/models/refign_acdc_source.json.

From the Linux repository directory:

```bash
python3 scripts/download_refign_acdc.py
.venv-segmentation/bin/python scripts/segment_morai_capture.py --segmenter refign --capture data/morai/capture_windows_007 --output data/morai/segmentation_refign_probe_007 --max-frames 1
```

If downloading manually, save data/models/refign_hrda_star_acdc.ckpt, then run the
download script again; it will verify the existing file instead of downloading it.
Only after the four-camera probe succeeds, use a new output folder and omit
--max-frames to process all 40 images.

Adapter: model/segmenters/refign_hrda.py.
Reuses upstream MiT-B5, DAFormerHead, scale attention and official HRDA decorators.
Only inference modules are instantiated; no teacher/alignment/training dependencies.
Original RGB ImageNet normalization, 19 Cityscapes train IDs -> ten Cam2BEV channels.
Native dimensions must be divisible by 8; all four current MORAI resolutions are.
Outer sliding windows use crop1080/stride420; inner crop batches run sequentially on CPU.
Original B2 remains default. No ongoing training jobs were changed.

Validation completed:
- Official modules import in .venv-segmentation (PyTorch CPU).
- Synthetic 64x64 HRDA structural test with the small MiT-B0 backbone.
- Sequential vs official batched HRDA computation parity: max error 2.384e-7.
- Syntax checks pass.
Not yet validated: checkpoint deserialization/key matching, B5 inference memory,
native-resolution trained outputs, speed and MORAI segmentation quality.
These synthetic tests do not establish trained-model accuracy.

Next: verify checkpoint, run four-camera probe, visually compare B2 and Refign on
identical frames (including the box/pedestrian), then process 40 frames if useful.
