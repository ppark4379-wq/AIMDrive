# Pretrained RGB segmentation frontend

Use NVIDIA's Cityscapes-finetuned SegFormer B0 without additional training:
https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024

The adapter exposes `segment(rgb_image, camera_name)` as required by
`model/predict_udp.py`. It loads weights on first use (internet required unless a
local model directory is provided), uses eval/inference mode, maps the 19 Cityscapes
classes to the 10 Cam2BEV channels and returns IDs at the original camera resolution.

CPU is the default to avoid competing with ongoing GPU training. This is a functional
test setting, not a real-time speed claim. `CAM2BEV_SEGMENTER_DEVICE=cuda` explicitly
selects GPU if a compatible PyTorch installation and free GPU memory are available.
`CAM2BEV_SEGMENTER_MODEL=/path/to/model` selects a local Hugging Face model directory
with the same 19 classes.

## Test segmentation independently

From the repository root, create a separate Python 3.11 environment. This does not
modify the running training environment:

```bash
.venv-gpu/bin/python -m venv .venv-segmentation
.venv-segmentation/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-segmentation/bin/python -m pip install "transformers>=4.46,<5" "numpy<2" pillow
.venv-segmentation/bin/python model/segmenters/segformer_cityscapes.py \
  data/morai/capture_001/front/00000000.jpg --output data/morai/segmentation_front_001
```

The input path above is an example: capture MORAI RGB frames first with predict_udp.py.
Use a new output directory for each test. Outputs are class_ids.png (IDs 0..9),
semantic.png (Cam2BEV palette), overlay.png and road_mask.png. These are camera-view
results, not BEV. Check all four views, especially road/sidewalk boundaries and vehicles.

Full UDP BEV inference requires this adapter's PyTorch/Transformers dependencies and
Cam2BEV's TensorFlow dependencies in the SAME inference environment. The standalone
segmentation environment above intentionally only tests the RGB frontend. Set
`inference.segmenter` to `segmenters/segformer_cityscapes.py`; calibration matrices
and a frozen Cam2BEV checkpoint are still required. Do not point an ongoing inference
run at a checkpoint being overwritten by training: copy a completed checkpoint first.

## Class mapping

Road -> road; sidewalk -> sidewalk; person -> person; car -> car; truck -> truck;
bus -> bus; rider/motorcycle/bicycle -> bike; vegetation/terrain -> vegetation;
sky -> sky; building/wall/fence/pole/traffic light/traffic sign/train -> obstacle.

Mapping is by numeric semantic class IDs, not exact RGB color matching. In particular,
the custom pedestrian/rider colors in the original Cam2BEV data do not change this
semantic mapping. Model output logits are restored to native HxW before argmax.

## Calibration status

User supplied positions and rotations are saved in config.morai_udp.json. Rotations
are recorded as MORAI roll/pitch/yaw degrees in their original form, including right
yaw=290 (equivalent angle -70). These source values are not yet converted to the
Cam2BEV projection convention. The user confirmed ideal pinhole cameras with horizontal FOV. Native-resolution
intrinsics are recorded in config.morai_udp.json using a centered principal point
and square pixels: fx=fy=width/(2*tan(horizontal_FOV/2)). Extrinsic rotation convention
conversion and the resulting BEV homographies still need to be completed/validated.
The BEV extent should initially match the trained 1_FRLR output geometry.
