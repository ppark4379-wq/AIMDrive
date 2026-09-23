# MORAI RGB camera UDP input

ROS is not required. From the repository root:

```bash
python3 model/predict_udp.py --preview
# Headless capture (choose a new or empty output directory):
python3 model/predict_udp.py --output data/morai/capture_001 --max-frames 100
```

Capture needs NumPy and OpenCV (`pip install numpy opencv-python` in your environment).
It does not import TensorFlow or require model weights. `q`, Esc or Ctrl+C stops preview;
Ctrl+C stops headless capture. Port binding errors are reported immediately.

## Simulator configuration

Set each camera to RGB, UDP, and the following destination ports.
These ports are defaults chosen for this integration, not MORAI defaults.

| Camera | Width | Height | FOV (degrees) | Destination port |
|---|---:|---:|---:|---:|
| front | 1280 | 720 | 90 | 5001 |
| rear | 640 | 480 | 90 | 5002 |
| left | 640 | 480 | 130 | 5003 |
| right | 640 | 480 | 130 | 5004 |

Set Destination IP to the reachable address of the computer running this receiver.
`0.0.0.0` in `config.morai_udp.json` is a local bind address, NOT a destination IP.
When running in WSL2, use an address reachable from MORAI; Windows localhost is not
automatically the WSL2 receiver. Check routing and UDP firewall rules for these four
ports. No ROS bridge or ROS master is involved.

The supplied release name is R1.H3; its year is unconfirmed. Official documentation
describes a timestamp field added in 23.R1 and changed timestamp units in 24.R1:
https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35

The receiver checks MOR header, block size, sequential zero-based block index,
AI/EI tail, and JPEG start/end markers. It supports:
- Legacy: MOR + index(4) + size(4) + JPEG + AI/EI.
- Timestamped: MOR + timestamp(8) + index(4) + size(4) + JPEG + AI/EI.

Both exact-length datagrams and fixed 65,000-byte datagrams with arbitrary unused buffer bytes
before AI/EI are supported. Only the declared JPEG block size is appended.
Index/size are interpreted as little-endian integers. `protocol: auto` selects by
valid packet length; explicit `legacy` and `timestamped` are also available.
Actual R1.H3 packets still need simulator validation. Other layouts are rejected
instead of silently decoding incorrect offsets. Timestamp bytes remain opaque because
their representation depends on the release.

Incomplete, duplicate or out-of-order frames are dropped, with recovery at index 0.
No retransmission or reordering is attempted. Timestamped fragments must have the same
timestamp. Legacy packets lack a frame identifier, so sufficiently delayed fragments
cannot always be distinguished across frames.

One receiver thread per camera keeps draining UDP during inference. Bounded queues
are paired by host completion time (`max_skew_ms`, default 100 ms), rejecting stale
frames (`max_age_ms`, default 500 ms) and previously used frames. This is approximate
pairing, NOT exposure synchronization. The opaque simulator timestamps and host
arrival times are saved alongside every captured batch. For moving scenes, validate
the release's timestamp format and implement simulator-time pairing before evaluating
temporal accuracy.

Every five seconds the script reports completed JPEG counts and paired batches.
Zero counts indicate a network/protocol problem; increasing per-camera counts with
zero batches suggests a stopped camera or pairing tolerance problem.

## Cam2BEV inference

Capture works without calibration and weights; BEV inference requires all three
paths under `inference` in `config.morai_udp.json`:

- `segmenter`: Python file exposing `segment(rgb_image, camera_name)`.
  Return a native-resolution HxW integer NumPy array with Cam2BEV input IDs:
  0 road, 1 sidewalk (including parking under the original palette), 2 person,
  3 car, 4 truck, 5 bus, 6 bike, 7 obstacle, 8 vegetation, 9 sky.
  Load your segmentation model once at module initialization. Map its classes
  explicitly; RGB values are not semantic class IDs.
- `homographies`: Python file containing four NumPy 3x3 STN matrices in `H`,
  ordered front, rear, left, right.
- `weights`: compatible four-input, ten-input-class, ten-output-class uNetXST
  checkpoint. This checkout has no pretrained weights. The project README documents
  training and then loading `output/<timestamp>/Checkpoints/best_weights.hdf5`.

Relative paths resolve against the JSON config directory. Once populated:

```bash
python3 model/predict_udp.py --infer --preview --output data/morai/inference_001
```

TensorFlow and the original model dependencies are required for inference; see
the repository requirements. Without the three files the command fails before binding
ports. No dummy segmentation or random model predictions are used.

RGB is decoded in each camera's native resolution. Segmentation IDs are individually
center-cropped/resized to `image_shape` (default H=256, W=512), using the repository's
`resize_image_op` and each camera's own source dimensions. This preserves the existing
homography-converter convention. Do not use the front-camera 720x1280 resolution
for side/rear transforms: their original resolution is 480x640.

FOV metadata is recorded but is not sufficient to generate homographies. Set/confirm
horizontal vs vertical FOV (`fov_axis`), pinhole vs fisheye model, each camera's position
and rotation, coordinate conventions, and the desired metric BEV extent. Unknown
values are intentionally null. For a centered, square-pixel pinhole camera, horizontal
FOV gives fx=fy=width/(2*tan(FOV/2)); vertical FOV uses height instead. Do not assume
an axis or use a pinhole transform on an unrectified fisheye image.

Generate OpenCV matrices using `preprocessing/ipm/ipm.py -v`, then convert each
with `preprocessing/homography_converter/homography_converter.py`, specifying the
correct source/output resolutions. See that directory's README. Changing camera
geometry also requires model validation and potentially retraining.

Outputs are paired RGB JPGs, JSON timing metadata and, with `--infer`, BEV and
drivable PNGs. Road channel 0 must win argmax and pass `road_threshold` (default 0.7).
The mask is a road candidate mask, not a vehicle-clearance/collision guarantee.
Occluded cells are not marked drivable. Output directories must be empty to prevent
overwriting previous captures.

## Verification

```bash
cd model
python3 -m unittest test_morai_udp -v
```

Tests cover both protocol layouts, split frames, malformed/lost/duplicate/reordered
packets, timestamp mismatch, size/time limits, stale pairing and four real loopback
UDP sockets. MORAI live transport and trained BEV quality require simulator testing.
