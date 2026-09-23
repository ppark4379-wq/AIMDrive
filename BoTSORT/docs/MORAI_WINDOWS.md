# Windows MORAI camera inference

Environment: Python 3.10 in the botsort Conda environment.
The setup uses torch 2.7.1 + torchvision 0.22.1 with CUDA 12.8 for RTX 5070.

## Install (Anaconda Prompt)

    conda activate botsort
    cd /d "<AIMDrive clone path>\BoTSORT"
    python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements-morai-windows.txt

Use this camera-only requirements file, not the old root requirements/setup.py.
YOLOX training/export and FastReID are not needed for this entry point.
The optional cython_bbox extension falls back to a NumPy implementation with
the same float64, inclusive (+1) pixel IoU convention.

The YOLOv7 loader explicitly uses weights_only=False for model-object
checkpoints on recent PyTorch. Only use checkpoints from a trusted source.

## Run (Anaconda Prompt)

    conda activate botsort
    cd /d "<AIMDrive clone path>\BoTSORT"
    python tools\morai_udp_mot.py --weights pretrained\yolov7.pt --port 15001 --device 0 --fp16 --img-size 640 --view --show-detections --output runs\morai\tracks.jsonl

Start with 640 because MORAI shares the GPU. Increase --img-size to 1280
when memory and processing time permit. The output boxes always use the
original camera resolution (1280 x 720). Lower inference resolution can
affect detection accuracy; keep it fixed when comparing experiments.

MORAI Camera-1:
- Host IP 127.0.0.1, Host Port 5010
- Destination IP 127.0.0.1, Destination Port 15001
- UDP; RGB/Ground Truth=None; 1280 x 720; 20 Hz
- Capture Mode disabled, network connected, simulation unpaused.

Only one receiver may use port 15001; close the PowerShell packet test first.
Press Q/ESC in the viewer or Ctrl+C in the terminal to stop.
Green boxes show track IDs; orange boxes show YOLO detections.
JSONL contains tracks, postprocessed raw detections, timing and drop counters.
The output file is overwritten on each run; use a new path for each experiment.

Weights: pretrained/yolov7.pt, official YOLOv7 COCO model.
Download provenance and local SHA256: pretrained/yolov7.source.json.

References:
- https://pytorch.org/blog/pytorch-2-7/
- https://pytorch.org/get-started/previous-versions/
- https://github.com/WongKinYiu/yolov7/releases/tag/v0.1

## Validation on this machine

RTX 5070: CUDA matrix operation and official YOLOv7 model load passed.
Windows unit tests: 12 passed; pip check: no broken requirements.
Live MORAI UDP run: 20 processed frames, 1280 x 720 input, 640 FP16 inference.
Last frame: 4 tracks; no invalid packets or incomplete frames during the run.
Median processing after first frame: 125.0 ms.
60 complete frames arrived; 39 were overwritten while inference was busy.
This run does not meet full 20 Hz processing. Do not treat it as a full-rate MOT benchmark.
Results: runs/morai/windows_smoke.jsonl and windows_smoke_summary.json.

## YOLO26 COCO detector with the original BoT-SORT

The default detector remains YOLOv7. To try the downloaded official YOLO26n
COCO detection checkpoint, stop the running receiver and run:

    python tools\morai_udp_mot.py --detector yolo26 --weights pretrained\yolo26n.pt --port 15001 --device 0 --fp16 --img-size 640 --view --show-detections --output runs\morai\yolo26_tracks.jsonl

The adapter uses Ultralytics predict(), not track(). Original-image boxes,
confidence and class IDs go into this repository's existing BoT-SORT.
UDP settings and the JSONL track ID / 2D bbox output are unchanged.
Both official checkpoints have COCO's 80 classes; there is no dedicated
cardboard-box/pallet class. A newer model does not guarantee better MORAI
accuracy. Compare the same images and inference size before choosing.
Ultralytics 8.4.160 is pinned in requirements-morai-windows.txt; --fp16 maps
to its quantize=16 prediction option.

Validation: 14 unit tests passed. YOLO26n CUDA FP16 detection plus original
BoT-SORT ran on 10 saved MORAI images; the last frame had 3 confirmed tracks.
Results: runs/morai/yolo26_recorded_validation.jsonl.
This is recorded-image validation, not a live UDP or 20 Hz speed benchmark.
Weights provenance: pretrained/yolo26n.source.json.
Official model documentation: https://docs.ultralytics.com/models/yolo26/
