# Model weights

Model binaries are excluded from Git. The default MORAI detector is YOLOv7.
From the BoTSORT directory, download the official COCO weights in PowerShell:

```powershell
New-Item -ItemType Directory -Force pretrained | Out-Null
Invoke-WebRequest -Uri https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -OutFile pretrained/yolov7.pt
Get-FileHash pretrained/yolov7.pt -Algorithm SHA256
```

Expected SHA256: `d1ac3c74eb96a3eec77949c0f37a06bc272756606ff011d9353f7abff4e0c71d`.
See the adjacent source JSON files for provenance of previously tested models.
YOLO26n is optional; the current preferred MORAI baseline is YOLOv7.
