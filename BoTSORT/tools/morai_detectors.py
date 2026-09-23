"""Detector adapters returning original-image [x1,y1,x2,y2,score,class] rows."""
from pathlib import Path

import numpy as np


def ultralytics_detections(result):
    """Keep Ultralytics' original-image coordinates; do not apply NMS twice."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack((
        boxes.xyxy.detach().cpu().numpy(),
        boxes.conf.detach().cpu().numpy(),
        boxes.cls.detach().cpu().numpy(),
    )).astype(np.float32, copy=False)


class YOLOv7Detector:
    def __init__(self, args):
        import torch
        from yolov7.models.experimental import attempt_load
        from yolov7.utils.general import check_img_size
        from yolov7.utils.torch_utils import select_device

        self.args = args
        self.device = select_device(args.device)
        self.model = attempt_load(str(Path(args.weights).expanduser().resolve()), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.size = check_img_size(args.img_size, s=self.stride)
        self.half = args.fp16 and self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()
        self.names = self.model.names

    def warmup(self):
        import torch
        if self.device.type != "cpu":
            with torch.no_grad():
                self.model(torch.zeros(1, 3, self.size, self.size, device=self.device)
                           .type_as(next(self.model.parameters())))
            torch.cuda.synchronize(self.device)

    def detect(self, image):
        import torch
        from yolov7.utils.datasets import letterbox
        from yolov7.utils.general import non_max_suppression, scale_coords

        tensor_image = letterbox(image, self.size, stride=self.stride)[0]
        tensor_image = np.ascontiguousarray(tensor_image[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(tensor_image).to(self.device)
        tensor = (tensor.half() if self.half else tensor.float()) / 255.0
        with torch.no_grad():
            pred = self.model(tensor.unsqueeze(0))[0]
            detections = non_max_suppression(
                pred, self.args.conf_thres, self.args.iou_thres,
                classes=self.args.classes, agnostic=self.args.agnostic_nms
            )[0]
        if len(detections):
            detections[:, :4] = scale_coords(tensor.shape[1:], detections[:, :4], image.shape)
            return detections.detach().cpu().numpy()
        return np.empty((0, 6), dtype=np.float32)


class YOLO26Detector:
    def __init__(self, args):
        from ultralytics import YOLO
        self.args = args
        self.model = YOLO(str(Path(args.weights).expanduser().resolve()))
        if self.model.task != "detect":
            raise ValueError("YOLO26 adapter requires object-detection weights, not segment/pose/obb")
        self.names = self.model.names

    def warmup(self):
        # Initialize predictor, fuse model and select device before UDP reception.
        self.detect(np.zeros((self.args.img_size, self.args.img_size, 3), dtype=np.uint8))

    def detect(self, image):
        result = self.model.predict(
            source=image, imgsz=self.args.img_size, device=self.args.device,
            quantize=16 if self.args.fp16 and self.args.device != "cpu" else 32,
            conf=self.args.conf_thres, iou=self.args.iou_thres,
            classes=self.args.classes, agnostic_nms=self.args.agnostic_nms,
            verbose=False, save=False, stream=False,
        )[0]
        # Use predict(), not track(): tracking belongs to this repository's BoT-SORT.
        return ultralytics_detections(result)


def create_detector(args):
    if args.detector == "yolov7":
        return YOLOv7Detector(args)
    if args.detector == "yolo26":
        return YOLO26Detector(args)
    raise ValueError("Unknown detector: " + args.detector)
