"""Detector adapters returning original-image [x1,y1,x2,y2,score,class] rows."""
from pathlib import Path
import logging

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
        self.runtime = getattr(args, "yolov7_runtime", "eager")
        self.postprocess_device = getattr(args, "postprocess_device", "model")
        if self.runtime not in ("eager", "cuda-graph"):
            raise ValueError("Unknown YOLOv7 runtime: " + self.runtime)
        if self.postprocess_device not in ("model", "cpu"):
            raise ValueError("Unknown postprocessing device: " + self.postprocess_device)
        self.device = select_device(args.device)
        if self.runtime == "cuda-graph" and self.device.type != "cuda":
            raise ValueError("--yolov7-runtime cuda-graph requires a CUDA device")
        self.model = attempt_load(str(Path(args.weights).expanduser().resolve()), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.size = check_img_size(args.img_size, s=self.stride)
        self.half = args.fp16 and self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()
        self.names = self.model.names
        self._graphs = {}
        self._eager_fallbacks = 0
        if self.runtime == "cuda-graph":
            self.model.to(memory_format=torch.channels_last)

    def _tensor(self, image):
        import torch
        from yolov7.utils.datasets import letterbox

        array = letterbox(image, self.size, stride=self.stride)[0]
        array = np.ascontiguousarray(array[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(array).to(self.device)
        tensor = ((tensor.half() if self.half else tensor.float()) / 255.0).unsqueeze(0)
        if self.runtime == "cuda-graph":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _capture(self, tensor):
        """Capture only configured shapes at startup; never grow the cache in detect()."""
        import torch
        shape = tuple(tensor.shape)
        if shape in self._graphs:
            return
        with torch.cuda.device(self.device), torch.no_grad():
            static = torch.empty_like(tensor)
            static.copy_(tensor)
            stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(static)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                prediction = self.model(static)[0]
            # YOLO replaces these tensors on shape changes. Keep each capture's
            # grids alive so another camera (or an eager fallback) cannot free them.
            grids = list(self.model.model[-1].grid)
            self._graphs[shape] = (graph, static, prediction, grids)

    def warmup(self, image_sizes=None):
        """Prepare (width, height) camera sizes before starting reception/timing."""
        import torch
        sizes = sorted(set(tuple(size) for size in (image_sizes or [(self.size, self.size)])))
        for width, height in sizes:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            if self.runtime == "cuda-graph":
                self._capture(self._tensor(image))
            with torch.no_grad():
                self.detect(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, tensor):
        """Single-worker API: the prediction buffer is reused on the next call."""
        if self.runtime == "cuda-graph":
            entry = self._graphs.get(tuple(tensor.shape))
            if entry is not None:
                graph, static, prediction, _ = entry
                static.copy_(tensor)
                graph.replay()
                return prediction
            self._eager_fallbacks += 1
            if self._eager_fallbacks == 1:
                logging.getLogger(__name__).warning(
                    "Unprepared camera tensor shape %s; using eager inference", tuple(tensor.shape))
        return self.model(tensor)[0]

    def runtime_info(self):
        return {"mode": self.runtime,
                "postprocess_device": self.postprocess_device,
                "channels_last": self.runtime == "cuda-graph",
                "graph_shapes": [list(shape) for shape in sorted(self._graphs)],
                "eager_fallback_frames": self._eager_fallbacks}

    def detect(self, image):
        import torch
        from yolov7.utils.general import non_max_suppression, scale_coords

        tensor = self._tensor(image)
        with torch.no_grad():
            pred = self._forward(tensor)
            if self.postprocess_device == "cpu":
                pred = pred.cpu()
            detections = non_max_suppression(
                pred, self.args.conf_thres, self.args.iou_thres,
                classes=self.args.classes, agnostic=self.args.agnostic_nms
            )[0]
        if len(detections):
            detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], image.shape)
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

    def warmup(self, image_sizes=None):
        # Initialize predictor, fuse model and select device before UDP reception.
        for width, height in sorted(set(tuple(size) for size in
                (image_sizes or [(self.args.img_size, self.args.img_size)]))):
            self.detect(np.zeros((height, width, 3), dtype=np.uint8))

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
    runtime = getattr(args, "yolov7_runtime", "eager")
    if getattr(args, "engine_manifest", None) and (args.detector != "yolov7" or runtime != "tensorrt"):
        raise ValueError("--engine-manifest applies only to --detector yolov7 --yolov7-runtime tensorrt")
    if args.detector == "yolov7":
        if runtime == "tensorrt":
            from tools.morai_tensorrt import TensorRTYOLOv7Detector
            return TensorRTYOLOv7Detector(args)
        return YOLOv7Detector(args)
    if args.detector == "yolo26":
        if getattr(args, "yolov7_runtime", "eager") != "eager":
            raise ValueError("--yolov7-runtime applies only to --detector yolov7")
        if getattr(args, "postprocess_device", "model") != "model":
            raise ValueError("--postprocess-device applies only to --detector yolov7")
        return YOLO26Detector(args)
    raise ValueError("Unknown detector: " + args.detector)
