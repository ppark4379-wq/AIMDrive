#!/usr/bin/env python3
"""Build YOLOv7 FP16 engines for the camera config on the deployment GPU."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'yolov7')]


def build(args):
    import cv2
    import numpy as np
    import onnx
    import tensorrt as trt
    import torch
    from camera_mot.config import load_config
    from tools.morai_detectors import YOLOv7Detector
    from tools.morai_tensorrt import TensorRTYOLOv7Detector

    if not trt.__version__.startswith('10.'):
        raise ValueError('This builder uses TensorRT 10.x APIs; verified with 10.9.0.34')
    if args.device == 'cpu' or ',' in args.device or args.img_size <= 0 or args.workspace_mib <= 0:
        raise ValueError('Require one CUDA device and positive input/workspace sizes')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config = load_config(args.config, args.cameras)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = SimpleNamespace(weights=args.weights, device=args.device, img_size=args.img_size,
        fp16=True, yolov7_runtime='eager', postprocess_device='model', conf_thres=.05,
        iou_thres=.5, classes=None, agnostic_nms=False)
    detector = YOLOv7Detector(options)
    sizes = sorted({tuple(c['image_size']) for c in config['cameras'].values()})
    shapes = sorted({tuple(detector._tensor(np.zeros((h, w, 3), np.uint8)).shape) for w, h in sizes})
    names = detector.names
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    metadata = {'schema_version': 1, 'precision': 'fp16', 'size': detector.size,
        'weights_sha256': hashlib.sha256(Path(args.weights).expanduser().read_bytes()).hexdigest(),
        'names': names, 'stride': detector.stride, 'torch': torch.__version__,
        'torch_cuda': torch.version.cuda, 'onnx': onnx.__version__, 'tensorrt': trt.__version__,
        'gpu': torch.cuda.get_device_name(detector.device), 'platform': platform.system(),
        'machine': platform.machine(), 'camera_image_sizes': sizes,
        'workspace_mib': args.workspace_mib, 'engines': []}

    class PredictionOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            return self.model(image)[0]

    model = PredictionOnly(detector.model).eval()
    logger = trt.Logger(trt.Logger.WARNING)
    cache_bytes = Path(args.timing_cache).read_bytes() if args.timing_cache else b''
    with torch.cuda.device(detector.device), torch.no_grad():
        for shape in shapes:
            stem = 'yolov7_%dx%d' % shape[2:]
            onnx_path, engine_path = output / (stem + '.onnx'), output / (stem + '.engine')
            sample = torch.zeros(shape, device=detector.device, dtype=torch.float16)
            model(sample)
            torch.onnx.export(model, sample, str(onnx_path), opset_version=12,
                input_names=['images'], output_names=['predictions'], dynamo=False,
                do_constant_folding=True)
            onnx.checker.check_model(onnx.load(str(onnx_path)))
            started = time.monotonic()
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            build_config = builder.create_builder_config()
            build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
            build_config.set_flag(trt.BuilderFlag.FP16)
            build_config.builder_optimization_level = 3
            cache = build_config.create_timing_cache(cache_bytes)
            if not build_config.set_timing_cache(cache, False):
                raise ValueError('Incompatible timing cache; rebuild without --timing-cache')
            serialized = builder.build_serialized_network(network, build_config)
            if serialized is None:
                raise RuntimeError('TensorRT engine build failed')
            engine_path.write_bytes(bytes(serialized))
            cache_bytes = bytes(build_config.get_timing_cache().serialize())
            (output / 'tactics.cache').write_bytes(cache_bytes)
            metadata['engines'].append({'shape': list(shape), 'path': engine_path.name,
                'sha256': hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                'onnx_sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                'build_s': time.monotonic() - started})
            print('Built', engine_path.name, flush=True)
    # Only a complete and loadable set receives the final manifest name.
    pending = output / 'engines.pending.json'
    pending.write_text(json.dumps(metadata, indent=2) + '\n')
    options.img_size = detector.size
    options.engine_manifest = str(pending)
    candidate = TensorRTYOLOv7Detector(options)
    candidate.warmup(sizes)
    pending.rename(output / 'engines.json')
    print('Ready:', output / 'engines.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True, help='New directory; never overwrites prior engines')
    parser.add_argument('--config', default=str(ROOT / 'configs/morai_three_cameras.yaml'))
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--img-size', type=int, default=960)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workspace-mib', type=int, default=1024)
    parser.add_argument('--timing-cache', help='Optional compatible cache; never modifies the source cache')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
