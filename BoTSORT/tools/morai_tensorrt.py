"""Optional static-shape YOLOv7 TensorRT execution; reuse existing preprocessing/NMS."""
import hashlib
import json
import platform
from pathlib import Path

from tools.morai_detectors import YOLOv7Detector


def load_engine_manifest(path, weights, image_size):
    """Reject stale/mismatched artifacts before deserializing an engine."""
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    if metadata.get('schema_version') != 1 or metadata.get('precision') != 'fp16':
        raise ValueError('Unsupported engine manifest; rebuild with build_morai_tensorrt.py')
    if metadata['size'] != image_size:
        raise ValueError('Engine input size differs from --img-size; rebuild or match the input size')
    if hashlib.sha256(Path(weights).expanduser().read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise ValueError('Engine/source weights mismatch; rebuild for the selected weights')
    names, stride = metadata['names'], metadata['stride']
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError('Engine manifest requires an ordered class-name list')
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError('Invalid engine stride')
    specs = metadata['engines']
    if not specs:
        raise ValueError('Engine manifest has no engines')
    shapes = set()
    for spec in specs:
        shape = tuple(spec['shape'])
        if (len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape)
                or shape[:2] != (1, 3) or any(n % stride or n > image_size for n in shape[2:])):
            raise ValueError('Invalid static engine input shape: ' + str(shape))
        if shape in shapes:
            raise ValueError('Duplicate engine input shape: ' + str(shape))
        shapes.add(shape)
    return path, metadata


def read_engine_bytes(manifest_path, spec):
    path = (manifest_path.parent / spec['path']).resolve()
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec['sha256']:
        raise ValueError('Engine hash mismatch: ' + str(path))
    return blob


class TensorRTYOLOv7Detector(YOLOv7Detector):
    def __init__(self, args):
        if not getattr(args, 'engine_manifest', None):
            raise ValueError('--yolov7-runtime tensorrt requires --engine-manifest')
        if not args.fp16 or args.device == 'cpu' or ',' in args.device:
            raise ValueError('TensorRT requires --fp16 and one CUDA device')
        self.manifest_path, self.metadata = load_engine_manifest(
            args.engine_manifest, args.weights, args.img_size)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError('TensorRT is not installed in this Python environment; '
                               'see docs/MORAI_MULTI_CAMERA.md') from exc
        import torch
        from yolov7.utils.torch_utils import select_device

        self.args = args
        self.device = select_device(args.device)
        self.runtime = 'tensorrt'
        self.postprocess_device = getattr(args, 'postprocess_device', 'model')
        if self.postprocess_device not in ('model', 'cpu'):
            raise ValueError('Unknown postprocessing device: ' + self.postprocess_device)
        self.size, self.stride, self.half = args.img_size, self.metadata['stride'], True
        self.names = self.metadata['names']
        expected = {'tensorrt': trt.__version__, 'gpu': torch.cuda.get_device_name(self.device),
                    'platform': platform.system(), 'machine': platform.machine()}
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError('Engine %s mismatch (%r != %r); rebuild on this environment'
                                 % (key, self.metadata.get(key), value))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.logger)
        self.execution_stream = torch.cuda.Stream(device=self.device)
        self.engines = {}
        with torch.cuda.device(self.device):
            for spec in self.metadata['engines']:
                engine = self.trt_runtime.deserialize_cuda_engine(read_engine_bytes(self.manifest_path, spec))
                if engine is None:
                    raise RuntimeError('TensorRT could not load ' + spec['path'])
                shape = tuple(spec['shape'])
                if (engine.num_io_tensors != 2 or
                        {engine.get_tensor_name(i) for i in range(2)} != {'images', 'predictions'} or
                        engine.get_tensor_mode('images') != trt.TensorIOMode.INPUT or
                        engine.get_tensor_mode('predictions') != trt.TensorIOMode.OUTPUT or
                        tuple(engine.get_tensor_shape('images')) != shape or
                        engine.get_tensor_dtype('images') != trt.float16):
                    raise ValueError('Unexpected TensorRT input/output contract: ' + spec['path'])
                output_shape = tuple(engine.get_tensor_shape('predictions'))
                if (len(output_shape) != 3 or output_shape[0] != 1 or output_shape[1] <= 0 or
                        output_shape[2] != 5 + len(self.names)):
                    raise ValueError('TensorRT output does not match the declared classes')
                dtype = {trt.float16: torch.float16, trt.float32: torch.float32}.get(
                    engine.get_tensor_dtype('predictions'))
                if dtype is None:
                    raise ValueError('Unsupported TensorRT output dtype')
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError('TensorRT execution context allocation failed')
                output = torch.empty(output_shape, device=self.device, dtype=dtype)
                if not context.set_tensor_address('predictions', output.data_ptr()):
                    raise RuntimeError('TensorRT output binding failed')
                self.engines[shape] = (engine, context, output)

    def _forward(self, tensor):
        import torch
        shape = tuple(tensor.shape)
        if shape not in self.engines:
            raise ValueError('No TensorRT engine for input shape %s; rebuild for the camera config' % (shape,))
        if tensor.device != self.device or tensor.dtype != torch.float16 or not tensor.is_contiguous():
            raise ValueError('TensorRT input must be contiguous FP16 on the selected CUDA device')
        _, context, output = self.engines[shape]
        if not context.set_tensor_address('images', tensor.data_ptr()):
            raise RuntimeError('TensorRT input binding failed')
        current = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(current)
        if not context.execute_async_v3(self.execution_stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        current.wait_stream(self.execution_stream)
        # Preserve the PyTorch FP16 prediction contract before the shared NMS.
        return output.to(torch.float16)

    def runtime_info(self):
        return {'mode': self.runtime, 'postprocess_device': self.postprocess_device,
                'precision': 'fp16', 'dedicated_cuda_stream': True,
                'tensorrt': self.metadata['tensorrt'], 'gpu': self.metadata['gpu'],
                'engine_manifest': str(self.manifest_path),
                'engine_manifest_sha256': hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                'engine_shapes': [list(s) for s in sorted(self.engines)],
                'engine_sha256': {e['path']: e['sha256'] for e in self.metadata['engines']}}
