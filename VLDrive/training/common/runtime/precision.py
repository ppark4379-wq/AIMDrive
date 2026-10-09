"""Training precision; existing implementations relocated without changing calculations."""
import torch
from contextlib import contextmanager


def autocast(model):
    device = next(model.parameters()).device.type
    return torch.autocast(device_type=device, dtype=torch.bfloat16,
                          enabled=model.config.precision == 'bf16')


@contextmanager
def causality_precision(model):
    """Compare variable-length forwards in FP32, restoring exact training tensors.

    BF16 GEMM/softmax rounding can differ when future text changes padding width.
    That numerical difference is not evidence of future-to-past attention.
    Gradient validation still runs in the configured training precision.
    """
    saved = []
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.dtype in (torch.float16, torch.bfloat16):
                saved.append((tensor, tensor.data))
                tensor.data = tensor.data.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        device = next(model.parameters()).device.type
        with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
            yield
    finally:
        for tensor, original in saved:
            tensor.data = original
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

