"""Training precision; existing implementations relocated without changing calculations."""
import torch


def autocast(model):
    device = next(model.parameters()).device.type
    return torch.autocast(device_type=device, dtype=torch.bfloat16,
                          enabled=model.config.precision == 'bf16')
