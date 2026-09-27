"""Training logging; existing implementations relocated without changing calculations."""
import math
import torch
from utils.visualization.training_dashboard.telemetry import TrainingTelemetry


def gradient_report(model):
    values = {}
    for group, prefix in model.gradient_groups().items():
        gradients = [p.grad.detach().float() for n, p in model.named_parameters()
                     if n.startswith(prefix) and p.requires_grad and p.grad is not None]
        norm = math.sqrt(sum(g.square().sum().item() for g in gradients))
        values[group] = dict(l2=norm, finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                             passed=bool(gradients) and math.isfinite(norm) and norm > 0)
    bad_parameters = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
                      and not bool(torch.isfinite(p.grad).all())]
    return dict(groups=values, nonfinite_parameters=bad_parameters,
                passed=all(v['passed'] for v in values.values()) and not bad_parameters)
