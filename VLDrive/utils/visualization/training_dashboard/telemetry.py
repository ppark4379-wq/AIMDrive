"""Detached observations only: no forward, random draws, or gradient mutation."""
import math
import platform


class TrainingTelemetry:
    def __init__(self, model, device, *, enabled=True, gradient_interval=100):
        self.model, self.device = model, device
        self.enabled, self.gradient_interval = enabled, gradient_interval
        self.handle, self.pending, self.error = None, None, None

    def __enter__(self):
        if self.enabled and hasattr(self.model, 'connector'):
            try:
                self.handle = self.model.connector.register_forward_hook(self._observe)
            except Exception as exc:
                self.error = type(exc).__name__ + ': ' + str(exc)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
        self.pending = None

    def _observe(self, module, inputs, output):
        if not module.training:
            return
        try:
            import torch
            with torch.no_grad():
                probability = output['selector']['probabilities'][..., 1].detach().float()
                count = output['selected_count_per_frame'].detach().float()
                self.pending = torch.stack((probability.std(unbiased=False), count.mean(), count.min(), count.max()))
        except Exception as exc:
            self.pending = None
            self.error = type(exc).__name__ + ': ' + str(exc)

    def gradients(self, update):
        if not self.enabled or not self.gradient_interval or update % self.gradient_interval:
            return None
        try:
            import torch
            declared = self.model.gradient_groups() if hasattr(self.model, 'gradient_groups') else {}
            groups = {name: declared[key] for name, key in (
                ('selector', 'selector'), ('mefa', 'mefa'), ('qformer', 'qformer'),
                ('connector_projection', 'connector_projection'), ('waypoint_head', 'waypoint')) if key in declared}
            # The published gradient_groups probes only the first/last LM layers.
            # A full TinyLlama bar must include all trainable decoder/embedding gradients.
            if 'text_embeddings' in declared:
                groups['tinyllama'] = declared['text_embeddings'].split('.model.')[0] + '.'
            result = {}
            with torch.no_grad():
                for group, prefix in groups.items():
                    norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
                             for name, p in self.model.named_parameters()
                             if name.startswith(prefix) and p.requires_grad and p.grad is not None]
                    if norms:
                        value = float(torch.linalg.vector_norm(torch.stack(norms)))
                        result[group] = value if math.isfinite(value) else None
            return result or None
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            return None

    def values(self, *, first=False):
        if not self.enabled:
            return {}
        result = {}
        try:
            import torch
            if self.pending is not None:
                result.update(zip(('p_keep_std', 'selected_tokens_mean', 'selected_tokens_min',
                                   'selected_tokens_max'),
                                  (v if math.isfinite(v) else None for v in self.pending.cpu().tolist())))
                self.pending = None
            cfg = getattr(self.model, 'config', None)
            retention = getattr(cfg, 'target_retention', None)
            connector_cfg = getattr(getattr(self.model, 'connector', None), 'config', None)
            tokens = getattr(connector_cfg, 'num_visual_tokens', None)
            if retention is not None:
                result['target_retention'] = retention
            if tokens is not None:
                result['num_visual_tokens'] = tokens
                if retention is not None:
                    result['selected_tokens_target'] = tokens * retention
            device = torch.device(self.device)
            if device.type == 'cuda':
                result['gpu_memory_gb'] = torch.cuda.memory_allocated(device) / 1024**3
            if first:
                result['system'] = dict(python=platform.python_version(), pytorch=str(torch.__version__),
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
        if self.error:
            result['telemetry_error'] = self.error[:300]
            self.error = None
        return result
