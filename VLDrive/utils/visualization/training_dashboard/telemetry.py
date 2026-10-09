"""Detached observations only: no forward, random draws, or gradient mutation."""
import math
import platform


class TrainingTelemetry:
    def __init__(self, model, device, *, enabled=True, gradient_interval=100):
        self.model, self.device = model, device
        self.enabled, self.gradient_interval = enabled, gradient_interval
        self.handle, self.pending, self.error = None, None, None
        self.language_handle, self.language_model = None, None
        self.prediction, self.tokens = None, None

    def __enter__(self):
        if self.enabled and hasattr(self.model, 'connector'):
            try:
                self.handle = self.model.connector.register_forward_hook(self._observe)
            except Exception as exc:
                self.error = type(exc).__name__ + ': ' + str(exc)
        if self.enabled and hasattr(self.model, 'language_model'):
            try:
                self.language_model = self.model.language_model
                self.previous_collection = getattr(self.language_model, 'collect_training_metrics', None)
                self.language_model.collect_training_metrics = True
                self.language_handle = self.language_model.register_forward_hook(self._observe_language)
            except Exception as exc:
                self.error = type(exc).__name__ + ': ' + str(exc)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
        if self.language_handle is not None:
            self.language_handle.remove()
        if self.language_model is not None:
            if self.previous_collection is None:
                delattr(self.language_model, 'collect_training_metrics')
            else:
                self.language_model.collect_training_metrics = self.previous_collection
        self.pending = None
        self.prediction, self.tokens = None, None

    def _observe_language(self, module, inputs, output):
        if not module.training:
            return
        try:
            prediction = output.predicted_waypoints
            self.prediction = None
            if prediction.ndim == 3 and prediction.shape[-1] == 10:
                prediction = prediction.reshape(*prediction.shape[:2], 5, 2)
            if prediction.ndim == 4 and tuple(prediction.shape[-2:]) == (5, 2):
                self.prediction = prediction[:, -1].detach()
            self.tokens = getattr(output, 'token_metrics', None)
        except Exception as exc:
            self.prediction, self.tokens = None, None
            self.error = type(exc).__name__ + ': ' + str(exc)

    def _driving_values(self, item):
        import torch
        result = {}
        if self.prediction is None or item is None:
            return result
        with torch.no_grad():
            target = item['training_targets']['future_trajectory'][:, -1].detach().float()
            error = torch.linalg.vector_norm(self.prediction.float() - target, dim=-1)
            ade, fde = error.mean(-1), error[:, -1]
            # One anchor per window. History predictions do not multiply samples.
            values = torch.stack((ade, fde), dim=-1).cpu().tolist()
            if not all(math.isfinite(v) for row in values for v in row):
                raise ValueError('Nonfinite trajectory telemetry')
            result.update(train_ade_m=sum(v[0] for v in values) / len(values),
                          train_fde_m=sum(v[1] for v in values) / len(values),
                          trajectory_samples=len(values))
            zones = item['model_inputs'].get('zone_ids')
            if zones is not None:
                grouped = {}
                for zone, (a, f) in zip(zones[:, -1].detach().cpu().tolist(), values):
                    if zone < 0:
                        continue
                    entry = grouped.setdefault(str(zone), dict(samples=0, ade_sum_m=0., fde_sum_m=0.))
                    entry['samples'] += 1
                    entry['ade_sum_m'] += a
                    entry['fde_sum_m'] += f
                result['zone_metrics'] = grouped
        return result

    def _token_values(self):
        import torch
        if self.tokens is None:
            return {}
        with torch.no_grad():
            values = torch.stack([self.tokens[key].detach().float().mean()
                                  for key in ('navigation', 'situation', 'zone', 'visual', 'valid')] +
                                 [self.tokens['valid'].detach().float().max()]).cpu().tolist()
        result = dict(zip(('navigation_tokens', 'situation_tokens', 'zone_tokens', 'visual_tokens',
                           'llm_valid_tokens', 'llm_max_valid_tokens'), values))
        padded = self.tokens['padded_length']
        result.update(llm_padded_tokens=padded, llm_context_limit=self.tokens['context_limit'],
                      llm_padding_ratio=1 - result['llm_valid_tokens'] / padded)
        return result

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
                ('connector_projection', 'connector_projection'), ('waypoint_head', 'waypoint'),
                ('navigation_mlp', 'navigation_mlp'), ('zone_embedding', 'zone_embedding')) if key in declared}
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
                    else:
                        result[group] = None
            return result or None
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            return None

    def values(self, *, first=False, item=None):
        if not self.enabled:
            return {}
        result = {}
        try:
            import torch
            result.update(self._driving_values(item))
            result.update(self._token_values())
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
        finally:
            self.prediction, self.tokens = None, None
        if self.error:
            result['telemetry_error'] = self.error[:300]
            self.error = None
        return result
