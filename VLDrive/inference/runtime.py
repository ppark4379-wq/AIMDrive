"""One offline anchor: causal Situation text and Joint latest-frame trajectory."""
from contextlib import nullcontext
import time

from .results import trajectory_metrics


class JointPredictor:
    def __init__(self, model, situation, *, device='cuda'):
        import torch
        self.torch, self.model, self.situation = torch, model, situation
        self.device = torch.device(device)
        self.model.language_model.token_cache_size = 4096

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def predict(self, dataset, identity, position):
        torch = self.torch
        frames, reason = dataset.window(identity, position)
        if frames is None:
            return dict(position=position, trajectory=None, reason=reason)
        self.synchronize()
        start = time.perf_counter()
        situations, reason = self.situation.window_predictions(dataset, identity, position)
        if reason:
            return dict(position=position, trajectory=None, reason=reason)
        self.synchronize()
        after_situation = time.perf_counter()
        inputs, reason = dataset.model_inputs(identity, position, [p['text'] for p in situations])
        if reason:
            return dict(position=position, trajectory=None, reason=reason)
        def move(value):
            if torch.is_tensor(value):
                return value.to(self.device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value
        inputs = {key: move(value) for key, value in inputs.items()}
        self.synchronize()
        before_model = time.perf_counter()
        context = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                   if self.model.config.precision == 'bf16' else nullcontext())
        with torch.inference_mode(), context:
            output = self.model.predict(inputs, compute_auxiliary=False,
                                        token_selection_mode=self.model.config.inference_selection_mode)
        self.synchronize()
        after_model = time.perf_counter()
        trajectory = output['trajectory'][0, -1].float().cpu()
        if trajectory.shape != (5, 2) or not torch.isfinite(trajectory).all():
            raise ValueError('Invalid Joint latest-frame trajectory')
        # Future ground truth is read only after the model has returned.
        target = dataset.ground_truth(identity, position)['trajectory']
        timing = dict(situation=(after_situation-start)*1000,
                      input=(before_model-after_situation)*1000,
                      model=(after_model-before_model)*1000,
                      total=(after_model-start)*1000)
        return dict(position=position, frame_index=frames[-1]['index'],
                    sensor_source_ns=frames[-1]['sensor_source_ns'], trajectory=trajectory.tolist(),
                    **trajectory_metrics(trajectory.numpy(), target),
                    navigation=inputs['instruction'][0], situation=situations[-1],
                    situation_input_texts=inputs['situation_instructions'][0],
                    situation_input_times_ns=[f['sensor_source_ns'] for f in frames],
                    selected_visual_tokens=output['selected_count'][0].detach().cpu().tolist(),
                    timing_ms=timing, reason=None)
