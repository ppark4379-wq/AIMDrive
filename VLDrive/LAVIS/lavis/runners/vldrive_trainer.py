"""Small-data VLDrive training, independent validation and exact resume.

The CPU optimizer option offloads AdamW state/master copies, not model layers.
It uses the same PyTorch FP32 AdamW calculation as the device optimizer.
"""
import copy
import json
import math
from pathlib import Path
import time

import torch
from .morai_trainer import (autocast, batch, gradient_report, parameter_hash, seed_all,
    subset_indices, source_hashes, save_checkpoint, restore_checkpoint)
from .morai_metrics import TrajectoryMetrics


class CPUAdamW:
    """FP32 AdamW masters/state for a single-device FP32 or BF16 model."""
    def __init__(self, parameters, *, lr=1e-5, weight_decay=.06):
        self.parameters = [p for p in parameters if p.requires_grad]
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in self.parameters):
            raise ValueError('CPU AdamW supports FP32 or BF16 model parameters')
        self.masters = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay, foreach=False)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        for parameter, master in zip(self.parameters, self.masters):
            master.grad = None if parameter.grad is None else parameter.grad.detach().to(device='cpu', dtype=torch.float32, copy=True)
        self.optimizer.step()
        for parameter, master in zip(self.parameters, self.masters):
            if master.grad is not None:
                parameter.copy_(master)
            master.grad = None

    def state_dict(self):
        return dict(implementation='cpu-fp32-adamw-v1', optimizer=self.optimizer.state_dict(),
                    masters=[p.detach() for p in self.masters])

    def load_state_dict(self, state):
        if state.get('implementation') != 'cpu-fp32-adamw-v1':
            raise ValueError('Optimizer implementation mismatch')
        if len(state['masters']) != len(self.masters):
            raise ValueError('Master parameter count mismatch')
        with torch.no_grad():
            for parameter, master, saved in zip(self.parameters, self.masters, state['masters']):
                if saved.shape != master.shape or saved.dtype != torch.float32:
                    raise ValueError('Master parameter contract mismatch')
                if not torch.equal(parameter.detach().cpu(), saved.to(parameter.dtype)):
                    raise ValueError('Restored model disagrees with FP32 optimizer masters')
                master.copy_(saved)
        self.optimizer.load_state_dict(state['optimizer'])


def make_optimizer(model, *, lr=1e-5, weight_decay=.06, state_device='cpu'):
    if state_device == 'cpu':
        return CPUAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if state_device != 'model':
        raise ValueError('optimizer.state_device must be cpu or model')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('BF16 training requires CPU FP32 master optimizer; use precision=fp32 for model-state AdamW')
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=lr, weight_decay=weight_decay, foreach=False)


@torch.no_grad()
def evaluate(model, dataset, indices, device, *, save_predictions=None):
    model.eval()
    metrics = TrajectoryMetrics()
    losses, predictions = [], []
    for index in indices:
        item = batch(dataset[index], device)
        with autocast(model):
            output = model.predict(item['model_inputs'], compute_auxiliary=False)
        target = item['training_targets']['future_trajectory']
        # Last frame is used for independent-window metrics to avoid multiplying
        # overlapping-history labels. Training uses every valid real frame.
        losses.append(torch.nn.functional.l1_loss(output['trajectory'][:, -1].float(), target[:, -1].float()).item())
        sample_id = item['metadata']['sample_ids'][-1]
        metrics.update(output['trajectory'][:, -1], target[:, -1], item['metadata']['run_id'], sample_id)
        predictions.append(dict(sample_id=sample_id[0], trajectory=output['trajectory'][0, -1].float().cpu().tolist(),
            selected_count=output['selected_count'][0].tolist()))
    if not losses:
        raise ValueError('No evaluation windows selected')
    report = metrics.compute()
    report.update(waypoint_l1=sum(losses) / len(losses), selection_policy='dynamic_argmax', windows=len(losses))
    if save_predictions:
        Path(save_predictions).write_text(json.dumps(predictions, allow_nan=False) + '\n')
    return report


def preflight(model, sample, device):
    item = batch(sample, device)
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast(model):
        # Exercise the actual samples -> loss dictionary, used by training/task.
        loss = model(item)
    loss['loss'].backward()
    gradients = gradient_report(model)
    initial = {k: float(v.detach()) for k, v in loss.items()}
    del loss
    model.zero_grad(set_to_none=True)
    model.eval()
    inputs = item['model_inputs']
    key = 'sensors' if 'sensors' in inputs else 'visual_feature'
    def perturb(value, history=False):
        changed = copy.deepcopy(value)
        tensors = changed[key].values() if key == 'sensors' else [changed[key]]
        for tensor in tensors:
            if history:
                tensor[:, :-1].zero_()
            else:
                tensor[:, -1].zero_()
        return changed
    with torch.no_grad(), autocast(model):
        baseline = model.predict(inputs, compute_auxiliary=False)
        future = model.predict(perturb(inputs), compute_auxiliary=False)
        delta = (baseline['trajectory'][:, :-1] - future['trajectory'][:, :-1]).abs().max().item()
        history = model.predict(perturb(inputs, history=True), compute_auxiliary=False)
        history_delta = (baseline['trajectory'][:, -1] - history['trajectory'][:, -1]).abs().max().item()
    rejected = {}
    for name, bad in [('teacher', dict(inputs, ego_teacher=torch.zeros(1))),
                      ('missing_instruction', dict(inputs, instruction=[''] * len(inputs['instruction'])))]:
        try:
            model.predict(bad)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    return dict(passed=gradients['passed'] and delta < 1e-5 and history_delta > 0 and all(rejected.values()),
        optimizer_updates=0, gradients=gradients, initial_losses=initial,
        future_to_past_max_delta=delta, history_dependency_delta=history_delta,
        rejected=rejected, selected_count=baseline['selected_count'].tolist(),
        real_history_frames=min(model.config.memory_size, sample['training_targets']['future_trajectory'].shape[0] - 1))


def train(model, optimizer, dataset, indices, *, start_update, updates, schedule_updates,
          seed, device, log_path, lr=1e-5, on_update=None):
    if not indices or not 0 <= start_update <= updates <= schedule_updates:
        raise ValueError('Invalid update range or empty training split')
    model.train()
    start = time.monotonic()
    with Path(log_path).open('x') as log:
        for step in range(start_update, updates):
            epoch, offset = divmod(step, len(indices))
            order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            index = indices[order[offset]]
            current_lr = lr * .5 * (1. + math.cos(math.pi * step / schedule_updates))
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            item = batch(dataset[index], device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(model):
                output = model(item)
            if not bool(torch.isfinite(output['loss'])):
                raise RuntimeError('Nonfinite VLDrive loss')
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                 10., error_if_nonfinite=True)
            optimizer.step()
            values = {k: float(v.detach()) for k, v in output.items()}
            values.update(update=step + 1, index=index, lr=current_lr, grad_norm=float(norm), elapsed_s=time.monotonic() - start)
            log.write(json.dumps(values, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(values), flush=True)
            del output
            optimizer.zero_grad(set_to_none=True)
            if on_update:
                on_update(step + 1)
