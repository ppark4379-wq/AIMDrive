"""Connector and LLM joint training, independent validation and exact resume.

The CPU optimizer option offloads AdamW state/master copies, not model layers.
It uses the same PyTorch FP32 AdamW calculation as the device optimizer.
"""
import copy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from training.common.runtime.precision import autocast
from training.common.runtime.batching import batch, batch_samples, subset_indices
from training.common.logging import gradient_report
from training.common.runtime.reproducibility import seed_all
from training.common.checkpoints import (parameter_hash, source_hashes,
    save_checkpoint, restore_checkpoint)
from utils.evaluation.metrics import TrajectoryMetrics
from utils.common.source_compatibility import equivalent_provenance
from training.common.logging import TrainingTelemetry

ROOT = Path(__file__).resolve().parents[2]

from training.common.runtime.batching import stride_indices, balanced_training_indices
from training.common.runtime.reproducibility import preserve_rng_state
from training.common.optimizers import _optimizer_parameter_groups, set_cosine_learning_rate, CPUAdamW, make_optimizer
from training.common.checkpoints import periodic_checkpoint_path, save_model_checkpoint, restore_model_checkpoint, warm_start_model_checkpoint


@torch.no_grad()
@preserve_rng_state()
def evaluate(model, dataset, indices, device, *, save_predictions=None, seed=607):
    """Evaluate with live-inference Gumbel selection and a repeatable RNG stream.

    Seed once per evaluation, never once per frame. The decorator restores all
    caller RNG states so evaluation cannot change subsequent training samples.
    """
    indices = list(indices)
    if indices != sorted(set(indices)):
        raise ValueError('Evaluation needs unique chronological indices, not balanced resampled draws')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Evaluation seed must be an integer in [0, 2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    metrics = TrajectoryMetrics(coordinate_convention=getattr(
        dataset, 'trajectory_coordinate_convention', 'forward_left'))
    losses, predictions, selected_counts = [], [], []
    selector_diagnostics = {key: [] for key in ('sampled_retention', 'soft_retention', 'saturated_keep_fraction')}
    evaluation_sample = getattr(dataset, 'evaluation_sample', dataset.__getitem__)
    for index in indices:
        item = batch(evaluation_sample(index), device)
        with autocast(model):
            output = model.predict(item['model_inputs'], compute_auxiliary=False,
                                   token_selection_mode='gumbel')
        target = item['training_targets']['future_trajectory']
        # Last frame is used for independent-window metrics to avoid multiplying
        # overlapping-history labels. Training uses every valid real frame.
        losses.append(torch.nn.functional.l1_loss(output['trajectory'][:, -1].float(), target[:, -1].float()).item())
        sample_id = item['metadata']['sample_ids'][-1]
        metrics.update(output['trajectory'][:, -1], target[:, -1], item['metadata']['run_id'], sample_id)
        predictions.append(dict(sample_id=sample_id[0], trajectory=output['trajectory'][0, -1].float().cpu().tolist(),
            selected_count=output['selected_count'][0].tolist()))
        selected_counts.extend(output['selected_count'].detach().cpu().reshape(-1).tolist())
        for key in selector_diagnostics:
            if key in output:
                selector_diagnostics[key].extend(output[key].detach().float().cpu().reshape(-1).tolist())
    if not losses:
        raise ValueError('No evaluation windows selected')
    report = metrics.compute()
    if 'situation_instructions' in item['model_inputs']:
        report['situation_conditioning'] = 'ground_truth_text'
    if 'zone_ids' in item['model_inputs']:
        report['zone_conditioning'] = 'recorded_per_frame_zone_ids'
    report.update(waypoint_l1=sum(losses) / len(losses),
        selection_policy='gumbel', evaluation_seed=seed, stochastic_passes=1, windows=len(losses),
        selected_count_mean=float(np.mean(selected_counts)), selected_count_min=int(min(selected_counts)),
        selected_count_max=int(max(selected_counts)))
    report.update({key: float(np.mean(values)) for key, values in selector_diagnostics.items() if values})
    if save_predictions:
        Path(save_predictions).write_text(json.dumps(predictions, allow_nan=False) + '\n')
    return report


@preserve_rng_state()
def preflight(model, sample, device):
    item = batch(sample, device)
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast(model):
        # Exercise the actual samples -> loss dictionary, used by training/task.
        loss = model(item)
    loss['loss'].backward()
    gradients = gradient_report(model, inputs=item['model_inputs'])
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
        path_delta = 0.
        situation_delta = 0.
        zone_delta = 0.
        if 'path_features' in inputs:
            changed_path = copy.deepcopy(inputs)
            changed_path['path_features']['ego_relation'][:, -1] += 1.
            changed_path['path_features']['path_context'][:, -1] += 1.
            path_future = model.predict(changed_path, compute_auxiliary=False)
            path_delta = (baseline['trajectory'][:, :-1] - path_future['trajectory'][:, :-1]).abs().max().item()
        if 'situation_instructions' in inputs:
            changed_text = copy.deepcopy(inputs)
            for row in changed_text['situation_instructions']:
                row[-1] = 'Maintain a stopped state.'
            situation_future = model.predict(changed_text, compute_auxiliary=False)
            situation_delta = (baseline['trajectory'][:, :-1] - situation_future['trajectory'][:, :-1]).abs().max().item()
        if 'zone_ids' in inputs:
            changed_zone = copy.deepcopy(inputs)
            changed_zone['zone_ids'][:, -1] = (changed_zone['zone_ids'][:, -1] + 1) % 6
            zone_future = model.predict(changed_zone, compute_auxiliary=False)
            zone_delta = (baseline['trajectory'][:, :-1] - zone_future['trajectory'][:, :-1]).abs().max().item()
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
    return dict(passed=gradients['passed'] and delta < 1e-5 and path_delta < 1e-5 and situation_delta < 1e-5
                and zone_delta < 1e-5 and history_delta > 0 and all(rejected.values()),
        optimizer_updates=0, gradients=gradients, initial_losses=initial,
        trajectory_shape=list(baseline['trajectory'].shape),
        future_to_past_max_delta=delta, history_dependency_delta=history_delta,
        future_path_to_past_max_delta=path_delta,
        future_situation_to_past_max_delta=situation_delta,
        future_zone_to_past_max_delta=zone_delta,
        rejected=rejected, selected_count=baseline['selected_count'].tolist(),
        real_history_frames=min(model.config.memory_size, sample['training_targets']['future_trajectory'].shape[0] - 1))


def train(model, optimizer, dataset, indices, *, start_update, updates, schedule_updates,
          seed, device, log_path, lr=1e-5, on_update=None, on_epoch=None, sampling_step_offset=0,
          telemetry_enabled=True, gradient_log_interval=100, batch_size=1):
    if (not indices or not 0 <= start_update <= updates <= schedule_updates
            or not isinstance(sampling_step_offset, int) or not 0 <= sampling_step_offset <= start_update):
        raise ValueError('Invalid update range or empty training split')
    if type(gradient_log_interval) is not int or gradient_log_interval < 0:
        raise ValueError('gradient_log_interval must be a nonnegative integer')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    steps_per_epoch = math.ceil(len(indices) / batch_size)
    model.train()
    start = time.monotonic()
    with TrainingTelemetry(model, device, enabled=telemetry_enabled,
                           gradient_interval=gradient_log_interval) as telemetry, Path(log_path).open('x') as log:
        for step in range(start_update, updates):
            update_start = time.monotonic()
            # A caller may draw a fresh balanced bag for each global epoch.
            # Only its shuffle cursor is local; LR, checkpoints and callbacks
            # continue to use global update numbers for exact resume.
            shuffle_epoch, offset = divmod(step - sampling_step_offset, steps_per_epoch)
            epoch = step // steps_per_epoch
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
            order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + shuffle_epoch)).tolist()
            batch_indices = [indices[i] for i in order[offset * batch_size:(offset + 1) * batch_size]]
            index = batch_indices[0]
            current_lr = set_cosine_learning_rate(optimizer, lr=lr, step=step, schedule_updates=schedule_updates)
            item = batch_samples([dataset[i] for i in batch_indices], device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(model):
                output = model(item)
            if not bool(torch.isfinite(output['loss'])):
                raise RuntimeError('Nonfinite VLDrive loss')
            output['loss'].backward()
            module_gradients = telemetry.gradients(step + 1)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                 10., error_if_nonfinite=True)
            optimizer.step()
            values = {k: float(v.detach()) for k, v in output.items()}
            values.update(telemetry.values(first=step == start_update, item=item))
            if module_gradients is not None:
                values['module_gradients'] = module_gradients
            values.update(update=step + 1, index=index, batch_indices=batch_indices, batch_size=len(batch_indices), lr=current_lr,
                group_lrs=[float(group['lr']) for group in optimizer.param_groups],
                grad_norm=float(norm), elapsed_s=time.monotonic() - start,
                epoch=epoch + 1, total_epochs=math.ceil(schedule_updates / steps_per_epoch),
                total_updates=schedule_updates, session_start_update=start_update,
                timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                update_time_s=time.monotonic() - update_start)
            log.write(json.dumps(values, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(values), flush=True)
            del output
            optimizer.zero_grad(set_to_none=True)
            if on_update:
                on_update(step + 1)
            if on_epoch and (step + 1) % steps_per_epoch == 0:
                on_epoch(epoch, step + 1)
                model.train()
