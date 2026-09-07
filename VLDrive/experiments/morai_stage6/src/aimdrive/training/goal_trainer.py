"""One offline training/evaluation path for the complete MORAI goal stack."""
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def batch(sample, device):
    return move(default_collate([sample]), device)


def subset_indices(length, count):
    if count <= 0 or count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def autocast(model):
    device = next(model.parameters()).device.type
    return torch.autocast(device_type=device, dtype=torch.bfloat16,
                          enabled=model.config.precision == 'bf16')


def objective(output, targets, config):
    trajectory = F.smooth_l1_loss(output['trajectory'][:, -1].float(), targets['future_trajectory'][:, -1].float())
    speed = F.smooth_l1_loss(output['speed'][:, -1].float(), targets['future_speed'][:, -1].float())
    total = trajectory + config.speed_weight * speed
    total = total + config.pruning_weight * output['pruning_loss'] + config.reconstruction_weight * output['reconstruction_loss']
    return total, dict(trajectory=trajectory.detach().item(), speed=speed.detach().item(),
        pruning=output['pruning_loss'].detach().item(), reconstruction=output['reconstruction_loss'].detach().item())


def parameter_hash(model, *, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != trainable:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not trainable:
        for name, buffer in model.named_buffers():
            digest.update(name.encode())
            digest.update(buffer.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


@torch.no_grad()
def evaluate(model, dataset, indices, device, *, save_predictions=None, on_progress=None):
    from .trajectory_metrics import TrajectoryMetrics
    model.eval()
    metrics = TrajectoryMetrics()
    speed_abs = torch.zeros(5, dtype=torch.float64)
    total_loss = 0.
    predictions = []
    for completed, index in enumerate(indices, 1):
        item = batch(dataset[index], device)
        with autocast(model):
            output = model(item['model_inputs'], selection_mode='topk', compute_auxiliary=False)
            loss, _ = objective(output, item['training_targets'], model.config)
        last_sample = item['metadata']['sample_ids'][-1]
        metrics.update(output['trajectory'][:, -1], item['training_targets']['future_trajectory'][:, -1],
                       item['metadata']['run_id'], last_sample)
        speed_abs += (output['speed'][0, -1].double().cpu() - item['training_targets']['future_speed'][0, -1].double().cpu()).abs()
        total_loss += loss.item()
        if save_predictions is not None:
            predictions.append(dict(sample_id=last_sample[0], trajectory=output['trajectory'][0, -1].float().cpu().tolist(),
                speed=output['speed'][0, -1].float().cpu().tolist()))
        if completed % 50 == 0 or completed == len(indices):
            if on_progress is not None:
                on_progress(completed, len(indices))
    result = metrics.compute()
    result.update(speed_mae_mps=(speed_abs / len(indices)).mean().item(),
                  speed_horizon_mae_mps=(speed_abs / len(indices)).tolist(), objective=total_loss / len(indices))
    if save_predictions is not None:
        path = Path(save_predictions)
        pending = path.with_suffix(path.suffix + '.pending')
        pending.write_text(json.dumps(predictions, indent=2, allow_nan=False) + '\n')
        pending.replace(path)
    return result


def preflight(model, sample, device):
    """Real-data gradient, dependency and temporal-causality probes; zero updates."""
    item = batch(sample, device)
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast(model):
        output = model(item['model_inputs'], selection_mode='gumbel', retain_visual_grad=True)
        loss, losses = objective(output, item['training_targets'], model.config)
    loss.backward()
    gradients = gradient_report(model)
    history = output['visual_feature'].grad
    history_norm = history[:, :-1].float().norm().item() if history is not None else None
    del output, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    original = item['model_inputs']
    with torch.no_grad(), autocast(model):
        baseline = model(original, selection_mode='topk', compute_auxiliary=False)
        # Mutate ONLY the last/future frame, including its route and receive time.
        changed = copy.deepcopy(original)
        changed['route'][:, -1, :, 1] += 7.
        changed['elapsed_s'][:, -1] += .1
        if 'sensors' in changed:
            for key in ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar'):
                changed['sensors'][key][:, -1].zero_()
        else:
            changed['visual_feature'][:, -1].zero_()
        future = model(changed, selection_mode='topk', compute_auxiliary=False)
        past_delta = max((baseline[k][:, :-1] - future[k][:, :-1]).abs().max().item() for k in ('trajectory', 'speed'))
        changed = copy.deepcopy(original)
        if 'sensors' in changed:
            for key in ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar'):
                changed['sensors'][key].zero_()
        else:
            changed['visual_feature'].zero_()
        no_visual = model(changed, selection_mode='topk', compute_auxiliary=False)
        visual_delta = (baseline['trajectory'][:, -1] - no_visual['trajectory'][:, -1]).abs().max().item()
        sensor_dependencies = {}
        if 'sensors' in original:
            for name, keys in [('front', ('rgb_front', 'rgb_center')), ('left', ('rgb_left',)),
                               ('right', ('rgb_right',)), ('rear', ('rgb_rear',)), ('lidar', ('lidar',))]:
                changed = copy.deepcopy(original)
                for key in keys:
                    changed['sensors'][key].zero_()
                perturbed = model(changed, selection_mode='topk', compute_auxiliary=False)
                sensor_dependencies[name] = (baseline['trajectory'][:, -1] - perturbed['trajectory'][:, -1]).abs().max().item()
        changed = copy.deepcopy(original)
        changed['route'][:, :, :, 1] += 3.
        route_changed = model(changed, selection_mode='topk', compute_auxiliary=False)
        route_delta = (baseline['trajectory'][:, -1] - route_changed['trajectory'][:, -1]).abs().max().item()
        if model.config.mefa and original['route'].shape[1] > 1:
            changed = copy.deepcopy(original)
            if 'sensors' in changed:
                for key in ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar'):
                    changed['sensors'][key][:, :-1].zero_()
            else:
                changed['visual_feature'][:, :-1].zero_()
            history_changed = model(changed, selection_mode='topk', compute_auxiliary=False)
            history_delta = (baseline['trajectory'][:, -1] - history_changed['trajectory'][:, -1]).abs().max().item()
        else:
            history_delta = None
    rejected = False
    try:
        model(dict(original, ego_teacher=torch.zeros(1, device=device)))
    except ValueError:
        rejected = True
    passed = gradients['passed'] and past_delta <= 1e-5 and visual_delta > 0 and route_delta > 0 and rejected
    if sensor_dependencies:
        passed &= all(delta > 0 for delta in sensor_dependencies.values())
    if model.config.mefa:
        passed &= history_delta is not None and history_delta > 0
    return dict(passed=bool(passed), optimizer_updates=0, gradient=gradients, initial_losses=losses,
                future_to_past_max_delta=past_delta, visual_dependency_delta_m=visual_delta,
                route_dependency_delta_m=route_delta, history_dependency_delta_m=history_delta,
                sensor_dependency_delta_m=sensor_dependencies,
                past_visual_gradient_l2=history_norm, privileged_key_rejected=rejected,
                semantic_training=any(original['instruction']))


def make_optimizer(model, *, head_lr=1e-3, encoder_lr=1e-5, language_lr=1e-5):
    groups = {'heads': [], 'encoder': [], 'language': []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = 'encoder' if name.startswith('visual_encoder.') else ('language' if name.startswith('language.') else 'heads')
            groups[key].append(parameter)
    rates = dict(heads=head_lr, encoder=encoder_lr, language=language_lr)
    return torch.optim.AdamW([dict(params=p, lr=rates[k], name=k) for k, p in groups.items() if p], weight_decay=.01)


def source_hashes(root):
    from aimdrive.model.goal_backends import sha256
    root = Path(root)
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'scripts/train_goal_stack.py']
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def save_checkpoint(path, model, optimizer, *, updates, provenance, settings):
    from aimdrive.model.goal_backends import sha256
    path = Path(path)
    trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    state = dict(schema='morai-goal-stack-v1', model_config=asdict(model.config), trainable_state=trainable,
        trainable_hash=parameter_hash(model, trainable=True), frozen_hash=parameter_hash(model, trainable=False),
        optimizer=optimizer.state_dict(), updates=updates, provenance=provenance, settings=settings,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    partial = path.with_suffix('.partial.pt')
    torch.save(state, partial)
    partial.replace(path)
    path.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=sha256(path), bytes=path.stat().st_size)) + '\n')


def restore_checkpoint(path, model, optimizer, *, provenance, restore_rng=True):
    from aimdrive.model.goal_backends import sha256
    path = Path(path)
    integrity = json.loads(path.with_suffix('.sha256.json').read_text())
    if path.stat().st_size != integrity['bytes'] or sha256(path) != integrity['sha256']:
        raise ValueError('Checkpoint file integrity failure')
    state = torch.load(path, map_location='cpu')
    if state['schema'] != 'morai-goal-stack-v1' or state['model_config'] != asdict(model.config):
        raise ValueError('Checkpoint/model contract mismatch')
    if state['provenance'] != provenance:
        raise ValueError('Checkpoint data, code or backend provenance changed')
    if parameter_hash(model, trainable=False) != state['frozen_hash']:
        raise ValueError('Frozen base weights/buffers mismatch')
    parameters = dict(model.named_parameters())
    if set(state['trainable_state']) != {n for n, p in parameters.items() if p.requires_grad}:
        raise ValueError('Checkpoint trainability policy mismatch')
    with torch.no_grad():
        for name, value in state['trainable_state'].items():
            parameters[name].copy_(value)
    if parameter_hash(model, trainable=True) != state['trainable_hash']:
        raise ValueError('Restored parameter hash mismatch')
    optimizer.load_state_dict(state['optimizer'])
    if restore_rng:
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda']:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def train(model, optimizer, dataset, indices, *, start_update, updates, seed, device, log_path, on_update=None):
    model.train()
    start = time.monotonic()
    with Path(log_path).open('x') as log:
        for step in range(start_update, updates):
            epoch, offset = divmod(step, len(indices))
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(indices), generator=generator).tolist()
            index = indices[order[offset]]
            item = batch(dataset[index], device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(model):
                output = model(item['model_inputs'])
                loss, values = objective(output, item['training_targets'], model.config)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10., error_if_nonfinite=True)
            optimizer.step()
            values.update(update=step + 1, objective=loss.item(), grad_norm=norm.item(), index=index,
                          selected_tokens_mean=output['selected_count'].float().mean().item(),
                          selected_tokens_min=int(output['selected_count'].min()),
                          selected_tokens_max=int(output['selected_count'].max()),
                          elapsed_s=time.monotonic() - start)
            log.write(json.dumps(values, allow_nan=False) + '\n')
            log.flush()
            if step == start_update or (step + 1) % 10 == 0:
                print(json.dumps(values), flush=True)
            if on_update is not None:
                on_update(step + 1)
