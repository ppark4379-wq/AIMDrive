"""Small-data VLDrive training, independent validation and exact resume.

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
from .training import (autocast, batch, batch_samples, gradient_report, parameter_hash, seed_all,
    subset_indices, source_hashes, save_checkpoint, restore_checkpoint)
from .metrics import TrajectoryMetrics
from .training_dashboard.telemetry import TrainingTelemetry


@contextmanager
def preserve_rng_state():
    """Diagnostics must not alter the resumed Gumbel/dropout sequence."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def stride_indices(length, stride):
    """Cover a chronological route uniformly without an arbitrary tiny subset."""
    stride = int(stride)
    if length < 1 or stride < 1:
        raise ValueError('length and stride must be positive')
    values = list(range(0, length, stride))
    if values[-1] != length - 1:
        values.append(length - 1)
    return values


def balanced_training_indices(dataset, *, indices=None, draws=640, seed=607,
                              stationary_distance_m=.5, curved_lateral_m=2.):
    """Balance train anchors, never mutate the unique chronological eval list.

    This resamples existing clips, not new driving events. Classification uses
    only the train anchor's actual future trajectory in [left,-forward] axes.
    """
    if getattr(dataset, 'split', None) != 'train':
        raise ValueError('Balanced sampling is restricted to the training split')
    if getattr(dataset, 'trajectory_coordinate_convention', None) != 'lmdrive_left_negative_forward':
        raise ValueError('Balanced sampling requires [left,-forward] trajectories')
    if (int(draws) != draws or draws < 4 or draws % 4
            or not 0 < stationary_distance_m < curved_lateral_m):
        raise ValueError('Use positive thresholds and draws divisible by four')
    candidates = list(range(len(dataset))) if indices is None else list(indices)
    if (not candidates or len(candidates) != len(set(candidates))
            or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(dataset) for i in candidates)):
        raise ValueError('Training candidates must be unique valid dataset indices')
    categories = {key: [] for key in ('stationary', 'moving_straight', 'moving_left', 'moving_right')}
    for index in sorted(candidates):
        # The cached dataset can read just five GT points without loading an
        # entire 11-frame visual window merely to choose its sampling category.
        if hasattr(dataset, 'entries') and hasattr(dataset, 'windows') and hasattr(dataset, '_array'):
            _, end = dataset.windows[index]
            target = dataset._array(dataset.entries[end-1]['future_trajectory'], (5, 2), 'future trajectory')
        else:
            target = dataset[index]['training_targets']['future_trajectory'][-1]
        target = torch.as_tensor(target).detach().cpu().float().numpy()
        if target.shape != (5, 2) or not np.isfinite(target).all():
            raise ValueError('Invalid train trajectory for balanced sampling')
        if float(np.linalg.norm(target, axis=1).max()) < stationary_distance_m:
            key = 'stationary'
        elif abs(float(target[-1, 0])) >= curved_lateral_m:
            key = 'moving_left' if target[-1, 0] > 0 else 'moving_right'
        else:
            key = 'moving_straight'
        categories[key].append(int(index))
    empty = [key for key, values in categories.items() if not values]
    if empty:
        raise ValueError('Cannot balance empty training categories: ' + ', '.join(empty))
    generator = random.Random(int(seed))
    per_category = int(draws) // 4
    sampled = []
    for values in categories.values():
        sampled.extend(generator.choices(values, k=per_category))
    generator.shuffle(sampled)
    audit = dict(policy='train-anchor-four-category-with-replacement-v1', split='train',
        seed=int(seed), draws=int(draws), unique_candidates=len(candidates),
        unique_sampled=len(set(sampled)), category_indices=categories,
        candidate_counts={key: len(values) for key, values in categories.items()},
        draw_counts={key: per_category for key in categories},
        stationary_distance_m=float(stationary_distance_m), curved_lateral_m=float(curved_lateral_m),
        note='Repeated clips are oversampling, not independent driving events; evaluate unique chronological indices separately.')
    return sampled, audit


def _optimizer_parameter_groups(model, lr, lr_multipliers):
    multipliers = dict(lr_multipliers or {})
    for prefix, value in multipliers.items():
        if not isinstance(prefix, str) or not prefix or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError('LR multipliers need nonempty prefixes and finite positive values')
    selected = {'default': []}
    selected.update({prefix: [] for prefix in multipliers})
    if 'default' in multipliers:
        raise ValueError('The LR prefix "default" is reserved')
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [prefix for prefix in multipliers if name.startswith(prefix)]
        if len(matches) > 1:
            raise ValueError('Overlapping LR prefixes for ' + name)
        selected[matches[0] if matches else 'default'].append(parameter)
    if any(not selected[prefix] for prefix in multipliers):
        raise ValueError('An LR multiplier prefix matched no trainable parameters')
    return [dict(params=parameters, lr=float(lr)*float(multipliers.get(name, 1.)),
                 base_lr_mult=float(multipliers.get(name, 1.)), group_name=name)
            for name, parameters in selected.items() if parameters]


def set_cosine_learning_rate(optimizer, *, lr, step, schedule_updates):
    """Keep explicit parameter-group multipliers throughout the cosine schedule."""
    if not 0 <= step <= schedule_updates or schedule_updates < 1:
        raise ValueError('Invalid cosine schedule position')
    current_lr = float(lr) * .5 * (1. + math.cos(math.pi * step / schedule_updates))
    for group in optimizer.param_groups:
        group['lr'] = current_lr * float(group.get('base_lr_mult', 1.))
    return current_lr


class CPUAdamW:
    """FP32 AdamW masters/state for a single-device FP32 or BF16 model."""
    def __init__(self, parameters, *, lr=1e-5, weight_decay=.06, fused=False):
        if type(fused) is not bool:
            raise ValueError('CPU AdamW fused must be a boolean')
        parameters = list(parameters)
        grouped = bool(parameters) and isinstance(parameters[0], dict)
        if any(isinstance(p, dict) != grouped for p in parameters):
            raise ValueError('Optimizer parameters cannot mix tensors and groups')
        groups = parameters if grouped else [dict(params=parameters)]
        groups = [dict(group, params=[p for p in group['params'] if p.requires_grad]) for group in groups]
        self.parameters = [p for group in groups for p in group['params']]
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError('Duplicate optimizer parameter')
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in self.parameters):
            raise ValueError('CPU AdamW supports FP32 or BF16 model parameters')
        self.masters = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.parameters]
        master_groups, offset = [], 0
        for group in groups:
            count = len(group['params'])
            if count:
                master_groups.append(dict(group, params=self.masters[offset:offset+count]))
            offset += count
        # CPU fused kernels avoid separate passes/intermediates for each AdamW
        # operation. FP32 masters, moments and CPU/GPU transfers stay unchanged.
        self.optimizer = torch.optim.AdamW(master_groups, lr=lr, weight_decay=weight_decay,
                                           foreach=False, fused=fused)

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


def make_optimizer(model, *, lr=1e-5, weight_decay=.06, state_device='cpu', lr_multipliers=None,
                   cpu_fused=False):
    if type(cpu_fused) is not bool:
        raise ValueError('optimizer.cpu_fused must be a boolean')
    groups = _optimizer_parameter_groups(model, lr, lr_multipliers)
    if state_device == 'cpu':
        return CPUAdamW(groups, lr=lr, weight_decay=weight_decay, fused=cpu_fused)
    if state_device != 'model':
        raise ValueError('optimizer.state_device must be cpu or model')
    if cpu_fused:
        raise ValueError('optimizer.cpu_fused requires state_device=cpu')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('BF16 training requires CPU FP32 master optimizer; use precision=fp32 for model-state AdamW')
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, foreach=False)


def periodic_checkpoint_path(output, step, interval, slots=0):
    """Rotate bounded snapshots; the other slot survives an interrupted write."""
    if interval < 1 or step < interval or step % interval or slots < 0 or slots == 1:
        raise ValueError('Checkpoint rotation requires a valid interval and zero or >=2 slots')
    name = ('checkpoint_slot%d.pt' % ((step // interval - 1) % slots)
            if slots else 'checkpoint_%06d.pt' % step)
    return Path(output) / name


def save_model_checkpoint(path, model, *, updates, provenance, settings, metrics):
    """Save best/inference weights without the much larger AdamW state."""
    from .runtime import sha256
    path = Path(path)
    state = dict(
        kind='vldrive-model-only-v1',
        schema=model.checkpoint_schema,
        model_config=asdict(model.config),
        trainable_state={
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        },
        trainable_hash=parameter_hash(model, trainable=True),
        frozen_hash=parameter_hash(model, trainable=False),
        updates=int(updates),
        provenance=provenance,
        settings=settings,
        metrics=metrics,
    )
    pending = path.with_suffix(path.suffix + '.pending')
    torch.save(state, pending)
    pending.replace(path)
    digest = dict(sha256=sha256(path), bytes=path.stat().st_size)
    path.with_suffix(path.suffix + '.sha256.json').write_text(
        json.dumps(digest, indent=2) + '\n')


def restore_model_checkpoint(path, model, *, provenance, evaluation_source_compatibility=False):
    """Restore local model-only or full training weights without an optimizer."""
    from .runtime import sha256
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256.json')
    model_only = sidecar.is_file()
    if not model_only:
        sidecar = path.with_suffix('.sha256.json')
    digest = json.loads(sidecar.read_text())
    if path.stat().st_size != digest['bytes'] or sha256(path) != digest['sha256']:
        raise ValueError('Model-only checkpoint hash mismatch')
    # Full local checkpoints also contain optimizer/RNG Python objects. mmap
    # avoids eagerly allocating their multi-GB optimizer state for evaluation.
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    expected_provenance = provenance
    source_changes = []
    if evaluation_source_compatibility and state.get('provenance') != provenance:
        # Evaluation-only source changes must be explicitly reviewed; dataset,
        # asset, backend and model checks remain exact. Training resume stays strict.
        from inference.vldrive_runtime import verify_sources
        recorded = state.get('provenance', {})
        expected_provenance = copy.deepcopy(provenance)
        expected_provenance['source'] = recorded.get('source')
        backend_sources = recorded.get('backend', {}).get('sources')
        if backend_sources is not None and 'backend' in expected_provenance:
            expected_provenance['backend']['sources'] = backend_sources
        if recorded != expected_provenance or not isinstance(recorded.get('source'), dict):
            raise ValueError('Model-only checkpoint contract/provenance mismatch')
        source_changes = verify_sources(recorded)
        if backend_sources is not None:
            source_changes.extend(verify_sources({'source': backend_sources}))
    if ((model_only and state.get('kind') != 'vldrive-model-only-v1')
            or state.get('schema') != model.checkpoint_schema
            or state.get('model_config') != asdict(model.config)
            or state.get('provenance') != expected_provenance):
        raise ValueError('Model-only checkpoint contract/provenance mismatch')
    if parameter_hash(model, trainable=False) != state.get('frozen_hash'):
        raise ValueError('Model-only checkpoint frozen asset mismatch')
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state.get('trainable_state', {})) != expected:
        raise ValueError('Model-only checkpoint parameter set mismatch')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state['trainable_state'][name])
    if parameter_hash(model, trainable=True) != state['trainable_hash']:
        raise ValueError('Restored model-only weights differ')
    if evaluation_source_compatibility:
        state['evaluation_source_compatibility'] = source_changes
    return state


def warm_start_model_checkpoint(path, model, *, provenance, reset_prefixes=(), transfer_reason):
    """Explicit weight transfer into a new experiment, never an exact resume.

    Source/config/dataset changes require a recorded reason. Frozen assets stay
    identical; all non-reset trainable keys, shapes and dtypes must match. The
    caller constructs fresh reset modules before this function. No optimizer,
    scheduler counters or random-generator states are restored.
    """
    from .runtime import sha256
    if not isinstance(transfer_reason, str) or not transfer_reason.strip():
        raise ValueError('Warm start requires a documented transfer_reason')
    prefixes = tuple(reset_prefixes)
    if (len(prefixes) != len(set(prefixes))
            or any(not isinstance(p, str) or not p or not p.endswith('.') for p in prefixes)
            or any(a != b and b.startswith(a) for a in prefixes for b in prefixes)):
        raise ValueError('Reset prefixes must be unique nonoverlapping module prefixes ending in a dot')
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256.json')
    model_only = sidecar.is_file()
    if not model_only:
        sidecar = path.with_suffix('.sha256.json')
    digest = json.loads(sidecar.read_text())
    if path.stat().st_size != digest['bytes'] or sha256(path) != digest['sha256']:
        raise ValueError('Warm-start checkpoint integrity mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if ((model_only and state.get('kind') != 'vldrive-model-only-v1')
            or state.get('schema') != model.checkpoint_schema
            or not isinstance(state.get('model_config'), dict)
            or not isinstance(state.get('provenance'), dict)):
        raise ValueError('Warm-start checkpoint schema mismatch')
    frozen_hash = parameter_hash(model, trainable=False)
    if frozen_hash != state.get('frozen_hash'):
        raise ValueError('Warm-start checkpoint frozen asset mismatch')
    new_config = asdict(model.config)
    for field in ('visual_sha256',):
        old, new = state['model_config'].get(field), new_config.get(field)
        if (old is not None or new is not None) and old != new:
            raise ValueError('Warm-start vision checkpoint hash mismatch')
    def vision_fingerprints(value):
        return {entry['checkpoint_sha256'] for entry in value.get('datasets', {}).values()
                if isinstance(entry, dict) and entry.get('checkpoint_sha256') is not None}
    if vision_fingerprints(state['provenance']) != vision_fingerprints(provenance):
        raise ValueError('Warm-start cached vision checkpoint hash mismatch')
    saved = state.get('trainable_state')
    if not isinstance(saved, dict) or not saved:
        raise ValueError('Warm-start checkpoint has no trainable state')
    saved_digest = hashlib.sha256()
    for name, value in saved.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('Invalid warm-start trainable tensor')
        saved_digest.update(name.encode())
        saved_digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if saved_digest.hexdigest() != state.get('trainable_hash'):
        raise ValueError('Warm-start trainable state hash mismatch')
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    reset = lambda name: any(name.startswith(prefix) for prefix in prefixes)
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in saved) or not any(name.startswith(prefix) for name in expected):
            raise ValueError('Reset prefix must match old and new trainable parameters: ' + prefix)
    old_loaded = {name for name in saved if not reset(name)}
    new_loaded = {name for name in expected if not reset(name)}
    if old_loaded != new_loaded:
        raise ValueError('Warm-start parameter set mismatch outside explicit reset prefixes')
    if not old_loaded:
        raise ValueError('Warm start must transfer at least one trainable parameter')
    for name in old_loaded:
        if saved[name].shape != expected[name].shape or saved[name].dtype != expected[name].dtype:
            raise ValueError('Warm-start parameter shape/dtype mismatch: ' + name)
    # Every check above precedes mutation, avoiding a partially loaded model
    # on a known key/shape/asset contract error.
    with torch.no_grad():
        for name in old_loaded:
            expected[name].copy_(saved[name])
    return dict(policy='explicit-model-weight-transfer-v1', transfer_reason=transfer_reason.strip(),
        source_checkpoint=str(path.resolve()), source_checkpoint_sha256=digest['sha256'],
        source_updates=int(state.get('updates', 0)), old_provenance=state['provenance'],
        new_provenance=copy.deepcopy(provenance), old_model_config=state['model_config'],
        new_model_config=new_config, reset_prefixes=list(prefixes),
        reset_old_keys=sorted(name for name in saved if reset(name)),
        reset_new_keys=sorted(name for name in expected if reset(name)),
        loaded_keys=sorted(old_loaded), loaded_parameter_count=sum(expected[name].numel() for name in old_loaded),
        frozen_hash=frozen_hash, optimizer_restored=False, rng_restored=False,
        note='New experiment starts at update zero with a new optimizer; source/config/data changes are audited, not exact-resumed.')


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
        trajectory_shape=list(baseline['trajectory'].shape),
        future_to_past_max_delta=delta, history_dependency_delta=history_delta,
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
            values.update(telemetry.values(first=step == start_update))
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
