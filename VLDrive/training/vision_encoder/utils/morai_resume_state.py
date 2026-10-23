"""Stateful continuation of the batch-one MORAI transfer experiment.

Legacy progress files have AdamW moments but no sampler/RNG state. Their first
continuation explicitly resets the sampler, retaining only the remaining epoch
budget. New checkpoints record the permutation, consumed cursor and RNG state.
Restore model RNG only after constructing the next DataLoader iterator; workers
must use deterministic per-frame preprocessing and a separate loader generator.
"""
import copy
import math
import random

import numpy as np
import torch


SCHEMA = 'morai_transfer_training_state_v1'
# Audited live 20260930_122537 trainer: one AdamW group from all requires_grad
# model.parameters(), no optimizer resume, no alternative parameter ordering.
LEGACY_TRAINER_SHA256 = '623d66223ee1c1581d4bb4c996e013d53571c8aeae4ca7b63dddec0f067bfcbb'
PROTECTED_IMPLEMENTATIONS = (
    'training/vision_encoder/setup.py', 'models/ve/vision_encoder.py',
    'models/ve/pointpillars.py', 'models/ve/resnet.py', 'models/ve/transformer.py',
    'data/src/loaders/morai_transfer_adapter.py',
    'data/src/loaders/morai_transfer_targets.py',
    'data/src/loaders/morai_gps_speed.py', 'data/src/loaders/morai_ego_speed.py',
    'data/src/loaders/morai_map_geometry.py', 'data/src/loaders/morai_map_derivation.py',
    'training/vision_encoder/losses/morai_transfer.py',
    'training/vision_encoder/losses/morai_bbox_auxiliary.py',
)
SEMANTIC_FIELDS = (
    'schema', 'publications', 'coverage', 'trainable_scope', 'loss', 'batch_size',
    'frame_sampling', 'augmentation', 'velocity_source', 'velocity_contract',
    'waypoint_source', 'target_point', 'batchnorm', 'stop_sign',
)
AUGMENTATION_RNG_POLICY = 'main_process_torch_generators_v1'
AUGMENTATION_GENERATOR_NAMES = ('photometric', 'target_point_dropout')


def _augmentation_contract(provenance):
    """Only the explicitly saved, main-process camera/target RNG is supported."""
    policy = provenance.get('augmentation_rng_contract')
    if policy is None:
        if provenance['augmentation'] != 'none' or provenance.get('args', {}).get('target_point_dropout', 0):
            raise ValueError('Exact augmentation continuation requires saved main-process generators')
        return None
    if (not isinstance(policy, dict) or policy.get('policy') != AUGMENTATION_RNG_POLICY
            or set(policy) != {'policy', 'generator_names', 'photometric_probability',
                              'target_point_dropout_probability'}
            or tuple(policy.get('generator_names', ())) != AUGMENTATION_GENERATOR_NAMES):
        raise ValueError('Unsupported augmentation RNG contract')
    photo, dropout = policy['photometric_probability'], policy['target_point_dropout_probability']
    if (not math.isfinite(photo) or not 0 <= photo <= 1
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError('Invalid augmentation probabilities')
    expected = 'photometric_shared_per_frame p=%g' % photo if photo else 'none'
    if provenance['augmentation'] != expected:
        raise ValueError('Photometric augmentation contradicts RNG contract')
    return _canonical_sequences(policy)


def _validate_augmentation_rng(state, names):
    if not isinstance(state, dict) or set(state) != set(names):
        raise ValueError('Missing/unexpected augmentation generator state')
    for name in names:
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError('Invalid augmentation generator state: ' + name)
        try:
            torch.Generator().set_state(value.cpu())
        except RuntimeError as exc:
            raise ValueError('Invalid augmentation generator state: ' + name) from exc


def apply_augmentation_rng_state(state, generators):
    """Restore independent training augmentation generators after loader setup."""
    if state is None:
        return
    _validate_augmentation_rng(state, generators)
    for name, generator in generators.items():
        generator.set_state(state[name].cpu())


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid resume ' + name)
    return value


def epoch_order(dataset_size, seed, epoch_index):
    """Isolated deterministic permutation; never advances the model RNG."""
    _integer(dataset_size, 'dataset_size', 1)
    _integer(seed, 'seed')
    _integer(epoch_index, 'epoch_index')
    generator = torch.Generator().manual_seed((seed + epoch_index * 1000003) % (2**63 - 1))
    return torch.randperm(dataset_size, generator=generator).tolist()


def _canonical_sequences(value):
    """Normalize JSON array representation only; preserve every scalar value."""
    if isinstance(value, dict):
        return {key: _canonical_sequences(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_sequences(item) for item in value]
    return copy.deepcopy(value)


def _contract(provenance, encoder_contract):
    result = {}
    for key in SEMANTIC_FIELDS:
        if key not in provenance:
            raise ValueError('Missing resume contract: ' + key)
        result[key] = copy.deepcopy(provenance[key])
    if _canonical_sequences(provenance.get('encoder_contract')) != _canonical_sequences(encoder_contract):
        raise ValueError('Encoder contract contradicts provenance')
    result['encoder_contract'] = copy.deepcopy(encoder_contract)
    # The stored current permutation is insufficient on its own: future epochs
    # are constructed from args.seed, so changing it is not exact continuation.
    result['sampler_seed'] = _integer(provenance.get('args', {}).get('seed'), 'seed')
    optimizer = provenance.get('optimizer', {})
    if optimizer.get('name') != 'AdamW' or optimizer.get('schedule') != 'constant':
        raise ValueError('Resume requires constant AdamW')
    result['optimizer'] = {k: optimizer[k] for k in ('name', 'learning_rate', 'weight_decay', 'schedule')}
    if result['batch_size'] != 1 or result['frame_sampling'].get('policy') != 'all':
        raise ValueError('Resume currently requires batch one and all-frame sampling')
    augmentation = _augmentation_contract(provenance)
    if augmentation is not None:
        result['augmentation_rng_contract'] = augmentation
    hashes = provenance.get('implementation_sha256', {})
    if augmentation is not None:
        # These transforms live in the recipe, outside the native model/data
        # implementation list protected by historical no-augmentation runs.
        recipe = 'training/vision_encoder/recipes/transfer.py'
        if not isinstance(hashes.get(recipe), str) or not hashes[recipe]:
            raise ValueError('Missing augmentation implementation hash')
        result['augmentation_implementation_sha256'] = hashes[recipe]
    for key in PROTECTED_IMPLEMENTATIONS:
        if not isinstance(hashes.get(key), str) or not hashes[key]:
            raise ValueError('Missing protected implementation hash: ' + key)
    result['protected_implementation_sha256'] = {k: hashes[k] for k in PROTECTED_IMPLEMENTATIONS}
    return _canonical_sequences(result)


def _optimizer_names(model, optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Resume optimizer must be AdamW')
    named = dict(model.named_parameters())
    names = {id(p): name for name, p in named.items()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        current = []
        for parameter in group['params']:
            name = names.get(id(parameter))
            if name is None or name in seen or not parameter.requires_grad:
                raise ValueError('Optimizer parameter mapping is invalid')
            seen.add(name)
            current.append(name)
        groups.append(current)
    if seen != {name for name, p in named.items() if p.requires_grad}:
        raise ValueError('Optimizer must include exactly the trainable parameters')
    return groups


def _validate_optimizer_contract(optimizer, contract):
    for group in optimizer.param_groups:
        if (group['lr'] != contract['optimizer']['learning_rate']
                or group['weight_decay'] != contract['optimizer']['weight_decay']):
            raise ValueError('Optimizer hyperparameters contradict training contract')


def _validate_position(order, cursor, epoch_index, global_step, dataset_size):
    _integer(cursor, 'cursor')
    _integer(epoch_index, 'epoch_index')
    _integer(global_step, 'global_step')
    if (not isinstance(order, list) or len(order) != dataset_size
            or any(type(i) is not int for i in order)
            or sorted(order) != list(range(dataset_size)) or cursor > dataset_size
            or global_step != epoch_index * dataset_size + cursor):
        raise ValueError('Invalid epoch permutation/cursor/global step')


def _capture_rng(include_cuda):
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=algorithm, keys=keys.tolist(),
                position=int(position), has_gauss=int(has_gauss), cached_gauss=float(cached_gauss)),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if include_cuda else [],
                cuda_required=bool(include_cuda))


def _validate_rng(state):
    # Validate with isolated generators; inspection must not advance live RNGs.
    try:
        random.Random().setstate(state['python'])
        n = state['numpy']
        np.random.RandomState().set_state((n['algorithm'], np.array(n['keys'], dtype=np.uint32),
                                          n['position'], n['has_gauss'], n['cached_gauss']))
        torch.Generator().set_state(state['torch_cpu'].cpu())
        if type(state['cuda_required']) is not bool or not isinstance(state['torch_cuda'], list):
            raise ValueError('Invalid CUDA RNG contract')
        if state['cuda_required'] != bool(state['torch_cuda']):
            raise ValueError('Incomplete CUDA RNG state')
        for tensor in state['torch_cuda']:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
                raise ValueError('Invalid CUDA RNG tensor')
    except (KeyError, TypeError, RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid saved RNG state') from exc


def apply_rng_state(state):
    """Apply after creating the resumed loader iterator, before next forward."""
    if state is None:
        return
    _validate_rng(state)
    if state['cuda_required'] and (not torch.cuda.is_available()
                                  or torch.cuda.device_count() != len(state['torch_cuda'])):
        raise ValueError('CUDA RNG device count differs from checkpoint')
    n = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((n['algorithm'], np.array(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gauss']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['cuda_required']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def capture_progress(model, optimizer, provenance, encoder_contract, *, epoch_index,
                     order, cursor, global_step, include_cuda=True, continuity=None,
                     augmentation_generators=None):
    """Return safe state for payload['training_state']; call after optimizer.step."""
    contract = _contract(provenance, encoder_contract)
    _validate_optimizer_contract(optimizer, contract)
    size = _integer(contract['frame_sampling']['selected_frames'], 'selected_frames', 1)
    _validate_position(order, cursor, epoch_index, global_step, size)
    result = dict(schema=SCHEMA, contract=contract,
                optimizer_parameter_names=_optimizer_names(model, optimizer),
                dataset_size=size, epoch_index=epoch_index, order=list(order), cursor=cursor,
                global_step=global_step, rng_state=_capture_rng(include_cuda),
                continuity=copy.deepcopy(continuity or {'origin': 'stateful_sampler'}))
    augmentation = contract.get('augmentation_rng_contract')
    if augmentation is not None:
        generators = augmentation_generators or {}
        state = {name: generator.get_state().clone() for name, generator in generators.items()}
        _validate_augmentation_rng(state, augmentation['generator_names'])
        result['augmentation_rng_state'] = state
    elif augmentation_generators:
        raise ValueError('Augmentation generators require an explicit contract')
    return result


def _validate_weights(payload, model):
    saved, current = payload.get('state_dict'), model.state_dict()
    if not isinstance(saved, dict) or list(saved) != list(current):
        raise ValueError('Model state parameter/buffer order differs')
    for name, expected in current.items():
        value = saved[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('Model tensor mismatch: ' + name)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite model tensor: ' + name)


def _validate_optimizer(saved, model, optimizer, names, global_step):
    current = optimizer.state_dict()
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('Missing AdamW optimizer state')
    if len(saved['param_groups']) != len(current['param_groups']):
        raise ValueError('Optimizer group count changed')
    parameters, known = dict(model.named_parameters()), set()
    for sg, cg, group_names in zip(saved['param_groups'], current['param_groups'], names):
        if {k:v for k,v in sg.items() if k != 'params'} != {k:v for k,v in cg.items() if k != 'params'}:
            raise ValueError('Optimizer hyperparameters changed')
        if sg['params'] != cg['params'] or len(sg['params']) != len(group_names):
            raise ValueError('Optimizer parameter mapping changed')
        for pid, name in zip(sg['params'], group_names):
            if pid in known:
                raise ValueError('Duplicate optimizer parameter')
            known.add(pid)
            entry = saved['state'].get(pid)
            if entry is None:
                continue  # Adam initializes a parameter lazily on its first gradient.
            required = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if sg.get('amsgrad') else set())
            if set(entry) != required:
                raise ValueError('Incomplete AdamW moments: ' + name)
            step = entry['step']
            if not isinstance(step, torch.Tensor) or step.numel() != 1:
                raise ValueError('Invalid optimizer step: ' + name)
            number = float(step)
            if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= global_step:
                raise ValueError('Invalid optimizer step: ' + name)
            for key in required - {'step'}:
                value, expected = entry[key], parameters[name]
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                        or value.dtype != expected.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError('Invalid optimizer moment: ' + name + ':' + key)
    if not set(saved['state']).issubset(known) or (global_step > 0 and not saved['state']):
        raise ValueError('Optimizer state has missing/unknown parameters')


def restore_progress(payload, model, optimizer, provenance, encoder_contract, *,
                     dataset_size, seed, allow_legacy_sampler_reset=False):
    """Validate all contracts before loading model/moments; defer RNG application.

    A legacy continuation preserves weights, AdamW moments, global update count,
    and the unfinished epoch's sample *budget*. It cannot preserve which legacy
    frames were consumed; repeats/omissions are explicitly recorded.
    """
    expected = _contract(provenance, encoder_contract)
    if _integer(seed, 'seed') != expected['sampler_seed']:
        raise ValueError('Resume sampler seed contradicts training contract')
    _validate_optimizer_contract(optimizer, expected)
    if _contract(payload.get('pretraining', {}), payload.get('morai_encoder_contract')) != expected:
        raise ValueError('Resume data/training/encoder contract changed')
    size = _integer(dataset_size, 'dataset_size', 1)
    if expected['frame_sampling']['selected_frames'] != size:
        raise ValueError('Resume dataset size changed')
    steps = _integer(payload.get('steps'), 'global_step')
    names = _optimizer_names(model, optimizer)
    state = payload.get('training_state')
    augmentation = expected.get('augmentation_rng_contract')
    augmentation_rng = None
    if state is None:
        if augmentation is not None:
            raise ValueError('Legacy checkpoint has no augmentation generator state')
        if not allow_legacy_sampler_reset:
            raise ValueError('Legacy checkpoint lacks sampler/RNG; explicitly allow one-time sampler reset')
        hashes = payload['pretraining'].get('implementation_sha256', {})
        if hashes.get('training/vision_encoder/recipes/transfer.py') != LEGACY_TRAINER_SHA256:
            raise ValueError('Legacy optimizer parameter ordering is not audited')
        if len(names) != 1 or names[0] != [n for n,p in model.named_parameters() if p.requires_grad]:
            raise ValueError('Legacy optimizer must preserve filtered model parameter order')
        epoch_index = max(0, (steps - 1) // size)
        cursor = steps - epoch_index * size
        if payload.get('epoch') != epoch_index + 1:
            raise ValueError('Legacy checkpoint epoch/global step disagree')
        order = epoch_order(size, seed + 917431, epoch_index)
        rng = None
        continuity = dict(origin='legacy_sampler_reset', reset_at_step=steps,
                          exact_legacy_next_batch=False,
                          caveat='Legacy permutation and RNG were not saved; remaining epoch may repeat or omit previously consumed frames.')
        kind = 'legacy_weights_optimizer_remaining_budget'
    else:
        if (state.get('schema') != SCHEMA or _canonical_sequences(state.get('contract')) != expected
                or state.get('dataset_size') != size or state.get('global_step') != steps):
            raise ValueError('Saved training state contract changed')
        if state.get('optimizer_parameter_names') != names:
            raise ValueError('Named optimizer parameter mapping changed')
        epoch_index, order, cursor = state['epoch_index'], state['order'], state['cursor']
        rng = state['rng_state']
        _validate_rng(rng)
        if augmentation is not None:
            augmentation_rng = state.get('augmentation_rng_state')
            _validate_augmentation_rng(augmentation_rng, augmentation['generator_names'])
        elif state.get('augmentation_rng_state'):
            raise ValueError('Unexpected augmentation RNG state without contract')
        if rng['cuda_required'] and (not torch.cuda.is_available()
                                    or torch.cuda.device_count() != len(rng['torch_cuda'])):
            raise ValueError('CUDA RNG device count differs from checkpoint')
        continuity = copy.deepcopy(state['continuity'])
        kind = 'saved_order_and_rng'
    _validate_position(order, cursor, epoch_index, steps, size)
    _validate_weights(payload, model)
    _validate_optimizer(payload.get('optimizer'), model, optimizer, names, steps)
    # Everything above is read-only; a rejected mapping/hparam must leave the
    # caller's weights, optimizer and RNG exactly as they were.
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    return dict(epoch_index=epoch_index, order=list(order), cursor=cursor, global_step=steps,
                remaining_epoch_samples=size-cursor, pending_rng_state=copy.deepcopy(rng),
                pending_augmentation_rng_state=copy.deepcopy(augmentation_rng),
                resume_kind=kind, continuity=continuity,
                exact_next_batch=rng is not None, optimizer_restored=True)
