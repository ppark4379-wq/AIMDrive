"""Pieces shared by the training paths: seeding, batching, autocast, checkpoints.

The stage-2 loop itself lives in training/training.py; this module only holds what
that loop and the tools reuse.

Formerly ``LAVIS/lavis/runners/morai_trainer.py``.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
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
    return batch_samples([sample], device)


def batch_samples(samples, device):
    return move(default_collate(samples), device)


def subset_indices(length, count):
    if count <= 0 or count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def autocast(model):
    device = next(model.parameters()).device.type
    return torch.autocast(device_type=device, dtype=torch.bfloat16,
                          enabled=model.config.precision == 'bf16')


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


def source_hashes(root):
    """Hash every source file a run depends on, for the checkpoint provenance."""
    from utils.common.runtime import sha256
    root = Path(root).resolve()
    paths = {p for d in ('models', 'data/src/loaders', 'utils', 'navigation', 'training', 'utils/common', 'utils/training', 'utils/evaluation', 'utils/config') for p in (root / d).glob('*.py')}
    paths.update(root / p for p in ('train.py', 'evaluate.py', 'utils/visualization/vision_encoder_visualization.py'))
    # Only the training-side observer belongs in checkpoint provenance.
    # UI/export/plot changes must not invalidate an otherwise identical resume.
    paths.update(root / 'utils/visualization/training_dashboard' / p for p in ('__init__.py', 'telemetry.py'))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def save_checkpoint(path, model, optimizer, *, updates, provenance, settings):
    from utils.common.runtime import sha256
    path = Path(path)
    trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    state = dict(schema=getattr(model, 'checkpoint_schema', 'morai-goal-stack-v1'), model_config=asdict(model.config), trainable_state=trainable,
        trainable_hash=parameter_hash(model, trainable=True), frozen_hash=parameter_hash(model, trainable=False),
        optimizer=optimizer.state_dict(), updates=updates, provenance=provenance, settings=settings,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    partial = path.with_suffix('.partial.pt')
    torch.save(state, partial)
    partial.replace(path)
    path.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=sha256(path), bytes=path.stat().st_size)) + '\n')


def restore_checkpoint(path, model, optimizer, *, provenance, restore_rng=True):
    from utils.common.runtime import sha256
    path = Path(path)
    integrity = json.loads(path.with_suffix('.sha256.json').read_text())
    if path.stat().st_size != integrity['bytes'] or sha256(path) != integrity['sha256']:
        raise ValueError('Checkpoint file integrity failure')
    # Only locally produced, integrity-checked training checkpoints are supported.
    # RNG state includes NumPy objects, beyond torch's weights-only allowlist.
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state['schema'] != getattr(model, 'checkpoint_schema', 'morai-goal-stack-v1') or state['model_config'] != asdict(model.config):
        raise ValueError('Checkpoint/model contract mismatch')
    from utils.common.source_compatibility import equivalent_provenance
    if not equivalent_provenance(state['provenance'], provenance, Path(__file__).resolve().parents[2]):
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
