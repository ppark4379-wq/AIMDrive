"""Training checkpoints; existing implementations relocated without changing calculations."""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from utils.common.source_compatibility import equivalent_provenance

ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_parameter(model, name, parameter):
    # Temporarily frozen warmup weights are not immutable external assets:
    # save them too, so resume/transition does not reconstruct random heads.
    return parameter.requires_grad or name in getattr(model, '_stage_base_trainable_names', ())


def parameter_hash(model, *, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if _checkpoint_parameter(model, name, parameter) != trainable:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not trainable:
        for name, buffer in model.named_buffers():
            digest.update(name.encode())
            digest.update(buffer.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _snapshot_hash(state):
    """Hash the exact CPU snapshot using parameter_hash's name/byte order."""
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes(root):
    """Hash every source file a run depends on, for the checkpoint provenance."""
    from utils.common.runtime import sha256
    root = Path(root).resolve()
    paths = {p for d in ('models', 'data/src/loaders', 'utils', 'navigation', 'training', 'utils/common', 'utils/evaluation', 'utils/config',
                        'training/common', 'training/common/runtime', 'training/connector_llm', 'training/connector_llm/utils', 'training/vision_encoder/recipes',
                        'training/vision_encoder/losses', 'training/vision_encoder/utils',
                        'training/vision_encoder') for p in (root / d).glob('*.py')}
    paths.update((root / 'models').rglob('*.py'))
    paths.update((root / 'instruction').rglob('*.py'))
    paths.update((root / 'instruction').rglob('*.json'))
    paths.update((root / 'data/src/processing').glob('build_path_features.py'))
    paths.add(root / 'training/situation/labels.py')
    paths.update(root / p for p in ('train.py', 'evaluate.py', 'utils/visualization/vision_encoder_visualization.py'))
    # Only the training-side observer belongs in checkpoint provenance.
    # UI/export/plot changes must not invalidate an otherwise identical resume.
    paths.update(root / 'utils/visualization/training_dashboard' / p for p in ('__init__.py', 'telemetry.py'))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def save_checkpoint(path, model, optimizer, *, updates, provenance, settings):
    from utils.common.runtime import sha256
    path = Path(path)
    trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if _checkpoint_parameter(model, n, p)}
    state = dict(schema=getattr(model, 'checkpoint_schema', 'morai-goal-stack-v1'), model_config=asdict(model.config), trainable_state=trainable,
        trainable_hash=_snapshot_hash(trainable), frozen_hash=parameter_hash(model, trainable=False),
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
    if set(state['trainable_state']) != {n for n, p in parameters.items() if _checkpoint_parameter(model, n, p)}:
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


def periodic_checkpoint_path(output, step, interval, slots=0):
    """Rotate bounded snapshots; the other slot survives an interrupted write."""
    if interval < 1 or step < interval or step % interval or slots < 0 or slots == 1:
        raise ValueError('Checkpoint rotation requires a valid interval and zero or >=2 slots')
    name = ('checkpoint_slot%d.pt' % ((step // interval - 1) % slots)
            if slots else 'checkpoint_%06d.pt' % step)
    return Path(output) / name


def save_model_checkpoint(path, model, *, updates, provenance, settings, metrics):
    """Save best/inference weights without the much larger AdamW state."""
    from utils.common.runtime import sha256
    path = Path(path)
    state = dict(
        kind='vldrive-model-only-v1',
        schema=model.checkpoint_schema,
        model_config=asdict(model.config),
        trainable_state={
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if _checkpoint_parameter(model, name, parameter)
        },
        trainable_hash=None,
        frozen_hash=parameter_hash(model, trainable=False),
        updates=int(updates),
        provenance=provenance,
        settings=settings,
        metrics=metrics,
    )
    state['trainable_hash'] = _snapshot_hash(state['trainable_state'])
    pending = path.with_suffix(path.suffix + '.pending')
    torch.save(state, pending)
    pending.replace(path)
    digest = dict(sha256=sha256(path), bytes=path.stat().st_size)
    path.with_suffix(path.suffix + '.sha256.json').write_text(
        json.dumps(digest, indent=2) + '\n')


def restore_model_checkpoint(path, model, *, provenance, evaluation_source_compatibility=False):
    """Restore local model-only or full training weights without an optimizer."""
    from utils.common.runtime import sha256
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
    if evaluation_source_compatibility and not equivalent_provenance(state.get('provenance', {}), provenance, ROOT):
        # Evaluation-only source changes must be explicitly reviewed; dataset,
        # asset, backend and model checks remain exact. Training resume stays strict.
        from inference.runtime.vldrive_runtime import verify_sources
        recorded = state.get('provenance', {})
        expected_provenance = copy.deepcopy(provenance)
        expected_provenance['source'] = recorded.get('source')
        backend_sources = recorded.get('backend', {}).get('sources')
        if backend_sources is not None and 'backend' in expected_provenance:
            expected_provenance['backend']['sources'] = backend_sources
        if not equivalent_provenance(recorded, expected_provenance, ROOT) or not isinstance(recorded.get('source'), dict):
            raise ValueError('Model-only checkpoint contract/provenance mismatch')
        source_changes = verify_sources(recorded)
        if backend_sources is not None:
            source_changes.extend(verify_sources({'source': backend_sources}))
    if ((model_only and state.get('kind') != 'vldrive-model-only-v1')
            or state.get('schema') != model.checkpoint_schema
            or state.get('model_config') != asdict(model.config)
            or not equivalent_provenance(state.get('provenance', {}), expected_provenance, ROOT)):
        raise ValueError('Model-only checkpoint contract/provenance mismatch')
    if parameter_hash(model, trainable=False) != state.get('frozen_hash'):
        raise ValueError('Model-only checkpoint frozen asset mismatch')
    expected = {name for name, parameter in model.named_parameters() if _checkpoint_parameter(model, name, parameter)}
    if set(state.get('trainable_state', {})) != expected:
        raise ValueError('Model-only checkpoint parameter set mismatch')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if _checkpoint_parameter(model, name, parameter):
                parameter.copy_(state['trainable_state'][name])
    if parameter_hash(model, trainable=True) != state['trainable_hash']:
        raise ValueError('Restored model-only weights differ')
    if evaluation_source_compatibility:
        state['evaluation_source_compatibility'] = source_changes
    return state


def warm_start_model_checkpoint(path, model, *, provenance, reset_prefixes=(), transfer_reason,
                                new_prefixes=()):
    """Explicit weight transfer into a new experiment, never an exact resume.

    Source/config/dataset changes require a recorded reason. Frozen assets stay
    identical; all non-reset trainable keys, shapes and dtypes must match. The
    caller constructs fresh reset modules before this function. No optimizer,
    scheduler counters or random-generator states are restored.
    """
    from utils.common.runtime import sha256
    if not isinstance(transfer_reason, str) or not transfer_reason.strip():
        raise ValueError('Warm start requires a documented transfer_reason')
    prefixes = tuple(reset_prefixes)
    additions = tuple(new_prefixes)
    if (len(additions) != len(set(additions)) or any(not isinstance(p, str) or not p or not p.endswith('.') for p in additions)
            or any(a != b and b.startswith(a) for a in additions + prefixes for b in additions + prefixes)
            or set(additions) & set(prefixes)):
        raise ValueError('New module prefixes must be unique, nonoverlapping and end in a dot')
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
    expected = {name: parameter for name, parameter in model.named_parameters() if _checkpoint_parameter(model, name, parameter)}
    reset = lambda name: any(name.startswith(prefix) for prefix in prefixes)
    new = lambda name: any(name.startswith(prefix) for prefix in additions)
    for prefix in additions:
        if any(name.startswith(prefix) for name in saved) or not any(name.startswith(prefix) for name in expected):
            raise ValueError('New prefix must exist only in the destination: ' + prefix)
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in saved) or not any(name.startswith(prefix) for name in expected):
            raise ValueError('Reset prefix must match old and new trainable parameters: ' + prefix)
    old_loaded = {name for name in saved if not reset(name)}
    new_loaded = {name for name in expected if not reset(name) and not new(name)}
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
        initialized_new_keys=sorted(name for name in expected if new(name)),
        reset_old_keys=sorted(name for name in saved if reset(name)),
        reset_new_keys=sorted(name for name in expected if reset(name)),
        loaded_keys=sorted(old_loaded), loaded_parameter_count=sum(expected[name].numel() for name in old_loaded),
        frozen_hash=frozen_hash, optimizer_restored=False, rng_restored=False,
        note='New experiment starts at update zero with a new optimizer; source/config/data changes are audited, not exact-resumed.')
