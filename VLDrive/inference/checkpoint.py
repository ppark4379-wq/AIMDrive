"""Strict model-only checkpoint loading; no optimizer or training mutations."""
from dataclasses import asdict
from pathlib import Path

from .dataset import ROOT, read_json, sha256


def local_path(value):
    text = str(value).strip().strip('"').replace('\\', '/')
    if len(text) > 2 and text[1:3] == ':/' and Path('/mnt').is_dir():
        text = '/mnt/' + text[0].lower() + '/' + text[3:]
    elif text.startswith('/mnt/') and len(text) > 7 and text[6] == '/':
        if not Path('/mnt').is_dir():
            text = text[5].upper() + ':/' + text[7:]
    path = Path(text).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def inspect_checkpoint(path):
    """Read metadata on CPU without allocating CUDA memory or optimizer state."""
    import torch
    path = local_path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256.json')
    metadata = read_json(sidecar)
    before = path.stat()
    if before.st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
        raise ValueError('Joint checkpoint integrity check failed')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Joint checkpoint changed while loading; retry after saving completes')
    if state.get('kind') != 'vldrive-model-only-v1' or state.get('schema') != 'vldrive-training-v1':
        raise ValueError('Select a Joint model-only checkpoint, such as model-best.pt')
    info = dict(path=str(path), sha256=metadata['sha256'], updates=state['updates'],
                epoch=state.get('metrics', {}).get('epoch'), model_config=state['model_config'])
    return state, info


def verify_model_sources(provenance):
    """Do not silently run old weights with a changed forward/tokenization graph."""
    sources = provenance.get('source', {})
    expected = {k: v for k, v in sources.items() if k.startswith(('models/', 'instruction/'))}
    if 'models/vldrive.py' not in expected:
        raise ValueError('Joint checkpoint has no model source provenance')
    changed = [name for name, value in expected.items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != value]
    if changed:
        raise ValueError('Joint forward source differs from checkpoint: ' + ', '.join(changed))


def load_joint(path, cache_meta, *, device='cuda', asset_paths=None):
    """Preserve all training dimensions, load every learned parameter exactly."""
    import torch
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from training.common.checkpoints import parameter_hash
    from training.common.runtime.reproducibility import seed_all
    state, info = inspect_checkpoint(path)
    verify_model_sources(state['provenance'])
    config = dict(state['model_config'])
    if config['input_mode'] != 'cached':
        raise ValueError('This replay expects a Joint model trained with VE caches')
    if config['visual_sha256'] != cache_meta['checkpoint_sha256']:
        raise ValueError('VE cache and Joint checkpoint use different vision weights')
    if not all(config.get(k) for k in ('path_navigation', 'zone_instruction', 'situation_instruction')):
        raise ValueError('Expected the Navigation/Path/Zone/Situation Joint model')
    recorded_meta = state['provenance']['datasets']['train']['cache_meta']
    if recorded_meta['encoder_contract'] != cache_meta['encoder_contract']:
        raise ValueError('VE preprocessing contract differs from Joint training')
    # Relocation changes file locations only. Asset manifests, frozen parameters,
    # learned parameter names/shapes/dtypes and hashes remain strict below.
    for key in ('language_checkpoint', 'qformer_checkpoint', 'visual_checkpoint'):
        value = (asset_paths or {}).get(key, config[key])
        config[key] = str(local_path(value)) if value else ''
    seed_all(state['settings']['seed'])
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**config), device=device)
    if {k: v for k, v in asdict(model.config).items() if k not in ('language_checkpoint', 'qformer_checkpoint', 'visual_checkpoint')} != {
            k: v for k, v in state['model_config'].items() if k not in ('language_checkpoint', 'qformer_checkpoint', 'visual_checkpoint')}:
        raise ValueError('Model structure/config changed during reconstruction')
    if parameter_hash(model, trainable=False) != state['frozen_hash']:
        raise ValueError('Frozen base assets differ from the saved Joint model')
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = state['trainable_state']
    if set(expected) != set(saved):
        raise ValueError('Joint checkpoint learned parameter names differ')
    with torch.no_grad():
        for name, parameter in expected.items():
            value = saved[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError('Joint parameter dimensions/dtype differ: ' + name)
            parameter.copy_(value)
    if parameter_hash(model, trainable=True) != state['trainable_hash']:
        raise ValueError('Joint learned weights failed exact restoration')
    model.requires_grad_(False).eval()
    info.update(selection_mode=model.config.inference_selection_mode,
                device=str(device), timing_scope='cached_VE_excludes_vision_encoder')
    contract = dict(state['provenance']['datasets']['train']['situation_instruction'])
    del state
    return model, info, contract
