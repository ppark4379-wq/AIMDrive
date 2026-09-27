"""Pinned local backends, explicit paths and scoped numerical adapters.

Everything that decides *where* code, weights and caches come from lives
here, so a training run can be reproduced from its recorded provenance.

Formerly ``LAVIS/lavis/common/morai_runtime.py``.
"""
import hashlib
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import sys


def sorted_pillar_reduce(values, groups, *, reduce):
    """Equivalent grouped reduction with deterministic CSR ordering, dim=0."""
    import torch
    if values.ndim != 2 or groups.ndim != 1 or values.shape[0] != groups.shape[0]:
        raise ValueError('PointPillar reduction expects values[P,C] and groups[P]')
    if groups.dtype != torch.long:
        raise ValueError('PointPillar groups must be torch.long')
    try:
        from torch_scatter import segment_csr
    except (ImportError, OSError):
        size = int(groups.max().item()) + 1 if groups.numel() else 0
        if reduce == 'mean':
            output = values.new_zeros((size,) + tuple(values.shape[1:]))
            output.index_add_(0, groups, values)
            counts = values.new_zeros(size)
            counts.index_add_(0, groups, values.new_ones(groups.shape[0]))
            return output / counts.clamp_min(1).view((size,) + (1,) * (values.ndim - 1))
        if reduce == 'max':
            output = values.new_full((size,) + tuple(values.shape[1:]), -float('inf'))
            expanded = groups.view((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
            return output.scatter_reduce_(0, expanded, values, reduce='amax', include_self=True)
        raise ValueError('Unsupported PointPillar reduction: ' + reduce)
    order = torch.argsort(groups, stable=True)
    counts = torch.bincount(groups)
    pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    # CSR float32 also works with older torch_scatter builds without BF16 CSR.
    return segment_csr(values[order].float(), pointers, reduce=reduce).to(values.dtype)


@contextmanager
def deterministic_pointpillar_reductions():
    """Scope the numeric adapter to the vendored PointPillar's two call sites."""
    pointpillar = __import__('importlib').import_module('models.pointpillars')
    try:
        __import__('torch_scatter')
    except (ImportError, OSError):
        # Keep the differentiable native fallback installed by pointpillar.py.
        yield
        return
    original_mean, original_max = pointpillar.scatter_mean, pointpillar.scatter_max
    def mean(values, groups, dim=0):
        if dim != 0:
            raise ValueError('Unexpected PointPillar reduction dimension')
        return sorted_pillar_reduce(values, groups, reduce='mean')
    def maximum(values, groups, dim=0):
        if dim != 0:
            raise ValueError('Unexpected PointPillar reduction dimension')
        # This vendored call site consumes only [0], never the argmax tensor.
        return sorted_pillar_reduce(values, groups, reduce='max'), None
    pointpillar.scatter_mean, pointpillar.scatter_max = mean, maximum
    try:
        yield
    finally:
        pointpillar.scatter_mean, pointpillar.scatter_max = original_mean, original_max


@contextmanager
def complete_visual_checkpoint_initialization():
    """Skip the redundant ImageNet fetch before a STRICT complete state load.

    The encoder builds its ResNet with ImageNet weights by default, and they
    are immediately overwritten in full by the trained checkpoint.  Inside this
    scope the constructor skips the fetch, so no download and no shared-cache
    write happens.  models.create_vision_encoder also takes
    pretrained_backbone=False directly; this context manager stays for callers
    that build the encoder several levels down.
    """
    encoder = __import__('importlib').import_module('models.vision_encoder')
    original = encoder.create_resnet_features
    def initialize(name, pretrained=True, **kwargs):
        return original(name, pretrained=False, **kwargs)
    encoder.create_resnet_features = initialize
    try:
        yield
    finally:
        encoder.create_resnet_features = original


def sha256(path):
    if '#' in str(path):
        from utils.config.config_files import document_bytes
        return hashlib.sha256(document_bytes(path)).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(repository):
    """Record exactly which code and library versions a run is about to use."""
    root = Path(repository).resolve()
    required = [root / 'models/qformer.py',
                root / 'models/tinyllama_backbone.py',
                root / 'models/ddia_attention.py',
                root / 'models/vision_encoder.py']
    if not all(path.is_file() for path in required):
        raise ValueError('Expected a complete VLDrive checkout: ' + str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import models as package
    if Path(package.__file__).resolve().parent != root / 'models':
        raise RuntimeError('A different models/ package is already imported')
    import torch
    import transformers
    implementation = sorted(set(required + list((root / 'models').glob('*.py'))
                                 + list((root / 'data/src/loaders').glob('*.py'))
                                 + [p for folder in ('utils', 'training', 'utils/common', 'utils/training', 'utils/evaluation', 'utils/config') for p in (root / folder).glob('*.py')]
                                 + [root / 'utils/visualization/vision_encoder_visualization.py']))
    def package_version(name):
        try:
            return version(name)
        except Exception:
            return 'not-installed-native-fallback' if name == 'torch-scatter' else 'not-installed'
    return dict(repository=str(root), torch=str(torch.__version__), transformers=str(transformers.__version__),
                packages={name: package_version(name) for name in ('numpy', 'torchvision', 'Pillow', 'pyproj',
                                                                   'safetensors', 'tokenizers', 'torch-scatter')},
                sources={str(p.relative_to(root)): sha256(p) for p in implementation})


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_runtime():
    """Configure local caches before importing model backends."""
    import os
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.runtime/torch'))
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / '.runtime/hf_cache'))


def add_path_arguments(parser):
    parser.add_argument('--dataset-root', type=Path, help='Existing MORAI dataset root (never modified)')
    parser.add_argument('--asset-root', type=Path,
                        help='Optional root containing vision_encoder/, tinyllama-1.1b-3T/ and bert-base-uncased/')
    parser.add_argument('--path-root', type=Path, default=PROJECT_ROOT,
                        help='Base for relative paths in JSON; defaults to this VLDrive checkout')


def resolve_settings(settings, args):
    import copy
    value = copy.deepcopy(settings)
    root = args.path_root.expanduser().resolve()
    def resolve(path):
        return str((root / Path(path).expanduser()).resolve())
    value['dataset'] = str(args.dataset_root.expanduser().resolve()) if args.dataset_root else resolve(value['dataset'])
    for key in ('instructions', 'feature_cache', 'warm_start_checkpoint'):
        if value.get(key):
            value[key] = resolve(value[key])
    navigation = value.get('navigation_sampling', {})
    if navigation.get('token_cache'):
        navigation['token_cache'] = resolve(navigation['token_cache'])
    model = value.get('model', value)
    for key, suffix in [('visual_checkpoint', 'vision_encoder/vision-encoder-r50.pth.tar'),
                        ('language_checkpoint', 'TinyLlama-1.1B-3T'),
                        ('qformer_checkpoint', 'bert-base-uncased')]:
        if key in model and model[key]:
            model[key] = str((args.asset_root.expanduser().resolve() / suffix)) if args.asset_root else resolve(model[key])
    return value
