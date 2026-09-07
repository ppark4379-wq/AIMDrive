"""Pinned local backends, explicit paths and scoped numerical adapters."""
import hashlib
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import sys


def sorted_pillar_reduce(values, groups, *, reduce):
    """Equivalent grouped reduction with deterministic CSR ordering, dim=0."""
    import torch
    from torch_scatter import segment_csr
    if values.ndim != 2 or groups.ndim != 1 or values.shape[0] != groups.shape[0]:
        raise ValueError('PointPillar reduction expects values[P,C] and groups[P]')
    order = torch.argsort(groups, stable=True)
    counts = torch.bincount(groups)
    pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    # CSR float32 also works with older torch_scatter builds without BF16 CSR.
    return segment_csr(values[order].float(), pointers, reduce=reduce).to(values.dtype)


@contextmanager
def deterministic_pointpillar_reductions():
    """Scope the numeric adapter to the vendored PointPillar's two call sites."""
    pointpillar = __import__('importlib').import_module('timm.models.pointpillar')
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
    """Skip redundant ImageNet fetch before STRICT complete LMDrive state load.

The vendored Memfuser constructor hardcodes resnet50d(pretrained=True). Its
entire state is immediately overwritten by the verified LMDrive checkpoint.
This scoped constructor adaptation performs no download or shared-cache write.
    """
    memfuser = __import__('importlib').import_module('timm.models.memfuser')
    original = memfuser.resnet50d
    def initialize(*args, **kwargs):
        kwargs['pretrained'] = False
        return original(*args, **kwargs)
    memfuser.resnet50d = initialize
    try:
        yield
    finally:
        memfuser.resnet50d = original


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(repository):
    root = Path(repository).resolve()
    lavis = root / 'LAVIS' / 'lavis'
    vision = root / 'vision_encoder'
    required = [lavis / 'models/blip2_models/Qformer.py',
                lavis / 'models/vldrive/tinyllama_backbone.py',
                lavis / 'models/vldrive/ddia_attention.py',
                vision / 'timm/models/memfuser.py']
    if not all(path.is_file() for path in required):
        raise ValueError('Expected a complete team VLDrive checkout: ' + str(root))
    import lavis as package
    if Path(package.__file__).resolve().parent != lavis:
        raise RuntimeError('A different LAVIS checkout is already imported')
    if str(vision) not in sys.path:
        sys.path.insert(0, str(vision))
    import timm
    if vision not in Path(timm.__file__).resolve().parents:
        raise RuntimeError('timm must come from the selected team VLDrive repository')
    import torch
    import transformers
    implementation = sorted(set(required + list((vision / 'timm').rglob('*.py'))
                                 + list((lavis / 'models/vldrive').glob('*.py'))))
    return dict(repository=str(root), torch=torch.__version__, transformers=transformers.__version__,
                packages={name: version(name) for name in ('numpy', 'torchvision', 'Pillow', 'pyproj',
                                                           'safetensors', 'tokenizers', 'torch-scatter')},
                sources={str(p.relative_to(root)): sha256(p) for p in implementation})


LAVIS_ROOT = Path(__file__).resolve().parents[2]


def configure_runtime():
    """Configure local caches before importing model backends."""
    import os
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TORCH_HOME', str(LAVIS_ROOT / '.runtime/torch'))
    os.environ.setdefault('HF_HOME', str(LAVIS_ROOT / '.runtime/hf_cache'))


def add_path_arguments(parser):
    parser.add_argument('--dataset-root', type=Path, help='Existing Stage6E run directory (never modified)')
    parser.add_argument('--asset-root', type=Path, help='Directory containing vision_encoder/ and TinyLlama-1.1B-3T/')
    parser.add_argument('--path-root', type=Path, default=LAVIS_ROOT,
                        help='Base for relative paths in JSON; defaults to this LAVIS checkout')


def resolve_settings(settings, args):
    import copy
    value = copy.deepcopy(settings)
    root = args.path_root.expanduser().resolve()
    def resolve(path):
        return str((root / Path(path).expanduser()).resolve())
    value['dataset'] = str(args.dataset_root.expanduser().resolve()) if args.dataset_root else resolve(value['dataset'])
    if value.get('instructions'):
        value['instructions'] = resolve(value['instructions'])
    model = value.get('model', value)
    for key, suffix in [('visual_checkpoint', 'vision_encoder/vision-encoder-r50.pth.tar'),
                        ('language_checkpoint', 'TinyLlama-1.1B-3T')]:
        if key in model:
            model[key] = str((args.asset_root.expanduser().resolve() / suffix)) if args.asset_root else resolve(model[key])
    return value
