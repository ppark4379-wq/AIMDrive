"""Load the team's real Q-Former/TinyLlama code without CARLA registry imports.

Only package namespaces are constructed here. Qformer, DDIA and timm modules
are loaded from the selected VLDrive repository, never substituted by mocks.
"""
import hashlib
from contextlib import contextmanager
import importlib.machinery
from importlib.metadata import version
from pathlib import Path
import sys
import types


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
    for name, path in [('lavis', lavis), ('lavis.models', lavis / 'models'),
                       ('lavis.models.blip2_models', lavis / 'models/blip2_models'),
                       ('lavis.models.vldrive', lavis / 'models/vldrive')]:
        if name in sys.modules:
            if list(getattr(sys.modules[name], '__path__', [])) != [str(path)]:
                raise RuntimeError('A different LAVIS package was already imported: ' + name)
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
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
