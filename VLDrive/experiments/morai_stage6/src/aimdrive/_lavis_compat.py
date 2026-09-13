"""Legacy collector imports delegate to the canonical LAVIS implementation."""
import importlib
import os
from pathlib import Path
import sys


def load_module(name):
    root = Path(__file__).resolve().parents[4] / 'LAVIS'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault('LAVIS_IMPORT_PROFILE', 'morai')
    return importlib.import_module(name)


def export(namespace, module, aliases=None):
    target = load_module(module)
    namespace.update({k: v for k, v in vars(target).items() if not k.startswith('__')})
    for old, new in (aliases or {}).items():
        namespace[old] = getattr(target, new)
