"""Compatibility import; implementation lives in training.vision_encoder.recipes.pretrain."""
import sys as _sys
from training.vision_encoder.recipes import pretrain as _implementation

# Keep module identity, private helpers and monkeypatch targets unchanged.
_sys.modules[__name__] = _implementation
