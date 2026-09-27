"""Compatibility import; implementation lives in training.vision_encoder.recipes.transfer."""
import sys as _sys
from training.vision_encoder.recipes import transfer as _implementation

# Keep module identity, private helpers and monkeypatch targets unchanged.
_sys.modules[__name__] = _implementation
