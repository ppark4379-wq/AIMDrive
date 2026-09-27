"""Compatibility import; implementation lives in training.joint.training."""
import sys as _sys
from training.joint import training as _implementation

# Keep module identity, private helpers and monkeypatch targets unchanged.
_sys.modules[__name__] = _implementation
