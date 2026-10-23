"""Compatibility import; implementation lives in network.ros."""
import sys
from .ros import input_timing as _implementation
sys.modules[__name__] = _implementation
