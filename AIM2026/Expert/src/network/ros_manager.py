"""Compatibility import; implementation lives in network.ros."""
import sys
from .ros import ros_manager as _implementation
sys.modules[__name__] = _implementation
