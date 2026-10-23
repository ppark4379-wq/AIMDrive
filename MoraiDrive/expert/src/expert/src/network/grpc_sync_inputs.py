"""Compatibility import; implementation lives in network.ros."""
import sys
from .ros import grpc_sync_inputs as _implementation
sys.modules[__name__] = _implementation
