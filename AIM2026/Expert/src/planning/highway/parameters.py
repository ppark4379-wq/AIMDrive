"""Compatibility import: tune config/parameters/highway.py."""
import sys
from ...config.parameters import highway as _parameters
sys.modules[__name__] = _parameters
