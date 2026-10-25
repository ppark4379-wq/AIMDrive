"""Compatibility import: tune config/parameters/lattice.py."""
import sys
from ...config.parameters import lattice as _parameters
sys.modules[__name__] = _parameters
