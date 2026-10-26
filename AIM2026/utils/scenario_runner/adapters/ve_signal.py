"""Compatibility entrypoint for scenario_runner.adapters.ve.signal."""
from scenario_runner._compat import redirect_module as _redirect_module
_redirect_module(__name__, 'scenario_runner.adapters.ve.signal')
