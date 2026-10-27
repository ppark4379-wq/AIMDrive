"""Compatibility entrypoint for scenario_runner.adapters.udp.gps_ve_driver."""
# Direct and legacy bare-module entrypoints need the canonical package root.
if __name__ == "__main__" or "." not in __name__:
    from pathlib import Path as _EntryPath
    import sys as _entry_sys
    for _entry_ancestor in _EntryPath(__file__).resolve().parents:
        if (_entry_ancestor / "project_paths.py").is_file():
            _entry_sys.path.insert(0, str(_entry_ancestor.parent))
            break
    from scenario_runner.project_paths import activate_project_paths as _activate_entry_paths
    _activate_entry_paths()

from scenario_runner._compat import redirect_module as _redirect_module
_redirect_module(__name__, 'scenario_runner.adapters.udp.gps_ve_driver')
