"""Standalone MORAI 26.R1 K-City 2025 scenario automation package."""

from pathlib import Path as _Path
import sys as _sys

_aim_parent = str(_Path(__file__).resolve().parents[3])
if _aim_parent not in _sys.path:
    _sys.path.insert(0, _aim_parent)

from .project_paths import activate_project_paths as _activate_paths

_activate_paths()

if __name__ != "scenario_runner":
    # Direct package-qualified imports must also reuse canonical classes.
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("scenario_runner")
else:
    from importlib import import_module as _import_module, reload as _reload
    from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
    from importlib.util import spec_from_loader as _spec_from_loader
    from types import ModuleType as _ModuleType

    class _VerificationModule(_ModuleType):
        """Retain each former public namespace while sharing its implementation."""
        def __getattribute__(self, name):
            namespace = _ModuleType.__getattribute__(self, "__dict__")
            if name in namespace.get("__all__", ()):
                return getattr(namespace["_implementation"], name)
            return _ModuleType.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name in self.__dict__.get("__all__", ()):
                setattr(self._implementation, name, value)
            _ModuleType.__setattr__(self, name, value)

        def __delattr__(self, name):
            if name in self.__dict__.get("__all__", ()):
                delattr(self._implementation, name)
            _ModuleType.__delattr__(self, name)

    class _VerificationLoader(_Loader):
        def __init__(self, exports):
            self.exports = exports

        def create_module(self, spec):
            module = _VerificationModule(spec.name)
            module._implementation = _import_module("scenario_runner.tools.verify_separation")
            module.__all__ = self.exports
            return module

        def exec_module(self, module):
            if module.__dict__.get("_loaded"):
                _reload(module._implementation)
            for name in module.__all__:
                _ModuleType.__setattr__(module, name, getattr(module._implementation, name))
            module._loaded = True

        def get_code(self, fullname):
            from .tools import _TunnelLegacyLoader
            return _TunnelLegacyLoader("scenario_runner.tools.verify_separation").get_code(fullname)

    class _VerificationAliases(_MetaPathFinder):
        """Keep former validation imports on the consolidated implementation."""
        _names = {
            "scenario_runner.tools.feature_contract": ("require", "verify_feature_contract"),
            "scenario_runner.tools.verify_segment_modes": (
                "Path", "asdict", "get_mission_mode", "polyline_length",
                "validate_window", "verify_mission_mode",
            ),
        }

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self._names:
                return None
            spec = _spec_from_loader(fullname, _VerificationLoader(self._names[fullname]))
            spec.origin = str(_Path(__file__).resolve().parent / "tools" / "verify_separation.py")
            spec.has_location = True
            return spec

    _sys.meta_path.append(_VerificationAliases())
