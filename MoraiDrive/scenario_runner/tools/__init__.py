"""Scenario tooling with lazy compatibility for relocated tunnel commands.

Tunnel runtime lives with its zone. Development commands live in tools.tunnel_static.
Legacy imports refer to the same canonical module object; legacy python -m names
execute its CLI without keeping a second implementation in this directory.
"""
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

_CORE = "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts."
_DEV = "scenario_runner.tools.tunnel_static."
_TUNNEL_LEGACY_MODULES = {
    "replay_tunnel_static_library": _CORE + "replay",
    "run_tunnel_static_search": _CORE + "execution",
    "tunnel_static_search_runtime": _CORE + "runtime",
    "tunnel_static_batch_safety": _CORE + "safety",
    "run_tunnel_static_trial": _CORE + "trial",
    "run_tunnel_static_batch": _DEV + "batch",
    "verify_tunnel_models": _DEV + "models",
    "audit_tunnel_models": _DEV + "audit",
    "survey_tunnel_boundaries": _DEV + "survey",
    "tunnel_static_candidates": _DEV + "candidates",
    "supervise_tunnel_static_search": _DEV + "supervise",
    "tunnel_static_search_guardian": _DEV + "guardian",
    "tunnel_static_search_preflight_review": _DEV + "review",
}


class _TunnelLegacyLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        self.canonical_loader = module.__loader__
        self.canonical_package = module.__package__
        return module

    def exec_module(self, module):
        # create_module already imported the one authoritative implementation.
        module.__spec__ = self.canonical_spec
        module.__loader__ = self.canonical_loader
        module.__package__ = self.canonical_package
        return None

    def get_code(self, fullname):
        source = ("import importlib, sys\n"
                  "_module = importlib.import_module(" + repr(self.target) + ")\n"
                  "if __name__ == '__main__' and hasattr(_module, 'main'):\n"
                  "    sys.exit(_module.main())\n")
        return compile(source, "<legacy tunnel command " + fullname + ">", "exec")


class _TunnelLegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix):
            return None
        canonical = _TUNNEL_LEGACY_MODULES.get(fullname[len(prefix):])
        if canonical is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, _TunnelLegacyLoader(canonical))
        spec.origin = str(Path(__file__).resolve().parents[1].joinpath(*canonical.split(".")[1:]).with_suffix(".py"))
        spec.has_location = True
        return spec


if not any(isinstance(finder, _TunnelLegacyFinder) for finder in sys.meta_path):
    sys.meta_path.append(_TunnelLegacyFinder())


def __getattr__(name):
    if name in _TUNNEL_LEGACY_MODULES:
        module = importlib.import_module(__name__ + "." + name)
        globals()[name] = module
        return module
    raise AttributeError(name)
