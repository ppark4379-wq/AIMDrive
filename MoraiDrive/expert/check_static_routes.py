"""Compatibility entry point; implementation lives in tools/paths/check_static_routes.py."""
import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location(
    "expert_check_static_routes", Path(__file__).resolve().parent / "tools/paths/check_static_routes.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
globals().update({name: value for name, value in vars(_module).items()
                  if not name.startswith("__")})

if __name__ == "__main__":
    main()
