"""Keep historical imports and command entrypoints on one implementation."""
from importlib import import_module
import sys


def redirect_module(name, target):
    """Alias imports by identity; dispatch direct files and python -m commands."""
    module = import_module(target)
    if name == "__main__":
        main = getattr(module, "main", None)
        if main is not None:
            raise SystemExit(main())
    else:
        sys.modules[name] = module
