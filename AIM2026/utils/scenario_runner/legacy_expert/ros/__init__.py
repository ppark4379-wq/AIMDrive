"""Optional ROS and synchronous gRPC adapters, never imported by UDP startup."""

from importlib.abc import MetaPathFinder as _MetaPathFinder
from importlib.util import spec_from_loader as _spec_from_loader
import sys as _sys

# Also bootstrap project paths for the historical bare ``ros`` package import.
import scenario_runner as _runner
from scenario_runner.project_paths import _RunnerAliasLoader


class _HelperAliases(_MetaPathFinder):
    """Keep old helper imports lazy and on the canonical module objects."""

    _legacy_ros_aliases = {
        __name__ + "." + name: _runner.__name__ + ".adapters.ros." + name
        for name in ("input_timing", "synchronized_inputs", "grpc_sync_inputs")
    }

    def find_spec(self, fullname, path=None, target=None):
        canonical = self._legacy_ros_aliases.get(fullname)
        if canonical is not None:
            return _spec_from_loader(fullname, _RunnerAliasLoader(canonical), is_package=False)
        return None


if not any(getattr(finder, "_legacy_ros_aliases", None) == _HelperAliases._legacy_ros_aliases
           for finder in _sys.meta_path):
    _sys.meta_path.insert(0, _HelperAliases())
