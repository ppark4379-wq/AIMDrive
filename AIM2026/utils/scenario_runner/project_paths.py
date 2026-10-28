"""Physical locations for the self-contained AIM2026 project.

Historical paths in recorded evidence are identifiers. Resolve them here when
opening files; never rewrite the recorded strings or their content hashes.
"""

import os
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


RUNNER_ROOT = PACKAGE_ROOT = Path(__file__).resolve().parent
AIM_ROOT = AIM2026_ROOT = RUNNER_ROOT.parents[1]
REPO_ROOT = AIM_ROOT.parent
PACKAGE_PARENT = RUNNER_ROOT.parent
EXPERT_ROOT = AIM_ROOT / "Expert"
SDK_ROOT = RUNNER_ROOT / "third_party" / "grpc_inha_univ"
SDK_SRC = SDK_ROOT / "src"
SDK_PROTO = SDK_SRC / "proto"
GEOMETRY_ROOT = RUNNER_ROOT / "data" / "geometry"
RUNTIME_ROOT = RUNNER_ROOT / ".runtime" / "integration"

_DEPENDENCY_MAPPINGS = json.loads(
    (RUNNER_ROOT / "third_party" / "legacy_path_mappings.json").read_text())["mappings"]

# B's sealed evidence was captured from this checkout. The current parent is
# also recognized so an isolated copy resolves its own logical project paths.
RECORDED_REPO_ROOT = Path("/home/jun/AIMDrive")
RECORDED_MORAI_ROOT = RECORDED_REPO_ROOT / "MoraiDrive"


class _RunnerAliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        module = importlib.import_module(self.canonical)
        self.original_spec = module.__spec__
        self.original_loader = module.__loader__
        return module

    def exec_module(self, module):
        # An alias must share globals, mocks and class identities with the
        # canonical module; executing the source again would duplicate them.
        module.__spec__ = self.original_spec
        module.__loader__ = self.original_loader

    def get_code(self, fullname):
        source = ("import runpy\n"
                  "runpy.run_module(" + repr(self.canonical) +
                  ", run_name='__main__', alter_sys=True)\n")
        return compile(source, "<scenario runner alias " + fullname + ">", "exec")


class _RunnerAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "MoraiDrive":
            # Only the historical runner import remains supported. An empty
            # namespace cannot accidentally find the original project nearby.
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        for prefix in ("MoraiDrive.scenario_runner", "AIM2026.utils.scenario_runner"):
            if fullname == prefix or fullname.startswith(prefix + "."):
                canonical = "scenario_runner" + fullname[len(prefix):]
                spec = importlib.util.find_spec(canonical)
                if spec is None:
                    return None
                return importlib.util.spec_from_loader(
                    fullname, _RunnerAliasLoader(canonical),
                    is_package=spec.submodule_search_locations is not None)
        return None


_runner_alias_finder = _RunnerAliasFinder()


def _relative_to(path, root):
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _morai_relative(relative):
    for mapping in _DEPENDENCY_MAPPINGS:
        suffix = _relative_to(relative, Path(mapping["logical_prefix"]))
        if suffix is not None:
            return RUNNER_ROOT / mapping["target_runner_relative"] / suffix
    for prefix, root in (("scenario_runner", RUNNER_ROOT),
                         ("expert", EXPERT_ROOT / "ros"),
                         (".runtime", RUNTIME_ROOT)):
        suffix = _relative_to(relative, Path(prefix))
        if suffix is not None:
            return root / suffix
    raise ValueError("No AIM2026 integration for historical MoraiDrive path: " + str(relative))


def _repo_relative(relative):
    parts = relative.parts
    if not parts:
        return AIM_ROOT
    if parts[0] == "..":
        # Old relative configuration must not bypass the same mapping applied
        # to recorded absolute paths and fall back to a neighbouring checkout.
        return project_path(Path(os.path.normpath(str(AIM_ROOT / relative))))
    if parts[0] == "AIM2026":
        if len(parts) > 2 and parts[1] == "_support":
            return _repo_relative(Path(*parts[2:]))
        return AIM_ROOT.joinpath(*parts[1:])
    if parts[0] == "MoraiDrive":
        return _morai_relative(Path(*parts[1:]))
    if parts[0] == "VLDrive":
        return REPO_ROOT.joinpath(*parts)
    if parts[0] == "scenario_runner":
        return RUNNER_ROOT.joinpath(*parts[1:])
    return AIM_ROOT / relative


def project_path(path):
    """Map known historical repository locations, retaining external paths."""
    path = Path(os.path.normpath(os.path.expanduser(str(path))))
    # On Windows the recorded POSIX root (/home/jun/AIMDrive) has no drive, so it is not "absolute"; it is still a
    # recorded location, not a path relative to this checkout.
    if not path.is_absolute() and not path.root:
        return _repo_relative(path)
    for mapping in _DEPENDENCY_MAPPINGS:
        for recorded in mapping["legacy_absolute_prefixes"]:
            suffix = _relative_to(path, Path(recorded))
            if suffix is not None:
                return RUNNER_ROOT / mapping["target_runner_relative"] / suffix
    inside_aim = _relative_to(path, AIM_ROOT)
    if inside_aim is not None:
        return _repo_relative(Path("AIM2026") / inside_aim)
    for root in (RECORDED_REPO_ROOT, REPO_ROOT):
        relative = _relative_to(path, root)
        if relative is not None:
            return _repo_relative(relative)
    return path


def morai_path(path):
    """Resolve a path historically relative to the MoraiDrive directory."""
    path = Path(path)
    if path.is_absolute():
        return project_path(path)
    return project_path(RECORDED_MORAI_ROOT / path)


def evidence_path(path, base=None):
    """Resolve evidence for reading without modifying its recorded identity."""
    path = Path(path)
    if path.is_absolute():
        return project_path(path)
    if base is None:
        return morai_path(path)
    return project_path(Path(base) / path)


def runtime_path(*parts):
    return RUNTIME_ROOT.joinpath(*parts)


def logical_source_path(path):
    """Return B's MoraiDrive-relative source key for a physical location."""
    path = Path(path).resolve()
    # SDK and evidence retain their original identities even though they now
    # live inside the runner. Native implementation files retain native keys.
    mappings = tuple((RUNNER_ROOT / item["target_runner_relative"], item["logical_prefix"])
                     for item in _DEPENDENCY_MAPPINGS)
    mappings += ((RUNTIME_ROOT, ".runtime"), (RUNNER_ROOT, "scenario_runner"),
                 (AIM_ROOT, "../AIM2026"))
    for root, prefix in mappings:
        relative = _relative_to(path, root)
        if relative is not None:
            return (Path(prefix) / relative).as_posix()
    return os.path.relpath(str(path), str(RECORDED_MORAI_ROOT))


def activate_project_paths():
    """Expose the runner and shared AIM2026 modules without external projects."""
    paths = (REPO_ROOT, PACKAGE_PARENT)
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    if _runner_alias_finder not in sys.meta_path:
        sys.meta_path.insert(0, _runner_alias_finder)
