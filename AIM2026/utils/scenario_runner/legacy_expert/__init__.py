"""Historical entrypoints for the runner's transport adapters."""

# Older tools import legacy_expert directly from the runner directory.
if __name__ == "legacy_expert":
    import importlib as _importlib
    from pathlib import Path as _LegacyPath
    import sys as _legacy_sys

    for _legacy_ancestor in _LegacyPath(__file__).resolve().parents:
        if (_legacy_ancestor / "project_paths.py").is_file():
            _legacy_sys.path.insert(0, str(_legacy_ancestor.parent))
            break
    from scenario_runner.project_paths import activate_project_paths as _activate_legacy_paths

    _activate_legacy_paths()
    _legacy_sys.modules[__name__] = _importlib.import_module("scenario_runner.legacy_expert")


def _find_running_command(paths, tokens):
    """Return the first matching command; tolerate processes exiting mid-read."""
    for path in paths:
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if any(token in command for token in tokens):
            return command
    return None
