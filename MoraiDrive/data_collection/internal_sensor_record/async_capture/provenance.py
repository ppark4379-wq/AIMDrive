"""Small software identity records; no copies of environments or weights."""
import platform
from pathlib import Path

from async_capture.journal import digest

ROOT=Path(__file__).resolve().parents[1]


def software_identity():
    paths=set()
    for folder in ('async_capture','morai_pretraining','ros','native_capture'):
        paths.update((ROOT/folder).rglob('*.py'))
    paths.update((ROOT/'tools').glob('*.py'))
    paths.update(
        path for path in (ROOT/name for name in (
            'ssd_storage.py','nas_storage.py','process_guard.py'
        )) if path.is_file()
    )
    return dict(python=platform.python_version(),files={str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)})
