"""Resolve one native source namespace without relocating previous recordings."""
import json
from pathlib import Path, PureWindowsPath
import re

from async_capture.sensor_record import ROLES
from ssd_storage import WINDOWS_ROOT

WINDOWS_BASE = PureWindowsPath(WINDOWS_ROOT) / 'sensor_record_auto_v1'


def native_root_from_links(rows, ssd_root):
    if len(rows) != len(ROLES) or {r['role'] for r in rows} != set(ROLES):
        raise ValueError('Expected exactly seven distinct sensor junctions')
    roots = set()
    for row in rows:
        targets = row.get('target', [])
        if row.get('link_type') != 'Junction' or len(targets) != 1:
            raise ValueError('Native source must have one verified junction target')
        target = PureWindowsPath(targets[0])
        if '..' in target.parts or target.name != row['role']:
            raise ValueError('Invalid native target path')
        relative = target.parent.relative_to(WINDOWS_BASE)
        if relative.parts != ('native',) and not (
            len(relative.parts) == 2 and relative.parts[0] == 'native_sessions'
            and re.fullmatch(r'[A-Za-z0-9_]{1,80}', relative.parts[1])
        ):
            raise ValueError('Native target is outside an approved source namespace')
        roots.add(tuple(relative.parts))
    if len(roots) != 1:
        raise ValueError('Partial session switch: sensor targets belong to different sessions')
    return Path(ssd_root) / 'sensor_record_auto_v1' / Path(*next(iter(roots)))


def recorded_native_root(run, ssd_root):
    """Use capture-time identity, never today's live junction configuration."""
    run = Path(run)
    roots = set()
    metadata = [run / name for name in ('native_setup.json', 'before.json', 'episode.json')]
    # Scenario metadata belongs to the run, while audits can target its episode.
    # Never silently select the legacy source just because the caller passed
    # episode_000 instead of its parent run directory.
    if re.fullmatch(r'episode_[0-9]+', run.name):
        metadata.append(run.parent / 'native_setup.json')
    for path in metadata:
        if not path.is_file():
            continue
        info = json.loads(path.read_text())
        if info.get('native_root'):
            roots.add(str(Path(info['native_root'])))
        elif info.get('links'):
            roots.add(str(native_root_from_links(info['links'], ssd_root)))
    if len(roots) > 1:
        raise ValueError('Conflicting native source identities in run metadata')
    if not roots:
        # Explicit compatibility for the previously recorded legacy probes.
        legacy = [run.parent/'native', run.parents[1]/'native']
        found = [p for p in legacy if p.is_dir()]
        if len(found) != 1:
            raise ValueError('No unambiguous recorded native source root')
        return found[0]
    root = Path(next(iter(roots)))
    relative = root.relative_to(Path(ssd_root)/'sensor_record_auto_v1')
    if relative.parts != ('native',) and not (
        len(relative.parts) == 2 and relative.parts[0] == 'native_sessions'
        and re.fullmatch(r'[A-Za-z0-9_]{1,80}', relative.parts[1])
    ):
        raise ValueError('Recorded native root is outside the approved namespaces')
    return root
