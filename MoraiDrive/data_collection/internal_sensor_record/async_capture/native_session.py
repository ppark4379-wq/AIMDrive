"""Resolve one native source namespace without relocating previous recordings."""
import json
from pathlib import Path, PureWindowsPath
import re

from async_capture.sensor_record import ROLES
from ssd_storage import WINDOWS_ROOT

WINDOWS_BASE = PureWindowsPath(WINDOWS_ROOT) / 'sensor_record_auto_v1'


def _direct_relative(root, ssd_root):
    """Return a validated direct dataset run relative path or ``None``."""
    try:
        relative = Path(root).resolve().relative_to(Path(ssd_root).resolve() / 'dataset')
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) == 3:
        category, campaign, run = parts
    elif len(parts) == 4 and parts[0] == '.collecting':
        _, category, campaign, run = parts
    else:
        return None
    if not all(re.fullmatch(r'[A-Za-z0-9_]{1,80}', item)
                   for item in (category, campaign, run)):
        return None
    return relative


def _approved_native_root(root, ssd_root):
    root = Path(root).resolve()
    legacy = Path(ssd_root).resolve() / 'sensor_record_auto_v1'
    try:
        relative = root.relative_to(legacy)
    except ValueError:
        return _direct_relative(root, ssd_root) is not None
    return relative.parts == ('native',) or (
        len(relative.parts) == 2 and relative.parts[0] == 'native_sessions'
        and re.fullmatch(r'[A-Za-z0-9_]{1,80}', relative.parts[1]))


def _transferred_native_mirror(run, recorded_root, ssd_root):
    """Resolve a copied run without rewriting capture-time identity evidence.

    Team transfers place the seven native role directories directly below the
    copied run.  The signed capture metadata still names the original PC's
    ``native_sessions/<run>`` path.  Accept the local mirror only when the run
    name, original session identity and completed quick barrier all agree.
    """
    run=Path(run)
    run_root=run.parent if re.fullmatch(r'episode_[0-9]+',run.name) else run
    base=Path(ssd_root)/'sensor_record_auto_v1'
    canonical=Path(ssd_root)/'dataset'
    is_canonical=False
    try:
        run_root.relative_to(base)
    except ValueError:
        try:
            relative=run_root.relative_to(canonical)
        except ValueError:
            return None
        direct = _direct_relative(run_root, ssd_root)
        if direct is None:
            return None
        is_canonical=True
    recorded_name=Path(recorded_root).name
    session_path=run_root/'SESSION.json'
    result_path=run_root/'NATIVE_CAPTURE_RESULT.json'
    if not session_path.is_file() or not result_path.is_file():
        return None
    if is_canonical:
        marker_path=run_root/'.LOCAL_DATASET_COMPLETE.json'
        if not marker_path.is_file():
            return None
        marker=json.loads(marker_path.read_text())
        if (marker.get('schema') not in {
                    'morai_local_dataset_materialization_v1',
                    'morai_local_dataset_materialization_v2'}
                or marker.get('kind')!='run'
                or marker.get('run_name')!=run_root.name
                or marker.get('source_run_name',run_root.name)!=recorded_name
                or (marker.get('hardlinked') is not True
                    and marker.get('self_contained') is not True)):
            return None
    elif recorded_name != run_root.name:
        return None
    session=json.loads(session_path.read_text())
    result=json.loads(result_path.read_text())
    if session.get('plan',{}).get('session') != recorded_name:
        return None
    quick=result.get('quick_native_capture',{})
    if result.get('error') is not None or quick.get('passed') is not True:
        return None
    if not all((run_root/role).is_dir() for role in ROLES):
        return None
    return run_root


def same_recorded_native_source(run, audited_root, contract_root, ssd_root):
    """Compare source identity while allowing one verified transferred mirror."""
    if not audited_root or not contract_root:
        return False
    if Path(audited_root) == Path(contract_root):
        return True
    mirror=_transferred_native_mirror(run,Path(contract_root),ssd_root)
    return mirror is not None and mirror.resolve()==Path(audited_root).resolve()


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
        windows_storage = PureWindowsPath(WINDOWS_ROOT)
        try:
            relative_storage = target.parent.relative_to(windows_storage)
        except ValueError:
            raise ValueError('Native target is outside the configured storage root')
        local_root = Path(ssd_root).joinpath(*relative_storage.parts)
        if not _approved_native_root(local_root, ssd_root):
            raise ValueError('Native target is outside an approved source namespace')
        roots.add(tuple(relative_storage.parts))
    if len(roots) != 1:
        raise ValueError('Partial session switch: sensor targets belong to different sessions')
    return Path(ssd_root).joinpath(*next(iter(roots)))


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
    if not _approved_native_root(root, ssd_root) or not root.is_dir():
        mirror=_transferred_native_mirror(run,root,ssd_root)
        if mirror is None:
            raise ValueError('Recorded native root belongs to another storage and no verified local mirror exists')
        return mirror
    return root
