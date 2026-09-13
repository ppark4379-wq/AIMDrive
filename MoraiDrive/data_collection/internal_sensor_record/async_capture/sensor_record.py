"""Sensor Record file-time hypotheses, never exposure timestamps or training GT.

The observed build formats DateTime(0)+simulation_time with a 12-hour clock
and no AM/PM. Resolve only against an independently recorded, continuous
simulation interval. mtime/arrival time is not used to select a scene.
"""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re
import stat

ROLES = {'CAMERA_1': 'camera_1', 'CAMERA_2': 'camera_2',
         'CAMERA_3': 'camera_3', 'CAMERA_4': 'camera_4',
         'LIDAR_5': 'lidar', 'GPS_6': 'gps', 'IMU_7': 'imu'}
PATTERN = re.compile(r'^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{6})(_Intensity)?\.(png|bin|txt)$')
FILENAME_QUANTUM_NS = 1000  # six decimal digits; retain all original values


def gps_text_matches_ros(fields, ros_values):
    """Allow only .NET's text double formatting precision, not motion error."""
    names=('Latitude','Longitude','Altitude','EastOffset','NorthOffset')
    return all(math.isfinite(fields[k]) and math.isfinite(ros_values[k]) and
               math.isclose(fields[k],ros_values[k],rel_tol=5e-14,abs_tol=1e-12) for k in names)


def filename_candidates(name, role):
    match = PATTERN.fullmatch(name)
    if match is None:
        raise ValueError('Unrecognized native filename: ' + name)
    yy, month, day, hour, minute, second, micro = map(int, match.groups()[:7])
    expected = ('_Intensity', 'png') if role.startswith('camera_') else (
        ('_Intensity', 'bin') if role == 'lidar' else (None, 'txt'))
    if match.groups()[7:] != expected or role not in ROLES.values():
        raise ValueError('Filename does not match sensor role')
    if not 1 <= hour <= 12 or yy != 1:
        raise ValueError('Unsupported filename epoch or 12-hour clock')
    epoch = datetime(1, 1, 1)
    values = []
    for h in (hour % 12, hour % 12 + 12):
        delta = datetime(1, month, day, h, minute, second, micro) - epoch
        values.append((delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000)
    return tuple(values)


def resolve_filename(name, role, start_ns, end_ns):
    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns < start_ns:
        raise ValueError('Independent continuous simulation interval required')
    candidates = filename_candidates(name, role)
    matches = [t for t in candidates if start_ns <= t <= end_ns]
    if len(matches) != 1:
        raise ValueError('Ambiguous/out-of-window file time; do not infer from mtime')
    return matches[0]


def inspect_directory(root, *, max_files=100000, progress=None):
    """Read metadata only; distinguish file creation/completion from exposure."""
    root = Path(root)
    result = []
    for folder, role in ROLES.items():
        directory = root / folder
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('Expected real native target directory: ' + folder)
        if progress is not None:
            progress(role, 'listing', 0, None)
        paths = sorted(directory.iterdir())
        if progress is not None:
            progress(role, 'listed', 0, len(paths))
        for index, path in enumerate(paths, 1):
            if len(result) >= max_files:
                raise ValueError('Bounded file inventory exceeded')
            # lstat once.  is_file(), is_symlink(), and stat() separately cause
            # three metadata round trips on drvfs/RaiDrive and dominate large
            # Sensor Record sessions.
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError('Unexpected native target entry: ' + str(path))
            # Validate identity now, retain both possible clock interpretations.
            candidates = filename_candidates(path.name, role)
            result.append(dict(role=role, relative_path=str(path.relative_to(root)),
                               size_bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                               filename_time_candidates_ns=list(candidates)))
            if progress is not None and (index % 1000 == 0 or index == len(paths)):
                progress(role, 'scanning', index, len(paths))
    return result


def audit_assets(root, start_ns, end_ns, *, validator, baseline=None, cadence_ns=100000000):
    """Hash + decode same bytes after recording is OFF; no source modification.

    A caller may give a previous catalog to detect changed originals. Repeated
    images are counted, not automatically rejected (a stationary scene repeats).
    Even a successful result only admits files for exposure investigation.
    """
    root = Path(root)
    rows = inspect_directory(root)
    baseline = {r['relative_path']: r for r in (baseline or [])}
    by_time = defaultdict(dict)
    errors, assets, outside = [], [], []
    seen = set()
    for row in rows:
        relative, role = row['relative_path'], row['role']
        path = root / relative
        seen.add(relative)
        try:
            stamp = resolve_filename(path.name, role, start_ns, end_ns)
        except ValueError as exc:
            if relative in baseline:
                errors.append(dict(relative_path=relative, error='Baseline asset falls outside the audited clock interval'))
            outside.append(dict(**row, reason=str(exc)))
            continue
        try:
            raw = path.read_bytes()
            checked = validator(role, raw)
            after = path.stat()
            if len(raw) != row['size_bytes'] or after.st_mtime_ns != row['mtime_ns'] or after.st_size != len(raw):
                raise ValueError('Source changed during read')
            sha = hashlib.sha256(raw).hexdigest()
            if relative in baseline and baseline[relative]['sha256'] != sha:
                raise ValueError('Source changed since previous catalog')
            asset = dict(row, **{k: v for k, v in checked.items() if k != 'size_bytes'},
                         filename_simulation_time_hypothesis_ns=stamp)
            assets.append(asset)
            if role in by_time[stamp]:
                raise ValueError('Duplicate sensor at file-time key')
            by_time[stamp][role] = relative
        except Exception as exc:
            errors.append(dict(relative_path=relative, error=type(exc).__name__ + ': ' + str(exc)))
    for relative in baseline.keys() - seen:
        errors.append(dict(relative_path=relative, error='Previously cataloged source missing'))
    bundles = [dict(filename_simulation_time_hypothesis_ns=t, assets=v,
                    missing=sorted(set(ROLES.values()) - v.keys())) for t, v in sorted(by_time.items())]
    times = sorted(by_time)
    steps = [dict(before_ns=a, after_ns=b, delta_ns=b-a) for a,b in zip(times,times[1:]) if b-a != cadence_ns]
    cadence = [r for r in steps if abs(r['delta_ns']-cadence_ns)>FILENAME_QUANTUM_NS]
    precision_steps = [r for r in steps if abs(r['delta_ns']-cadence_ns)<=FILENAME_QUANTUM_NS]
    counts = Counter(r['role'] for r in assets)
    return dict(schema='morai_sensor_record_file_audit_v1', source_root=str(root),
                independent_window_ns=[start_ns,end_ns], assets=assets, bundles=bundles,
                outside_window=outside, errors=errors, cadence_exceptions=cadence,
                filename_precision_steps=precision_steps, cadence_tolerance_ns=FILENAME_QUANTUM_NS,
                counts=dict(counts), complete_bundles=sum(not b['missing'] for b in bundles),
                files_valid=bool(assets) and not errors,
                complete_cadence=bool(bundles) and not errors and not cadence and all(not b['missing'] for b in bundles),
                source_completion_evidence='post-record read; metadata stable during validation',
                producer_reopen_impossible=False, filenames_are_exposure_proof=False,
                physical_simultaneity_verified=False, teacher_time_alignment_verified=False,
                training_visible=False)
