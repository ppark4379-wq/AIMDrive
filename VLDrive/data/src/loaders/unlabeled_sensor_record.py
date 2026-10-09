"""Read sealed Sensor Record frames without inventing Situation instructions."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
ROLES = ('camera_1', 'camera_2', 'camera_3', 'camera_4', 'lidar')


def episode_directories(root, source='situation_labels'):
    """Stop at episode roots instead of walking millions of sensor payload files."""
    pending, found = [Path(root)], []
    while pending:
        folder = pending.pop()
        labeled = (folder / 'situation_labels/labeled_frames.jsonl').is_file()
        recorded = (folder / 'episode.json').is_file()
        if labeled or recorded:
            if labeled if source == 'situation_labels' else recorded:
                found.append(folder)
            continue
        with os.scandir(folder) as entries:
            pending.extend(Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)
                           and not entry.name.startswith(('.', '_'))
                           and not entry.name.startswith(('CAMERA_', 'LIDAR_', 'GPS_', 'IMU_'))
                           and entry.name not in ('software_snapshot', 'ros_logs', 'raw', 'runtime'))
    return sorted(found)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fingerprint(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sensor_record_episode(folder, spec, root):
    """Return the existing loader's row contract entirely in memory; raw stays untouched."""
    from .situation_dataset import sha256
    folder, root = Path(folder).resolve(), Path(root).resolve()
    if (folder / 'situation_labels/labeled_frames.jsonl').exists():
        raise ValueError('Situation labels now exist; select labeled frames explicitly: ' + str(folder))
    files, evidence = {}, {}

    def track(key, path):
        files[key] = Path(path)
        evidence[key] = sha256(path)
        return path

    def sealed(directory, name, seal='COMPLETE.json'):
        metadata = read(track(directory + '/' + seal, folder / directory / seal))
        path = track(directory + '/' + name, folder / directory / name)
        if metadata.get('files', {}).get(name) != evidence[directory + '/' + name]:
            raise ValueError('Changed sealed Sensor Record file: ' + str(path))
        return path

    admission = read(sealed('admission_v2', 'frames.json'))
    gt = read(sealed('gt_timeline_v1', 'frames.json'))
    sealed('quality', 'active_signals.json', 'QUALITY_COMPLETE.json')
    track('episode.json', folder / 'episode.json')
    track('sensor_contract.json', folder.parent / 'sensor_contract.json')
    track('raw/raw_manifest.json', folder / 'raw/raw_manifest.json')
    track('runtime/expert/path.csv', folder / 'runtime/expert/path.csv')
    name = 'data/src/loaders/unlabeled_sensor_record.py'
    track('implementation:' + name, ROOT / name)
    evidence['frame_source'] = 'sensor_record_without_situation_v1'
    evidence['zone_policy'] = 'unlabeled_sensor_record_defaults_to_normal'
    if not admission or len(admission) != len(gt):
        raise ValueError('Sensor Record frame timelines are empty or have different lengths')
    rows = []
    previous = None
    for frame, target in zip(admission, gt):
        index, stamp = frame['index'], frame['lidar_source_time_ns']
        if index != target['index'] or stamp != target['lidar_source_ns']:
            raise ValueError('Sensor Record frame timestamps/indices differ')
        if (type(index) is not int or type(stamp) is not int or index < 0 or stamp < 0
                or previous is not None and (index <= previous[0] or stamp <= previous[1])):
            raise ValueError('Nonadvancing Sensor Record frames')
        previous = index, stamp
        assets = frame.get('assets', {})
        for role in ROLES:
            ref = assets.get(role)
            if ref is not None:
                name = PurePosixPath(str(ref['relative_path']).replace('\\', '/'))
                if name.is_absolute() or '..' in name.parts or ':' in str(name):
                    raise ValueError('Unsafe relative asset path: ' + str(name))
                if (type(ref.get('size_bytes')) is not int or ref['size_bytes'] <= 0
                        or not isinstance(ref.get('sha256'), str) or len(ref['sha256']) != 64):
                    raise ValueError('Invalid sensor asset metadata')
                # Actual bytes, size and SHA256 are checked by read_asset when
                # exporting VE tokens. Avoid repeating 5 filesystem stats/frame
                # whenever training reads only cached tokens and Zone metadata.
        valid = frame.get('sensor_bundle_valid') is True and all(role in assets for role in ROLES)
        rows.append(dict(frame_id=index, sensor_time_ns=stamp, assets=assets,
            sensor_input_valid=valid, frame_valid=False, instruction=None,
            zone='normal', zone_instruction=0,
            labeling=dict(index=index, zone_id='normal', instruction=None, instruction_id=None,
                          instruction_consistency='unlabeled'),
            frame_invalid_reasons=[]))
    return dict(identity=spec['namespace'] + '/' + folder.relative_to(root).as_posix(),
        path=folder, native=folder.parent, rows=rows, label_path=None,
        label_sha256=fingerprint(evidence), source_evidence=evidence, source_files=files,
        frame_source='sensor_record')
