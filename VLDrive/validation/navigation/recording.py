"""Read real recorded poses and images; never invent timestamps or ego headings."""
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def child(root, name):
    """Recorded asset references must remain inside their explicit image root."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('Missing or unsafe recorded image: ' + str(name))
    return path


def timestamp(value):
    return value if type(value) is int and value >= 0 else None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def synchronization(frame, max_skew_ms=50.):
    """Check declared source times, not wall-clock file modification times."""
    if not math.isfinite(max_skew_ms) or max_skew_ms <= 0:
        raise ValueError('max_skew_ms must be positive')
    # Only the audited native adapter creates this flag. Its camera association
    # is a scene association, not a measured camera hardware timestamp.
    if frame.get('audited_scene_association'):
        scene, pose = timestamp(frame.get('sensor_source_ns')), timestamp(frame.get('pose_source_ns'))
        valid = scene is not None and pose is not None and 0 <= scene - pose <= max_skew_ms * 1e6
        return dict(valid=valid, status='audited_scene_association' if valid else 'stale_audited_pose', skew_ms=None)
    scene, camera, pose = (timestamp(frame.get(key)) for key in
                           ('sensor_source_ns', 'camera_source_ns', 'pose_source_ns'))
    if any(value is None for value in (scene, camera, pose)):
        return dict(valid=False, status='missing_source_timestamps', skew_ms=None)
    skew = (camera - pose) / 1e6
    if pose > camera or camera > scene:
        return dict(valid=False, status='future_pose_or_camera', skew_ms=skew)
    valid = max(camera - pose, scene - camera) <= max_skew_ms * 1e6
    return dict(valid=valid, status='declared_timestamps_match' if valid else 'stale_pose_or_camera',
                skew_ms=skew)


def _collection(root, camera):
    files = sorted((p for p in (root / 'measurements_full').glob('*.json') if p.stem.isdigit()),
                   key=lambda p: int(p.stem))
    if not files:
        raise ValueError('No recorded measurements_full frames')
    frames, sources = [], []
    for path in files:
        index, row = int(path.stem), read_json(path)
        if row.get('frame_id', index) != index:
            raise ValueError('Recorded frame_id disagrees with measurement filename')
        localization = row.get('localization') or {}
        times = row.get('sensor_timestamps_ns') or {}
        source = 'gnss' if localization.get('source') == 'gnss_mgeo' else 'ego'
        # theta is the recorded CCW-radian ego heading, never the route tangent.
        frame = dict(index=index, sensor_source_ns=timestamp(row.get('timestamp_ns')),
                     camera_source_ns=timestamp(times.get('cam_' + camera)),
                     pose_source_ns=timestamp(times.get(source)),
                     x=localization.get('x', row.get('gps_x')),
                     y=localization.get('y', row.get('gps_y')),
                     yaw_rad=localization.get('heading_rad', row.get('theta')),
                     speed_mps=row.get('speed'), valid=(row.get('navigation') or {}).get('route_valid') is not False,
                     pose_source=source, measurement_file=str(path),
                     image_path=str(child(root, 'rgb_' + camera + '/' + path.stem + '.jpg')))
        frames.append(frame)
        sources.append(dict(path=str(path), sha256=digest(path)))
    route = None
    route_file = root / 'route.json'
    if route_file.is_file():
        metadata = read_json(route_file)
        if metadata.get('nav_waypoints'):
            points = metadata['nav_waypoints']
            route = dict(points=points, closed=math.dist(points[0][:2], points[-1][:2]) <= .01)
            sources.append(dict(path=str(route_file), sha256=digest(route_file)))
    return dict(format='morai_collection', episode_id=root.name, frames=frames,
                route=route, sources=sources, source_path=str(root))


def _portable(path, camera, image_root):
    document = read_json(path)
    if document.get('schema') != 'navigation-episode-v1':
        raise ValueError('Expected navigation-episode-v1')
    root = Path(image_root).resolve() if image_root else path.parent
    frames = []
    for original in document['frames']:
        frame = dict(original)
        frame.pop('audited_scene_association', None)
        image = (frame.get('camera_files') or {}).get(camera)
        if not image:
            raise ValueError('Portable frame needs camera_files.' + camera)
        frame['image_path'] = str(child(root, image))
        frames.append(frame)
    return dict(format='portable', episode_id=document['episode_id'], frames=frames,
                route=document.get('route'), sources=[dict(path=str(path), sha256=digest(path))],
                source_path=str(path))


def _native(root, camera, config):
    from navigation.adapters import sensor_record, json_array_items
    document = sensor_record(root, config)
    needed = {frame.get('ego_signal_record_id') for frame in document['frames']}
    signals = {row['record_id']: row for row in json_array_items(root / 'quality/active_signals.json')
               if row.get('record_id') in needed and row.get('role') == 'ego'}
    role = dict(front='camera_1', left='camera_2', right='camera_3', rear='camera_4')[camera]
    frames = []
    for original in document['frames']:
        frame = dict(original)
        frame['image_path'] = str(child(root.parent, frame['camera_files'][role]))
        frame['camera_source_ns'] = None
        frame['pose_source_ns'] = timestamp(signals.get(frame.get('ego_signal_record_id'), {}).get('source_timestamp_ns'))
        stamp, pose_stamp = frame['sensor_source_ns'], frame['pose_source_ns']
        frame['audited_scene_association'] = (pose_stamp is not None and
                                               0 <= stamp - pose_stamp <= 50_000_000 and
                                               finite(frame.get('yaw_rad')))
        frames.append(frame)
    return dict(format='audited_sensor_record', episode_id=document['episode_id'], frames=frames,
                route=document['route'], sources=document['sources'], source_path=str(root))


def load_recording(source, *, camera='front', image_root=None, map_root=None, route_definition=None):
    if camera not in ('front', 'left', 'right', 'rear'):
        raise ValueError('Unsupported camera view')
    root = Path(source).resolve()
    if root.is_file():
        recording = _portable(root, camera, image_root)
    elif (root / 'measurements_full').is_dir():
        recording = _collection(root, camera)
    elif (root / 'episode.json').is_file():
        recording = _native(root, camera, dict(map_root=map_root, route_definition=route_definition))
    else:
        raise ValueError('Expected a MORAI recording folder or portable episode JSON')
    if not recording['frames']:
        raise ValueError('Empty recording')
    previous_index = previous_stamp = None
    for frame in recording['frames']:
        index, stamp = frame['index'], frame.get('sensor_source_ns')
        if (type(index) is not int or index < 0 or
                (previous_index is not None and index <= previous_index)):
            raise ValueError('Recorded frame indices must strictly increase')
        if stamp is not None:
            if timestamp(stamp) is None or (previous_stamp is not None and stamp <= previous_stamp):
                raise ValueError('Recorded timestamps must strictly increase')
            previous_stamp = stamp
        previous_index = index
    return recording
