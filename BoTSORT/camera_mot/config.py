"""Configuration shared by live capture, inference and replay."""
from pathlib import Path
import re
import yaml


def load_config(path, selected=None):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping')
    cameras = config.get('cameras', {})
    if selected:
        missing = set(selected) - set(cameras)
        if missing:
            raise ValueError('Unknown cameras: ' + ', '.join(sorted(missing)))
        cameras = {key: cameras[key] for key in selected}
    if not cameras:
        raise ValueError('No cameras configured')
    ids = []
    ports = []
    for key, value in cameras.items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
            raise ValueError('Invalid camera name: ' + key)
        if not isinstance(value.get('frame_id'), str) or not value['frame_id']:
            raise ValueError('Camera frame_id must be a nonempty string')
        ids.append(value['frame_id'])
        port = value.get('udp_port')
        if port is not None:
            if not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError('Invalid UDP port for ' + key)
            ports.append(port)
        size = value.get('image_size', [])
        if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError('Invalid image_size for ' + key)
        if value.get('expected_hz', 0) <= 0:
            raise ValueError('Invalid expected_hz for ' + key)
        for polygon in value.get('ego_mask_polygons', []):
            if len(polygon) < 3:
                raise ValueError('Ego mask requires at least 3 vertices')
            for point in polygon:
                if (len(point) != 2 or any(not isinstance(v, int) for v in point)
                        or not 0 <= point[0] < size[0] or not 0 <= point[1] < size[1]):
                    raise ValueError('Ego mask vertices must be integer original-image pixels')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source frame_id')
    if len(set(ports)) != len(ports):
        raise ValueError('Duplicate UDP port')
    if config.get('transport', 'udp') not in ('udp', 'ros'):
        raise ValueError('Unsupported transport')
    if config.get('index_base', 0) not in (0, 1):
        raise ValueError('index_base must be 0 or 1')
    return {**config, 'cameras': cameras}
