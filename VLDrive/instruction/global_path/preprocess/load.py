"""Read planned paths, not future ego trajectories. Coordinates must be metric XY."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_path(source):
    """Accept a JSON path, route dictionary, or an existing portable episode.

    XY and pose must share the same metric map frame; GPS degrees are not XY.
    Existing MGeo adapters may export this format without changing their code.
    """
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == '.txt':
        # Competition path: whitespace-separated metric x y [z]. Never GPS degrees.
        coordinates = np.loadtxt(source, ndmin=2)
        if coordinates.shape[0] < 2 or coordinates.shape[1] < 2:
            raise ValueError('Path text must contain at least two metric XY rows')
        data = dict(points=coordinates.tolist(),
                    closed=bool(np.linalg.norm(coordinates[0, :2] - coordinates[-1, :2]) <= .01))
    else:
        data = json.loads(Path(source).read_text(encoding='utf-8')) if isinstance(source, (str, Path)) else source
    data = data.get('route', data)
    points = np.asarray(data['points'], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError('Planned path requires at least two finite metric XY points')
    points = points[:, :2]
    # Remove consecutive duplicate vertices before differentiation.
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    if len(points) < 2:
        raise ValueError('Planned path has zero length')
    closed = data.get('closed', False)
    if type(closed) is not bool or (closed and np.linalg.norm(points[0] - points[-1]) > .01):
        raise ValueError('Closed route must repeat its initial vertex')
    result = dict(points=points.tolist(), closed=closed)
    result['route_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def load_episodes(bundle):
    """Verify and read existing navigation-bundle episodes without legacy rules."""
    root = Path(bundle).resolve()
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'lmdrive-navigation-labels-v1':
        raise ValueError('Expected a verified navigation episode bundle')
    episodes = []
    for entry in manifest['episodes']:
        path = (root / entry['episode']).resolve()
        if root not in path.parents:
            raise ValueError('Unsafe episode path')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry['episode_sha256']:
            raise ValueError('Episode hash mismatch')
        episode = json.loads(content)
        if episode['episode_id'] != entry['episode_id']:
            raise ValueError('Episode identity mismatch')
        episodes.append(episode)
    return episodes, hashlib.sha256(manifest_bytes).hexdigest()
