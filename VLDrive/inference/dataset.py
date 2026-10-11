"""Read-only offline replay of VE caches and recorded navigation/Zone inputs."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def child(root, name):
    name = PurePosixPath(str(name).replace('\\', '/'))
    if name.is_absolute() or '..' in name.parts or ':' in str(name):
        raise ValueError('Expected a safe relative path: ' + str(name))
    root = Path(root).resolve()
    result = (root / str(name)).resolve()
    if root not in result.parents:
        raise ValueError('Path leaves its data directory')
    return result


def json_lines(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def local_trajectory(points, pose):
    """Stage-2 axes: [left, negative forward], metres in current ego frame."""
    delta = np.asarray(points, dtype=np.float64) - [pose['x'], pose['y']]
    c, s = math.cos(pose['yaw_rad']), math.sin(pose['yaw_rad'])
    return np.stack((-s * delta[:, 0] + c * delta[:, 1],
                     -c * delta[:, 0] - s * delta[:, 1]), axis=-1)


class ReplayDataset:
    """A selected drive is complete; gaps are displayed rather than bridged.

    Model inputs use only cached observations at or before the anchor frame.
    Future ego coordinates and GT Situation text are exposed separately for
    the viewer, never through model_inputs().
    """

    def __init__(self, bundle, episode_ids, *, sequence_length=11, frame_interval=2):
        from instruction.global_path.encoder import PathFeatureCache
        if (type(sequence_length) is not int or type(frame_interval) is not int
                or min(sequence_length, frame_interval) < 1):
            raise ValueError('Invalid replay window')
        if not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError('Select unique, nonempty episode IDs')
        self.bundle = Path(bundle).resolve()
        self.cache = self.bundle / 'vision_tokens'
        self.length, self.interval = sequence_length, frame_interval
        self.meta = read_json(self.cache / 'cache_meta.json')
        if (self.meta.get('schema') != 'navigation-token-cache-v1' or
                self.meta.get('trajectory_coordinate_convention') != 'lmdrive_left_negative_forward'):
            raise ValueError('Unsupported VE cache or trajectory coordinate convention')
        manifest_path = self.bundle / 'instructions/manifest.json'
        if sha256(manifest_path) != self.meta['labels_manifest_sha256']:
            raise ValueError('Navigation and VE cache bundles differ')
        self.manifest = {e['episode_id']: e for e in read_json(manifest_path)['episodes']}
        if set(episode_ids) - set(self.manifest):
            raise ValueError('Selected episode is not in the prepared bundle')
        self.paths = PathFeatureCache(self.bundle / 'navigation_tokens/path_features.json',
                                      self.meta['labels_manifest_sha256'])
        self.ids = list(episode_ids)
        selected = set(episode_ids)
        self.visual, self.targets = {}, {}
        # features.jsonl includes usable observations without future GT. Such
        # frames remain inferable; their ADE/FDE is simply unavailable.
        for name, key, destination in (
                ('features.jsonl', 'features_manifest_sha256', self.visual),
                ('manifest.jsonl', 'manifest_sha256', self.targets)):
            path = self.cache / name
            if sha256(path) != self.meta[key]:
                raise ValueError('Changed cache manifest: ' + name)
            for row in json_lines(path):
                if row['episode_id'] not in selected:
                    continue
                identity = (row['episode_id'], row['sensor_source_ns'])
                if identity in destination:
                    raise ValueError('Duplicate cached timestamp')
                destination[identity] = row
        self._episode_cache, self._arrays = OrderedDict(), OrderedDict()

    def episode(self, identity):
        from VLDrive.data.src.loaders.situation_dataset import labeled_episodes
        from VLDrive.instruction.zone.encoding import recorded_zone_id
        if identity not in self.ids:
            raise ValueError('Episode is outside this replay selection')
        if identity in self._episode_cache:
            self._episode_cache.move_to_end(identity)
            return self._episode_cache[identity]
        entry = self.manifest[identity]
        path = child(self.bundle / 'instructions', entry['episode'])
        if sha256(path) != entry['episode_sha256']:
            raise ValueError('Changed prepared episode')
        document = read_json(path)
        folder = child(ROOT, document['source_path'])
        namespace, relative = identity.split('/', 1)
        source_root = folder
        for _ in PurePosixPath(relative).parts:
            source_root = source_root.parent
        source = 'situation_labels' if (folder / 'situation_labels/labeled_frames.jsonl').exists() else 'sensor_record'
        raw = labeled_episodes([dict(root=str(source_root), namespace=namespace,
                                    include_episodes=[relative], frame_source=source)])[0]
        raw_by_time = {r['sensor_time_ns']: r for r in raw['rows']}
        for frame in document['frames']:
            row = raw_by_time.get(frame['sensor_source_ns'])
            if row is None or row['frame_id'] != frame['index']:
                raise ValueError('Recorded frame and prepared pose alignment differ')
            frame['zone_id'] = recorded_zone_id(row)
            record = self.visual.get((identity, frame['sensor_source_ns']))
            if record is not None and record['frame_index'] != frame['index']:
                raise ValueError('VE frame index and pose alignment differ')
        result = dict(document=document, raw=raw_by_time, native=raw['native'])
        self._episode_cache[identity] = result
        while len(self._episode_cache) > 2:
            self._episode_cache.popitem(last=False)
        return result

    def window(self, identity, position, *, length=None, interval=None, require_path=True):
        length = self.length if length is None else length
        interval = self.interval if interval is None else interval
        frames = self.episode(identity)['document']['frames']
        if not 0 <= position < len(frames):
            raise IndexError('Unknown frame position')
        start = position - (length - 1) * interval
        if start < 0:
            return None, 'insufficient_history'
        span = frames[start:position + 1]
        times = [f['sensor_source_ns'] for f in span]
        if any(b['index'] != a['index'] + 1 or not 0 < tb - ta <= self.paths.max_gap_ns
               for a, b, ta, tb in zip(span, span[1:], times, times[1:])):
            return None, 'source_frame_gap'
        if require_path and not self.paths.eligible(identity, times):
            return None, 'invalid_path_history'
        # Check the complete source span as in training, before subsampling.
        if any((identity, t) not in self.visual for t in times):
            return None, 'missing_visual_cache'
        return span[::interval], None

    def array(self, reference, shape):
        path = child(self.cache, reference['path'])
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, reference['sha256'])
        if key not in self._arrays:
            if sha256(path) != reference['sha256']:
                raise ValueError('Changed cached tensor: ' + str(path))
            value = np.load(path, allow_pickle=False)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid cached tensor')
            value.setflags(write=False)
            self._arrays[key] = value
            while len(self._arrays) > 256:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(key)
        return self._arrays[key].astype(np.float32, copy=True)

    def model_inputs(self, identity, position, situation_texts):
        import torch
        frames, reason = self.window(identity, position)
        if frames is None:
            return None, reason
        if len(situation_texts) != len(frames):
            raise ValueError('One optional Situation prediction is required per input frame')
        times = [f['sensor_source_ns'] for f in frames]
        selection, features = self.paths.sample(identity, times)
        visual = np.stack([self.array(self.visual[(identity, t)]['visual_feature'], (106, 256)) for t in times])
        return dict(visual_feature=torch.from_numpy(visual).unsqueeze(0),
                    instruction=[selection['text']], situation_instructions=[list(situation_texts)],
                    zone_ids=torch.tensor([[f['zone_id'] for f in frames]], dtype=torch.long),
                    path_features={k: torch.tensor([v], dtype=torch.bool if k == 'path_valid' else torch.float32)
                                   for k, v in features.items()}), None

    def ground_truth(self, identity, position):
        episode = self.episode(identity)
        frame = episode['document']['frames'][position]
        row = episode['raw'][frame['sensor_source_ns']]
        record = self.targets.get((identity, frame['sensor_source_ns']))
        ref = record.get('future_trajectory') if record else None
        trajectory = None
        if ref is not None:
            if ref.get('source') != 'future_synchronized_ego_positions' or ref.get('horizons_s') != [.5, 1., 1.5, 2., 2.5]:
                raise ValueError('Unexpected future ego target contract')
            trajectory = self.array(ref, (5, 2)).tolist()
        return dict(situation=row.get('labeling', {}).get('instruction'), trajectory=trajectory)

    def image_path(self, identity, position, role):
        if role not in ('camera_1', 'camera_2', 'camera_3', 'camera_4'):
            raise ValueError('Unknown camera')
        episode = self.episode(identity)
        frame = episode['document']['frames'][position]
        ref = episode['raw'][frame['sensor_source_ns']]['assets'].get(role)
        return child(episode['native'], ref['relative_path']) if ref else None

    def review_document(self, identity):
        """Viewer-only metadata. No ground-truth values enter model_inputs()."""
        episode = self.episode(identity)
        source = episode['document']['frames']
        origin = source[0]['sensor_source_ns']
        route = self.paths.encoders[identity].route['points']
        signal_root = child(ROOT, episode['document']['source_path']) / 'route_signal_audit_v1'
        signals = {}
        if (signal_root / 'frames.json').is_file():
            seal = read_json(signal_root / 'COMPLETE.json')
            if sha256(signal_root / 'frames.json') != seal['files']['frames.json']:
                raise ValueError('Changed traffic signal review evidence')
            signals = {row['sensor_source_ns']: row for row in read_json(signal_root / 'frames.json')}
        frames = []
        for position, frame in enumerate(source):
            window, reason = self.window(identity, position)
            navigation = None
            if window is not None:
                selection, _ = self.paths.sample(identity, [f['sensor_source_ns'] for f in window])
                navigation = selection['text']
            target = self.ground_truth(identity, position)
            images = {}
            for role in ('camera_1', 'camera_2', 'camera_3', 'camera_4'):
                path = self.image_path(identity, position, role)
                if path is not None:
                    images[role] = path.relative_to(ROOT / 'data/raw').as_posix()
            prepared = self.paths.frames.get((identity, frame['sensor_source_ns']), {})
            projection = prepared.get('projection')
            local = []
            if projection is not None and all(k in frame for k in ('x', 'y', 'yaw_rad')):
                segment = projection['segment']
                local = local_trajectory(route[max(0, segment - 10):segment + 100], frame).tolist()
            frames.append(dict(index=frame['index'], sensor_source_ns=frame['sensor_source_ns'],
                               elapsed_s=(frame['sensor_source_ns'] - origin) / 1e9,
                               zone_id=frame['zone_id'], navigation=navigation, reason=reason,
                               gt_trajectory=target['trajectory'], gt_situation=target['situation'],
                               path_local=local, images=images, speed_mps=frame.get('speed_mps'),
                               traffic_signal=signals.get(frame['sensor_source_ns'])))
        return dict(episode_id=identity, source_path=episode['document']['source_path'],
                    sequence_length=self.length, frame_interval=self.interval, frames=frames)
