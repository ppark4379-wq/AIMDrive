"""Read existing labeled sensor records and VE caches; never generate labels."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import Dataset

from VLDrive.models.llm.zone_embedding import ZONE_NAMES, ZONE_SCHEMA
from VLDrive.training.situation.labels import encode_targets, vocabulary

ROOT = Path(__file__).resolve().parents[3]
CACHE_SCHEMA = 'situation-ve-cache-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_path(path):
    return (ROOT / path).resolve()


def child(root, relative):
    """Ignore obsolete absolute paths; only use the supplied relative asset path."""
    name = PurePosixPath(str(relative).replace('\\', '/'))
    if name.is_absolute() or '..' in name.parts or ':' in str(name):
        raise ValueError('Unsafe relative asset path: ' + str(relative))
    root = Path(root).resolve()
    path = (root / str(name)).resolve()
    if root not in path.parents:
        raise ValueError('Asset escapes its episode/cache root')
    return path


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg.get('schema') != 'situation-training-v1':
        raise ValueError('Expected situation-training-v1 config')
    if cfg['sequence_length'] < 1 or cfg['max_frame_gap_ms'] <= 0:
        raise ValueError('Invalid window settings')
    return cfg


def labeled_episodes(specs):
    episodes, seen = [], set()
    for spec in specs:
        root = project_path(spec['root'])
        selected = spec.get('include_episodes')
        source = spec.get('frame_source', 'situation_labels')
        if source not in ('situation_labels', 'sensor_record'):
            raise ValueError('Unsupported frame_source: ' + str(source))
        if source == 'sensor_record':
            from .unlabeled_sensor_record import sensor_record_episode, episode_directories
            paths = (episode_directories(root, source) if selected is None
                     else [child(root, name) for name in selected])
            if not paths or len(paths) != len(set(paths)):
                raise ValueError('Missing or duplicate Sensor Record episodes')
            for path in paths:
                if path.resolve() in seen:
                    raise ValueError('Duplicate source episode')
                seen.add(path.resolve())
                episodes.append(sensor_record_episode(path, spec, root))
            continue
        if selected is None:
            from .unlabeled_sensor_record import episode_directories
            paths = [folder / 'situation_labels/labeled_frames.jsonl' for folder in episode_directories(root)]
        else:
            if (not isinstance(selected, list) or not selected or
                    len(selected) != len(set(selected))):
                raise ValueError('include_episodes must be a nonempty unique episode list')
            paths = [child(root, name) / 'situation_labels/labeled_frames.jsonl' for name in selected]
            if any(not path.is_file() for path in paths):
                raise ValueError('A snapshotted Situation episode is missing')
        if not paths:
            raise ValueError('No situation labels under ' + str(root))
        for path in paths:
            if path.resolve() in seen:
                raise ValueError('Duplicate labeled episode')
            seen.add(path.resolve())
            complete = json.loads((path.parent / 'COMPLETE.json').read_text())
            label_sha = sha256(path)
            if complete['files']['labeled_frames.jsonl'] != label_sha:
                raise ValueError('Changed situation labels: ' + str(path))
            rows = jsonl(path)
            if not rows:
                raise ValueError('Empty situation episode')
            for i, row in enumerate(rows):
                if row['frame_id'] != row['labeling']['index']:
                    raise ValueError('Situation GT/frame index mismatch')
                if i and (row['frame_id'] <= rows[i-1]['frame_id'] or
                          row['sensor_time_ns'] <= rows[i-1]['sensor_time_ns']):
                    raise ValueError('Duplicate or nonadvancing sensor frames')
                zone = row.get('zone_instruction')
                # Exported unlabeled areas use zone=None with a placeholder ID 0.
                # This is not evidence for the learned 'normal' zone.
                if row.get('zone') is not None:
                    if type(zone) is not int or not 0 <= zone < 6 or row.get('zone') != ZONE_NAMES[zone]:
                        raise ValueError('Unexpected Zone ID/name mapping')
                    if row['labeling'].get('zone_id') != ZONE_NAMES[zone]:
                        raise ValueError('Situation label Zone differs from input Zone')
            episode = path.parent.parent
            identity = spec['namespace'] + '/' + episode.relative_to(root).as_posix()
            episodes.append(dict(identity=identity, path=episode, rows=rows, label_sha256=label_sha,
                                 label_path=path, native=episode.parent))
    if len({e['identity'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode namespace/identity')
    return episodes


class SituationSensorFrames(Dataset):
    """Camera/LiDAR preprocessing and Ego speed for one labeled episode."""
    def __init__(self, episode, contract):
        from .morai_ego_speed import CausalEgoSpeed
        from .morai_transfer_adapter import MoraiLMDriveVisionAdapter
        from .morai_encoder_contract import adapter_config, validate_encoder_contract
        self.episode = episode
        recipe = validate_encoder_contract(contract)
        if recipe['velocity_input'] != 'causal_ego_topic_speed_mps_50ms':
            raise ValueError('Situation export requires the Ego-conditioned VE contract')
        sensor = json.loads((episode['native'] / 'sensor_contract.json').read_text())
        config = adapter_config(recipe['adapter'])
        if (sensor['raw_lidar_axes'] != 'FLU' or sensor['lidar_sweeps'] != 1
                or sensor['preprocessing'] != config.preprocessing
                or not np.allclose(sensor['lidar_to_base']['translation_xyz'], config.lidar_translation_xyz,
                                   atol=1e-6, rtol=0)
                or not np.allclose(sensor['lidar_to_base']['rotation_xyzw'], config.lidar_rotation_xyzw,
                                   atol=1e-6, rtol=0)):
            raise ValueError('Dataset calibration differs from VE training contract')
        self.adapter = MoraiLMDriveVisionAdapter(config)
        speeds = CausalEgoSpeed.from_episode(episode['path'])
        self.speed_rows, self.reasons = [], Counter()
        for row in episode['rows']:
            value, reason = (speeds.at(row['sensor_time_ns']) if row.get('sensor_input_valid', True)
                             else (None, 'invalid_sensor_bundle'))
            self.speed_rows.append(value)
            if reason:
                self.reasons[reason] += 1

    def __len__(self):
        return len(self.episode['rows'])

    def read_asset(self, ref):
        path = child(self.episode['native'], ref['relative_path'])
        data = path.read_bytes()
        if len(data) != ref['size_bytes'] or hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('Changed sensor file: ' + str(path))
        return data

    def __getitem__(self, index):
        from PIL import Image
        row = self.episode['rows'][index]
        speed = self.speed_rows[index]
        if speed is None:
            raise ValueError('Unavailable causal Ego speed')
        images = {}
        for role, view in zip(('camera_1', 'camera_2', 'camera_3', 'camera_4'),
                              ('front', 'left', 'right', 'rear')):
            with Image.open(io.BytesIO(self.read_asset(row['assets'][role]))) as image:
                images[view + '_rgb'] = np.asarray(image.convert('RGB')).copy()
        inputs = self.adapter.camera_tensors(**images)
        raw = self.read_asset(row['assets']['lidar'])
        if not raw or len(raw) % 16:
            raise ValueError('Invalid LiDAR XYZI payload')
        points = np.frombuffer(raw, dtype='<f4').reshape(-1, 4).copy()
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite LiDAR points')
        lidar, count = self.adapter.lidar_to_raw_features(points, rng_seed=row['frame_id'])
        if not count:
            raise ValueError('Empty preprocessed LiDAR')
        inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count),
                      velocity=torch.tensor([speed['velocity_mps']], dtype=torch.float32))
        return inputs


class SituationCacheDataset(Dataset):
    def __init__(self, cfg):
        self.root = project_path(cfg['cache_output'])
        meta = json.loads((self.root / 'cache_meta.json').read_text())
        if (meta.get('schema') != CACHE_SCHEMA or meta.get('feature_kind') != 'vision_encoder'
                or meta.get('encoder_contract', {}).get('velocity_input') != 'causal_ego_topic_speed_mps_50ms'
                or meta.get('zone_schema') != ZONE_SCHEMA
                or meta.get('checkpoint_sha256') != cfg['checkpoint_sha256']):
            raise ValueError('Wrong VE cache schema/checkpoint/Zone mapping')
        if sha256(self.root / 'manifest.jsonl') != meta['manifest_sha256']:
            raise ValueError('Changed VE cache manifest')
        if sha256(self.root / 'source.json') != meta['source_sha256']:
            raise ValueError('Changed VE cache provenance')
        self.feature_source = meta
        cached = {}
        for record in jsonl(self.root / 'manifest.jsonl'):
            key = (record['episode_id'], record['frame_index'])
            if key in cached:
                raise ValueError('Duplicate cached frame')
            cached[key] = record
        self.windows, self.counts, self._verified = [], Counter(), set()
        length = cfg['sequence_length']
        gap_ns = cfg['max_frame_gap_ms'] * 1e6
        for episode in labeled_episodes(cfg['datasets']):
            if meta['labels_sha256'].get(episode['identity']) != episode['label_sha256']:
                raise ValueError('Cache/label episode differs; rebuild the cache')
            rows = episode['rows']
            for end in range(length - 1, len(rows)):
                anchor = rows[end]
                if not anchor.get('frame_valid') or not anchor.get('instruction'):
                    self.counts['excluded_anchor_without_gt'] += 1
                    continue
                gt = anchor['labeling']
                if not gt.get('instruction_id'):
                    continue
                if gt.get('instruction_consistency') != 'consistent' and not (
                        cfg.get('allow_no_future', False) and gt.get('instruction_consistency') == 'no_future'):
                    self.counts['excluded_instruction_consistency'] += 1
                    continue
                window = rows[end-length+1:end+1]
                if any(r.get('zone') is None or r.get('zone_instruction') is None for r in window):
                    self.counts['excluded_unknown_zone'] += 1
                    continue
                if any(b['frame_id'] != a['frame_id'] + 1 or
                       not 0 < b['sensor_time_ns'] - a['sensor_time_ns'] <= gap_ns
                       for a, b in zip(window, window[1:])):
                    self.counts['excluded_gap'] += 1
                    continue
                records = [cached.get((episode['identity'], r['frame_id'])) for r in window]
                if any(r is None for r in records):
                    self.counts['excluded_uncached_or_invalid_speed'] += 1
                    continue
                if any(r['sensor_source_ns'] != w['sensor_time_ns'] for r, w in zip(records, window)):
                    raise ValueError('VE tokens and situation GT timestamps differ')
                self.windows.append((window, records))
                self.counts['zone_' + str(anchor['zone_instruction'])] += 1
        if not self.windows:
            raise ValueError('No eligible Situation windows')
        self.vocab = vocabulary([window[-1]['labeling'] for window, _ in self.windows])
        self.counts['windows'] = len(self.windows)
        self.class_counts = {name: Counter() for name in self.vocab}
        for window, _ in self.windows:
            ids, _ = encode_targets(window[-1]['labeling'], self.vocab)
            for name, index in ids.items():
                if index != -100:
                    self.class_counts[name][self.vocab[name][index]] += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window, records = self.windows[index]
        tokens = []
        for record in records:
            ref = record['visual_feature']
            path = child(self.root, ref['path'])
            if path not in self._verified:
                if sha256(path) != ref['sha256']:
                    raise ValueError('Changed cached VE token: ' + str(path))
                self._verified.add(path)
            feature = np.load(path, allow_pickle=False)
            if feature.shape != (106, 256) or not np.isfinite(feature).all():
                raise ValueError('Invalid cached VE token shape/value')
            tokens.append(torch.from_numpy(feature.astype(np.float32)))
        ids, action = encode_targets(window[-1]['labeling'], self.vocab)
        anchor_ns = window[-1]['sensor_time_ns']
        return dict(inputs=dict(visual_tokens=torch.stack(tokens),
                    visual_mask=torch.ones(len(window), 106, dtype=torch.bool),
                    timestamps=torch.tensor([(r['sensor_time_ns']-anchor_ns)/1e9 for r in window]),
                    zone_ids=torch.tensor([r['zone_instruction'] for r in window], dtype=torch.long)),
                    targets=dict(labels={key: torch.tensor(value) for key, value in ids.items()},
                                 action_valid=torch.tensor(action)))
