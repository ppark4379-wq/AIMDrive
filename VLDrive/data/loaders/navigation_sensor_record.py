"""Native Sensor Record inputs and timed ego targets for frozen Stage 2 caches.

This is a derived Stage 2 export, not a Stage 1 publication. Original admission
flags and seals are never rewritten. Visual features do not depend on language.
"""
import hashlib
import io
import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.loaders.morai_sensor_record_dataset import child, verify_seal
from data.loaders.morai_transfer_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig
from navigation.adapters import json_array_items


HORIZONS_NS = (500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000, 2_500_000_000)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def future_target(times, positions, index, yaw, max_gap_ns):
    """Interpolate actual ego positions, with no extrapolation or gap crossing."""
    if yaw is None or not math.isfinite(yaw) or positions[index] is None:
        return None
    anchor = times[index]
    stop = bisect_left(times, anchor + HORIZONS_NS[-1])
    if stop >= len(times):
        return None
    if any(positions[j] is None or times[j] - times[j-1] > max_gap_ns
           for j in range(index+1, stop+1)):
        return None
    future = []
    for horizon in HORIZONS_NS:
        stamp = anchor + horizon
        right = bisect_left(times, stamp)
        if times[right] == stamp:
            xy = positions[right]
        else:
            left = right-1
            ratio = (stamp-times[left]) / (times[right]-times[left])
            xy = np.asarray(positions[left]) * (1-ratio) + np.asarray(positions[right]) * ratio
        dx, dy = np.asarray(xy) - positions[index]
        # Stage 2 convention [left, -forward], in the current ego frame.
        future.append([-math.sin(yaw)*dx + math.cos(yaw)*dy,
                       -math.cos(yaw)*dx - math.sin(yaw)*dy])
    result = np.asarray(future, dtype=np.float32)
    return result if np.isfinite(result).all() else None


def temporal_split(stamp, boundary_ns):
    if stamp + HORIZONS_NS[-1] < boundary_ns:
        return 'train'
    if stamp >= boundary_ns:
        return 'validation'
    return None


class NativeNavigationSensors:
    def __init__(self, episode, recipe, namespace, max_gap_ms=150):
        self.episode = Path(episode).resolve()
        self.native = self.episode.parent
        self.identity = '/'.join(filter(None, (namespace, self.native.name, self.episode.name)))
        self.provenance = {}
        for folder in ('native_audit', 'native_associations', 'gt_timeline_v1', 'motion_time_v1'):
            seal = verify_seal(self.episode/folder)
            self.provenance[folder] = seal['files']
        motion = read_json(self.episode/'motion_time_v1/report.json')
        if motion.get('all_native_intervals_consistent') is not True:
            raise ValueError('Native ego motion audit failed: '+str(self.episode))
        contract_path = self.native/'sensor_contract.json'
        contract = read_json(contract_path)
        self.provenance['sensor_contract_sha256'] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        adapter = recipe['adapter']
        if (recipe.get('velocity_input') != 'constant_zero'
                or recipe.get('pillar_decoration') != 'legacy_lmdrive'
                or recipe.get('feature_shape') != [106, 256]
                or recipe.get('token_order') != 'bev100_light1_waypoint5'
                or adapter.get('use_previous_lidar_sweep') is not False):
            raise ValueError('Unsupported checkpoint recipe; do not silently change encoder inputs')
        if (contract.get('raw_lidar_axes') != 'FLU' or contract.get('lidar_sweeps') != 1
                or contract.get('preprocessing') != adapter['preprocessing']
                or not np.allclose(contract['lidar_to_base']['translation_xyz'], adapter['lidar_translation_xyz'])
                or not np.allclose(contract['lidar_to_base']['rotation_xyzw'], adapter['lidar_rotation_xyzw'])):
            raise ValueError('Native sensor calibration differs from checkpoint recipe')
        self.adapter = MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(**adapter))
        self.scenes = read_json(self.episode/'native_associations/associations.json')
        gt = read_json(self.episode/'gt_timeline_v1/frames.json')
        if len(gt) != len(self.scenes) or not gt:
            raise ValueError('Native/ego timeline length mismatch')
        self.times, self.positions = [], []
        for i, (scene, row) in enumerate(zip(self.scenes, gt)):
            stamp = scene['scene_time_hypothesis_ns']
            if (scene['index'] != i or row['index'] != i or row['lidar_source_ns'] != stamp
                    or (self.times and stamp <= self.times[-1])):
                raise ValueError('Native/ego time mismatch')
            xy = row.get('ego_xy')
            if xy is not None and (len(xy) != 2 or not np.isfinite(xy).all()):
                raise ValueError('Nonfinite ego position')
            self.times.append(stamp)
            self.positions.append(xy)
        needed = {record for row in self.scenes for record in row['ego_bracket_record_ids']}
        ego = {}
        signals = self.episode/'quality/active_signals.json'
        quality = verify_seal(self.episode/'quality', 'QUALITY_COMPLETE.json')
        actual_hash = hashlib.sha256(signals.read_bytes()).hexdigest()
        if quality['files'].get(signals.name) != actual_hash:
            raise ValueError('Ego source stream is not covered by quality seal')
        self.provenance['active_signals_sha256'] = actual_hash
        for row in json_array_items(signals):
            if row.get('record_id') in needed and row.get('role') == 'ego':
                ego[row['record_id']] = row
        self.yaws = []
        for scene, stamp in zip(self.scenes, self.times):
            candidates = [ego[k] for k in scene['ego_bracket_record_ids'] if k in ego
                          and ego[k].get('source_timestamp_ns') is not None
                          and 0 <= stamp-ego[k]['source_timestamp_ns'] <= 50_000_000]
            latest = max(candidates, key=lambda r:r['source_timestamp_ns']) if candidates else None
            self.yaws.append(latest['signal'].get('yaw') if latest else None)
        audit = read_json(self.episode/'native_audit/report.json')
        self.assets = {row['relative_path']: row for row in audit['assets']}
        if len(self.assets) != len(audit['assets']):
            raise ValueError('Duplicate native asset')
        self.max_gap_ns = int(max_gap_ms*1e6)

    def __len__(self):
        return len(self.scenes)

    def target(self, index):
        return future_target(self.times, self.positions, index, self.yaws[index], self.max_gap_ns)

    def inputs(self, index):
        scene = self.scenes[index]
        def read(name):
            raw = child(self.native, name).read_bytes()
            ref = self.assets[name]
            if len(raw) != ref['size_bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('Changed native sensor asset: '+name)
            return raw
        rgb = {}
        for role, view in zip(('camera_1','camera_2','camera_3','camera_4'), ('front','left','right','rear')):
            with Image.open(io.BytesIO(read(scene['camera_files'][role]))) as im:
                rgb[view+'_rgb'] = np.asarray(im.convert('RGB')).copy()
        inputs = self.adapter.camera_tensors(**rgb)
        raw = read(scene['lidar_file'])
        if not raw or len(raw) % 16:
            raise ValueError('Incomplete XYZI asset')
        points = np.frombuffer(raw, dtype='<f4').reshape(-1,4).copy()
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite raw LiDAR')
        lidar, count = self.adapter.lidar_to_raw_features(points, rng_seed=index)
        if count == 0:
            raise ValueError('Empty LiDAR')
        inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count), velocity=torch.zeros(1))
        return inputs
