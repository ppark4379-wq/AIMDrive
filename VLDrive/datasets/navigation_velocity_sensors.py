"""Native navigation frames with the transfer encoder's causal GPS speed input.

The legacy reader owns audited image/LiDAR geometry and Stage-2 future targets.
This reader replaces its scalar velocity with the same native-GPS differencing
used by SensorRecordDataset. It never substitutes Ego GT speed or a zero value
for an unavailable fix. Native GPS values can lag the paired sensor scene;
that recorded phase is preserved, not silently relabeled as instantaneous speed.
"""
from collections import Counter
import hashlib
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from datasets.morai_encoder_contract import validate_encoder_contract
from datasets.morai_gps_speed import parse_native_gps, causal_gps_speed
from datasets.morai_sensor_record_dataset import child, validate_relative_name
from datasets.navigation_sensor_record import NativeNavigationSensors, read_json


def gps_velocities(fixes):
    """Retain original indices and explicit rejection reasons across gaps."""
    previous = None
    speeds, reasons = [], []
    for current in fixes:
        speed, reason = None, None
        if current is None:
            reason = 'gps_missing_or_invalid_fix'
        elif previous is None:
            reason = 'gps_no_previous_fix'
        else:
            try:
                speed = causal_gps_speed(previous, current)
            except (ValueError, TypeError):
                reason = 'gps_invalid_interval'
        speeds.append(speed)
        reasons.append(reason)
        previous = current
    return speeds, reasons


class VerifiedNativeAssets:
    """Hash every read; resolve a camera/GPS directory once instead of per file.

    POSIX directory-relative opens reject symlink leaves. The fallback retains
    the existing full containment check. No open descriptors survive a call or
    get copied into DataLoader workers.
    """
    def __init__(self, native, assets):
        self.native=Path(native).resolve()
        self.assets=assets
        self.directories={}

    def __call__(self, name):
        validate_relative_name(name)
        relative=Path(name)
        if os.open in os.supports_dir_fd and hasattr(os,'O_NOFOLLOW'):
            folder=self.directories.get(relative.parent)
            if folder is None:
                folder=(self.native/relative.parent).resolve()
                if folder!=self.native and self.native not in folder.parents:
                    raise ValueError('Unsafe native asset directory')
                self.directories[relative.parent]=folder
            directory=os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                handle=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                with os.fdopen(handle,'rb') as stream:
                    raw=stream.read()
            finally:
                os.close(directory)
        else:
            raw=child(self.native,name).read_bytes()
        reference=self.assets[name]
        if len(raw)!=reference['size_bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('Changed native asset: '+name)
        return raw


class VelocityNativeNavigationSensors(NativeNavigationSensors):
    def __init__(self, episode, recipe, namespace, max_gap_ms=150):
        recipe = validate_encoder_contract(recipe)
        if (recipe['velocity_input'] != 'causal_gps_source_time'
                or recipe.get('feature_shape') != [106, 256]
                or recipe.get('token_order') != 'bev100_light1_waypoint5'):
            raise ValueError('Expected GPS-conditioned [106,256] feature encoder')
        # Reuse the unchanged legacy geometry/GT reader. Its zero scalar is
        # deliberately replaced below; this is not the exported encoder recipe.
        geometry_recipe = dict(recipe, velocity_input='constant_zero',
                               pillar_decoration='legacy_lmdrive')
        super().__init__(episode, geometry_recipe, namespace, max_gap_ms)
        self.encoder_contract = recipe
        self.read_asset = VerifiedNativeAssets(self.native,self.assets)
        audit = read_json(self.episode/'native_audit/report.json')
        bundles = {row['assets']['lidar']: row for row in audit['bundles']
                   if 'lidar' in row.get('assets', {})}
        if len(bundles) != len(audit['bundles']):
            raise ValueError('Missing or duplicate native sensor bundle')
        fixes, gps_refs = [], []
        for stamp, scene in zip(self.times, self.scenes):
            bundle = bundles.get(scene['lidar_file'])
            if bundle is None or bundle['filename_simulation_time_hypothesis_ns'] != scene['native_filename_time_ns']:
                raise ValueError('GPS bundle does not match the native scene')
            name = bundle['assets'].get('gps')
            if name is None:
                fixes.append(None)
                gps_refs.append(None)
                continue
            raw = self.read_asset(name)
            reference = self.assets[name]
            if (len(raw) != reference['size_bytes']
                    or hashlib.sha256(raw).hexdigest() != reference['sha256']):
                raise ValueError('Changed native GPS asset: '+name)
            try:
                fix = (stamp, *parse_native_gps(raw.decode('utf-8-sig')))
            except (ValueError, KeyError, UnicodeError):
                fix = None
            fixes.append(fix)
            gps_refs.append(dict(path=name, sha256=reference['sha256']))
        self.velocities, self.velocity_reasons = gps_velocities(fixes)
        self.gps_refs = gps_refs
        phase = read_json(self.episode/'native_associations/report.json')
        self.velocity_summary = dict(
            source='native_gps_consecutive_fixes', units='m/s',
            time_source='paired_sensor_source_ns; same convention as SensorRecordDataset',
            max_interval_s=.25, ego_fallback=False, invalid_zero_fallback=False,
            native_gps_lag_ns=phase.get('native_gps_lag_ns'),
            excluded=dict(Counter(r for r in self.velocity_reasons if r)),
            target_point_consumed_by_feature_encoder=False)

    def inputs(self, index):
        speed = self.velocities[index]
        if speed is None:
            raise ValueError('Unavailable encoder velocity: '+str(self.velocity_reasons[index]))
        inputs = self.geometry_inputs(index)
        inputs['velocity'] = torch.tensor([speed], dtype=torch.float32)
        # return_feature exits before the target-point-dependent GRU head.
        return inputs

    def geometry_inputs(self,index):
        scene=self.scenes[index]
        images={}
        for role,view in zip(('camera_1','camera_2','camera_3','camera_4'),('front','left','right','rear')):
            with Image.open(io.BytesIO(self.read_asset(scene['camera_files'][role]))) as image:
                images[view+'_rgb']=np.asarray(image.convert('RGB')).copy()
        inputs=self.adapter.camera_tensors(**images)
        raw=self.read_asset(scene['lidar_file'])
        if not raw or len(raw)%16:raise ValueError('Incomplete XYZI asset')
        points=np.frombuffer(raw,dtype='<f4').reshape(-1,4).copy()
        if not np.isfinite(points).all():raise ValueError('Nonfinite raw LiDAR')
        lidar,count=self.adapter.lidar_to_raw_features(points,rng_seed=index)
        if count==0:raise ValueError('Empty LiDAR')
        inputs.update(lidar=torch.from_numpy(lidar),num_points=torch.tensor(count))
        return inputs
