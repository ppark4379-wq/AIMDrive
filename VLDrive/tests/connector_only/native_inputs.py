"""Read the checkpoint's causal Ego speed without substituting GPS or zeros."""
from bisect import bisect_right
import math

import torch

from data.src.loaders.navigation_sensor_record import NativeNavigationSensors
from navigation.adapters import json_array_items


def causal_speed(rows, stamp, max_age_ns=50_000_000):
    """Rows are sorted (source_timestamp_ns, speed_mps), including invalid speeds."""
    index = bisect_right(rows, (stamp, float('inf'))) - 1
    if index < 0 or not 0 <= stamp - rows[index][0] <= max_age_ns:
        return None
    speed = rows[index][1]
    return speed if math.isfinite(speed) and speed >= 0 else None


class EgoNativeSensors(NativeNavigationSensors):
    def __init__(self, episode, recipe, namespace, max_gap_ms=150):
        if (recipe.get('schema') != 'morai_sensor_record_encoder_v2'
                or recipe.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms'
                or recipe.get('raw_lidar_frame') != 'sensor_flu'
                or recipe.get('lidar_tensor_frame') != 'base_fru'
                or recipe.get('camera_recipe') != 'lmdrive_compat_front_focus_resize800x600_center128'):
            raise ValueError('Unsupported encoder input contract')
        # Reuse the sealed geometry reader; overwrite its placeholder speed
        # BEFORE encoder execution. Preserve the real recipe in provenance.
        geometry = dict(recipe, velocity_input='constant_zero', pillar_decoration='legacy_lmdrive')
        super().__init__(episode, geometry, namespace, max_gap_ms)
        self.encoder_contract = recipe
        speeds = {}
        for row in json_array_items(self.episode / 'quality/active_signals.json'):
            if row.get('role') != 'ego' or row.get('source_timestamp_ns') is None:
                continue
            stamp = row['source_timestamp_ns']
            speed = (row.get('signal') or {}).get('speed')
            value = float(speed) if isinstance(speed, (float, int)) else float('nan')
            if stamp in speeds and speeds[stamp] != value:
                raise ValueError('Conflicting Ego speeds at one source timestamp')
            speeds[stamp] = value
        self.speed_rows = sorted(speeds.items())
        self.velocities = [causal_speed(self.speed_rows, stamp) for stamp in self.times]

    def inputs(self, index):
        speed = self.velocities[index]
        if speed is None:
            raise ValueError('No finite causal Ego speed within 50 ms')
        inputs = super().inputs(index)
        inputs['velocity'] = torch.tensor([speed], dtype=torch.float32)
        # The return_feature encoder exits before the route-dependent GRU.
        return inputs
