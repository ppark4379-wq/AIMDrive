"""Opt-in protected-turn labels over an already admitted SensorRecordDataset.

This module does not discover, admit, filter, or relabel recordings. The caller
must first construct the official SensorRecordDataset from sealed publications.
The legacy red/green target is never a fallback for movement permission.
"""
import copy
import math
from collections.abc import Mapping

import torch
from torch.utils.data import Dataset


SCHEMA = 'morai_route_movement_permission_v1'
MOVEMENT_INDICES = {'straight': 0, 'left': 1, 'right': 2}
# Explicit phases from async_capture.route_signal_binding.movement_permissions.
# Unknown/yellow phases have no permission supervision; right turn is unsupported.
PHASE_PERMISSIONS = {
    1: {'straight': False, 'left': False},
    16: {'straight': True, 'left': False},
    32: {'straight': False, 'left': True},
    33: {'straight': False, 'left': True},
    48: {'straight': True, 'left': True},
}


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _distance(value):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0.)


def permission_target(entry):
    """Parse a separate admitted movement label and retain its audit metadata.

    Missing/invalid labels return target=-1. An explicit valid=True promise
    must agree with phase semantics, the route-signal record and its sensor-time
    bracket; contradictions raise instead of becoming apparently good labels.
    Movement is route information. Signal bits, IDs, distance and provenance
    returned here are metadata only and must never be passed to the model.
    """
    if not isinstance(entry, Mapping):
        raise ValueError('Route permission requires a manifest entry mapping')
    permission = entry.get('route_permission')
    signal = entry.get('traffic_signal')
    permission = permission if isinstance(permission, Mapping) else {}
    signal = signal if isinstance(signal, Mapping) else {}
    claimed = permission.get('valid') is True
    movement = permission.get('signal_movement', signal.get('signal_movement'))
    movement = movement if isinstance(movement, str) and movement in MOVEMENT_INDICES else None
    raw = permission.get('raw_signal_bits', signal.get('raw_signal_bits'))
    light = permission.get('light_id', signal.get('light_id'))
    distance = signal.get('distance_to_stopline_m')
    result = dict(target=-1, valid=False, movement_index=MOVEMENT_INDICES.get(movement, -1),
                  movement=movement, raw_signal_bits=raw if _integer(raw) else None,
                  light_id=light if _text(light) else None,
                  stopline_distance_m=float(distance) if _distance(distance) else None,
                  provenance=dict(
                      schema=permission.get('schema'),
                      frame_id=entry.get('frame_id'), sensor_time_ns=entry.get('sensor_time_ns'),
                      signal_route_link_id=signal.get('signal_route_link_id'),
                      signal_stop_node_id=signal.get('signal_stop_node_id'),
                      source_record_ids=copy.deepcopy(permission.get('source_record_ids', [])),
                      source_bracket_ns=copy.deepcopy(permission.get('source_bracket_ns', [])),
                      separate_from_color_head=permission.get('separate_from_color_head') is True,
                      label_source='route_permission_only; never traffic_light color'))
    if not claimed:
        return result

    def require(condition, reason):
        if not condition:
            raise ValueError('Invalid claimed-valid route permission: ' + reason)

    require(permission.get('schema') == SCHEMA and signal.get('schema') == SCHEMA, 'schema')
    require(permission.get('separate_from_color_head') is True, 'separate label contract')
    require(type(permission.get('target')) is bool, 'target must be boolean')
    require(_integer(entry.get('frame_id')) and _integer(entry.get('sensor_time_ns'), 1), 'frame identity')
    require(movement in ('straight', 'left') and movement == signal.get('signal_movement'), 'route movement')
    require(_integer(raw) and raw in PHASE_PERMISSIONS
            and _integer(signal.get('raw_signal_bits'))
            and raw == signal.get('raw_signal_bits'), 'raw signal phase')
    require(_text(light) and light == signal.get('light_id'), 'bound light identity')
    require(_text(signal.get('signal_route_link_id')) and _text(signal.get('signal_stop_node_id')), 'route/stop-node binding')
    require(_distance(distance), 'stop-line distance')
    require(signal.get('permission_valid') is True, 'route signal permission mask')
    expected = PHASE_PERMISSIONS[raw][movement]
    require(permission['target'] is expected and signal.get('signal_permission') is expected, 'phase/movement permission')
    for direction in ('straight', 'left'):
        require(signal.get(direction + '_permitted') is PHASE_PERMISSIONS[raw][direction], 'phase ' + direction + ' bit')
    bracket = permission.get('source_bracket_ns')
    records = permission.get('source_record_ids')
    require(isinstance(bracket, (list, tuple)) and len(bracket) == 2
            and all(_integer(v, 1) for v in bracket), 'signal-time bracket')
    require(bracket[0] <= entry['sensor_time_ns'] <= bracket[1], 'signal bracket does not contain sensor time')
    require(list(bracket) == signal.get('light_source_bracket_ns'), 'signal-time provenance mismatch')
    require(isinstance(records, (list, tuple)) and len(records) == 2
            and all(_integer(v) for v in records), 'source record IDs')
    require(list(records) == signal.get('light_source_record_ids'), 'source record provenance mismatch')
    result.update(target=int(expected), valid=True)
    return result


class RoutePermissionDataset(Dataset):
    """Add movement input and permission target without changing admitted rows.

    ``metadata`` is aligned to ``base_dataset.rows`` for evaluation. It contains
    target provenance only; ``__getitem__`` exposes just the route movement as
    a new model input. ``manifest_scenarios`` is optional publication metadata.
    """
    def __init__(self, base_dataset, manifest_scenarios=None):
        if not hasattr(base_dataset, 'rows') or not hasattr(base_dataset, 'row_manifest_paths'):
            raise ValueError('Wrap an already admitted SensorRecordDataset with aligned rows')
        if len(base_dataset) != len(base_dataset.rows) or len(base_dataset) != len(base_dataset.row_manifest_paths):
            raise ValueError('Base dataset row/manifest alignment changed')
        self.base_dataset = base_dataset
        self._metadata = []
        for row, manifest in zip(base_dataset.rows, base_dataset.row_manifest_paths):
            if len(row) < 2 or not isinstance(row[1], Mapping):
                raise ValueError('Invalid admitted base dataset row')
            entry = row[1]
            meta = permission_target(entry)
            if not _integer(entry.get('frame_id')) or not _integer(entry.get('sensor_time_ns'), 1):
                raise ValueError('Missing admitted frame identity')
            meta.update(frame_id=entry['frame_id'], sensor_time_ns=entry['sensor_time_ns'],
                        manifest_path=str(manifest))
            if manifest_scenarios is not None:
                scenario = manifest_scenarios.get(str(manifest))
                if not _text(scenario):
                    raise ValueError('Missing publication scenario for permission metadata')
                meta['scenario_name'] = scenario
            self._metadata.append(meta)

    @property
    def rows(self):
        return self.base_dataset.rows

    @property
    def row_manifest_paths(self):
        return self.base_dataset.row_manifest_paths

    @property
    def metadata(self):
        # Callers may annotate reports without changing subsequent training GT.
        return tuple(copy.deepcopy(self._metadata))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        inputs, targets = self.base_dataset[index]
        meta = self._metadata[index]
        for key in ('frame_id', 'sensor_time_ns'):
            value = targets.get(key)
            if (not isinstance(value, torch.Tensor) or value.numel() != 1
                    or value.dtype != torch.int64 or int(value.item()) != meta[key]):
                raise ValueError('Permission metadata differs from admitted sample: ' + key)
        if 'route_movement' in inputs or any(key in targets for key in ('route_permission', 'route_permission_valid')):
            raise ValueError('Route permission wrapper cannot overwrite existing fields')
        inputs, targets = dict(inputs), dict(targets)
        inputs['route_movement'] = torch.tensor(meta['movement_index'], dtype=torch.int64)
        targets['route_permission'] = torch.tensor(meta['target'], dtype=torch.int64)
        targets['route_permission_valid'] = torch.tensor(meta['valid'], dtype=torch.bool)
        return inputs, targets
