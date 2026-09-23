#!/usr/bin/env python3
"""Replay sealed native bytes through dataset and live build on CPU.

Checks all nine model input tensors for identical raw samples, target and speed.
Does not assert live GPS timing/localization, UDP/native simultaneity, or model
performance. PNG packet replay isolates preprocessing from JPEG recompression.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.morai_sensor_record_dataset import SensorRecordDataset, PathResolver, digest, read_json
from datasets.morai_encoder_contract import LiveInputAdapter, sensor_record_encoder_contract
from inference.live_frame import LiveFrameSource
from inference.morai_route import navigation_recipe, official_route_tracker


class RecordedReceiver:
    def __init__(self, value):
        self.value = value

    def latest(self):
        return self.value, 0.


def make_route_tracker(document, encoder_contract):
    nav = document['navigation_contract']
    encoder_contract['navigation_recipes'] = [navigation_recipe(nav)]
    paths = nav['route_source_resolution']['selected_paths']
    return official_route_tracker(encoder_contract, paths['global_path'],
                                  map_root=Path(paths['link_set']).parent,
                                  route_definition=paths['route_definition'])


def check_route_targets(document, encoder_contract):
    tracker = make_route_tracker(document, encoder_contract)
    for entry in document['entries']:
        ego, nav = entry['ego_pose'], entry['navigation']
        progress = tracker.update(ego['x'], ego['y'], ego['yaw'])
        if (progress.route_index != nav['route_index']
                or progress.target_route_index != nav['target_route_index']
                or (progress.target_x, progress.target_y) != (nav['target_x'], nav['target_y'])):
            raise ValueError('Live official-route target differs from manifest frame ' + str(entry['frame_id']))
    return dict(frames=len(document['entries']), exact_target_and_route_index_match=True)


def check_frame(dataset, index, document):
    native, entry, adapter, speed = dataset.rows[index]
    expected, _ = dataset[index]
    payloads = {role: dataset.read_asset(native, entry, name) for role, name in entry['asset_roles'].items()}
    source = LiveFrameSource.__new__(LiveFrameSource)
    source.contract = sensor_record_encoder_contract([adapter.config], dataset.velocity_source)
    source.route_tracker = make_route_tracker(document, source.contract)
    for previous in document['entries']:
        if previous['frame_id'] == entry['frame_id']:
            break
        ego = previous['ego_pose']
        source.route_tracker.update(ego['x'], ego['y'], ego['yaw'])
    source.input_adapter = LiveInputAdapter(source.contract, lidar_frame="sensor_flu", lidar_to_base={
        "translation_xyz": adapter.config.lidar_translation_xyz,
        "rotation_xyzw": adapter.config.lidar_rotation_xyzw})
    source.cameras = {"rgb_" + name: RecordedReceiver(payloads[role]) for role, name in zip(
        ("camera_1", "camera_2", "camera_3", "camera_4"), ("front", "left", "right", "rear"))}
    source.lidar = RecordedReceiver(np.frombuffer(payloads['lidar'], dtype='<f4').reshape(-1, 4).copy())
    ego, nav = entry['ego_pose'], entry['navigation']
    source.status = RecordedReceiver(dict(speed=speed, yaw=ego['yaw'], yaw_degrees=np.degrees(ego['yaw'])))
    source.gps = RecordedReceiver({})
    source.route = source.route_tracker.points
    source._ego_xy = lambda fix: (ego['x'], ego['y'])
    source.frames = 0
    source.velocity_source = "recorded_causal_gps"
    source.velocity_mismatch = False  # Identical recorded speed supplied for this replay only.
    actual, metadata = source.build('cpu', rng_seed=entry['frame_id'])
    if actual is None:
        raise ValueError('Live replay rejected: ' + str(metadata))
    differences = {}
    for key, value in expected.items():
        live_value = actual[key][0]
        torch.testing.assert_close(live_value, value, rtol=0, atol=0)
        differences[key] = float((live_value - value).abs().max())
    return dict(frame_id=entry['frame_id'], max_abs_difference=differences,
                lidar_points=metadata['lidar_points'], target_point=metadata['target_point'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--frame-id', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if digest(args.manifest) != args.manifest_sha256:
        raise ValueError('Manifest SHA256 mismatch')
    document = read_json(args.manifest)
    dataset = SensorRecordDataset([(args.manifest, document)], PathResolver(), 'gps')
    if args.frame_id:
        by_id = {row[1]['frame_id']: index for index, row in enumerate(dataset.rows)}
        indices = [by_id[frame] for frame in args.frame_id]
    else:
        indices = sorted({0, len(dataset)//2, len(dataset)-1})
    contract = sensor_record_encoder_contract([dataset.rows[0][2].config], dataset.velocity_source)
    route_result = check_route_targets(document, contract)
    results = [check_frame(dataset, index, document) for index in indices]
    report = dict(schema='morai_native_live_input_parity_v1', passed=True, frames=results,
                  manifest=str(args.manifest), manifest_sha256=args.manifest_sha256,
                  official_route_parity=route_result,
                  scope='same native camera/LiDAR bytes and recorded GPS speed/ego pose; shared official route target; CPU tensors only',
                  unresolved=['live source-clock parity', 'GPS/status pose error', 'UDP/native capture timing'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
