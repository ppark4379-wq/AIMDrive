#!/usr/bin/env python3
"""Resumable native Sensor Record -> frozen visual features and Stage 2 manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.common.runtime import configure_runtime, deterministic_pointpillar_reductions, sha256
configure_runtime()


def write_json(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)


def tensor_ref(output, name, value):
    import numpy as np
    path = output/name
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    with pending.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    pending.replace(path)
    return dict(path=name, shape=list(value.shape), dtype=str(value.dtype), sha256=sha256(path))


def valid_record(path, output):
    import numpy as np
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    ref = record['visual_feature']
    asset = (output/ref['path']).resolve()
    if output not in asset.parents:
        raise ValueError('Unsafe cached feature path')
    if asset.is_file() and sha256(asset) == ref['sha256']:
        feature = np.load(asset, allow_pickle=False)
        if feature.shape == (106,256) and np.isfinite(feature).all():
            return record
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    fields = {'schema','dataset_root','namespace','instructions','checkpoint','checkpoint_sha256',
              'output','device','cpu_threads','train_fraction','max_sensor_gap_ms','log_interval'}
    if set(cfg) != fields or cfg['schema'] != 'native-navigation-token-cache-config-v1':
        raise ValueError('Invalid native cache config')
    if not 0 < cfg['train_fraction'] < 1 or min(cfg['cpu_threads'],cfg['log_interval'],cfg['max_sensor_gap_ms']) <= 0:
        raise ValueError('Invalid split/thread/gap configuration')
    import numpy as np
    import torch
    from data.src.loaders.navigation_sensor_record import NativeNavigationSensors, temporal_split
    from navigation.sampling import NavigationLabels
    from models.ve.vision_encoder import create_vision_encoder
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(42)
    checkpoint = (ROOT/cfg['checkpoint']).resolve()
    if sha256(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    recipe = blob['morai_encoder_contract']
    labels = NavigationLabels(ROOT/cfg['instructions'])
    raw = (ROOT/cfg['dataset_root']).resolve()
    episodes = sorted(p.parent for p in raw.rglob('episode.json'))
    if not episodes:
        raise ValueError('No Sensor Record episodes')
    datasets = []
    for episode in episodes:
        print('Checking native sources:', episode, flush=True)
        dataset = NativeNavigationSensors(episode, recipe, cfg['namespace'], cfg['max_sensor_gap_ms'])
        if dataset.identity not in labels.routes:
            raise ValueError('No matching navigation episode: '+dataset.identity)
        datasets.append(dataset)
    identity = dict(schema='native-visual-cache-source-v1',checkpoint_sha256=cfg['checkpoint_sha256'],
        encoder_contract=recipe, source={d.identity:d.provenance for d in datasets},
        sources_sha256={name:sha256(ROOT/name) for name in (
            'data/src/loaders/navigation_sensor_record.py','data/src/loaders/morai_transfer_adapter.py',
            'models/ve/vision_encoder.py','models/ve/pointpillars.py','models/ve/resnet.py','models/ve/transformer.py')},
        inference_precision='float32',storage_precision='float16')
    output = (ROOT/cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output/'visual_source.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing cache has different source/encoder/implementation')
    write_json(identity_path, identity)
    model = create_vision_encoder({'name':'memfuser_baseline_e1d3_return_feature'}, pretrained_backbone=False)
    strict = str(model.load_state_dict(blob['state_dict'], strict=True))
    del blob
    model.to(cfg['device']).eval()
    records, features, counts, splits = [], [], Counter(), {}
    started = time.monotonic()
    for dataset in datasets:
        boundary = dataset.times[int(len(dataset)*cfg['train_fraction'])]
        splits[dataset.identity] = dict(validation_start_ns=boundary,policy='chronological_with_future_horizon_purge',
            evaluation_scope='same-drive held-out time segment, not independent-drive generalization')
        for i, stamp in enumerate(dataset.times):
            key = hashlib.sha256(f'{dataset.identity}|{stamp}'.encode()).hexdigest()[:24]
            record_path = output/'records'/(key+'.json')
            record = valid_record(record_path, output)
            if record is None:
                inputs = dataset.inputs(i)
                with torch.inference_mode(), deterministic_pointpillar_reductions():
                    feature = model({k:v.unsqueeze(0).to(cfg['device']) for k,v in inputs.items()})[0].float().cpu().numpy()
                if feature.shape != (106,256) or not np.isfinite(feature).all():
                    raise ValueError('Invalid encoder output')
                feature = feature.astype(np.float16)
                if not np.isfinite(feature).all():
                    raise ValueError('Float16 feature overflow')
                record = dict(episode_id=dataset.identity, sensor_source_ns=stamp, frame_index=i,
                    visual_feature=tensor_ref(output,'visual_feature/'+key+'.npy',feature))
                record_path.parent.mkdir(exist_ok=True)
                write_json(record_path, record)
                counts['inferred'] += 1
            else:
                if record['episode_id'] != dataset.identity or record['sensor_source_ns'] != stamp:
                    raise ValueError('Cached scene identity mismatch')
                counts['reused'] += 1
            features.append(record)
            scene = labels.frames.get((dataset.identity, stamp))
            target = dataset.target(i)
            split = temporal_split(stamp, boundary)
            if scene is None or not scene['valid']:
                counts['excluded_navigation'] += 1
            elif target is None:
                counts['excluded_future_or_heading'] += 1
            elif split is None:
                counts['excluded_split_purge'] += 1
            else:
                if not np.allclose([scene['frame']['x'],scene['frame']['y']],dataset.positions[i],atol=1e-5):
                    raise ValueError('Navigation and sensor ego pose differ')
                ref = tensor_ref(output,'future_trajectory/'+key+'.npy',target)
                ref.update(source='future_synchronized_ego_positions', horizons_s=[.5,1,1.5,2,2.5])
                records.append(dict(record,split=split,future_trajectory=ref))
                counts[split] += 1
            if len(features) % cfg['log_interval'] == 0 or i == len(dataset)-1:
                print(json.dumps(dict(frames=len(features),counts=counts,elapsed_s=round(time.monotonic()-started,1))),flush=True)
    if not counts['train'] or not counts['validation']:
        raise ValueError('Both train and validation require valid targets and language')
    for name, rows in [('features.jsonl',features),('manifest.jsonl',records)]:
        path=output/name
        pending=path.with_suffix('.pending')
        pending.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
        pending.replace(path)
    write_json(output/'split.json',splits)
    meta = dict(schema='navigation-token-cache-v1',checkpoint_sha256=cfg['checkpoint_sha256'],
        trajectory_coordinate_convention='lmdrive_left_negative_forward',
        manifest_sha256=sha256(output/'manifest.jsonl'),features_manifest_sha256=sha256(output/'features.jsonl'),
        labels_manifest_sha256=sha256(labels.root/'manifest.json'),config_sha256=sha256(args.config),
        visual_source_sha256=sha256(identity_path),strict_load=strict,encoder_contract=recipe,
        visual_frames=len(features),training_frames=len(records),counts=counts,split=splits,
        known_label_limitations='navigation v2: unresolved roundabout/highway-exit/traffic-light semantics; see navigation_full_lap_audit_20260917',
        target_heading_source='latest causal ego yaw, age <=50ms; actual GT timeline positions',
        publication='derived Stage 2 native export; original raw admission/publication flags unchanged')
    write_json(output/'cache_meta.json',meta)
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == '__main__':
    main()
