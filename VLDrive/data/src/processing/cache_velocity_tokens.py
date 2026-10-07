#!/usr/bin/env python3
"""Cache GPS-conditioned Stage 2 features or Ego-conditioned Situation features."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    sys.path.insert(1, str(ROOT/'.runtime/stage2_wsl_deps'))
    sys.path.insert(2, str(ROOT/'.runtime/offline_viz_wsl_deps'))
from utils.common.runtime import configure_runtime, deterministic_pointpillar_reductions, sha256
configure_runtime()
from data.src.processing.cache_sensor_record_tokens import write_json, tensor_ref, valid_record


def verified_recipe(blob, checkpoint_sha, datasets):
    """Record explicit inputs without claiming absent historical publications.

    The checkpoint pins encoder/adapter/GPS implementations. Sensor calibration
    comes from the supplied, sealed current datasets. The original training
    publications may live on another PC; this distinction stays in the sidecar.
    """
    from data.src.loaders.morai_encoder_contract import sensor_record_encoder_contract, validate_encoder_contract
    from data.src.loaders.morai_transfer_adapter import MoraiVisionAdapterConfig
    from data.src.loaders.navigation_sensor_record import read_json
    provenance = blob.get('pretraining', {})
    if (provenance.get('schema') != 'morai_visual_transfer_v1'
            or provenance.get('velocity_source') != 'gps'
            or blob.get('feature_model_name') != 'memfuser_baseline_e1d3_return_feature'
            or list(blob.get('token_shape', [])) != [106, 256]):
        raise ValueError('Expected complete GPS-conditioned MORAI transfer checkpoint')
    checked = {}
    for name in ('models/ve/vision_encoder.py', 'data/src/loaders/morai_transfer_adapter.py', 'data/src/loaders/morai_gps_speed.py'):
        raw = (ROOT/name).read_bytes()
        legacy = name.replace('data/src/loaders/', 'datasets/', 1).replace('models/ve/', 'models/', 1)
        intermediate = name.replace('data/src/loaders/', 'data/loaders/', 1)
        expected = next((provenance['implementation_sha256'][key] for key in (name, intermediate, legacy)
                         if key in provenance['implementation_sha256']), None)
        candidates = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()}
        from utils.common.source_compatibility import source_matches
        if expected not in candidates and not source_matches(ROOT, name, expected):
            raise ValueError('Checkpoint input implementation mismatch: '+name)
        checked[name] = expected
    configs, evidence = [], []
    for episode, namespace in datasets:
        path = episode.parent/'sensor_contract.json'
        sensor = read_json(path)
        if (sensor.get('raw_lidar_axes') != 'FLU' or sensor.get('lidar_sweeps') != 1
                or sensor.get('preprocessing') != 'lmdrive_compat'):
            raise ValueError('Unsupported source sensor contract: '+str(path))
        extrinsic = sensor['lidar_to_base']
        config = MoraiVisionAdapterConfig(use_previous_lidar_sweep=False,
            lidar_intensity_policy='reflectance_uint8_unit',
            lidar_translation_xyz=tuple(extrinsic['translation_xyz']),
            lidar_rotation_xyzw=tuple(extrinsic['rotation_xyzw']))
        if config not in configs:
            configs.append(config)
        evidence.append(dict(path=str(path), sha256=sha256(path), namespace=namespace))
    if len(configs) != 1:
        raise ValueError('This export requires one consistent calibration across all datasets')
    contract = validate_encoder_contract(sensor_record_encoder_contract(configs, 'gps'))
    return dict(schema='morai_encoder_contract_sidecar_v1', checkpoint_sha256=checkpoint_sha,
        contract=contract, verified_implementation_sha256=checked, input_sensor_contracts=evidence,
        original_training_publications=provenance.get('publications', []),
        original_training_publications_verified=False,
        calibration_basis='explicit supplied dataset sensor contracts; historical training calibration not independently reverified',
        velocity_source='native GPS files, paired source timestamps, consecutive difference as in SensorRecordDataset',
        note='Feature export does not execute the GRU target_point head. GPS file phase is preserved and reported per episode.')


class PreparedFrames:
    def __init__(self, datasets, entries):
        self.datasets, self.entries = datasets, entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        dataset_index, frame_index = self.entries[index]
        return index, self.datasets[dataset_index].inputs(frame_index)


def discover(cfg):
    from navigation.adapters import sensor_record_episode_paths
    episodes = []
    seen = set()
    for spec in cfg['datasets']:
        if not {'root','namespace'}<=set(spec) or set(spec)-{'root','namespace','exclude_episodes'} or not spec['namespace']:
            raise ValueError('Each dataset requires an explicit root and namespace')
        found = sensor_record_episode_paths(ROOT/spec['root'],spec.get('exclude_episodes',()))
        if not found:
            raise ValueError('No episodes: '+spec['root'])
        for episode in found:
            if episode in seen:
                raise ValueError('Duplicate source episode')
            seen.add(episode)
            episodes.append((episode,spec['namespace']))
    return episodes


def reuse_situation_records(output, source, previous):
    """Copy verified records into a NEW cache; never alter a published version."""
    import os
    import shutil
    from data.src.loaders.situation_dataset import child
    previous = Path(previous).resolve()
    old = json.loads((previous / 'source.json').read_text())
    for name in ('schema', 'checkpoint_sha256', 'encoder_contract', 'implementations',
                 'inference_precision', 'storage_precision', 'diagnostic_frame_limit'):
        if old.get(name) != source.get(name):
            raise ValueError('Cannot reuse a different VE recipe: ' + name)
    matching = {name for name, evidence in source['episodes'].items()
                if old['episodes'].get(name) == evidence}
    count = 0
    for path in sorted((previous / 'records').glob('*.json')):
        row = json.loads(path.read_text())
        if row['episode_id'] not in matching:
            continue
        destination = output / 'records' / path.name
        if destination.exists():
            continue  # Normal cache validation below handles interrupted runs.
        record = valid_record(path, previous)
        if record is None:
            continue  # Recompute corrupt/missing features; never reuse them.
        ref = record['visual_feature']
        original, target = child(previous, ref['path']), child(output, ref['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != ref['sha256']:
                raise ValueError('Existing reuse target differs: ' + str(target))
        else:
            try:
                os.link(original, target)
            except OSError:
                shutil.copyfile(original, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, record)
        count += 1
    print(json.dumps(dict(reused_from=str(previous), imported_records=count)), flush=True)
    return count


def cache_situation(cfg, *, audit_only=False, max_frames_per_episode=None, reuse_cache=None):
    """VE-only export from existing Situation labels; no navigation/trajectory GT."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))
    from VLDrive.data.src.loaders.situation_dataset import (
        CACHE_SCHEMA, SituationSensorFrames, labeled_episodes, project_path,
    )
    from VLDrive.data.src.loaders.morai_encoder_contract import validate_encoder_contract
    from VLDrive.models.llm.zone_embedding import ZONE_SCHEMA
    from models.ve.vision_encoder import create_vision_encoder
    options = cfg['cache']
    if (type(options['batch_size']) is not int or options['batch_size'] < 1
            or options['workers'] < 0 or options['cpu_threads'] < 1
            or options['log_interval'] < 1
            or (max_frames_per_episode is not None and max_frames_per_episode < 1)):
        raise ValueError('Invalid Situation cache settings')
    torch.set_num_threads(options['cpu_threads'])
    checkpoint = project_path(cfg['checkpoint'])
    if sha256(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Situation VE checkpoint SHA256 mismatch')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (blob.get('feature_model_name') != 'memfuser_baseline_e1d3_return_feature'
            or blob.get('token_shape') != [106, 256]
            or blob.get('pretraining', {}).get('velocity_source') != 'ego_topic'):
        raise ValueError('Expected Ego-conditioned MORAI feature checkpoint')
    recipe = validate_encoder_contract(blob['morai_encoder_contract'])
    if (recipe['velocity_input'] != 'causal_ego_topic_speed_mps_50ms'
            or recipe.get('feature_shape') != [106, 256]
            or recipe.get('token_order') != 'bev100_light1_waypoint5'):
        raise ValueError('Unsupported Situation VE feature/velocity contract')
    implementations = {}
    for name in ('models/ve/vision_encoder.py', 'models/ve/pointpillars.py',
                 'models/ve/resnet.py', 'models/ve/transformer.py',
                 'data/src/loaders/morai_transfer_adapter.py'):
        data = (ROOT / name).read_bytes()
        expected = blob['pretraining']['implementation_sha256'].get(name)
        if expected not in {hashlib.sha256(data).hexdigest(),
                            hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()}:
            raise ValueError('VE training implementation differs: ' + name)
        implementations[name] = expected
    for name in ('data/src/loaders/morai_ego_speed.py', 'data/src/loaders/situation_dataset.py',
                 'data/src/processing/cache_velocity_tokens.py'):
        implementations[name] = sha256(ROOT / name)
    episodes = labeled_episodes(cfg['datasets'])
    sources = {e['identity']: dict(labels_sha256=e['label_sha256'],
                   sensor_contract_sha256=sha256(e['native'] / 'sensor_contract.json'),
                   raw_manifest_sha256=sha256(e['path'] / 'raw/raw_manifest.json')) for e in episodes}
    source = dict(schema=CACHE_SCHEMA, checkpoint_sha256=cfg['checkpoint_sha256'],
                  encoder_contract=recipe, implementations=implementations, episodes=sources,
                  inference_precision='float32', storage_precision='float16',
                  diagnostic_frame_limit=max_frames_per_episode)
    # Contract tuples become JSON arrays on disk; compare the serialized representation.
    source = json.loads(json.dumps(source, allow_nan=False))
    output = project_path(cfg['cache_output'])
    if not audit_only:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'source.json').exists() and json.loads((output / 'source.json').read_text()) != source:
            raise ValueError('Existing cache uses different weights, labels, sources or export settings')
        write_json(output / 'source.json', source)
        if reuse_cache is not None:
            reuse_situation_records(output, source, reuse_cache)
    model, strict = None, None
    if not audit_only:
        model = create_vision_encoder({'name': blob['feature_model_name']}, pretrained_backbone=False)
        strict = str(model.load_state_dict(blob['state_dict'], strict=True))
        model.requires_grad_(False).to(options['device']).eval()
    del blob
    totals, per_episode, exported = Counter(), {}, []
    for number, episode in enumerate(episodes, 1):
        print(f"[Situation cache] episode {number}/{len(episodes)}: {episode['identity']}", flush=True)
        dataset = SituationSensorFrames(episode, recipe)
        eligible = [i for i, speed in enumerate(dataset.speed_rows) if speed is not None]
        totals['source_frames'] += len(dataset)
        totals['valid_speed_frames'] += len(eligible)
        totals.update(dataset.reasons)
        if max_frames_per_episode is not None:
            eligible = eligible[:max_frames_per_episode]
        per_episode[episode['identity']] = dict(frames=len(dataset), valid_speed_frames=
            sum(s is not None for s in dataset.speed_rows), export_frames=len(eligible),
            exclusions=dict(dataset.reasons))
        if audit_only:
            continue
        pending, records = [], {}
        for i in eligible:
            row = episode['rows'][i]
            identity = dict(episode_id=episode['identity'], frame_index=row['frame_id'],
                            sensor_source_ns=row['sensor_time_ns'],
                            velocity_source='ego_topic', **dataset.speed_rows[i])
            key = hashlib.sha256(f"{episode['identity']}|{row['frame_id']}|{row['sensor_time_ns']}".encode()).hexdigest()
            cached = valid_record(output / 'records' / (key + '.json'), output)
            if cached is not None:
                if any(cached.get(k) != v for k, v in identity.items()):
                    raise ValueError('Cached Ego speed/frame identity differs')
                exported.append(cached)
                totals['reused'] += 1
            else:
                pending.append(i)
                records[i] = (key, identity)
        loader = DataLoader(Subset(dataset, pending), batch_size=options['batch_size'],
                            shuffle=False, num_workers=options['workers'],
                            pin_memory=options['device'].startswith('cuda'))
        cursor = 0
        for inputs in loader:
            count = inputs['rgb_front'].shape[0]
            indices = pending[cursor:cursor+count]
            cursor += count
            inputs = {k: v if k == 'num_points' else v.to(options['device'], non_blocking=True)
                      for k, v in inputs.items()}
            with torch.inference_mode(), deterministic_pointpillar_reductions():
                features = model(inputs).float().cpu().numpy()
            if features.shape != (count, 106, 256) or not np.isfinite(features).all():
                raise ValueError('Invalid VE output')
            features = features.astype(np.float16)
            if not np.isfinite(features).all():
                raise ValueError('FP16 cache overflow')
            for i, feature in zip(indices, features):
                key, record = records[i]
                record['visual_feature'] = tensor_ref(output, 'visual_feature/' + key + '.npy', feature)
                (output / 'records').mkdir(exist_ok=True)
                write_json(output / 'records' / (key + '.json'), record)
                exported.append(record)
                totals['inferred'] += 1
            if totals['inferred'] % options['log_interval'] < count:
                print(json.dumps(dict(totals)), flush=True)
    summary = dict(episodes=len(episodes), counts=dict(totals), per_episode=per_episode)
    if not audit_only:
        if not exported:
            raise ValueError('No VE tokens exported')
        exported.sort(key=lambda r: (r['episode_id'], r['sensor_source_ns']))
        pending = output / 'manifest.pending'
        pending.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n'
                                   for r in exported), encoding='utf-8')
        pending.replace(output / 'manifest.jsonl')
        meta = dict(schema=CACHE_SCHEMA, feature_kind='vision_encoder', zone_schema=ZONE_SCHEMA,
                    checkpoint_sha256=cfg['checkpoint_sha256'], encoder_contract=recipe,
                    manifest_sha256=sha256(output / 'manifest.jsonl'),
                    source_sha256=sha256(output / 'source.json'),
                    labels_sha256={e['identity']: e['label_sha256'] for e in episodes},
                    strict_load=strict, frames=len(exported), diagnostic_frame_limit=max_frames_per_episode,
                    audit=summary)
        write_json(output / 'cache_meta.json', meta)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_episode'}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path, help='Existing GPS/Stage 2 export config')
    mode.add_argument('--situation-config', type=Path, help='Ego/Situation VE-only export config')
    parser.add_argument('--audit-only', action='store_true', help='verify all metadata/scalars; do not run the VE')
    args = parser.parse_args()
    if args.situation_config is not None:
        sys.path.insert(0, str(ROOT.parent))
        from VLDrive.data.src.loaders.situation_dataset import load_config
        cache_situation(load_config(args.situation_config), audit_only=args.audit_only)
        return
    from utils.config.config_files import load_config
    cfg = load_config(args.config, 'vision')
    fields = {'schema','datasets','instructions','checkpoint','checkpoint_sha256','contract_sidecar',
        'output','device','cpu_threads','batch_size','workers','train_fraction','max_sensor_gap_ms','log_interval'}
    if set(cfg) != fields or cfg['schema'] != 'native-velocity-token-cache-config-v1':
        raise ValueError('Invalid native velocity cache config')
    if (cfg['train_fraction'] != 1.0 or cfg['batch_size'] not in (1,2,4)
            or cfg['workers'] not in (0,1,2) or not 1 <= cfg['cpu_threads'] <= 8):
        raise ValueError('Expected full-route cache and bounded batch/workers/threads')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data.src.loaders.navigation_velocity_sensors import VelocityNativeNavigationSensors
    from navigation.sampling import NavigationLabels
    from models.ve.vision_encoder import create_vision_encoder
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(42)
    checkpoint = (ROOT/cfg['checkpoint']).resolve()
    if sha256(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    discovered = discover(cfg)
    sidecar = json.loads(json.dumps(verified_recipe(blob, cfg['checkpoint_sha256'], discovered)))
    sidecar_path = ROOT/cfg['contract_sidecar']
    if sidecar_path.exists() and json.loads(sidecar_path.read_text()) != sidecar:
        raise ValueError('Existing encoder sidecar differs')
    write_json(sidecar_path, sidecar)
    recipe = sidecar['contract']
    labels = (NavigationLabels(ROOT/cfg['instructions'])
              if (ROOT/cfg['instructions']/'manifest.json').is_file() or not args.audit_only else None)
    datasets=[]; counts=Counter(); per_episode={}; splits={}; entries=[]
    for episode, namespace in discovered:
        print('Auditing native source:',episode,flush=True)
        dataset=VelocityNativeNavigationSensors(episode,recipe,namespace,cfg['max_sensor_gap_ms'])
        if labels is not None and dataset.identity not in labels.routes:
            raise ValueError('Missing navigation episode: '+dataset.identity)
        dataset_index=len(datasets); datasets.append(dataset)
        for i in range(len(dataset)):
            counts['source_frames'] += 1
            if dataset.velocities[i] is None:
                counts['excluded_'+dataset.velocity_reasons[i]] += 1
            else:
                entries.append((dataset_index,i))
        per_episode[dataset.identity]=dict(frames=len(dataset),velocity=dataset.velocity_summary)
        splits[dataset.identity]=dict(policy='all_valid_frames_train',validation_start_ns=None)
    identity = dict(schema='native-velocity-visual-cache-source-v1',checkpoint_sha256=cfg['checkpoint_sha256'],
        encoder_contract=recipe,contract_sidecar_sha256=sha256(sidecar_path),
        source={d.identity:d.provenance for d in datasets}, velocity=per_episode,
        sources_sha256={name:sha256(ROOT/name) for name in (
            'data/src/loaders/navigation_sensor_record.py','data/src/loaders/navigation_velocity_sensors.py',
            'data/src/loaders/morai_gps_speed.py','data/src/loaders/morai_transfer_adapter.py',
            'models/ve/vision_encoder.py','models/ve/pointpillars.py','models/ve/resnet.py','models/ve/transformer.py')},
        inference_precision='float32',storage_precision='float16')
    output=(ROOT/cfg['output']).resolve();output.mkdir(parents=True,exist_ok=True)
    identity_path=output/'visual_source.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing cache has different source/scalars/implementation')
    write_json(identity_path,identity)
    speeds=[d.velocities[i] for d in datasets for i in range(len(d)) if d.velocities[i] is not None]
    audit=dict(episodes=len(datasets),counts=counts,eligible_visual_frames=len(entries),
        velocity_mps=dict(min=min(speeds),max=max(speeds),mean=float(np.mean(speeds)),
                          nonzero=sum(v>0.05 for v in speeds)),per_episode=per_episode)
    write_json(output/'input_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='per_episode'}),flush=True)
    if args.audit_only:
        return
    records={}; pending=[]
    for entry in entries:
        di,i=entry;d=datasets[di];stamp=d.times[i]
        key=hashlib.sha256(f'{d.identity}|{stamp}'.encode()).hexdigest()[:24]
        record=valid_record(output/'records'/(key+'.json'),output)
        if record is not None:
            if (record['episode_id']!=d.identity or record['sensor_source_ns']!=stamp
                    or record.get('velocity_mps') != d.velocities[i]):
                raise ValueError('Cached frame identity/velocity changed')
            records[entry]=record;counts['reused']+=1
        else:
            pending.append(entry)
    if pending:
        model=create_vision_encoder({'name':'memfuser_baseline_e1d3_return_feature'},pretrained_backbone=False)
        strict=str(model.load_state_dict(blob['state_dict'],strict=True))
        del blob
        model.requires_grad_(False).to(cfg['device']).eval()
        loader=DataLoader(PreparedFrames(datasets,pending),batch_size=cfg['batch_size'],
            shuffle=False,num_workers=cfg['workers'],pin_memory=cfg['device']=='cuda',
            **(dict(prefetch_factor=2) if cfg['workers'] else {}))
        start=time.monotonic()
        for batch_indices,inputs in loader:
            inputs={k:v if k=='num_points' else v.to(cfg['device'],non_blocking=True) for k,v in inputs.items()}
            with torch.inference_mode(),deterministic_pointpillar_reductions():
                visual=model(inputs).float().cpu().numpy()
            if visual.shape!=(len(batch_indices),106,256) or not np.isfinite(visual).all():
                raise ValueError('Invalid VE feature batch')
            visual=visual.astype(np.float16)
            if not np.isfinite(visual).all(): raise ValueError('FP16 storage overflow')
            for batch_index,feature in zip(batch_indices.tolist(),visual):
                entry=pending[batch_index];di,i=entry;d=datasets[di];stamp=d.times[i]
                key=hashlib.sha256(f'{d.identity}|{stamp}'.encode()).hexdigest()[:24]
                record=dict(episode_id=d.identity,sensor_source_ns=stamp,frame_index=i,
                    velocity_mps=d.velocities[i],velocity_source='native_gps_consecutive_fixes',
                    gps_asset=d.gps_refs[i],previous_gps_asset=d.gps_refs[i-1],
                    visual_feature=tensor_ref(output,'visual_feature/'+key+'.npy',feature))
                path=output/'records'/(key+'.json');path.parent.mkdir(exist_ok=True)
                write_json(path,record);records[entry]=record;counts['inferred']+=1
            if counts['inferred']%cfg['log_interval']<len(batch_indices):
                elapsed=time.monotonic()-start
                print(json.dumps(dict(inferred=counts['inferred'],reused=counts['reused'],
                    total=len(entries),frames_per_s=counts['inferred']/elapsed,elapsed_s=elapsed)),flush=True)
    else:
        strict='All keys strictly loaded on original export; verified source and completed records reused'
    features=[];training=[]
    for entry in entries:
        di,i=entry;d=datasets[di];record=records[entry];features.append(record)
        scene=labels.frames.get((d.identity,d.times[i]));target=d.target(i)
        if scene is None or not scene['valid']:
            counts['excluded_navigation']+=1;continue
        if target is None:
            counts['excluded_future_or_heading']+=1;continue
        if not np.allclose([scene['frame']['x'],scene['frame']['y']],d.positions[i],atol=1e-5):
            raise ValueError('Navigation and GT positions differ')
        key=Path(record['visual_feature']['path']).stem
        ref=tensor_ref(output,'future_trajectory/'+key+'.npy',target)
        ref.update(source='future_synchronized_ego_positions',horizons_s=[.5,1.,1.5,2.,2.5])
        training.append(dict(record,split='train',future_trajectory=ref));counts['train']+=1
    if not training: raise ValueError('No valid training frames')
    for name,rows in [('features.jsonl',features),('manifest.jsonl',training)]:
        path=output/name;temp=path.with_suffix('.pending')
        temp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8');temp.replace(path)
    write_json(output/'split.json',splits)
    meta=dict(schema='navigation-token-cache-v1',checkpoint_sha256=cfg['checkpoint_sha256'],
        trajectory_coordinate_convention='lmdrive_left_negative_forward',
        manifest_sha256=sha256(output/'manifest.jsonl'),features_manifest_sha256=sha256(output/'features.jsonl'),
        labels_manifest_sha256=sha256(labels.root/'manifest.json'),config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        visual_source_sha256=sha256(identity_path),strict_load=strict,encoder_contract=recipe,
        contract_sidecar_sha256=sha256(sidecar_path),visual_frames=len(features),training_frames=len(training),
        counts=counts,split=splits,source_datasets=cfg['datasets'],velocity=audit['velocity_mps'],
        target_heading_source='latest causal ego yaw <=50ms; synchronized future GT positions',
        publication='derived Stage 2 export; raw publication flags/seals untouched; no validation holdout')
    write_json(output/'cache_meta.json',meta)
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
