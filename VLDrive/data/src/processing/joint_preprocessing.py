"""Versioned GP/ego preparation and resumable Ego-conditioned VE export."""
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'vldrive-joint-processing-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    pending.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\', '/')


def frozen_config(source, name):
    """Snapshot episode membership so additions cannot change a running version."""
    cfg = copy.deepcopy(source)
    cfg.pop('data_version', None)
    base = 'data/processed/' + name
    cfg.update(instructions=base + '/instructions', feature_cache=base + '/vision_tokens',
               split_file=base + '/vision_tokens/split.json')
    cfg['navigation_sampling'].pop('token_cache', None)
    cfg['path_instruction']['feature_cache'] = base + '/navigation_tokens/path_features.json'
    specs = []
    for source_spec in cfg['preprocessing']['datasets']:
        root = (ROOT / source_spec['root']).resolve()
        paths = sorted(root.glob('**/situation_labels/labeled_frames.jsonl'))
        if not paths:
            raise ValueError('No labeled episodes: ' + str(root))
        specs.append(dict(root=source_spec['root'], namespace=source_spec['namespace'],
                          include_episodes=[p.parent.parent.relative_to(root).as_posix() for p in paths]))
    if len({s['namespace'] for s in specs}) != len(specs):
        raise ValueError('Dataset namespaces must be unique')
    cfg['situation_instruction']['datasets'] = copy.deepcopy(specs)
    cfg['zone_instruction']['datasets'] = copy.deepcopy(specs)
    return cfg


def encoder_recipe(cfg):
    import torch
    checkpoint = ROOT / cfg['model']['visual_checkpoint']
    if digest(checkpoint) != cfg['model']['visual_sha256']:
        raise ValueError('Selected VE checksum differs from the actual checkpoint')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if blob.get('pretraining', {}).get('velocity_source') != 'ego_topic':
        raise ValueError('Joint labeled-data processing requires the Ego-conditioned VE')
    from data.src.loaders.morai_encoder_contract import validate_encoder_contract
    return json.loads(json.dumps(validate_encoder_contract(blob['morai_encoder_contract']), allow_nan=False))


def episode_evidence(episode):
    folder = episode['path']
    names = ('episode.json', 'runtime/expert/path.csv', 'quality/active_signals.json',
             'quality/QUALITY_COMPLETE.json', 'gt_timeline_v1/frames.json',
             'gt_timeline_v1/COMPLETE.json', 'situation_labels/labeled_frames.jsonl')
    evidence = {name: digest(folder / name) for name in names}
    for name, seal_name in (('quality/active_signals.json', 'quality/QUALITY_COMPLETE.json'),
                            ('gt_timeline_v1/frames.json', 'gt_timeline_v1/COMPLETE.json')):
        if read(folder / seal_name)['files'].get(Path(name).name) != evidence[name]:
            raise ValueError('Source is not covered by its recorded seal: ' + str(folder / name))
    if evidence['situation_labels/labeled_frames.jsonl'] != episode['label_sha256']:
        raise ValueError('Labels changed during preparation')
    if read(folder / 'episode.json')['expert_inputs'].get('runtime/expert/path.csv') != evidence['runtime/expert/path.csv']:
        raise ValueError('Recorded planned path changed: ' + str(folder))
    return evidence


def prepare_episode(episode, recipe, policy):
    """Causal inputs and future-position targets are kept in separate files."""
    import numpy as np
    from navigation.adapters import CausalEgoSamples, json_array_items
    from data.src.loaders.navigation_sensor_record import future_target
    from instruction.global_path.preprocess.load import load_path
    folder = episode['path']
    evidence = episode_evidence(episode)
    with (folder / 'runtime/expert/path.csv').open(encoding='utf-8-sig', newline='') as stream:
        points = [[float(row['x']), float(row['y'])] for row in csv.DictReader(stream)]
    route = load_path(dict(points=points, closed=bool(len(points) > 1 and math.dist(points[0], points[-1]) <= .01)))
    # Geometric turns are selected by PathFeatureCache. Do not invent junction,
    # exit-number or traffic semantics from this XY-only GP.
    route['events'] = []
    ego = CausalEgoSamples(json_array_items(folder / 'quality/active_signals.json'))
    geometry = None
    basis = recipe.get('map_basis')
    if basis:
        from data.src.loaders.morai_map_geometry import validate_map_basis
        geometry = validate_map_basis(basis)
    gt = read(folder / 'gt_timeline_v1/frames.json')
    rows = episode['rows']
    if len(gt) != len(rows):
        raise ValueError('Label and future-Ego timelines have different lengths')
    times, positions, frames, yaws = [], [], [], []
    for label_row, target_row in zip(rows, gt):
        stamp = label_row['sensor_time_ns']
        if target_row['index'] != label_row['frame_id'] or target_row['lidar_source_ns'] != stamp:
            raise ValueError('Label/trajectory source-frame mismatch')
        position = target_row.get('ego_xy')
        if position is not None and (len(position) != 2 or not np.isfinite(position).all()):
            raise ValueError('Nonfinite future Ego position')
        pose = ego.at(stamp)
        if pose['valid'] and geometry is not None:
            if basis['source_orientation_frame'] == 'true_enu':
                pose['yaw_rad'] += math.radians(geometry.convergence_degrees([pose['x'], pose['y'], 0.]))
        yaw = pose.get('yaw_rad') if pose['valid'] else None
        frames.append(dict(index=label_row['frame_id'], sensor_source_ns=stamp,
                           x=None, y=None, yaw_rad=None, **{k: v for k, v in pose.items()
                           if k not in ('x', 'y', 'yaw_rad')}))
        if pose['valid']:
            frames[-1].update(x=pose['x'], y=pose['y'], yaw_rad=yaw)
        times.append(stamp); positions.append(position); yaws.append(yaw)
    targets = []
    for i, row in enumerate(rows):
        target = future_target(times, positions, i, yaws[i], policy['max_sensor_gap_ms'] * 1e6)
        if target is not None:
            targets.append(dict(frame_index=row['frame_id'], sensor_source_ns=times[i],
                                future_trajectory=target.tolist()))
    document = dict(schema='navigation-episode-v1', episode_id=episode['identity'],
        route=route, frames=frames, sources=evidence, source_path=relative(folder),
        pose_policy=dict(method='latest_past_ego_xy_yaw', max_age_ms=50.,
                         future_samples_used=False, map_basis=basis))
    return document, targets, evidence


def prepare(cfg, folder, recipe, previous=None):
    from data.src.loaders.situation_dataset import labeled_episodes
    from VLDrive.data.src.loaders.situation_instructions import SituationInstructionIndex
    from navigation.build import build_episode
    from navigation.catalog import CATALOG_SHA256, CORRECTIONS
    from data.src.processing.build_path_features import build_path_features
    bundle = folder / 'instructions'
    policy = cfg['preprocessing']['navigation']['policy']
    records, entries, prepared = [], [], []
    # Fail on incompatible recorded GT before spending hours exporting VE features.
    label_index = SituationInstructionIndex(cfg['situation_instruction']['datasets'], include_zone=True)
    label_summary = dict(situation=dict(label_index.counts), zone=dict(label_index.zone_counts))
    del label_index
    episodes = labeled_episodes(cfg['situation_instruction']['datasets'])
    old = read(previous / 'preparation.json') if previous else None
    reusable = bool(old and old.get('recipe') == recipe and old.get('policy') == policy)
    old_entries = {e['episode_id']: e for e in old['episodes']} if reusable else {}
    for index, episode in enumerate(episodes, 1):
        identity = episode['identity']
        stem = hashlib.sha256(identity.encode()).hexdigest()[:20]
        checkpoint = bundle / 'prepared' / (stem + '.json')
        evidence = episode_evidence(episode)
        cached = read(checkpoint) if checkpoint.exists() else None
        if cached and (cached['evidence'] != evidence or cached['recipe'] != recipe or cached['policy'] != policy):
            raise ValueError('Prepared episode inputs changed; create a new version: ' + identity)
        if cached is None and identity in old_entries:
            path = previous / 'instructions/prepared' / (stem + '.json')
            if digest(path) != old_entries[identity]['prepared_sha256']:
                raise ValueError('Previous prepared episode changed')
            candidate = read(path)
            if candidate['evidence'] == evidence:
                cached = candidate
        if cached is None:
            document, targets, evidence = prepare_episode(episode, recipe, policy)
            instance_rows, trace, excluded = build_episode(document, policy)
            cached = dict(document=document, targets=targets, evidence=evidence,
                          recipe=recipe, policy=policy, instructions=instance_rows, trace=trace,
                          excluded=excluded)
        write(checkpoint, cached)
        episode_file, trace_file, target_file = ('episodes/' + stem + '.json',
                                               'traces/' + stem + '.jsonl', 'targets/' + stem + '.jsonl')
        write(bundle / episode_file, cached['document'])
        write_lines(bundle / trace_file, cached['trace'])
        write_lines(bundle / target_file, cached['targets'])
        records.extend(cached['instructions'])
        entries.append(dict(episode_id=identity, episode=episode_file, trace=trace_file,
                            episode_sha256=digest(bundle / episode_file), trace_sha256=digest(bundle / trace_file)))
        prepared.append(dict(episode_id=identity, targets=target_file, targets_sha256=digest(bundle / target_file),
                             prepared_sha256=digest(checkpoint), labels_sha256=episode['label_sha256'],
                             frames=len(episode['rows']), target_frames=len(cached['targets'])))
        print(f'[Joint prepare] {index}/{len(episodes)} {identity}: {len(cached["targets"])} targets', flush=True)
    write_lines(bundle / 'instructions.jsonl', records)
    write(bundle / 'manifest.json', dict(schema='lmdrive-navigation-labels-v1', episodes=entries,
          policy=policy, catalog_sha256=CATALOG_SHA256, corrections=CORRECTIONS,
          instruction_sha256=digest(bundle / 'instructions.jsonl')))
    target = ROOT / cfg['path_instruction']['feature_cache']
    path_options = cfg['preprocessing'].get('path', {})
    # build_path_features deliberately uses exclusive creation. A resumed,
    # unfinished version may have a completed deterministic cache already.
    if target.exists():
        from instruction.global_path.encoder import PathFeatureCache
        PathFeatureCache(target, digest(bundle / 'manifest.json'))
    else:
        build_path_features(bundle, target, geometry_options=path_options.get('geometry'),
            encoder_options=path_options.get('encoder'), instruction_options=path_options.get('instruction'),
            max_sensor_gap_ms=policy['max_sensor_gap_ms'])
    result = dict(schema=SCHEMA, recipe=recipe, policy=policy, episodes=prepared, labels=label_summary,
                  path_features_sha256=digest(target), navigation_sha256=digest(bundle / 'manifest.json'))
    write(folder / 'preparation.json', result)
    return result


def vision_config(cfg):
    options = cfg['preprocessing']['vision']
    return dict(schema='situation-training-v1', datasets=cfg['situation_instruction']['datasets'],
        checkpoint=cfg['model']['visual_checkpoint'], checkpoint_sha256=cfg['model']['visual_sha256'],
        cache_output=cfg['feature_cache'], sequence_length=cfg['sequence_length'],
        max_frame_gap_ms=options['max_sensor_gap_ms'],
        cache={key: options[key] for key in ('device', 'batch_size', 'workers', 'cpu_threads', 'log_interval')})


def finish(cfg, folder):
    """Join VE features to actual future Ego targets without crossing gaps."""
    import numpy as np
    from data.src.processing.cache_sensor_record_tokens import tensor_ref
    from instruction.global_path.encoder import PathFeatureCache
    from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
    vision = ROOT / cfg['feature_cache']
    prep = read(folder / 'preparation.json')
    bundle = folder / 'instructions'
    nav_sha = digest(bundle / 'manifest.json')
    if prep['navigation_sha256'] != nav_sha:
        raise ValueError('Prepared Navigation bundle changed')
    path_file = ROOT / cfg['path_instruction']['feature_cache']
    if digest(path_file) != prep['path_features_sha256']:
        raise ValueError('Prepared Path cache changed')
    paths = PathFeatureCache(path_file, nav_sha)
    targets = {}
    for episode in prep['episodes']:
        path = bundle / episode['targets']
        if digest(path) != episode['targets_sha256']:
            raise ValueError('Prepared future target changed')
        for line in path.read_text().splitlines():
            row = json.loads(line)
            targets[(episode['episode_id'], row['sensor_source_ns'])] = row
    features = [read(p) for p in sorted((vision / 'records').glob('*.json'))]
    features.sort(key=lambda r: (r['episode_id'], r['sensor_source_ns']))
    training = []
    for row in features:
        key = (row['episode_id'], row['sensor_source_ns'])
        target = targets.get(key)
        if target is None or not paths.eligible(key[0], [key[1]]):
            continue
        if target['frame_index'] != row['frame_index']:
            raise ValueError('VE/target frame indices differ')
        stem = hashlib.sha256(f'{key[0]}|{row["frame_index"]}|{key[1]}'.encode()).hexdigest()
        ref = tensor_ref(vision, 'future_trajectory/' + stem + '.npy', np.asarray(target['future_trajectory'], dtype=np.float32))
        ref.update(source='future_synchronized_ego_positions', horizons_s=[.5, 1., 1.5, 2., 2.5])
        training.append(dict(row, split='train', future_trajectory=ref))
    if not training:
        raise ValueError('No jointly valid VE/Path/future-Ego frames')
    write_lines(vision / 'features.jsonl', features)
    write_lines(vision / 'manifest.jsonl', training)
    source = read(vision / 'source.json')
    write(vision / 'split.json', dict(schema='joint-full-route-split-v1', policy='all_train',
                                    episodes=sorted(source['episodes'])))
    write(vision / 'cache_meta.json', dict(schema='navigation-token-cache-v1',
        checkpoint_sha256=cfg['model']['visual_sha256'], encoder_contract=prep['recipe'],
        trajectory_coordinate_convention='lmdrive_left_negative_forward', labels_manifest_sha256=nav_sha,
        manifest_sha256=digest(vision / 'manifest.jsonl'), features_manifest_sha256=digest(vision / 'features.jsonl'),
        visual_source_sha256=digest(vision / 'source.json'), visual_frames=len(features), training_frames=len(training)))
    dataset = NavigationCacheDataset(vision, bundle, 'train', sequence_length=cfg['sequence_length'],
        frame_sampling_interval=cfg['frame_sampling_interval'], include_validation_in_train=True,
        path_feature_cache=path_file, situation_datasets=cfg['situation_instruction']['datasets'],
        zone_datasets=cfg['zone_instruction']['datasets'])
    sample = dataset[0]
    summary = dict(visual_frames=len(features), training_frames=len(training), windows=len(dataset),
        sample_visual_shape=list(sample['model_inputs']['visual_feature'].shape),
        sample_target_shape=list(sample['training_targets']['future_trajectory'].shape),
        situation_windows=dataset.situation_window_counts)
    write(folder / 'readiness.json', summary)
    files = [folder / 'processing_config.json', folder / 'preparation.json', folder / 'readiness.json',
             path_file, bundle / 'manifest.json', bundle / 'instructions.jsonl',
             vision / 'manifest.jsonl', vision / 'features.jsonl', vision / 'cache_meta.json',
             vision / 'source.json', vision / 'split.json']
    write(folder / 'manifest.json', dict(schema='vldrive-data-version-v1', name=folder.name,
        config=cfg, assets=dict(instructions=cfg['instructions'], vision_tokens=cfg['feature_cache'],
                               path_features=cfg['path_instruction']['feature_cache']),
        fingerprints={relative(p): digest(p) for p in files}, **summary))
    return summary


def process(config, name=None, *, prepare_only=False, resume=False, reuse_version=None, activate=True):
    from data.src.processing.manage_data import component
    from data.src.processing.cache_velocity_tokens import cache_situation
    config = Path(config).resolve()
    original, original_sha = read(config), digest(config)
    name = component(name or datetime.now().strftime('joint_%Y%m%d_%H%M%S_%f'))
    folder = ROOT / 'data/processed' / name
    if resume:
        state = read(folder / 'status.json')
        if state.get('schema') != SCHEMA or state['status'] == 'complete':
            raise ValueError('Resume requires an unfinished joint-processing version')
        cfg = read(folder / 'processing_config.json')
        if digest(folder / 'processing_config.json') != state['config_sha256']:
            raise ValueError('Processing snapshot changed')
        if reuse_version is not None and reuse_version != state.get('reuse_version'):
            raise ValueError('Resume must keep the original reuse version')
        reuse_version = state.get('reuse_version')
    else:
        if folder.exists():
            raise FileExistsError('Use --resume for this version, or choose a new --id')
        cfg = frozen_config(original, name)
        if cfg.get('validation_enabled') or not cfg.get('include_validation_in_train'):
            raise ValueError('This preparer follows the configured all-train policy; validation must be disabled')
        folder.mkdir(parents=True)
        for kind in ('instructions', 'navigation_tokens', 'vision_tokens', 'traffic_labels'):
            (folder / kind).mkdir()
        write(folder / 'processing_config.json', cfg)
        state = dict(schema=SCHEMA, status='preparing', reuse_version=reuse_version,
                     config_sha256=digest(folder / 'processing_config.json'))
        write(folder / 'status.json', state)
    previous = ROOT / 'data/processed' / component(reuse_version) if reuse_version else None
    if previous:
        if previous == folder or read(previous / 'status.json').get('status') != 'complete':
            raise ValueError('Reuse requires a different completed processing version')
        if read(previous / 'status.json').get('schema') != SCHEMA:
            raise ValueError('Reuse requires an Ego/joint processing version')
        for name, expected in read(previous / 'manifest.json')['fingerprints'].items():
            path = (ROOT / name).resolve()
            if ROOT.resolve() not in path.parents or digest(path) != expected:
                raise ValueError('Previous version changed: ' + name)
    try:
        state.pop('error', None)
        recipe = encoder_recipe(cfg)
        prepare(cfg, folder, recipe, previous)
        write(folder / 'traffic_labels/status.json', dict(status='not_requested',
              reason='Joint waypoint training uses recorded Situation/Zone, not traffic auxiliary labels.'))
        state.update(status='prepared')
        write(folder / 'status.json', state)
        if prepare_only:
            return folder / 'status.json'
        state.update(status='caching')
        write(folder / 'status.json', state)
        cache_situation(vision_config(cfg), reuse_cache=previous / 'vision_tokens' if previous else None)
        summary = finish(cfg, folder)
        state.update(status='complete', **summary)
        write(folder / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write(folder / 'status.json', state)
        raise
    if activate:
        if digest(config) != original_sha:
            raise RuntimeError('Processing finished, but training config changed; select its manifest manually')
        original['data_version'] = relative(folder / 'manifest.json')
        write(config, original)
    return folder / 'manifest.json'
