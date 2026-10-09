"""Append new episodes without reopening or copying published VE tensor files."""
import copy

from data.src.processing import joint_preprocessing as joint


def cache_additions(cfg, pending, previous, recipe):
    from data.src.processing.cache_velocity_tokens import cache_situation, compatible_export_sources

    old = joint.read(previous / 'preparation.json')
    new = joint.read(pending / 'preparation.json')
    if old['recipe'] != recipe or new['recipe'] != recipe or old['policy'] != new['policy']:
        raise ValueError('Append requires unchanged encoder and preprocessing settings')
    before = {e['episode_id']: e for e in old['episodes']}
    after = {e['episode_id']: e for e in new['episodes']}
    if any(after.get(key) != value for key, value in before.items()):
        raise ValueError('Append cannot modify existing prepared episodes')
    added = set(after) - set(before)
    if not added:
        raise ValueError('No new episodes to append')
    bundle = pending / 'instructions'
    if joint.digest(bundle / 'manifest.json') != new['navigation_sha256']:
        raise ValueError('Prepared navigation changed')
    if joint.digest(bundle / 'instructions.jsonl') != joint.read(bundle / 'manifest.json')['instruction_sha256']:
        raise ValueError('Prepared instructions changed')
    path_file = pending / 'navigation_tokens/path_features.json'
    if joint.digest(path_file) != new['path_features_sha256']:
        raise ValueError('Prepared path features changed')
    old_paths = joint.read(previous / 'navigation_tokens/path_features.json')
    new_paths = joint.read(path_file)
    for key in old_paths:
        if key not in ('episodes', 'labels_manifest_sha256') and old_paths[key] != new_paths.get(key):
            raise ValueError('Append cannot change path encoder settings')
    new_path_episodes = {e['episode_id']: e for e in new_paths['episodes']}
    for value in old_paths['episodes']:
        if new_path_episodes.get(value['episode_id']) != value:
            raise ValueError('Append cannot change existing path features')

    selected = copy.deepcopy(cfg)
    datasets = []
    for spec in cfg['situation_instruction']['datasets']:
        episodes = [e for e in spec['include_episodes'] if spec['namespace'] + '/' + e in added]
        if episodes:
            datasets.append(dict(spec, include_episodes=episodes))
    identities = {s['namespace'] + '/' + e for s in datasets for e in s['include_episodes']}
    if identities != added:
        raise ValueError('New episode membership does not match the prepared data')
    base_sha = joint.digest(previous / 'manifest.json')
    addition = pending / 'vision_tokens' / ('appended_' + base_sha[:12])
    if (previous / 'vision_tokens' / addition.name).exists():
        raise ValueError('Append destination already exists')
    selected['situation_instruction']['datasets'] = datasets
    selected['zone_instruction']['datasets'] = copy.deepcopy(datasets)
    selected['feature_cache'] = joint.relative(addition)
    storage_dirs = []
    for path in (previous / 'vision_tokens').iterdir():
        if path.is_symlink():
            raise ValueError('Published cache must not use symlink storage')
        if path.is_dir():
            storage_dirs.append(path.name)
    joint.write(pending / 'append.json', dict(base_manifest_sha256=base_sha,
        storage_dirs=sorted(storage_dirs), episodes=sorted(added), cache=addition.name))
    print(f'[Joint append] Preserving {len(before)} episodes; caching only {len(added)} new episodes', flush=True)
    cache_addition_config = joint.vision_config(selected)
    cache_situation(cache_addition_config)
    old_source = joint.read(previous / 'vision_tokens/source.json')
    new_source = joint.read(addition / 'source.json')
    for key in set(old_source) | set(new_source):
        if key == 'episodes':
            continue
        if key == 'implementations' and compatible_export_sources(old_source[key], new_source[key]):
            continue
        if old_source.get(key) != new_source.get(key):
            raise ValueError('Cannot append a different VE recipe: ' + key)
    if set(new_source['episodes']) != added:
        raise ValueError('Exported episodes differ from requested additions')
    for key, evidence in new_source['episodes'].items():
        if evidence.get('labels_sha256') != after[key]['labels_sha256']:
            raise ValueError('Raw labels changed after preprocessing')
    combined = dict(new_source, episodes={**old_source['episodes'], **new_source['episodes']})
    joint.write(pending / 'vision_tokens/source.json', combined)
    print('[Joint append] New VE cache complete; joining manifests and new Ego targets', flush=True)
    return addition


def storage_names(backup, pending):
    journal = pending / 'append.json'
    if not journal.exists():
        return []
    info = joint.read(journal)
    if info['base_manifest_sha256'] != joint.digest(backup / 'manifest.json'):
        raise ValueError('Append storage journal belongs to a different base')
    names = info['storage_dirs']
    for name in names:
        if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('Unsafe cache storage directory')
        for parent in (backup / 'vision_tokens', pending / 'vision_tokens', backup / 'discarded_partial_cache'):
            path = parent / name
            if path.is_symlink() or path.resolve().parent != parent.resolve():
                raise ValueError('Unsafe cache storage path')
    return names


def transfer_storage(backup, pending):
    """Rename whole tensor directories on the same volume; no per-file IO."""
    for name in storage_names(backup, pending):
        source, target = backup / 'vision_tokens' / name, pending / 'vision_tokens' / name
        if target.exists():
            # Discard files copied by a previously interrupted, older extension.
            discarded = backup / 'discarded_partial_cache' / name
            discarded.parent.mkdir(exist_ok=True)
            target.rename(discarded)
        source.rename(target)


def restore_storage(backup, pending):
    """Recover exceptions or process interruption during directory publication."""
    for name in reversed(storage_names(backup, pending)):
        original, staged = backup / 'vision_tokens' / name, pending / 'vision_tokens' / name
        if not original.exists():
            if not staged.is_dir():
                raise ValueError('Missing original cache storage during recovery')
            staged.rename(original)
        discarded = backup / 'discarded_partial_cache' / name
        if discarded.exists():
            discarded.rename(staged)
