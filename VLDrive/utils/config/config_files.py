"""Select a stage from the single training configuration; paths use project root."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def config_path(path):
    """Resolve exact former config locations for saved run snapshots."""
    path = Path(path)
    if path.is_file():
        return path
    aliases = {
        'vldrive_train.json': ROOT / 'training/vldrive_train.json',
        'vldrive_realtime.json': ROOT.parent / 'AIM2026/Inference/vldrive_realtime.json',
        'vldrive_control.json': ROOT.parent / 'AIM2026/Control/vldrive_control.json',
    }
    if path.resolve().parent == ROOT / 'configs' and path.name in aliases:
        return aliases[path.name]
    return path


def read_document(path):
    filename, _, selector = str(path).partition('#')
    value = json.loads(config_path(filename).read_text(encoding='utf-8-sig'))
    for key in selector.replace('\\', '/').split('/') if selector else ():
        value = value[key]
    return value


def document_bytes(path):
    if '#' not in str(path):
        return config_path(path).read_bytes()
    # Preserve the original split files' byte identity for existing cache hashes.
    if '#split_manifests/' in str(path).replace('\\', '/'):
        return (json.dumps(read_document(path), indent=2) + '\n').replace('\n', '\r\n').encode()
    return json.dumps(read_document(path), sort_keys=True).encode()


def load_config(path, role='training'):
    value = read_document(path)
    value.pop('split_manifests', None)
    data_version = value.pop('data_version', None)
    if data_version:
        bundle = read_document(ROOT / data_version)
        if bundle.get('schema') != 'vldrive-data-version-v1':
            raise ValueError('Unsupported data version')
        if bundle['config']['model']['visual_sha256'] != value['model']['visual_sha256']:
            raise ValueError('Selected data version was made with a different VE')
        for relative, expected in bundle['fingerprints'].items():
            source_file = (ROOT / relative).resolve()
            if ROOT not in source_file.parents:
                raise ValueError('Data version fingerprint escapes project')
            digest = hashlib.sha256()
            with source_file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('Versioned data changed: ' + relative)
        # The version stores relative paths in its original training config;
        # do not import machine-specific absolute asset paths into WSL.
        original = bundle['config']
        for key in ('feature_cache', 'instructions', 'split_file'):
            value[key] = original[key]
        if value.get('model', {}).get('path_navigation') and original.get('path_instruction', {}).get('feature_cache'):
            value['path_instruction']['feature_cache'] = original['path_instruction']['feature_cache']
        # A completed version pins episode membership, including after more raw
        # episodes are downloaded. Processing must publish a new or extended bundle.
        for key in ('situation_instruction', 'zone_instruction'):
            if value.get('model', {}).get(key) and original.get(key, {}).get('datasets'):
                value[key]['datasets'] = original[key]['datasets']
        if not (value.get('model', {}).get('path_navigation', False) or
                value.get('model', {}).get('situation_instruction', False)):
            value['navigation_sampling']['token_cache'] = original['navigation_sampling']['token_cache']
    if 'preprocessing' not in value:
        return value  # Existing standalone configs and saved run snapshots.
    preparation = value.pop('preprocessing')
    for source in preparation['datasets']:
        registration = ROOT / source['root'] / 'registration.json'
        if registration.is_file():
            record = read_document(registration)
            report = Path(record['validation_report'])
            if (record.get('schema') != 'vldrive-raw-registration-v1'
                    or hashlib.sha256(report.read_bytes()).hexdigest() != record['validation_sha256']
                    or hashlib.sha256((report.parent/'COMPLETE.json').read_bytes()).hexdigest() != record['seal_sha256']):
                raise ValueError('Registered source validation changed')
            for episode in record['episodes']:
                metadata = Path(record['source']) / episode['path'] / 'episode.json'
                if hashlib.sha256(metadata.read_bytes()).hexdigest() != episode['episode_sha256']:
                    raise ValueError('Registered episode changed')
            source['root'] = record['source']
    encoder_training = value.pop('vision_encoder_training', None)
    if role == 'training':
        return value
    if role == 'vision_encoder_training':
        if encoder_training is None:
            raise ValueError('Missing vision_encoder_training section')
        return encoder_training
    if role == 'navigation':
        return dict(preparation['navigation'], datasets=preparation['datasets'],
                    output=value['instructions'])
    if role == 'vision':
        datasets = [{k: v for k, v in source.items()
                     if k in ('root', 'namespace', 'exclude_episodes')}
                    for source in preparation['datasets']]
        return dict(preparation['vision'], datasets=datasets,
                    instructions=value['instructions'], output=value['feature_cache'],
                    checkpoint=value['model']['visual_checkpoint'],
                    checkpoint_sha256=value['model']['visual_sha256'])
    raise ValueError('Unknown config role: ' + role)
