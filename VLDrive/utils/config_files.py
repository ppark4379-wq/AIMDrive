"""Select a stage from the single training configuration; paths use project root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_document(path):
    filename, _, selector = str(path).partition('#')
    value = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    for key in selector.replace('\\', '/').split('/') if selector else ():
        value = value[key]
    return value


def document_bytes(path):
    if '#' not in str(path):
        return Path(path).read_bytes()
    # Preserve the original split files' byte identity for existing cache hashes.
    if '#split_manifests/' in str(path).replace('\\', '/'):
        return (json.dumps(read_document(path), indent=2) + '\n').replace('\n', '\r\n').encode()
    return json.dumps(read_document(path), sort_keys=True).encode()


def load_config(path, role='training'):
    value = read_document(path)
    value.pop('split_manifests', None)
    if 'preprocessing' not in value:
        return value  # Existing standalone configs and saved run snapshots.
    preparation = value.pop('preprocessing')
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
