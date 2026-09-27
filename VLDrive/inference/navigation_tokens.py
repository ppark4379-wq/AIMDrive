"""Optional verified text-ID cache for live navigation; no model weight changes."""
import json

from inference.vldrive_runtime import local_path
from navigation.token_cache import CachedNavigationTokenizer
from utils.common.runtime import sha256


def attach_navigation_tokens(model, cfg, navigation):
    path = cfg.get('navigation_token_config')
    if not path:
        return None
    from utils.config.config_files import read_document, document_bytes
    config = read_document(local_path(path))
    labels = local_path(config['instructions'])
    manifest = json.loads((labels/'manifest.json').read_text())
    if navigation is not None:
        # Live sensor timeout intentionally differs from the recording interval.
        for key, value in manifest['policy'].items():
            if key != 'max_sensor_gap_ms' and navigation.policy.get(key) != value:
                raise ValueError('Live navigation and cached instruction policy differ: '+key)
    cache = CachedNavigationTokenizer(model.language_model.tokenizer, local_path(config['output']),
                                     sha256(labels/'manifest.json'),
                                     sha256(local_path(cfg['visual_cache'])/'manifest.jsonl'))
    model.language_model.tokenizer = cache
    return cache.describe()
