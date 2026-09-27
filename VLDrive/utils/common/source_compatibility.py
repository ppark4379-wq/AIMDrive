"""Exact source identities for the utility-package relocation.

Only the recorded before/after hashes are equivalent. Later edits invalidate
the equivalence; model weights, datasets and other provenance remain strict.
"""
import copy
import hashlib
import json
from pathlib import Path


def _manifest():
    return json.loads(Path(__file__).with_name('source_relocations.json').read_text(encoding='utf-8'))


def source_path(root, name):
    root = Path(root)
    name = name.replace('\\', '/')
    path = root / name
    if path.is_file():
        return path
    entry = _manifest()['files'].get(name)
    return root / entry['path'] if entry else path


def source_matches(root, name, expected):
    name = name.replace('\\', '/')
    path = source_path(root, name)
    if not path.is_file():
        return False
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        return True
    entry = _manifest()['files'].get(name)
    return bool(entry and actual in entry['after'] and expected in entry['before'])


def equivalent_provenance(left, right, root):
    """Normalize only approved source maps, never data/model/backend metadata."""
    manifest = _manifest()

    def normalize(value):
        result = copy.deepcopy(value)
        maps = [result.get('source'), result.get('backend', {}).get('sources')]
        for sources in maps:
            if not isinstance(sources, dict):
                continue
            normalized = {}
            for name, digest in sources.items():
                name = name.replace('\\', '/')
                entry = manifest['files'].get(name)
                if entry and digest in entry['before'] and source_matches(root, name, digest):
                    name = entry['path']
                    digest = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
                # New package markers and compatibility machinery were absent
                # from old manifests. Ignore only their exact migration version.
                if digest in manifest['added'].get(name, []):
                    if source_matches(root, name, digest):
                        continue
                normalized[name.replace('\\', '/')] = digest
            sources.clear()
            sources.update(normalized)
        return result

    return normalize(left) == normalize(right)
