"""Relocation preserves old source identities without accepting later edits."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from utils.common.source_compatibility import equivalent_provenance, source_matches, source_path
from utils.training.training_helper import source_hashes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'utils/common/source_relocations.json').read_text())


@pytest.mark.parametrize('name', [
    'utils/trainer.py', 'utils/training.py', 'utils/runtime.py',
    'training/training.py', 'training/vision_encoder_training.py',
    'training/morai_transfer_training.py',
    'tools/build_instructions.py', 'tools/cache_sensor_record_tokens.py',
])
def test_old_paths_accept_only_exact_relocated_source(tmp_path, name):
    entry = MANIFEST['files'][name]
    target = tmp_path / entry['path']
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / entry['path']).read_bytes())
    assert source_path(tmp_path, name) == target
    assert all(source_matches(tmp_path, name, digest) for digest in entry['before'])
    assert source_matches(tmp_path, name.replace('/', '\\'), entry['before'][0])
    assert not source_matches(tmp_path, name, '0' * 64)
    target.write_bytes(target.read_bytes() + b'\n# unrelated edit\n')
    assert not source_matches(tmp_path, name, entry['before'][0])


def test_resume_normalizes_paths_but_keeps_dataset_identity_strict():
    name = 'utils/trainer.py'
    entry = MANIFEST['files'][name]
    before = {'source': {name: entry['before'][0]}, 'datasets': {'train': 'original'}}
    after = {'source': {entry['path']: hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest()},
             'datasets': {'train': 'original'}}
    assert equivalent_provenance(before, after, ROOT)
    changed = copy.deepcopy(after)
    changed['datasets']['train'] = 'different'
    assert not equivalent_provenance(before, changed, ROOT)
    changed = copy.deepcopy(after)
    changed['source'][entry['path']] = '0' * 64
    assert not equivalent_provenance(before, changed, ROOT)


def test_training_manifest_includes_nested_sources_without_ui():
    names = {name.replace('\\', '/') for name in source_hashes(ROOT)}
    assert {'training/joint/training.py', 'utils/training/training_helper.py',
            'utils/common/runtime.py', 'utils/evaluation/metrics.py',
            'utils/config/vision_encoder_config.py',
            'utils/visualization/training_dashboard/telemetry.py'} <= names
    assert 'utils/visualization/training_dashboard/app.py' not in names


def test_config_and_runtime_roots_still_point_to_repository():
    from utils.common.runtime import PROJECT_ROOT
    from utils.config.vision_encoder_config import ROOT as config_root
    assert PROJECT_ROOT == config_root == ROOT
