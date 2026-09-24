"""Reviewed compatibility must not become a blanket source-check bypass."""
import hashlib

import pytest

from inference import vldrive_runtime as runtime


@pytest.mark.parametrize('name,expected', [
    ('datasets/morai_sensor_record_dataset.py',
     '9af9b8b1186a6b3158a880094faf0df8abb6cee27d8efd22d595d5e07b96caf5'),
    ('utils/morai_transfer_training.py',
     'cf0e8b89f74ef435dba10aeb318bb3efa48de432ad9b5b3837e72f39e55359e7'),
    ('utils/vision_encoder_visualization.py',
     '5c9e02e8828266961264c8f7073845abae75199ea276eecbc4463837d15de8fd'),
])
def test_only_reviewed_old_new_pair_is_accepted(tmp_path, monkeypatch, name, expected):
    current = (runtime.ROOT / name).read_bytes()
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_bytes(current)
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    changes = runtime.verify_sources({'source': {name: expected}})
    assert len(changes) == 1 and changes[0].startswith(name + ': ')
    # Even this reviewed filename cannot authorize another checkpoint version.
    with pytest.raises(ValueError, match='Training source changed'):
        runtime.verify_sources({'source': {name: '0' * 64}})
    # A subsequent edit to an approved file must still block startup.
    target.write_bytes(current + b'\n# unreviewed edit\n')
    with pytest.raises(ValueError, match='Training source changed'):
        runtime.verify_sources({'source': {name: expected}})


def test_unreviewed_model_edit_still_rejected(tmp_path, monkeypatch):
    target = tmp_path / 'models' / 'connector.py'
    target.parent.mkdir()
    original = b'weight = 1\n'
    target.write_bytes(original)
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    provenance = {'source': {'models/connector.py': hashlib.sha256(original).hexdigest()}}
    assert runtime.verify_sources(provenance) == []
    target.write_bytes(b'weight = 2\n')
    with pytest.raises(ValueError, match='Training source changed'):
        runtime.verify_sources(provenance)
