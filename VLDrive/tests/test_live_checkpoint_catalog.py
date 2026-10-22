"""New live model labels preserve old configs and exact checkpoint identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference import realtime_ve_objects as live


class CheckpointCatalogTests(unittest.TestCase):
    def load(self, catalog, label):
        config = dict(schema='morai_live_ve_objects_v1', checkpoints=catalog,
                      map_basis=live.MAP_BASIS, object_topic='/ve/Object_topic',
                      health_topic='/ve/perception_health')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps(config))
            with patch.object(Path, 'is_file', return_value=True):
                return live.load_config(path, label)

    def test_old_catalog_still_selects_original_checkpoint(self):
        old = {key:live.CHECKPOINTS[key] for key in ('E3','E4')}
        for label in old:
            self.assertEqual(self.load(old,label)['checkpoint'],old[label])

    def test_new_label_requires_explicit_identity_in_config(self):
        old = {key:live.CHECKPOINTS[key] for key in ('E3','E4')}
        with self.assertRaisesRegex(ValueError,'identities'):
            self.load(old,'P1')

    def test_new_label_selects_registered_candidate(self):
        self.assertEqual(self.load(live.CHECKPOINTS,'P1')['checkpoint'],live.CHECKPOINTS['P1'])

    def test_tampered_or_unknown_identity_is_rejected(self):
        for kind in ('path','sha256','unknown'):
            catalog = copy.deepcopy(live.CHECKPOINTS)
            if kind == 'unknown':
                catalog['arbitrary'] = dict(catalog['P1'])
            else:
                catalog['P1'][kind] = 'different'
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError,'identities'):
                self.load(catalog,'P1')


if __name__ == '__main__':
    unittest.main()
