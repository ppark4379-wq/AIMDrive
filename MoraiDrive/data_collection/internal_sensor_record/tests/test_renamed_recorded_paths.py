import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from async_capture.journal import digest
from async_capture.portable_paths import recorded_episode_directory

class RenamedRecordingTests(unittest.TestCase):
    def test_hash_bound_rename_and_tampering(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'new_run'/'episode_000';raw=episode/'raw';raw.mkdir(parents=True)
            quality=episode/'quality';quality.mkdir()
            manifest=raw/'raw_manifest.json';manifest.write_text('{}')
            (raw/'RAW_COMPLETE.json').write_text(json.dumps({'status':'completed','manifest_sha256':digest(manifest)}))
            old='/old_disk/old_run/episode_000/raw'
            index=quality/'INDEX_COMPLETE.json';index.write_text(json.dumps({'raw_path':old,'raw_manifest_sha256':digest(manifest)}))
            seal=quality/'QUALITY_COMPLETE.json';seal.write_text(json.dumps({'files':{index.name:digest(index)}}))
            self.assertEqual(recorded_episode_directory(old,episode,'raw',root),raw)
            with self.assertRaises(ValueError):recorded_episode_directory('/another/run/episode_000/raw',episode,'raw',root)
            manifest.write_text('{"changed":true}')
            with self.assertRaises(ValueError):recorded_episode_directory(old,episode,'raw',root)
            manifest.write_text('{}');index.write_text('{}')
            with self.assertRaises(ValueError):recorded_episode_directory(old,episode,'raw',root)
if __name__=='__main__':unittest.main()
