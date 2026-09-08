"""Exercise CLI defaults and session selection without starting ROS or collection."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect import _parse_args
from config import COLLECTOR, DATASET
from sources.sync_capture_manifest import resolve_capture_manifest


class CollectorConfigTest(unittest.TestCase):
    def test_duration_only_uses_config_and_newest_session_not_last_write(self):
        with tempfile.TemporaryDirectory() as root:
            old = Path(root) / 'sync-20260101-010000-old/capture_manifest.jsonl'
            new = Path(root) / 'sync-20260101-020000-new/capture_manifest.jsonl'
            for path in (old, new):
                path.parent.mkdir()
                path.touch()
            os.utime(new, (1, 1))
            with patch.object(COLLECTOR, 'manifest_root', root), patch.object(
                COLLECTOR, 'capture_manifest', ''
            ):
                args = _parse_args(['--duration', '17'])
            self.assertEqual(args.duration, 17)
            self.assertEqual(args.mode, 'sync_native_files')
            self.assertEqual(args.capture_manifest, str(new))
            self.assertEqual(args.dataset_root, DATASET.dataset_root)
            self.assertTrue(Path(args.global_path).is_absolute())
            self.assertEqual(resolve_capture_manifest(str(old), root), str(old))

    def test_explicit_storage_and_mode_overrides(self):
        args = _parse_args(['--mode', 'realtime', '--job', 'custom',
                            '--morai-sync-root', '/tmp/sensor-session'])
        self.assertEqual(args.mode, 'realtime')
        self.assertEqual(args.job, 'custom')
        self.assertEqual(args.morai_sync_root, '/tmp/sensor-session')
        self.assertEqual(args.morai_camera_sync_base, '')
        self.assertEqual(args.morai_lidar_sync_base, '')

    def test_missing_manifest_fails_before_collection(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'Start Sync Runtime'):
                resolve_capture_manifest('', root)
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                resolve_capture_manifest(str(Path(root) / 'missing.jsonl'), root)


if __name__ == '__main__':
    unittest.main()
