import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(ROOT))

from dataset_layout import (collecting_campaign, final_campaign,
                            finalize_collecting_campaign, find_campaign,
                            is_direct_campaign_metadata)


class DirectDatasetLayoutTests(unittest.TestCase):
    def test_finalize_is_same_tree_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            source = collecting_campaign(storage, "full_loop_dynamic", "dynamic_lap2_1")
            metadata = source / "_campaign"
            run = source / "dynamic_lap2_1_lap_001"
            metadata.mkdir(parents=True)
            run.mkdir()
            payload = run / "sensor.bin"
            payload.write_bytes(b"sensor")
            inode = payload.stat().st_ino
            destination = finalize_collecting_campaign(
                metadata, storage, "full_loop_dynamic", "dynamic_lap2_1")
            self.assertEqual(destination,
                             final_campaign(storage, "full_loop_dynamic", "dynamic_lap2_1"))
            self.assertFalse(source.exists())
            self.assertEqual((destination / run.name / payload.name).stat().st_ino, inode)
            self.assertTrue(is_direct_campaign_metadata(destination / "_campaign", storage))
            self.assertEqual(find_campaign(storage, "dynamic_lap2_1"),
                             destination / "_campaign")


if __name__ == "__main__":
    unittest.main()
