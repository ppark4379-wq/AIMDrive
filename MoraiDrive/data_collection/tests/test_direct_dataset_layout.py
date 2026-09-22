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
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from validate_sensor_record_campaign import finalize_direct_local_publication


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

    def test_direct_publication_is_written_once_after_finalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            campaign_id = "static_lap1_1"
            source = collecting_campaign(
                storage, "full_loop_static", campaign_id)
            metadata = source / "_campaign"
            output = metadata / "validations" / "validation_1"
            output.mkdir(parents=True)
            plan = {
                "campaign_id": campaign_id,
                "dataset_scenario_folder": "static_obstacle",
            }

            campaign, final_output, publication, promoted = (
                finalize_direct_local_publication(
                    metadata, output, storage, "full_loop_static", plan))

            destination = final_campaign(
                storage, "full_loop_static", campaign_id)
            self.assertTrue(promoted)
            self.assertEqual(campaign, destination / "_campaign")
            self.assertEqual(
                final_output,
                destination / "_campaign/validations/validation_1")
            self.assertFalse(source.exists())
            publication_path = final_output / "local_dataset_publication.json"
            self.assertEqual(
                json.loads(publication_path.read_text())["campaign"],
                str(destination.resolve()))

            rerun_output = destination / "_campaign/validations/validation_2"
            rerun_output.mkdir()
            campaign, rerun_output, publication, promoted = (
                finalize_direct_local_publication(
                    destination / "_campaign", rerun_output, storage,
                    "full_loop_static", plan))
            self.assertFalse(promoted)
            self.assertEqual(publication["campaign"], str(destination.resolve()))
            self.assertTrue(
                (rerun_output / "local_dataset_publication.json").is_file())


if __name__ == "__main__":
    unittest.main()
