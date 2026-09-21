import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection" / "internal_sensor_record"
TOOLS = INTERNAL / "tools"
sys.path[:0] = [str(TOOLS), str(INTERNAL), str(PROJECT)]

from async_capture.journal import digest
from build_sensor_record_training_dataset import planned_split_laps


class StaticCollectionSplitTest(unittest.TestCase):
    def write_plan(self, directory):
        path = Path(directory) / "static_collection_plan.json"
        path.write_text(json.dumps({
            "schema": "morai_static_collection_plan_v1",
            "scenario": "full_loop_static",
            "ready": True,
            "target_runs": 4,
            "runs": [
                {"run": 1, "split": "train"},
                {"run": 2, "split": "train"},
                {"run": 3, "split": "val"},
                {"run": 4, "split": "test"},
            ],
        }), encoding="utf-8")
        return path

    def test_plan_assigns_whole_runs_to_three_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_plan(directory)
            splits = planned_split_laps(path, digest(path), {1, 2, 3, 4}, 4)
        self.assertEqual(splits, {
            "train": {1, 2}, "val": {3}, "test": {4},
        })

    def test_plan_hash_and_surviving_split_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_plan(directory)
            with self.assertRaisesRegex(ValueError, "changed after capture"):
                planned_split_laps(path, "0" * 64, {1, 2, 3, 4}, 4)
            with self.assertRaisesRegex(ValueError, "retain at least one"):
                planned_split_laps(path, digest(path), {1, 2, 3}, 4)

    def test_validator_forwards_sealed_split_plan(self):
        source = (
            TOOLS / "validate_sensor_record_campaign.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--split-plan", str(split_plan_path)', source)
        self.assertIn('"--split-plan-sha256", str(expected_split_sha)', source)


if __name__ == "__main__":
    unittest.main()
