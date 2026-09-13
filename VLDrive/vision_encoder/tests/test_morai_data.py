import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morai_data import build_chronological_split, expand_range


class MoraiSplitTest(unittest.TestCase):
    def test_4240_frame_split_has_no_future_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = root / "data" / "route_a"
            route.mkdir(parents=True)
            (root / "dataset_index.txt").write_text("data/route_a/ 4240\n", encoding="utf-8")

            manifest = build_chronological_split(root)
            spec = manifest["routes"][0]
            train = expand_range(spec["train"])
            purge = expand_range(spec["purge"])
            validation = expand_range(spec["validation"])

            self.assertEqual(len(train), 3352)
            self.assertEqual(len(purge), 25)
            self.assertEqual(len(validation), 838)
            self.assertEqual(validation[-1], 4214)
            self.assertLess(train[-1] + 25, validation[0])

    def test_invalid_nested_route_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "good").mkdir(parents=True)
            (root / "data" / "_invalid_bad").mkdir(parents=True)
            (root / "dataset_index.txt").write_text(
                "data/good/ 100\ndata/_invalid_bad/ 100\n", encoding="utf-8"
            )
            manifest = build_chronological_split(root)
            self.assertEqual([r["path"] for r in manifest["routes"]], ["good"])


if __name__ == "__main__":
    unittest.main()
