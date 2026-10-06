"""Review viewer builds a self-contained page from a run folder and its labels."""
import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from MoraiDrive.data_collection.situation_labeling import review_viewer


class ReviewViewerTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "campaign" / "campaign_lap_001"
        episode = run / "episode_000"
        features = episode / "situation_labels" / "features"
        (episode / "native_associations").mkdir(parents=True)
        features.mkdir(parents=True)
        associations = []
        for i in range(2):
            files = {}
            for k in range(1, 5):
                folder = run / ("CAMERA_%d" % k)
                folder.mkdir(exist_ok=True)
                Image.new("RGB", (64, 36), (40 * k, 20, 20)).save(folder / ("%d.png" % i))
                files["camera_%d" % k] = "CAMERA_%d/%d.png" % (k, i)
            associations.append({"index": i, "camera_files": files})
        (episode / "native_associations" / "associations.json").write_text(json.dumps(associations))
        frames = [{"index": i, "lidar_source_ns": 1_000_000_000 + i * 100_000_000,
                   "ego_world_x_m": -59.6, "ego_world_y_m": -150.0 + i} for i in range(2)]
        rules = [{"index": 0, "zone_id": "static_obstacle", "instruction_state": "approach_maintain",
                  "instruction": "There is an obstacle to the left of the ego vehicle. Maintain the current driving behavior.",
                  "instruction_consistency": "consistent"},
                 {"index": 1, "zone_id": "static_obstacle", "instruction_state": "approach_maintain",
                  "instruction": None, "instruction_consistency": "dropped_keep_going_while_stopped",
                  "instruction_dropped": "Continue driving."}]
        for name, rows in (("frame_features.jsonl", frames), ("situation_rules.jsonl", rules)):
            (features / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return run

    def test_page_index_and_frames_from_the_run_label_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            page = review_viewer.build(run, root / "out")
            review_viewer.build_index(root / "out")
            text = page.read_text(encoding="utf-8")
            rows = json.loads(re.search(r"const ROWS = (\[.*?\]);\n", text, re.S).group(1))
            self.assertEqual([r["i"] for r in rows], [0, 1])
            self.assertEqual(rows[0]["zone_instruction"], 1)
            self.assertEqual(rows[1]["dropped"], "Continue driving.")
            self.assertNotIn(str(root), text.split("const ROWS")[0])   # no absolute paths in the page
            self.assertTrue((page.parent / "frames" / "00001_rear.jpg").is_file())
            self.assertIn("campaign_lap_001/index.html", (root / "out" / "index.html").read_text(encoding="utf-8"))

    def test_labels_root_layout_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            moved = root / "labels" / "campaign" / run.name
            moved.mkdir(parents=True)
            (run / "episode_000" / "situation_labels").rename(moved / "situation_labels")
            (moved / "situation_labels" / "features").rename(moved / "features")
            self.assertEqual(review_viewer.features_dir(run, root / "labels"), moved / "features")
            with self.assertRaises(FileNotFoundError):
                review_viewer.features_dir(run)


if __name__ == "__main__":
    unittest.main()
