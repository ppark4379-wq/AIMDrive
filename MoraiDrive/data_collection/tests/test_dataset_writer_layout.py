import json
import os
import tempfile
import unittest

import numpy as np

from ros.sensor_types import GNSSData, IMUData
from writer.dataset_writer import DatasetWriter, ROUTE_DIRS


class DatasetWriterLayoutTest(unittest.TestCase):
    def test_creates_layout_and_writes_aligned_raw_files(self):
        with tempfile.TemporaryDirectory() as root:
            map_source = os.path.join(root, "source_map")
            os.makedirs(map_source)
            for filename in ("global_info.json", "link_set.json", "node_set.json"):
                with open(os.path.join(map_source, filename), "w", encoding="utf-8") as f:
                    json.dump({"file": filename}, f)

            writer = DatasetWriter(root, "job", "route", "kcity", map_source, False)
            for dirname in ROUTE_DIRS:
                self.assertTrue(os.path.isdir(os.path.join(writer.output_dir, dirname)))

            measurement = {"gps_x": 1.0, "gps_y": 2.0, "theta": 0.1, "speed": 3.0,
                           "actors_data": {}}
            writer.write_frame(
                0, measurement, dict(measurement, timestamp_ns=123),
                gnss=GNSSData(1, 37.0, 126.0, 10.0),
                imu=IMUData(1, 1, 2, 3, 4, 5, 6),
            )
            writer.finalize()

            self.assertEqual(np.load(os.path.join(writer.output_dir, "gnss", "0000.npy")).shape, (6,))
            self.assertEqual(np.load(os.path.join(writer.output_dir, "imu", "0000.npy")).shape, (6,))
            with open(os.path.join(root, "job", "dataset_index.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "data/route/ 1\n")
            self.assertTrue(os.path.isfile(os.path.join(root, "job", "maps", "kcity", "manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(root, "job", "results", "episode_summary.json")))

    def test_empty_or_double_finalize_does_not_duplicate_index(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(root, "job", "route", append_timestamp=False)
            writer.finalize()
            writer.finalize()
            with open(os.path.join(root, "job", "dataset_index.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "")


if __name__ == "__main__":
    unittest.main()
