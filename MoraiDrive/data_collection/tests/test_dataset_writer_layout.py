import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from schema import EgoState, build_measurement_dict
from ros.sensor_types import GNSSData, IMUData
from writer.dataset_writer import DatasetWriter, ROUTE_DIRS
from tools.build_dataset_splits import FRAME_DIRS


class DatasetWriterLayoutTest(unittest.TestCase):
    def test_measurement_schema_emits_lmdrive_navigation_and_hazard_keys(self):
        measurement = build_measurement_dict(
            EgoState(
                x=1.0, y=2.0, theta=0.0, speed=3.0,
                command=4, gt_command=3,
                x_command=25.0, y_command=2.0,
                near_node_x=5.0, near_node_y=2.0,
                far_node_x=25.0, far_node_y=2.0,
            ),
            is_vehicle_present=[10],
            is_lane_vehicle_present=[10],
            is_pedestrian_present=[20],
            should_slow=1,
            should_brake=1,
        )
        self.assertEqual(measurement["near_node_x"], 5.0)
        self.assertEqual(measurement["far_node_x"], measurement["x_command"])
        self.assertEqual(measurement["gt_command"], 3)
        self.assertEqual(measurement["is_vehicle_present"], [10])
        self.assertEqual(measurement["is_lane_vehicle_present"], [10])
        self.assertEqual(measurement["is_pedestrian_present"], [20])
        self.assertEqual(measurement["should_slow"], 1)
        self.assertEqual(measurement["should_brake"], 1)

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
                           "actors_data": {},
                           "future_waypoints": [[10.0, 20.0, 4]]}
            writer.write_frame(
                0, measurement, dict(measurement, timestamp_ns=123),
                gnss=GNSSData(1, 37.0, 126.0, 10.0),
                imu=IMUData(1, 1, 2, 3, 4, 5, 6),
                actors={"7": {
                    "loc": [1, 2, 0], "ori": [1, 0], "box": [2, 1],
                    "vel": [0, 0, 0], "tpe": 0,
                }},
                affordances={"traffic_light": 4, "stop_sign": False},
            )
            second_measurement = dict(
                measurement,
                gps_x=100.0,
                gps_y=200.0,
                future_waypoints=[[11.0, 21.0, 4]],
            )
            writer.write_frame(
                1, second_measurement,
                dict(second_measurement, timestamp_ns=124),
            )
            writer.finalize()

            self.assertEqual(np.load(os.path.join(writer.output_dir, "gnss", "0000.npy")).shape, (6,))
            self.assertEqual(np.load(os.path.join(writer.output_dir, "imu", "0000.npy")).shape, (6,))
            with open(os.path.join(root, "job", "dataset_index.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "data/route/ 2\n")
            self.assertTrue(os.path.isfile(os.path.join(root, "job", "maps", "kcity", "manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(root, "job", "results", "episode_summary.json")))
            with open(os.path.join(writer.output_dir, "measurements", "0000.json"),
                      encoding="utf-8") as f:
                self.assertEqual(
                    json.load(f)["future_waypoints"], [[10.0, 20.0, 4]]
                )
            with open(os.path.join(writer.output_dir, "measurements_full", "0000.json"),
                      encoding="utf-8") as f:
                merged = json.load(f)
            self.assertEqual(merged["actors_data"]["7"]["tpe"], 0)
            self.assertEqual(merged["traffic_light"], 4)
            self.assertFalse(merged["stop_sign"])
            with open(os.path.join(writer.output_dir, "measurements", "0001.json"),
                      encoding="utf-8") as f:
                self.assertEqual(
                    json.load(f)["future_waypoints"], [[11.0, 21.0, 4]]
                )
            with open(os.path.join(writer.output_dir, "measurements_all.json"),
                      encoding="utf-8") as f:
                all_frames = json.load(f)
            self.assertEqual(
                [frame["future_waypoints"] for frame in all_frames],
                [[[10.0, 20.0, 4]], [[11.0, 21.0, 4]]],
            )

    def test_empty_or_double_finalize_does_not_duplicate_index(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(root, "job", "route", append_timestamp=False)
            writer.finalize()
            writer.finalize()
            with open(os.path.join(root, "job", "dataset_index.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "")

    def test_split_builder_excludes_failed_routes(self):
        with tempfile.TemporaryDirectory() as root:
            completed = DatasetWriter(root, "job", "ok", append_timestamp=False)
            failed = DatasetWriter(root, "job", "bad", append_timestamp=False)
            measurement = {"gps_x": 1.0, "gps_y": 2.0, "theta": 0.1, "speed": 3.0,
                           "actors_data": {}, "future_waypoints": [[10.0, 20.0, 4]]}
            for writer in (completed, failed):
                writer.write_frame(0, measurement, dict(measurement, timestamp_ns=123))
                for dirname in FRAME_DIRS:
                    directory = os.path.join(writer.output_dir, dirname)
                    os.makedirs(directory, exist_ok=True)
                    ext = ".npy" if dirname in ("lidar", "affordances") else ".json"
                    if dirname.startswith("rgb_"):
                        ext = ".jpg"
                    path = os.path.join(directory, "0000" + ext)
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            f.write(b"test")
            completed.finalize(success=True)
            failed.finalize(success=False)

            script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "tools", "build_dataset_splits.py",
            )
            subprocess.run(
                [
                    sys.executable,
                    script,
                    os.path.join(root, "job"),
                    "--val-ratio",
                    "0",
                    # This one-frame fixture verifies completion filtering, not
                    # route-progress quality (which requires multiple frames).
                    "--skip-route-quality-check",
                ],
                check=True,
            )
            with open(os.path.join(root, "job", "dataset_index.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "data/ok/ 1\n")


if __name__ == "__main__":
    unittest.main()
