"""
Writes one CARLA/LMDrive-style route folder:
    measurements_full/%04d.json   (per-frame)
    measurements_all.json         (whole-route list, for future-waypoint lookups)
    dataset_index.txt             (registers this route under $DATASET_ROOT)

Call finalize() once the route ends.
"""
import os
import json


class DatasetWriter:
    def __init__(self, output_root: str, route_name: str):
        self.output_root = output_root
        self.route_name = route_name
        self.route_dir = os.path.join(output_root, "data", route_name)
        self.meas_dir = os.path.join(self.route_dir, "measurements_full")
        os.makedirs(self.meas_dir, exist_ok=True)

        self._all_frames = []   # accumulates for measurements_all.json
        self._frame_count = 0

    @property
    def output_dir(self) -> str:
        return self.route_dir

    def write_frame(self, frame_id: int, measurement: dict):
        path = os.path.join(self.meas_dir, f"{frame_id:04d}.json")
        with open(path, "w") as f:
            json.dump(measurement, f)

        self._all_frames.append({
            "gps_x": measurement["gps_x"],
            "gps_y": measurement["gps_y"],
            "theta": measurement["theta"],
            "speed": measurement["speed"],
        })
        self._frame_count = max(self._frame_count, frame_id + 1)

    def finalize(self):
        with open(os.path.join(self.route_dir, "measurements_all.json"), "w") as f:
            json.dump(self._all_frames, f)

        index_path = os.path.join(self.output_root, "dataset_index.txt")
        with open(index_path, "a") as f:
            f.write(f"{self.route_name}/ {self._frame_count}\n")
