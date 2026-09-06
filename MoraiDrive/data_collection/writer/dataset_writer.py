"""LMDrive-compatible MORAI dataset layout and per-frame writers."""
import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Dict, Optional

import numpy as np


ROUTE_DIRS = (
    "rgb_front", "rgb_left", "rgb_right", "rgb_rear", "lidar", "gnss", "imu",
    "birdview", "topdown", "3d_bbs", "affordances", "actors_data",
    "measurements", "measurements_full",
)
MAP_FILES = (
    "global_info.json", "link_set.json", "node_set.json", "junction_set.json",
    "lane_boundary_set.json", "traffic_light_set.json",
)


class DatasetWriter:
    def __init__(self, dataset_root: str, job_name: str, route_name: str,
                 map_id: str = "unknown", map_source_dir: str = "",
                 append_timestamp: bool = True, route_metadata: Optional[dict] = None):
        self.dataset_root = os.path.abspath(dataset_root)
        self.job_name = job_name
        self.job_dir = os.path.join(self.dataset_root, job_name)
        timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.route_name = f"{route_name}_{timestamp}" if append_timestamp else route_name
        self.route_dir = os.path.join(self.job_dir, "data", self.route_name)
        self.map_id = map_id

        os.makedirs(os.path.join(self.job_dir, "results"), exist_ok=True)
        for dirname in ROUTE_DIRS:
            os.makedirs(os.path.join(self.route_dir, dirname), exist_ok=True)

        self._ensure_job_files()
        self._install_map(map_source_dir)
        self._write_route_metadata(route_metadata)
        self._all_frames = []
        self._frame_count = 0
        self._finalized = False

    @property
    def output_dir(self) -> str:
        return self.route_dir

    def _ensure_job_files(self):
        for filename in ("dataset_index.txt", "navigation_instruction_list.txt"):
            path = os.path.join(self.job_dir, filename)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8"):
                    pass
        notice_path = os.path.join(self.job_dir, "notice_instruction_list.json")
        if not os.path.exists(notice_path):
            self._write_json(notice_path, [])

    def _install_map(self, source_dir: str):
        map_dir = os.path.join(self.job_dir, "maps", self.map_id)
        os.makedirs(map_dir, exist_ok=True)
        copied = {}
        if source_dir:
            for filename in MAP_FILES:
                source = os.path.join(source_dir, filename)
                if not os.path.isfile(source):
                    continue
                destination = os.path.join(map_dir, filename)
                # Windows mapped/network drives can reject copystat() even when
                # ordinary file writes work. Dataset maps only need their bytes;
                # provenance and integrity are recorded in manifest.json.
                shutil.copyfile(source, destination)
                copied[filename] = self._sha256(destination)

        manifest_path = os.path.join(map_dir, "manifest.json")
        manifest = {
            "map_id": self.map_id,
            "source_dir": os.path.abspath(source_dir) if source_dir else None,
            "files": copied,
        }
        if copied or not os.path.exists(manifest_path):
            self._write_json(manifest_path, manifest)

    def _write_route_metadata(self, route_metadata: Optional[dict] = None):
        metadata = {
            "map_id": self.map_id,
            "map_path": f"../../maps/{self.map_id}",
            "nav_link_ids": [], "nav_node_ids": [], "nav_waypoints": [],
        }
        if route_metadata:
            metadata.update(route_metadata)
            # Dataset route metadata must remain relocatable.
            metadata["map_id"] = self.map_id
            metadata["map_path"] = f"../../maps/{self.map_id}"
        self._write_json(os.path.join(self.route_dir, "route.json"), metadata)

    def write_frame(self, frame_id: int, measurement: dict, full_measurement: dict,
                    gnss=None, imu=None, actors: Optional[Dict] = None,
                    affordances: Optional[Dict] = None):
        name = f"{frame_id:04d}"
        actors = actors or {}
        affordances = dict(affordances or {
            "traffic_light": None,
            "stop_sign": False,
        })
        affordances.setdefault("traffic_light", None)
        affordances.setdefault("stop_sign", False)

        # measurements_full은 LMDrive preprocessing과 동일하게 measurements,
        # actors_data, affordances를 한 frame JSON으로 합친 loader 입력이다.
        merged_measurement = dict(measurement)
        merged_measurement.update(full_measurement)
        merged_measurement["actors_data"] = actors
        merged_measurement["traffic_light"] = affordances["traffic_light"]
        merged_measurement["stop_sign"] = affordances["stop_sign"]

        self._write_json(os.path.join(self.route_dir, "measurements", name + ".json"), measurement)
        self._write_json(
            os.path.join(self.route_dir, "measurements_full", name + ".json"),
            merged_measurement,
        )
        self._write_json(os.path.join(self.route_dir, "actors_data", name + ".json"), actors)

        if gnss is not None:
            np.save(os.path.join(self.route_dir, "gnss", name + ".npy"), np.asarray([
                gnss.latitude, gnss.longitude, gnss.altitude,
                gnss.velocity_x, gnss.velocity_y, gnss.velocity_z,
            ], dtype=np.float64))
        if imu is not None:
            np.save(os.path.join(self.route_dir, "imu", name + ".npy"), np.asarray([
                imu.accel_x, imu.accel_y, imu.accel_z,
                imu.gyro_x, imu.gyro_y, imu.gyro_z,
            ], dtype=np.float32))

        np.save(
            os.path.join(self.route_dir, "affordances", name + ".npy"),
            affordances,
            allow_pickle=True,
        )
        vehicles = []
        pedestrians = []
        for actor_id, actor in actors.items():
            box = {
                "id": int(actor_id), "loc": actor["loc"], "ori": actor["ori"],
                "box": actor["box"], "vel": actor["vel"], "tpe": actor["tpe"],
            }
            (vehicles if actor["tpe"] == 0 else pedestrians).append(box)
        np.save(os.path.join(self.route_dir, "3d_bbs", name + ".npy"), {
            "traffic_lights": [], "stop_signs": [],
            "vehicles": vehicles, "pedestrians": pedestrians,
        }, allow_pickle=True)

        self._all_frames.append(merged_measurement)
        self._frame_count = max(self._frame_count, frame_id + 1)

    def finalize(self, success: bool = True):
        if self._finalized:
            return
        self._backfill_future_waypoints()
        self._write_json(os.path.join(self.route_dir, "measurements_all.json"), self._all_frames)
        if self._frame_count > 0 and success:
            relative_route = os.path.relpath(self.route_dir, self.job_dir).replace(os.sep, "/") + "/"
            index_path = os.path.join(self.job_dir, "dataset_index.txt")
            with open(index_path, "a+", encoding="utf-8") as f:
                f.seek(0)
                existing = {line.split()[0] for line in f if line.split()}
                if relative_route not in existing:
                    f.write(f"{relative_route} {self._frame_count}\n")
        if self._frame_count > 0:
            relative_route = os.path.relpath(self.route_dir, self.job_dir).replace(os.sep, "/") + "/"
            summary_path = os.path.join(self.job_dir, "results", "episode_summary.json")
            episodes = []
            if os.path.exists(summary_path):
                with open(summary_path, encoding="utf-8") as f:
                    previous = json.load(f)
                episodes = previous.get("episodes", [])
            if not any(item.get("route") == relative_route for item in episodes):
                episodes.append({
                    "route": relative_route,
                    "frames": self._frame_count,
                    "status": "completed" if success else "failed",
                })
            self._write_json(summary_path, {"episodes": episodes})
        self._finalized = True

    def _backfill_future_waypoints(self, count: int = 50):
        """Fill LMDrive future_waypoints after future frames become available."""
        for index, full in enumerate(self._all_frames):
            future = [
                [frame["gps_x"], frame["gps_y"], frame.get("command", 4)]
                for frame in self._all_frames[index:min(len(self._all_frames), index + count)]
            ]
            full["future_waypoints"] = future
            name = f"{index:04d}.json"
            full_path = os.path.join(self.route_dir, "measurements_full", name)
            measurement_path = os.path.join(self.route_dir, "measurements", name)
            self._write_json(full_path, full)
            if os.path.isfile(measurement_path):
                with open(measurement_path, encoding="utf-8") as f:
                    measurement = json.load(f)
                measurement["future_waypoints"] = future
                self._write_json(measurement_path, measurement)

    @staticmethod
    def _write_json(path: str, value):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
