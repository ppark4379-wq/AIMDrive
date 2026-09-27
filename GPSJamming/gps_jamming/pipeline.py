"""Independent LiDAR pipeline. Frame metadata survives every processing step."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from .lidar.ROI import ROI as preprocess
from .lidar.ransac import remove_ground
from .lidar.clustering import cluster_indices
from .lidar.features import build_clusters
from .geometry.transforms import transform_xyzi


class LidarPipeline:
    def __init__(self, config, capture_debug=False, roi_only=False, voxel_only=False, ground_only=False, measure_stages=False):
        if sum((roi_only, voxel_only, ground_only)) > 1:
            raise ValueError("Choose one preview stage")
        self.config = config
        self.capture_debug = capture_debug
        self.last_debug = None
        self.roi_only = roi_only
        self.voxel_only = voxel_only
        self.ground_only = ground_only
        self.measure_stages = measure_stages
        self.last_timings = {}

    def _timed(self, name, function, *args, **kwargs):
        if not self.measure_stages:
            return function(*args, **kwargs)
        started = perf_counter()
        result = function(*args, **kwargs)
        self.last_timings[name] = (perf_counter() - started) * 1000
        return result

    def process(self, frame):
        self.last_timings = {}
        timings = self.last_timings if self.measure_stages else None
        transform = self.config.get("transform")
        if transform and frame.coordinate_frame != transform["target_frame"]:
            source_frame = self.config.get("assembly", {}).get("coordinate_frame", "lidar")
            if frame.coordinate_frame != source_frame:
                raise ValueError(f"Expected {source_frame} points, got {frame.coordinate_frame}")
            frame = replace(frame, points=self._timed('transform', transform_xyzi,
                frame.points, transform["translation_m"], transform["rotation_rpy_deg"]),
                coordinate_frame=transform["target_frame"])
        if self.roi_only:
            points = preprocess(frame.points, **{**self.config["preprocessing"], "voxel_size": 0}, timings=timings)
            result = replace(frame, points=points, clusters=[], diagnostics={
                **frame.diagnostics, "processing_stage": "roi",
                "input_points": len(frame.points), "roi_points": len(points)})
            if self.capture_debug:
                self.last_debug = {"frame": result, "roi": points, "roi_only": True}
            return result
        points = preprocess(frame.points, **self.config["preprocessing"], timings=timings)
        if self.voxel_only:
            roi_points = self._timed('preview_prepare', preprocess, frame.points, **{**self.config["preprocessing"], "voxel_size": 0})
            result = replace(frame, points=points, clusters=[], diagnostics={
                **frame.diagnostics, "processing_stage": "voxel", "input_points": len(frame.points),
                "roi_points": len(roi_points), "voxel_points": len(points)})
            if self.capture_debug:
                self.last_debug = {"frame": result, "roi": roi_points, "voxel": points, "voxel_only": True}
            return result
        nonground, diagnostics = self._timed('ransac', remove_ground, points, **self.config["ground"])
        components = [] if self.ground_only else self._timed('clustering', cluster_indices, nonground, **self.config["clustering"])
        clusters = [] if self.ground_only else self._timed('features', build_clusters, nonground, components)
        diagnostics.update(input_points=len(frame.points), preprocessed_points=len(points),
                           nonground_points=len(nonground))
        if self.ground_only:
            diagnostics["processing_stage"] = "ground"
        result = replace(frame, points=nonground if self.ground_only else frame.points, clusters=clusters,
                         diagnostics={**frame.diagnostics, **diagnostics})
        if self.capture_debug:
            debug_started = perf_counter() if self.measure_stages else 0
            # ROI-only preview; keep the normal numerical processing path identical.
            roi_config = {**self.config["preprocessing"], "voxel_size": 0}
            roi_points = None if self.ground_only else preprocess(frame.points, **roi_config)
            ground_mask = np.zeros(len(points), dtype=bool)
            if diagnostics["ground_status"] == "plane_found":
                a, b, c, d = diagnostics["ground_plane"]
                distances = np.abs(points[:, :3].astype(np.float64) @ np.array([a, b, c]) + d)
                ground_mask = distances <= self.config["ground"].get("distance_threshold", 0.12)
            # Inspect this dictionary at a breakpoint, or pass it to DirectStageViewer.
            self.last_debug = {
                "frame": result, "raw": frame.points, "roi": roi_points,
                "voxel": points, "ground": points[ground_mask],
                "nonground": nonground, "components": components, "clusters": clusters,
            }
            if self.ground_only:
                self.last_debug = {"frame": result, "voxel": points, "ground": points[ground_mask],
                                   "nonground": nonground, "ground_only": True}
            if self.measure_stages:
                self.last_timings['preview_prepare'] = (perf_counter() - debug_started) * 1000
        return result
