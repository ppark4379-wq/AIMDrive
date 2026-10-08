"""Prepared road geometry must survive cold concurrent planner/runtime reads."""

import copy
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import shapely
from shapely.geometry import Polygon

from expert.src.localization.point import Point
from expert.src.planning.Jamming.road_boundary import RoadBoundary
from expert.src.planning.Jamming.recorded_planner import JammingPlanner


MAP_DIR = Path(__file__).resolve().parents[4] / "scenario_runner/data/R_KR_PR_K-city_2025"
BODY = {"rear_m": .860558, "front_m": 3.914257, "width_m": 2.366220}


def actual_boundary():
    return RoadBoundary.from_map_dir(MAP_DIR, body_policy=BODY)


def poses(count=1024):
    rows = json.loads((MAP_DIR / "link_set.json").read_text())
    own = np.asarray(next(row["points"] for row in rows if row["idx"] == "A2256W000126"))[:, :2]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(own, axis=0), axis=1))]
    station = np.linspace(10., distance[-1]-10., count)
    center = np.column_stack([np.interp(station, distance, own[:, axis]) for axis in (0, 1)])
    delta = np.gradient(center, axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    side = np.linspace(-2., 6., count)
    xy = center + np.column_stack((-np.sin(yaw), np.cos(yaw))) * side[:, None]
    return xy, yaw + .04 * np.sin(station)


class BoundaryThreadTests(unittest.TestCase):
    def test_shallow_and_deep_copies_share_native_polygon_lock_and_immutable_contract(self):
        boundary = actual_boundary()
        copies = [copy.copy(boundary), copy.deepcopy(boundary)]
        for clone in copies:
            self.assertIsNot(clone, boundary)
            self.assertIs(clone._polygon, boundary._polygon)
            self.assertIs(clone._predicate_lock, boundary._predicate_lock)
            self.assertIs(clone.origin, boundary.origin)
            self.assertFalse(clone.origin.flags.writeable)
            self.assertEqual(clone.policy, boundary.policy)
            self.assertEqual(clone.source_sha256, boundary.source_sha256)
            self.assertEqual(clone.policy_sha256, boundary.policy_sha256)
        # Exported metadata remains independent, including for a deep copy.
        metadata = copies[1].policy
        metadata["body"]["width_m"] = .1
        self.assertEqual(boundary.body_policy["width_m"], BODY["width_m"])

    def test_cold_shared_covers_and_observed_polygon_operations_enter_geos_serially(self):
        boundary = actual_boundary()  # prepared, with no predicate call yet
        clones = [boundary, copy.copy(boundary), copy.deepcopy(boundary), copy.copy(boundary)]
        xy, yaw = poses(64)
        polygon = Polygon(boundary.corners_world(xy[:1], yaw[:1])[0])
        state_lock = threading.Lock()
        state = {"active": 0, "max_active": 0, "calls": 0}
        original_covers, original_difference = shapely.covers, shapely.difference

        def guarded(operation):
            def invoke(*args, **kwargs):
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    state["calls"] += 1
                    overlap = state["active"] > 1
                try:
                    # Widen the entry race deterministically, while rejecting
                    # an overlap BEFORE calling unsafe shared native geometry.
                    if overlap:
                        raise AssertionError("Concurrent prepared GEOS access")
                    time.sleep(.0005)
                    return operation(*args, **kwargs)
                finally:
                    with state_lock:
                        state["active"] -= 1
            return invoke

        barrier = threading.Barrier(4)
        def run(index):
            owner = clones[index]
            barrier.wait(timeout=5.)
            for _ in range(24):
                if index == 0:
                    owner.mask_world(xy, yaw)
                elif index == 1:
                    owner.planning_mask_world(xy, yaw)
                elif index == 2:
                    owner.covers_geometry(polygon)
                else:
                    owner.outside_area_geometry(polygon)

        with patch.object(shapely, "covers", side_effect=guarded(original_covers)), \
                patch.object(shapely, "difference", side_effect=guarded(original_difference)):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run, index) for index in range(4)]
                for future in futures:
                    future.result(timeout=10.)
        self.assertEqual(state["active"], 0)
        self.assertEqual(state["max_active"], 1)
        self.assertGreaterEqual(state["calls"], 96)

    def test_two_and_four_thread_cold_masks_equal_independent_unprepared_geometry(self):
        xy, yaw = poses()
        reference = actual_boundary()
        unprepared = shapely.from_wkb(shapely.to_wkb(reference._polygon))
        expected_runtime = shapely.covers(unprepared, shapely.polygons(reference.corners_world(xy, yaw, .15)))
        expected_planning = shapely.covers(unprepared, shapely.polygons(reference.corners_world(xy, yaw, .35)))
        self.assertTrue(expected_runtime.any())
        self.assertFalse(expected_runtime.all())
        start = time.monotonic()
        for thread_count in (2, 4):
            # Each run starts on a new prepared polygon, not warmed by the
            # oracle's serial calls. Copies share the cache and its lock.
            boundary = actual_boundary()
            clones = [copy.copy(boundary) if i % 2 else copy.deepcopy(boundary)
                      for i in range(thread_count)]
            barrier = threading.Barrier(thread_count)
            def run(index):
                owner = clones[index]
                barrier.wait(timeout=5.)
                for _ in range(48):
                    runtime = owner.mask_world(xy, yaw)
                    planning = owner.planning_mask_world(xy, yaw)
                    np.testing.assert_array_equal(runtime, expected_runtime)
                    np.testing.assert_array_equal(planning, expected_planning)
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(run, index) for index in range(thread_count)]
                for future in futures:
                    future.result(timeout=12.)
        elapsed = time.monotonic()-start
        # 589,824 pose checks plus extra planning's runtime pass. This ceiling
        # matches the candidate worker budget and catches gross lock overhead.
        self.assertLess(elapsed, 15., "Concurrent mask stress exceeded screening budget")

    def test_runtime_validation_and_static_planner_snapshot_share_cold_boundary(self):
        boundary = actual_boundary()
        planner = JammingPlanner(background=False, road_boundary=boundary)
        snapshot = copy.copy(planner.local)
        runtime = copy.copy(planner)
        runtime.local = copy.copy(planner.local)
        self.assertIs(snapshot.road_boundary._predicate_lock, boundary._predicate_lock)
        self.assertIs(runtime.local.road_boundary._predicate_lock, boundary._predicate_lock)
        rows = json.loads((MAP_DIR / "link_set.json").read_text())
        points = next(row["points"] for row in rows if row["idx"] == "A2256W000126")[40:190]
        path = [Point(*row[:2]) for row in points]
        xy = np.asarray(path)
        delta = np.gradient(xy, axis=0)
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        oracle = actual_boundary()
        expected_runtime = oracle.mask_world(xy, yaw)
        expected_plan = oracle.planning_mask_world(xy, yaw)
        barrier = threading.Barrier(3)
        def run(role):
            barrier.wait(timeout=5.)
            for _ in range(32):
                if role == 0:
                    np.testing.assert_array_equal(boundary.mask_world(xy, yaw), expected_runtime)
                elif role == 1:
                    np.testing.assert_array_equal(snapshot.road_boundary.planning_mask_world(xy, yaw), expected_plan)
                else:
                    result = runtime.validate_world_path(path, [], global_path=path,
                                                         require_complete=False, planning=False)
                    self.assertTrue(result["passed"], result)
                    self.assertTrue(result["runtime_path_road_passed"])
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run, role) for role in range(3)]
                for future in futures:
                    future.result(timeout=12.)
        finally:
            planner.close()


if __name__ == "__main__":
    unittest.main()
