"""Stationary survey provenance, full rotations and conservative authority."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..scenarios.full_loop_parts.tunnel_static_parts.layout import OPPOSING_LINK, TUNNEL_LINK
from ..scenarios.full_loop_parts.tunnel_static_parts.survey import (APPROACH_LINK, DEPARTURE_LINK, analyze_clouds,
    cloud_to_world, ground_crosssection, project_cloud, rotation_rpy, sensor_calibration, station_pose, structure_extent,
    survey_route, validate_exposure)


def route():
    links = {APPROACH_LINK: {"idx": APPROACH_LINK, "from_node_idx": "a", "to_node_idx": "b", "points": [[-15., 0., 28.], [0., 0., 28.]]},
             TUNNEL_LINK: {"idx": TUNNEL_LINK, "from_node_idx": "b", "to_node_idx": "c", "road_type": "2", "width_start": 3.5,
                           "width_end": 3.5, "points": [[0., 0., 28.], [120., 0., 28.]]},
             DEPARTURE_LINK: {"idx": DEPARTURE_LINK, "from_node_idx": "c", "to_node_idx": "d", "points": [[120., 0., 28.], [120., 20., 28.]]},
             OPPOSING_LINK: {"road_type": "2", "width_start": 3.5, "width_end": 3.5, "points": [[120., 3.5, 28.], [0., 3.5, 28.]]}}
    return survey_route(SimpleNamespace(get_link=lambda identity: links[identity]))


def pose():
    return {"position_xyz": [0., 0., 28.], "rotation_xyz_deg": [0., 0., 0.],
            "speed_norm_raw": 0., "angular_velocity_raw": [0., 0., 0.]}


class StationarySurveyTests(unittest.TestCase):
    def test_observed_simulator_microsecond_clock_matches_ros_seconds(self):
        from ..tools.survey_tunnel_boundaries import _clock
        adapter = SimpleNamespace(_simulator_stub=SimpleNamespace(
            GetTimestamp=lambda message, timeout: SimpleNamespace(value=1791290288408035)))
        with patch.dict("sys.modules", {"proto.morai.common.type_pb2": SimpleNamespace(Empty=lambda: object())}):
            self.assertAlmostEqual(_clock(adapter), 1791290288.408035, places=5)

    def test_full_rpy_rotation_and_sensor_translation(self):
        calibration = {"rotation_ego_rpy_deg": [0., 0., 90.], "position_ego_flu_m": [1., 0., 1.]}
        value = pose(); value["rotation_xyz_deg"] = [90., 0., 0.]
        world = cloud_to_world([[2., 0., 0.], [0., 0., 0.], [float("nan"), 0., 0.]], value, calibration)
        np.testing.assert_allclose(world, [[1., -1., 30.]], atol=1e-10)
        np.testing.assert_allclose(rotation_rpy([0., 90., 0.]) @ np.array([1., 0., 0.]), [0., 0., -1.], atol=1e-10)

    def test_adjacent_map_link_preserves_exit_bend_and_signed_station(self):
        geometry = route()
        position, heading = station_pose(geometry, 125.)
        np.testing.assert_allclose(position, [120., 5., 28.])
        self.assertAlmostEqual(heading, math.pi/2)
        np.testing.assert_allclose(project_cloud([[-5., 1., 28.], [120., 5., 30.]], geometry), [[-5., 1., 0.], [125., 0., 2.]])

    def test_fresh_pose_bracket_and_stamp_validation(self):
        options = {"cloud_stamp": 101., "stamp_before": 100., "stamp_after": 102.,
                   "receipt_monotonic": 11., "pose_before_monotonic": 10., "pose_after_monotonic": 12.,
                   "frame_id": "Lidar3D-5", "expected_frame": "Lidar3D-5", "previous_stamp": 100.5}
        self.assertTrue(validate_exposure(pose(), pose(), **options))
        for key, value in (("cloud_stamp", 99.), ("previous_stamp", 101.), ("receipt_monotonic", 9.), ("frame_id", "unrelated")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_exposure(pose(), pose(), **dict(options, **{key: value}))
        moved = pose(); moved["rotation_xyz_deg"][0] = .5
        with self.assertRaisesRegex(RuntimeError, "RPY"):
            validate_exposure(pose(), moved, **options)
        moved = pose(); moved["position_xyz"][0] = .1
        with self.assertRaisesRegex(RuntimeError, "position"):
            validate_exposure(pose(), moved, **options)
        moved = pose(); moved["speed_norm_raw"] = .5
        with self.assertRaisesRegex(RuntimeError, "moving"):
            validate_exposure(pose(), moved, **options)

    def test_saved_sensor_hash_and_frame_without_runtime_mounting_claim(self):
        row = {"m_SensorUniqueID": 5, "pos": {"x": "1", "y": "0", "z": "1.48"}, "rot": {"roll": "0", "pitch": "0", "yaw": "0"},
               "lc": {"sensorPeriod": .1, "gnGenerator": {"isActivated": False}, "rosConfig": {"Topic": "/lidar3D", "frameID": "Lidar3D-5"}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"sensors.json"; path.write_text(json.dumps({"Lidar3DList": [row]}))
            calibration = sensor_calibration(path)
            self.assertEqual(calibration["position_ego_flu_m"], [1., 0., 1.48])
            self.assertFalse(calibration["active_mounting_verified"])
            path.write_text(json.dumps({"Lidar3DList": [row, copy.deepcopy(row)]}))
            with self.assertRaises(ValueError):
                sensor_calibration(path)

    def test_sparse_single_view_does_not_claim_portal_or_drivable_width(self):
        result = analyze_clouds([[40., -4.3, 2.], [40., 7.7, 2.], [3., 0., 6.8]], [0, 0, 0], route())
        self.assertEqual(result["portal_estimates"], {})
        self.assertFalse(result["physical_portal_edge_verified"])
        self.assertFalse(result["physical_drivable_width_verified"])
        self.assertEqual(result["planning_corridor"]["width_m"], 7.)

    def test_multi_view_structure_estimate_keeps_wall_and_drivable_width_separate(self):
        points, views = [], []
        for view in range(3):
            for station in np.arange(2.5, 120.6, .1):
                for lateral in np.arange(-1.5, 5.1, .5):
                    points.append([station, lateral, 6.8]); views.append(view)
            for station in np.arange(25., 95., .2):
                points.extend([[station, -4.3, 2.], [station, 7.7, 2.]])
                views.extend([view, view])
        result = analyze_clouds(points, views, route())
        self.assertAlmostEqual(result["observed_wall_span_m"], 12., places=8)
        self.assertLess(abs(result["portal_estimates"]["entrance"]["station_estimate_m"]-2.5), .3)
        self.assertFalse(result["physical_portal_edge_verified"])
        self.assertFalse(result["physical_drivable_width_verified"])
        self.assertEqual(result["planning_corridor"]["lateral_interval_m"], [-1.75, 5.25])

    def test_flat_floor_and_curb_like_returns_do_not_expand_planning_authority(self):
        points, views = [], []
        for view in range(3):
            for station in np.arange(35., 85., .5):
                for lateral in np.arange(-3.9, 7.1, .1):
                    height = .2 if lateral < -3.25 or lateral > 6.75 else 0.
                    points.append([station, lateral, height]); views.append(view)
        profile = ground_crosssection(np.array(points), np.array(views))
        self.assertTrue(profile["nominal_carriageway_inside_observed_flat_floor"])
        self.assertFalse(profile["physical_road_edge_verified"])
        self.assertGreater(profile["flat_floor_span_m"], 9.)
        self.assertEqual(set(profile["raised_floor_transitions"]), {"right", "left"})
        result = analyze_clouds(points, views, route())
        self.assertEqual(result["planning_corridor"]["width_m"], 7.)

    def test_high_structure_extent_excludes_isolated_external_clutter(self):
        points, views = [], []
        for view in range(3):
            for station in np.arange(2.8, 120.8, .1):
                for height in np.arange(2., 4.1, .2):
                    points.append([station, -4.3, height]); views.append(view)
            for station in np.arange(-5., -3., .05):
                for height in np.arange(2., 4.1, .2):
                    points.append([station, -4., height]); views.append(view)
        extent = structure_extent(np.asarray(points), np.asarray(views), route())
        self.assertEqual(extent["status"], "continuous_high_structure_observed")
        self.assertAlmostEqual(extent["portal_estimates"]["entrance"]["station_estimate_m"], 2.875)
        self.assertGreater(extent["observed_structure_station_interval_m"][1], 120.5)
        self.assertFalse(extent["physical_portal_edge_verified"])

    def test_interior_visibility_loss_cannot_be_called_tunnel_exit(self):
        points, views = [], []
        for view in range(3):
            for station in np.arange(2.8, 79., .1):
                for height in np.arange(2., 4.1, .2):
                    points.append([station, -4.3, height]); views.append(view)
        extent = structure_extent(np.asarray(points), np.asarray(views), route())
        self.assertEqual(extent["status"], "insufficient_continuous_structure")
        self.assertNotIn("portal_estimates", extent)


if __name__ == "__main__":
    unittest.main()
