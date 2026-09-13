import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from expert.src.localization.point import Point
from expert.src.planning.highway_lane_planning import lane_map, project_diagnostics


class ProjectDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.map = lane_map.HighwayLaneMap.__new__(lane_map.HighwayLaneMap)
        self.link = SimpleNamespace(id="test-link", points=[Point(0, 0), Point(10, 0)], cumulative_s=[0., 10.])

    def test_normal_projection_does_not_log(self):
        with patch.object(project_diagnostics, "write_project_failure") as log:
            result = self.map.project(Point(3, 4), self.link)
        self.assertEqual((result.s, result.distance, result.x, result.y), (3., 4., 3., 0.))
        log.assert_not_called()

    def test_array_failure_records_inputs_and_reraises(self):
        position = SimpleNamespace(x=np.array([1., 2.]), y=0.)
        with tempfile.TemporaryDirectory() as root:
            source = str(Path(root)/"one/two/three/four/five/lane_map.py")
            with patch.object(lane_map, "__file__", source):
                with self.assertRaises(ValueError):
                    self.map.project(position, self.link)
            paths = list((Path(root)/".runtime/diagnostics").glob("*.json"))
            self.assertEqual(len(paths), 1)
            data = json.loads(paths[0].read_text())
            self.assertEqual(data["exception_type"], "ValueError")
            self.assertTrue(data["locals"]["ratio"]["not_assigned"])
            helper = data["traceback_frames"][-1]
            self.assertEqual(helper["function"], "_projection_xy")
            self.assertEqual(helper["locals"]["coordinate"]["shape"], [2])
            self.assertEqual(data["locals"]["position.x"]["shape"], [2])
            self.assertEqual(data["lane_link"]["id"]["value_repr"], "'test-link'")
            self.assertIn("project", data["traceback"])
            self.assertTrue(data["python_executable"])
        np.testing.assert_array_equal(position.x, [1., 2.])

    def test_constructor_identity_is_captured_without_swallowing_error(self):
        class BrokenMatch:
            def __init__(self, lane_link, s, distance, x, y, yaw):
                x = self
                self.x = float(x)

        with tempfile.TemporaryDirectory() as root:
            source = str(Path(root)/"one/two/three/four/five/lane_map.py")
            with patch.object(lane_map, "__file__", source), patch.object(lane_map, "LaneMatch", BrokenMatch):
                with self.assertRaises(TypeError):
                    self.map.project(Point(3, 4), self.link)
            data = json.loads(next((Path(root)/".runtime/diagnostics").glob("*.json")).read_text())
            caller, constructor = data["traceback_frames"]
            self.assertEqual(caller["function"], "project")
            self.assertEqual(constructor["function"], "__init__")
            self.assertTrue(constructor["identity"]["x_is_self"])
            self.assertFalse(constructor["identity"]["x_is_project_x"])
            self.assertEqual(constructor["locals"]["x"]["object_id"], constructor["locals"]["self"]["object_id"])
            self.assertEqual(caller["locals"]["x"]["object_id"], data["locals"]["x"]["object_id"])
            self.assertEqual(caller["locals"]["x"]["value_repr"], "3.0")

    def test_projection_does_not_use_point_scalar_properties(self):
        class BadScalarPoint(Point):
            @property
            def x(self):
                raise AssertionError("Point.x must not be used for projection")
            @property
            def y(self):
                raise AssertionError("Point.y must not be used for projection")

        link = SimpleNamespace(id="scalar-test", cumulative_s=[0., 10.],
                               points=[BadScalarPoint(0, 0), BadScalarPoint(10, 0)])
        result = self.map.project(BadScalarPoint(3, 4), link)
        self.assertEqual((result.s, result.distance, result.x, result.y), (3., 4., 3., 0.))

    def test_projection_clamps_endpoints_and_skips_duplicate(self):
        link = SimpleNamespace(id="duplicates", cumulative_s=[0., 0., 10.],
                               points=[Point(0, 0), Point(0, 0), Point(10, 0)])
        for position, expected in [(Point(-2, 0), (0., 2.)), (Point(12, 0), (10., 2.))]:
            result = self.map.project(position, link)
            self.assertEqual((result.s, result.distance), expected)

    def test_projection_rejects_malformed_or_nonfinite_points(self):
        with patch.object(project_diagnostics, "write_project_failure"):
            for position in (Point(1, 2).reshape(1, 2), Point(float("nan"), 0)):
                with self.assertRaises(ValueError):
                    self.map.project(position, self.link)

    def test_logging_failure_keeps_original_exception(self):
        original = TypeError("original computation error")
        with patch.object(lane_map.math, "hypot", side_effect=original), patch.object(
                project_diagnostics, "write_project_failure", side_effect=OSError("disk failure")):
            with self.assertRaises(TypeError) as caught:
                self.map.project(Point(3, 4), self.link)
        self.assertIs(caught.exception, original)


if __name__ == "__main__":
    unittest.main()
