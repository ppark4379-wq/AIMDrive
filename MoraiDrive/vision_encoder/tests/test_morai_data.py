"""CPU regression checks; run with python -m unittest discover -s tests -v."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DATA = Path(__file__).resolve().parents[1] / "timm" / "data"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, DATA / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


routes = load_module("morai_routes")
try:
    import numpy as np
except ImportError:
    np = None


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "dataset_index.txt").write_text("alpha/ 3\ndata/beta/ 1\n", encoding="utf-8")
        for name, measurements in {
            "alpha": [{}, {"navigation": {"route_valid": False}},
                      {"navigation": {"route_valid": True}}],
            "beta": [{}],
        }.items():
            folder = self.root / "data" / name / "measurements_full"
            folder.mkdir(parents=True)
            for frame_id, measurement in enumerate(measurements):
                (folder / f"{frame_id:04d}.json").write_text(json.dumps(measurement))

    def test_validity_preserves_original_frame_ids_and_legacy_data(self):
        frames = routes.get_route_frames(self.root, ["alpha"])
        self.assertEqual([frame for _, frame in frames], [0, 2])

    def test_separate_validation_index(self):
        (self.root / "dataset_index_test.txt").write_text("beta/ 1\n")
        frames = routes.get_route_frames(self.root, ["beta"], index_filename="dataset_index_test.txt")
        self.assertEqual(frames, [(str(self.root / "data" / "beta"), 0)])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            routes.get_route_frames(self.root, ["alpha"], index_filename="dataset_index_test.txt")

    def test_missing_validation_index_does_not_fall_back(self):
        with self.assertRaises(FileNotFoundError):
            routes.get_route_frames(self.root, ["beta"], index_filename="dataset_index_test.txt")

    def test_data_prefix_and_split_selection(self):
        frames = routes.get_route_frames(self.root, ["data/beta/"])
        self.assertEqual(frames, [(str(self.root / "data" / "beta"), 0)])

    def test_overlap_alias_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            routes.validate_route_split(["alpha"], ["data/alpha/"])

    def test_missing_split_and_integer_towns_are_rejected(self):
        for train, val in [(None, ["beta"]), ([0], [1])]:
            with self.assertRaises(ValueError):
                routes.validate_route_split(train, val)

    def test_split_routes_can_be_derived_from_indexes(self):
        (self.root / "dataset_index.txt").write_text("alpha/ 3\n")
        (self.root / "dataset_index_test.txt").write_text("data/beta/ 1\n")
        self.assertEqual(routes.resolve_route_split(
            self.root, "dataset_index.txt", "dataset_index_test.txt"),
            (["alpha"], ["beta"]))

    def test_shared_index_requires_disjoint_explicit_selection(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            routes.resolve_route_split(self.root, "dataset_index.txt", "dataset_index.txt")
        self.assertEqual(routes.resolve_route_split(
            self.root, "dataset_index.txt", "dataset_index.txt", ["alpha"], ["beta"]),
            (["alpha"], ["beta"]))

    def test_disjoint_routes_are_accepted(self):
        routes.validate_route_split(["alpha"], ["beta"])

    def test_unknown_route_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            routes.get_route_frames(self.root, ["typo"])

    def test_all_invalid_is_rejected(self):
        path = self.root / "data" / "beta" / "measurements_full" / "0000.json"
        path.write_text('{"navigation": {"route_valid": false}}')
        with self.assertRaisesRegex(ValueError, "No valid"):
            routes.get_route_frames(self.root, ["beta"])

    def test_missing_measurement_does_not_fall_back(self):
        (self.root / "data" / "alpha" / "measurements_full" / "0001.json").unlink()
        with self.assertRaises(FileNotFoundError):
            routes.get_route_frames(self.root, ["alpha"])


@unittest.skipIf(np is None, "NumPy is required for LiDAR regression checks")
class LidarTests(unittest.TestCase):
    def setUp(self):
        self.preprocess = load_module("morai_utils").lidar_to_raw_features

    def test_more_than_40000_points(self):
        points = np.tile([10, 2, 3, 0.5], (50001, 1)).astype(np.float32)
        result, count = self.preprocess(points)
        self.assertEqual(result.shape, (40000, 4))
        self.assertEqual(count, 40000)
        np.testing.assert_array_equal(result, points[:40000])

    def test_axes_and_padding(self):
        points = np.array([[10, 2, 3, 0.5]], dtype=np.float32)
        result, count = self.preprocess(points)
        self.assertEqual(count, 1)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0], points[0])
        self.assertFalse(result[1:].any())

    def test_intensity_is_not_clipped_with_coordinates(self):
        points = np.array([[150, -150, 3, 240]], dtype=np.float32)
        result, count = self.preprocess(points)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(result[0], [0, 0, 3, 240])

    def test_empty_cloud(self):
        result, count = self.preprocess(np.empty((0, 4), dtype=np.float32))
        self.assertEqual(count, 0)
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()
