import csv
import importlib.util
import io
import math
from pathlib import Path
import unittest
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[3] / "record_static_route.py"
SPEC = importlib.util.spec_from_file_location("static_route_recorder", SCRIPT)
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def point(x=-59.65, y=-159.06, stamp=1.0):
    msg = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=29.96), heading=90.0,
        velocity=SimpleNamespace(x=1.0),
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp)))
    return recorder.sample_message(msg)


class RecorderTests(unittest.TestCase):
    def test_raw_enu_and_heading(self):
        sample = point()
        self.assertEqual((sample["x"], sample["y"], sample["z"]), (-59.65, -159.06, 29.96))
        self.assertAlmostEqual(sample["yaw_rad"], math.pi / 2)

    def test_spacing_and_last_endpoint(self):
        stream = io.StringIO()
        writer = recorder.RouteWriter(stream)
        writer.add(point(), 0.0)
        writer.add(point(y=-158.96), 0.1)
        writer.add(point(y=-158.76), 0.2)
        writer.add(point(y=-158.71), 0.3)
        self.assertEqual(writer.finish(), 3)
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertAlmostEqual(float(rows[-1]["y"]), -158.71)
        self.assertAlmostEqual(float(rows[-1]["s_m"]), 0.35)

    def test_parked_no_duplicate_endpoint(self):
        writer = recorder.RouteWriter(io.StringIO())
        writer.add(point(), 0)
        writer.add(point(), 0.1)
        self.assertEqual(writer.finish(), 1)

    def test_discontinuities_do_not_append(self):
        for changed, received in ((point(x=0), 0.1), (point(), 3), (point(stamp=0), 0.1)):
            writer = recorder.RouteWriter(io.StringIO())
            writer.add(point(), 0)
            with self.assertRaises(ValueError):
                writer.add(changed, received)
            self.assertEqual(writer.finish(), 1)

    def test_names_and_spacing(self):
        for name in ("../route", "a/b", "", "x.csv"):
            with self.assertRaises(ValueError):
                recorder.route_file(Path("routes"), name)
        self.assertEqual(recorder.route_file(Path("routes"), "right_01").name, "right_01.csv")
        for spacing in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                recorder.RouteWriter(io.StringIO(), spacing)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            point(x=float("nan"))


if __name__ == "__main__":
    unittest.main()
