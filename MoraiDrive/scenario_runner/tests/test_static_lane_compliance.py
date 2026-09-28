import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ..core.full_loop_drive_base import FullLoopDriveBase
from ..tools import static_lane_compliance as lane


FIRST = lane.SEGMENT_INDEX[0]
# A straight northbound official route: index i sits 0.5 m north of i-1.
ROUTE = (np.array([[0.0, 0.5 * k] for k in range(lane.SEGMENT_INDEX[1] - FIRST + 1)]), FIRST)
NORTH = math.pi / 2


def line(idx, lane_type, shape, points, color="white"):
    return {"idx": idx, "lane_type": [lane_type], "lane_color": [color],
            "lane_shape": [shape], "lane_width_m": 0.15, "points_xy": points}


def markings(*lines):
    return {"schema": "morai_static_segment_lane_markings_v1", "source": {"sha256": "x"},
            "lines": list(lines)}


# Yellow centre 1.7 m left, white edge 1.7 m right, and one stop line.
ROAD = markings(
    line("Y", 501, "solid", [[-1.7, 0.0], [-1.7, 60.0]], "yellow"),
    line("G", 525, "broken", [[-1.7, 60.0], [-1.7, 80.0]]),
    line("Y2", 501, "solid", [[-1.7, 80.0], [-1.7, 140.0]], "yellow"),
    line("W", 505, "solid", [[1.7, 0.0], [1.7, 140.0]]),
    line("S", 530, "solid", [[-1.7, 30.0], [1.7, 30.0]]),
    line("B", 503, "broken", [[5.0, 0.0], [5.0, 140.0]]),
)


def straight(x, y0=0.0, y1=134.0, heading=NORTH, step=0.5):
    return [(x, y, heading) for y in np.arange(y0, y1, step)]


def swing(peak_m, start=45.0, end=75.0, ramp=10.0):
    """Move right by peak_m between start and end with linear ramps."""
    rows = []
    for x, y, t in straight(0.0):
        weight = min(max((y - start) / ramp, 0.0), 1.0, max((end - y) / ramp, 0.0))
        rows.append((peak_m * weight, y, t))
    return rows


class StaticLaneComplianceTest(unittest.TestCase):
    def test_no_cross_lines_are_solid_lines_and_centre_guides_but_not_stop_lines(self):
        _, _, labels = lane.no_cross_segments(ROAD)
        kinds = {label.split(":")[0]: label.split(":")[2] for label in labels}
        self.assertEqual(kinds, {"Y": "solid", "G": "centre_guide", "Y2": "solid", "W": "solid"})

    def test_centred_lane_keeping_passes(self):
        result = lane.evaluate_trace(straight(0.0), ROAD, ROUTE)
        self.assertTrue(result["passed"], result["reasons"])
        # 1.7 m to the line centre - 0.945 m half body - 0.075 m half paint.
        self.assertAlmostEqual(result["min_marking_clearance_m"], 0.68, places=2)
        self.assertEqual(result["avoidance_class"], "small")

    def test_body_on_the_paint_is_rejected_on_either_side(self):
        # x grows to the east; heading north keeps +x on the Ego's right.
        right = lane.evaluate_trace(straight(0.70), ROAD, ROUTE)
        self.assertIn("solid_lane_marking_touched", right["reasons"])
        self.assertTrue(right["touched_markings"][0]["marking"].startswith("W:"))
        inside = lane.evaluate_trace(straight(0.66), ROAD, ROUTE)
        self.assertTrue(inside["passed"], inside["reasons"])
        # Crossing the intersection guide that continues the centre line counts.
        left = [(x if not 62.0 <= y <= 70.0 else -0.8, y, t) for x, y, t in straight(0.0)]
        result = lane.evaluate_trace(left, ROAD, ROUTE)
        self.assertEqual([row["marking"].split(":")[0] for row in result["touched_markings"]], ["G"])

    def test_swing_beyond_the_limit_is_too_far_and_classified(self):
        road = markings(line("Y", 501, "solid", [[-1.7, 0.0], [-1.7, 140.0]], "yellow"))
        medium = lane.evaluate_trace(swing(3.6), road, ROUTE)
        self.assertTrue(medium["passed"], medium["reasons"])
        self.assertEqual(medium["avoidance_class"], "medium")
        result = lane.evaluate_trace(swing(4.5), road, ROUTE)
        self.assertIn("lateral_deviation_limit_exceeded", result["reasons"])
        self.assertEqual(result["avoidance_class"], "large")

    def test_samples_after_the_goal_are_not_a_deviation(self):
        # After the goal the Ego may drive on; those samples project onto the end point.
        trace = straight(0.0) + [(0.3 * k, 134.0 + 0.3 * k, NORTH) for k in range(20)]
        result = lane.evaluate_trace(trace, markings(), ROUTE)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertLess(result["max_right_deviation_m"], 0.1)

    def test_missing_incomplete_or_gapped_traces_fail_closed(self):
        self.assertEqual(lane.evaluate_trace([], ROAD, ROUTE)["reasons"], ["ego_trace_missing"])
        short = lane.evaluate_trace(straight(0.0, 0.0, 40.0), ROAD, ROUTE)
        self.assertIn("ego_trace_incomplete", short["reasons"])
        gap = [row for row in straight(0.0) if not 70.0 < row[1] < 75.0]
        self.assertIn("ego_trace_gap", lane.evaluate_trace(gap, ROAD, ROUTE)["reasons"])

    def test_active_signal_rows_collapse_to_one_pose_per_source_time(self):
        row = {"role": "ego", "source_timestamp_ns": 2,
               "signal": {"position": [1.0, 2.0, 0.0], "yaw": 0.5}}
        rows = [row, dict(row), dict(row, source_timestamp_ns=1), {"role": "collision"}]
        self.assertEqual(lane.ego_trace_from_active_signals(rows),
                         [(1.0, 2.0, 0.5), (1.0, 2.0, 0.5)])

    def test_pinned_kcity_markings_match_the_route_and_centre_line(self):
        pinned = lane.load_markings()
        route = lane.load_route()
        self.assertEqual(pinned["route"]["segment_index"], list(lane.SEGMENT_INDEX))
        self.assertEqual(pinned["route"]["sha256"], lane.sha256_file(lane.DEFAULT_ROUTE))
        for index in (660, 670, 680):
            self.assertAlmostEqual(lane.centre_line_lateral(pinned, *route, index), 1.71, delta=0.05)

    def test_runner_writes_one_trace_row_per_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace" / "ego.csv"
            scenario = SimpleNamespace(cfg={"ego_trace_path": str(path)})
            for elapsed in (0.0, 0.05):
                FullLoopDriveBase.record_ego_trace(
                    scenario, elapsed, {"x": 1.0, "y": 2.0 + elapsed, "yaw_deg": 90.0, "speed": 3.0})
            scenario._ego_trace_stream.close()
            rows = lane.read_trace_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[1][1], 2.05)
            self.assertAlmostEqual(rows[0][2], NORTH)
            silent = SimpleNamespace(cfg={})
            FullLoopDriveBase.record_ego_trace(silent, 0.0, {})
            self.assertFalse(hasattr(silent, "_ego_trace_stream"))


if __name__ == "__main__":
    unittest.main()
