"""Route-relative tunnel candidate geometry and review status."""

import unittest

from ..scenarios.full_loop_parts.tunnel_dynamic_catalog import (
    generate_tunnel_candidates,
    route_offset_point,
)


class TunnelDynamicCatalogTest(unittest.TestCase):
    def setUp(self):
        self.route = [(float(x), 0.0, 10.0) for x in range(101)]
        self.spec = {
            "division_count": 4,
            "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5,
            "opposite_lane_1_offset_m": 2.5,
            "opposite_lane_2_offset_m": 0.0,
            "straight_length_m": 15.0,
            "cross_delta_s_m": 3.0,
            "diagonal_delta_s_m": 8.0,
            "approved_ids": ["straight_same_direction_01"],
            "disabled_ids": ["diagonal_cross_02"],
        }

    def test_left_positive_offset_and_route_height(self):
        self.assertEqual(route_offset_point(self.route, 35.0, 2.5),
                         {"x": 35.0, "y": 2.5, "z": 10.0})
        self.assertEqual(route_offset_point(self.route, 35.0, -2.5),
                         {"x": 35.0, "y": -2.5, "z": 10.0})

    def test_four_centered_sections_and_review_status(self):
        paths = generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        self.assertEqual(len(paths), 20)
        by_id = {item["id"]: item for item in paths}
        self.assertEqual(by_id["straight_same_direction_01"]["status"], "approved")
        self.assertEqual(by_id["diagonal_cross_02"]["status"], "disabled")
        self.assertEqual(by_id["cross_left_to_right_01"]["status"], "candidate")
        crossing = by_id["cross_left_to_right_01"]
        self.assertEqual(crossing["start_xyz"]["y"], 2.5)
        self.assertEqual(crossing["end_xyz"]["y"], -2.5)
        self.assertEqual([by_id[f"straight_same_direction_{i:02d}"]["anchor_s_m"]
                          for i in range(1, 5)], [27.5, 42.5, 57.5, 72.5])
        for path in paths:
            start, end = path["start_sd"]["s_m"], path["end_sd"]["s_m"]
            self.assertEqual((start + end) / 2, path["anchor_s_m"])
            self.assertGreaterEqual(min(start, end), path["section_s_range_m"]["start"])
            self.assertLessEqual(max(start, end), path["section_s_range_m"]["end"])
        reverse = by_id["straight_opposite_01"]
        self.assertGreater(reverse["start_sd"]["s_m"], reverse["end_sd"]["s_m"])

    def test_rejects_oversized_path_and_unknown_approval(self):
        self.spec["straight_length_m"] = 16.0
        with self.assertRaisesRegex(ValueError, "exceeds a tunnel section"):
            generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        self.spec["straight_length_m"] = 15.0
        self.spec["disabled_ids"] = ["diagonal_cross_05"]
        with self.assertRaisesRegex(ValueError, "Unknown tunnel candidate IDs"):
            generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)

    def test_opposite_paths_alternate_lanes_and_run_backwards(self):
        paths = generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        reverse = [item for item in paths if item["behavior"] == "straight_opposite"]
        self.assertEqual([item["lane_index"] for item in reverse], [1, 2, 1, 2])
        self.assertEqual([item["start_sd"]["d_m"] for item in reverse],
                         [2.5, 0.0, 2.5, 0.0])
        for item in reverse:
            self.assertEqual(item["start_sd"]["d_m"], item["end_sd"]["d_m"])
            self.assertGreater(item["start_sd"]["s_m"], item["end_sd"]["s_m"])
            self.assertGreater(item["start_xyz"]["x"], item["end_xyz"]["x"])

    def test_opposite_lane_offsets_are_independent_of_side_offset(self):
        self.spec.update(opposite_lane_1_offset_m=2.0, opposite_lane_2_offset_m=0.5)
        paths = generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        by_id = {item["id"]: item for item in paths}
        self.assertEqual(by_id["straight_opposite_01"]["start_xyz"]["y"], 2.0)
        self.assertEqual(by_id["straight_opposite_02"]["start_xyz"]["y"], 0.5)
        self.assertEqual(by_id["straight_same_direction_01"]["start_xyz"]["y"], 2.5)
        self.assertEqual(by_id["cross_left_to_right_01"]["end_xyz"]["y"], -2.5)

    def test_rejects_invalid_divisions_and_excessive_margin(self):
        for divisions in (0, -1, 1.5, True):
            with self.subTest(divisions=divisions):
                self.spec["division_count"] = divisions
                with self.assertRaisesRegex(ValueError, "division_count"):
                    generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        self.spec["division_count"] = 4
        self.spec["endpoint_margin_m"] = 40.0
        with self.assertRaisesRegex(ValueError, "no usable tunnel length"):
            generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)

    def test_anchors_follow_changed_tunnel_length(self):
        paths = generate_tunnel_candidates(self.route, 0.0, 100.0, self.spec)
        self.assertEqual([item["anchor_s_m"] for item in paths
                          if item["behavior"] == "straight_same_direction"],
                         [20.0, 40.0, 60.0, 80.0])

    def test_full_route_rounding_at_window_end_is_accepted(self):
        paths = generate_tunnel_candidates(
            self.route, 10.0, 100.0 + 1e-10, self.spec)
        self.assertEqual(len(paths), 20)


if __name__ == "__main__":
    unittest.main()
