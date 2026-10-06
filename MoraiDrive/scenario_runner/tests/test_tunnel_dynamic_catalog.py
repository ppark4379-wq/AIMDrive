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
            "positions_from_entry_m": [25.0, 40.0],
            "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5,
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

    def test_five_behaviors_per_position_and_review_status(self):
        paths = generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        self.assertEqual(len(paths), 10)
        by_id = {item["id"]: item for item in paths}
        self.assertEqual(by_id["straight_same_direction_01"]["status"], "approved")
        self.assertEqual(by_id["diagonal_cross_02"]["status"], "disabled")
        self.assertEqual(by_id["cross_left_to_right_01"]["status"], "candidate")
        crossing = by_id["cross_left_to_right_01"]
        self.assertEqual(crossing["start_xyz"]["y"], 2.5)
        self.assertEqual(crossing["end_xyz"]["y"], -2.5)

    def test_rejects_outside_margin_and_unknown_approval(self):
        self.spec["positions_from_entry_m"] = [70.0]
        with self.assertRaisesRegex(ValueError, "outside the tunnel margin"):
            generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)
        self.spec["positions_from_entry_m"] = [25.0]
        with self.assertRaisesRegex(ValueError, "Unknown tunnel candidate IDs"):
            generate_tunnel_candidates(self.route, 10.0, 90.0, self.spec)

    def test_full_route_rounding_at_window_end_is_accepted(self):
        self.spec["positions_from_entry_m"] = [25.0]
        self.spec["disabled_ids"] = []
        paths = generate_tunnel_candidates(
            self.route, 10.0, 100.0 + 1e-10, self.spec)
        self.assertEqual(len(paths), 5)


if __name__ == "__main__":
    unittest.main()
