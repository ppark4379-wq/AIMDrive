"""Broad placements preserve native geometry and all shared geometry gates."""

import copy
import math
import unittest

from .test_tunnel_static_layout import connected_sample_map
from ..scenarios.full_loop_parts.tunnel_static_parts.layout import (
    OPPOSING_LINK, TUNNEL_LINK, bbox_world, boxes_overlap,
    expanded_road_boundary, normalize_footprint, route_pose, tunnel_geometry, validate_candidate,
)
from ..scenarios.full_loop_parts.tunnel_static_parts.sampling import (
    SCHEMA, SamplingExhausted, generate_sampled_candidate, layout_signature,
)


def native_profile(width=1.2, length=2., *, scale=None):
    row = {"width_m": width, "length_m": length,
           "center_offset_spawn_xy": [1.3, -.21], "heading_offset_deg": 1.347,
           "geometry_verified": True, "geometry_source": {"type": "test_reviewed_native_geometry"},
           "expected_api_raw": {"size_lwh_m": [length, width, 1.5],
                                "center_offset_actor_flu_m": [1., .2, .3]}}
    if scale is not None:
        row["scale_xyz"] = scale
    return row


class TunnelStaticSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = connected_sample_map()
        cls.boundary = expanded_road_boundary(cls.loader)

    def candidate(self, seed=42, **kwargs):
        arguments = {"count": 2, "_road_boundary": self.boundary}
        arguments.update(kwargs)
        return generate_sampled_candidate(self.loader, {"Box": native_profile()},
                                          {"Car": native_profile(1.8, 4.5)}, seed, **arguments)

    def test_deterministic_and_catalog_order_independent_without_modifying_inputs(self):
        statics = {"BoxB": native_profile(), "BoxA": native_profile(.9, 1.6)}
        cars = {"CarB": native_profile(2., 4.), "CarA": native_profile(1.8, 4.5)}
        before = copy.deepcopy((statics, cars))
        arguments = {"count": 3, "allow_three": True, "_road_boundary": self.boundary}
        first = generate_sampled_candidate(self.loader, statics, cars, 63, **arguments)
        second = generate_sampled_candidate(self.loader, dict(reversed(list(statics.items()))),
                                           dict(reversed(list(cars.items()))), 63, **arguments)
        self.assertEqual(first, second)
        self.assertEqual((statics, cars), before)
        self.assertNotEqual(first, generate_sampled_candidate(self.loader, statics, cars, 63,
                            candidate_index=1, **arguments))

    def test_stations_independent_gaps_and_laterals_cover_the_broad_domain(self):
        stations, laterals, gaps = [], [], []
        unequal_gaps = 0
        for seed in range(120):
            candidate = self.candidate(seed, count=3, allow_three=True)
            checked = validate_candidate(candidate, self.loader, _road_boundary=self.boundary)
            actors = candidate["obstacles"]
            stations.extend(row["station_m"] for row in actors)
            laterals.extend(row["lateral_m"] for row in actors)
            adjacent = [b["station_m"]-a["station_m"] for a, b in zip(actors, actors[1:])]
            gaps.extend(adjacent)
            unequal_gaps += abs(adjacent[0]-adjacent[1]) > 5.
            self.assertEqual(checked["placement_geometry"]["all_pair_obb_check_count"], 3)
            self.assertTrue(all(0. <= gap <= 55. for gap in adjacent))
        self.assertLess(min(stations), 25.)
        self.assertGreater(max(stations), 100.)
        self.assertLess(min(laterals), -.7)
        self.assertGreater(max(laterals), 4.2)
        self.assertTrue(any(.5 < lateral < 1.4 for lateral in laterals))
        self.assertTrue(any(2.1 < lateral < 3. for lateral in laterals))
        self.assertLess(min(gaps), 8.)
        self.assertGreater(max(gaps), 45.)
        self.assertGreater(unequal_gaps, 70)

    def test_native_lane_yaw_and_pivot_compensation_work_on_a_rotated_map(self):
        loader = connected_sample_map()
        angle = math.radians(37.)
        for link in loader.links.values():
            for point in link["points"]:
                x, y = point[:2]
                point[0], point[1] = (100.+x*math.cos(angle)-y*math.sin(angle),
                                      -40.+x*math.sin(angle)+y*math.cos(angle))
        boundary = expanded_road_boundary(loader)
        geometry = tunnel_geometry(loader)
        profile = native_profile(1.8, 4.5)
        for intent, lane, native in (("own_lane_scatter", TUNNEL_LINK, 37.),
                                     ("opposing_lane_scatter", OPPOSING_LINK, 217.)):
            candidate = generate_sampled_candidate(loader, {}, {"Car": profile}, 11, count=1,
                actor_kind="vehicle", placement_intent=intent, _road_boundary=boundary)
            row = candidate["obstacles"][0]
            box = bbox_world(row)
            position, yaw = route_pose(geometry["own_points"], row["station_m"])
            expected = [position[0]-math.sin(yaw)*row["lateral_m"],
                        position[1]+math.cos(yaw)*row["lateral_m"]]
            self.assertLess(math.dist(box["center_xyz"][:2], expected), 1e-10)
            self.assertEqual(row["planned_bbox"], normalize_footprint(profile))
            self.assertEqual(row["expected_api_raw"], profile["expected_api_raw"])
            self.assertEqual(row["placement_lane_id"], lane)
            self.assertAlmostEqual(box["heading_deg"], native)
            self.assertAlmostEqual(row["native_forward_yaw_deg"], native)
            validate_candidate(candidate, loader, _road_boundary=boundary)

    def test_rotated_barriers_keep_their_native_boxes_and_rock_is_disabled(self):
        models = {"Steel_Barricade": native_profile(3.01, 1.08),
                  "concrete-barrier": native_profile(4., .8), "Rockslide": native_profile()}
        seen = set()
        for seed in range(15):
            candidate = generate_sampled_candidate(self.loader, models, {}, seed, count=1,
                max_width_m=1.1, _road_boundary=self.boundary)
            row = candidate["obstacles"][0]
            seen.add(row["model"])
            self.assertEqual(row["scale"], [1., 1., 1.])
            self.assertEqual(row["static_yaw_offset_deg"], 90.)
            self.assertEqual(row["planned_bbox"], normalize_footprint(models[row["model"]]))
            self.assertEqual(row["expected_api_raw"], models[row["model"]]["expected_api_raw"])
        self.assertEqual(seen, {"Steel_Barricade", "concrete-barrier"})
        for requested in ({"Rockslide": [0.]}, {"Steel_Barricade": [0.]}, {"Box": [30.]}):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "native permitted"):
                self.candidate(static_yaw_offsets_deg=requested)

    def test_three_actors_require_explicit_count_and_permission(self):
        with self.assertRaisesRegex(ValueError, "explicit count"):
            self.candidate(count=3)
        for seed in range(12):
            self.assertIn(self.candidate(seed, count=None, allow_three=True)["count"], (1, 2))
        candidate = self.candidate(count=3, allow_three=True)
        self.assertTrue(candidate["experimental_three_actor_trial"])
        candidate["experimental_three_actor_trial"] = False
        with self.assertRaisesRegex(ValueError, "experimental_three_actor_trial"):
            validate_candidate(candidate, self.loader, _road_boundary=self.boundary)

    def test_same_station_windows_require_every_pair_to_be_clear(self):
        for count in (2, 3):
            candidate = self.candidate(count=count, allow_three=count == 3,
                                       placement_intent="same_station_window")
            actors = candidate["obstacles"]
            self.assertEqual(actors[0]["station_m"], actors[1]["station_m"])
            self.assertNotEqual(actors[0]["placement_lane_id"], actors[1]["placement_lane_id"])
            self.assertEqual(candidate["placement"]["geometry"]["minimum_center_gap_m"], 0.)
            boxes = [bbox_world(row) for row in actors]
            for first in range(count):
                for last in range(first+1, count):
                    self.assertFalse(boxes_overlap(boxes[first], boxes[last], margin_m=.5))
            bad = copy.deepcopy(candidate)
            bad["placement"].pop("geometry")
            bad["obstacles"][-1] = copy.deepcopy(bad["obstacles"][0])
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_candidate(bad, self.loader, _road_boundary=self.boundary)

    def test_geometry_rejection_is_bounded_without_shrinking_or_faking_shapes(self):
        wide = {"Wide": native_profile(4., 2.)}
        before = copy.deepcopy(wide)
        with self.assertRaises(SamplingExhausted) as raised:
            generate_sampled_candidate(self.loader, wide, {}, 71, count=2,
                placement_intent="same_station_window", max_attempts=7, _road_boundary=self.boundary)
        self.assertEqual(raised.exception.attempts, 7)
        self.assertIn("overlap", raised.exception.last_rejection)
        self.assertEqual(wide, before)
        with self.assertRaises(SamplingExhausted) as raised:
            generate_sampled_candidate(self.loader, {"Long": native_profile(1., 12.)}, {}, 71,
                count=1, station_window_m=(.1, .2), max_attempts=3, _road_boundary=self.boundary)
        self.assertEqual(raised.exception.attempts, 3)

    def test_unreviewed_unknown_or_scaled_profiles_are_excluded(self):
        invalid = [native_profile(scale=[2., 1., 1.]), native_profile(scale=[True, 1., 1.]),
                   {"width_m": 1., "length_m": 1.},
                   dict(native_profile(), geometry_verified=False),
                   dict(native_profile(), geometry_source={})]
        for profile in invalid:
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "No verified native"):
                generate_sampled_candidate(self.loader, {"Unknown": profile}, {}, 2,
                    count=1, _road_boundary=self.boundary)

    def test_sampler_marker_lane_yaw_road_and_portal_constraints_are_hard(self):
        candidate = self.candidate(count=1, actor_kind="vehicle", placement_intent="opposing_lane_scatter")
        self.assertEqual(candidate["placement"]["sampling_schema"], SCHEMA)
        self.assertFalse(candidate["physical_walls_verified"])
        self.assertFalse(candidate["expert_verified"])
        self.assertFalse(candidate["live_avoidance_verified"])
        self.assertTrue(candidate["placement"]["actual_jamming_path_required"])
        for kind in ("marker", "domain", "lane", "yaw", "road", "portal"):
            bad = copy.deepcopy(candidate)
            bad["placement"].pop("geometry")
            row = bad["obstacles"][0]
            if kind == "marker":
                bad["placement"].pop("sampling_schema")
            elif kind == "domain":
                bad.pop("experimental_placement_domain")
            elif kind == "lane":
                row["placement_lane_id"] = TUNNEL_LINK
            elif kind == "yaw":
                row["yaw_deg"] += 180.
            elif kind == "road":
                row["planned_bbox"]["width_m"] = 8.
            else:
                row["position"]["x"] += .1-row["station_m"]
                row["station_m"] = .1
                bad["placement"]["station_min_m"] = .1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_candidate(bad, self.loader, _road_boundary=self.boundary)

    def test_signatures_deduplicate_geometry_independent_of_seed_order_and_intent(self):
        candidate = self.candidate()
        signature = layout_signature(candidate)
        self.assertEqual(candidate["placement"]["layout_signature"], signature)
        duplicate = copy.deepcopy(candidate)
        duplicate["seed"] += 1
        duplicate["candidate_index"] += 7
        duplicate["placement"]["placement_intent"] = "another_intent"
        duplicate["obstacles"].reverse()
        duplicate["obstacles"][0]["yaw_deg"] += 360.
        self.assertEqual(layout_signature(duplicate), signature)
        for field in ("model", "station_m", "lateral_m", "yaw_deg"):
            changed = copy.deepcopy(candidate)
            changed["obstacles"][0][field] = ("another_model" if field == "model"
                                              else changed["obstacles"][0][field]+.1)
            with self.subTest(field=field):
                self.assertNotEqual(layout_signature(changed), signature)


if __name__ == "__main__":
    unittest.main()
