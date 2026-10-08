"""Candidate generation must preserve map, full geometry and reproducibility."""

import copy
import hashlib
import json
import math
from types import SimpleNamespace
import unittest

from ..scenarios.full_loop_parts.tunnel_static_parts.layout import (OPPOSING_LINK, TUNNEL_LINK, bbox_world,
    boxes_overlap, footprint_corners, generate_candidate, normalized_geometry,
    project_polyline, validate_candidate, projected_footprint_dimensions,
    expanded_region_width_limit)


def sample_map():
    links = {
        TUNNEL_LINK:{"road_type":"2","width_start":3.5,"width_end":3.5,
                     "points":[[float(x),0.,28.] for x in range(121)]},
        OPPOSING_LINK:{"road_type":"2","width_start":3.5,"width_end":3.5,
                       "points":[[float(x),3.5,28.] for x in reversed(range(121))]},
    }
    return SimpleNamespace(get_link=lambda identity:links[identity],links=links)


def static_models():
    return {"models":{"WoodBox":{"width_m":1.2,"length_m":1.5,
                "center_offset_spawn_xy":[.4,-.12],"heading_offset_deg":1.347},
                      "TooWide":{"width_m":4.,"length_m":2.}}}


def vehicle_models():
    return {"VerifiedCar":{"width_m":2.,"length_m":4.5,"geometry_verified":True,
                           "center_offset_spawn_xy":[1.8,.05],"heading_offset_deg":1.347},
            "UnverifiedCar":{"width_m":1.8,"length_m":4.}}


class TunnelStaticLayoutTests(unittest.TestCase):
    def candidate(self,**kwargs):
        return generate_candidate(sample_map(),static_models(),vehicle_models(),seed=42,**kwargs)

    def test_reproducible_and_catalog_iteration_order_independent(self):
        first = self.candidate(count=2)
        vehicle = dict(reversed(list(vehicle_models().items())))
        second = generate_candidate(sample_map(),static_models(),vehicle,seed=42,count=2)
        self.assertEqual(first,second)
        self.assertNotEqual(first,self.candidate(count=2,candidate_index=1))
        self.assertFalse(first["expert_verified"])
        self.assertFalse(first["live_avoidance_verified"])

    def test_legacy_default_seed_documents_keep_their_exact_golden_hashes(self):
        expected = {None: "983ca8961d21e024e95f5858be40b8629b1255d082b31dcf7b736c2f5fbfeba5",
                    1: "7f1ad8f3c9761c766b0b48c6aa2fc343e40467abbbef245d3d14f9b776a2a7b1",
                    2: "2672566c2cecf40187e961b855c8f9332665db56f516aadfdbe41d1273063fc0"}
        for count, digest in expected.items():
            candidate = self.candidate(count=count)
            actual = hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            self.assertEqual(actual,digest)
            self.assertNotIn("experimental_placement_domain",candidate)

    def test_pivot_and_heading_offsets_are_compensated_in_spawn_frame(self):
        candidate = self.candidate(count=1,actor_kind="vehicle")
        obstacle = candidate["obstacles"][0]
        box = bbox_world(obstacle)
        self.assertAlmostEqual(box["heading_deg"],0.,places=10)
        self.assertAlmostEqual(box["center_xyz"][0],obstacle["station_m"],places=10)
        self.assertAlmostEqual(box["center_xyz"][1],obstacle["lateral_m"],places=10)
        self.assertNotAlmostEqual(obstacle["position"]["x"],obstacle["station_m"],places=2)

    def test_random_mixed_pairs_keep_opposite_lane_clear_and_fit_whole_box(self):
        seen = set()
        loader = sample_map()
        for seed in range(30):
            candidate = generate_candidate(loader,static_models(),vehicle_models(),seed,count=2)
            geometry = validate_candidate(candidate,loader)
            a,b = candidate["obstacles"]
            self.assertGreaterEqual(b["station_m"]-a["station_m"],18.)
            self.assertFalse(boxes_overlap(bbox_world(a),bbox_world(b),margin_m=.5))
            for obstacle in candidate["obstacles"]:
                seen.add(obstacle["actor_kind"])
                self.assertNotIn(obstacle["model"],("TooWide","UnverifiedCar"))
                box = bbox_world(obstacle)
                for corner in footprint_corners(box["center_xyz"],box["width_m"],box["length_m"],box["heading_deg"]):
                    _,lat,_ = project_polyline(geometry["own_points"],*corner)
                    self.assertLessEqual(abs(lat),1.6)
        self.assertEqual(seen,{"obstacle","vehicle"})

    def test_rejects_nonfinite_invalid_counts_and_unverified_only_vehicles(self):
        for kwargs in ({"count":3},{"count":True},{"station_min_m":float("nan")},
                       {"station_max_m":float("inf")},{"min_pair_gap_m":-2.},
                       {"lateral_limit_m":.4}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.candidate(**kwargs)
        with self.assertRaisesRegex(ValueError,"No eligible"):
            generate_candidate(sample_map(),{}, {"Unverified":{"width_m":2.,"length_m":4.}},
                               1,count=1,actor_kind="vehicle")

    def test_source_map_mutation_and_actor_outside_lane_are_rejected(self):
        candidate = self.candidate(count=1)
        loader = sample_map()
        loader.links[TUNNEL_LINK]["points"][0][2] += 1.
        with self.assertRaisesRegex(ValueError,"source geometry"):
            validate_candidate(candidate,loader)
        changed = copy.deepcopy(candidate)
        changed["obstacles"][0]["position"]["y"] += 2.
        with self.assertRaisesRegex(ValueError,"station/lateral"):
            validate_candidate(changed,sample_map())

    def test_measured_override_cannot_shrink_reviewed_geometry(self):
        candidate = self.candidate(count=1,actor_kind="vehicle")
        planned = bbox_world(candidate["obstacles"][0])
        measured = dict(planned,width_m=.1,length_m=.1)
        union = normalized_geometry(candidate,[measured])[0]
        self.assertGreaterEqual(union["width_m"],planned["width_m"]-1e-10)
        self.assertGreaterEqual(union["length_m"],planned["length_m"]-1e-10)
        measured["center_xyz"] = [planned["center_xyz"][0]+10., planned["center_xyz"][1],28.]
        union = normalized_geometry(candidate,[measured])[0]
        self.assertGreater(union["length_m"],10.)

    def test_vehicle_runtime_provenance_survives_generation_without_aliasing_catalog(self):
        vehicles = vehicle_models()
        vehicles["VerifiedCar"].update(
            geometry_source={"registry_sha256":"sealed-profile"},
            expected_api_raw={"size_lwh_m":[4.,2.,1.5],"center_offset_actor_flu_m":[1.,0.,.5]})
        candidate = generate_candidate(sample_map(),{},vehicles,seed=1,count=1,actor_kind="vehicle")
        item = candidate["obstacles"][0]
        self.assertEqual(item["geometry_source"],{"registry_sha256":"sealed-profile"})
        self.assertEqual(item["expected_api_raw"]["size_lwh_m"],[4.,2.,1.5])
        item["expected_api_raw"]["size_lwh_m"][0] = 99.
        self.assertEqual(vehicles["VerifiedCar"]["expected_api_raw"]["size_lwh_m"][0],4.)

    def test_three_actor_trials_are_explicit_and_default_stays_one_or_two(self):
        candidate = self.candidate(count=3,allow_three=True)
        self.assertIs(candidate["experimental_three_actor_trial"],True)
        self.assertEqual(len(candidate["obstacles"]),3)
        self.assertFalse(candidate["expert_verified"])
        for seed in range(20):
            ordinary = generate_candidate(sample_map(),static_models(),vehicle_models(),seed,allow_three=True)
            self.assertIn(ordinary["count"],(1,2))
            self.assertNotIn("experimental_three_actor_trial",ordinary)
        for marker in (None,False,"true",1):
            modified = copy.deepcopy(candidate)
            if marker is None:
                modified.pop("experimental_three_actor_trial")
            else:
                modified["experimental_three_actor_trial"] = marker
            with self.subTest(marker=marker),self.assertRaisesRegex(ValueError,"experimental_three_actor_trial"):
                validate_candidate(modified,sample_map())
        with self.assertRaisesRegex(ValueError,"boolean"):
            self.candidate(count=3,allow_three="true")

    def test_three_actor_spacing_uses_residual_window_and_complete_corner_extents(self):
        loader = sample_map()
        for seed in range(20):
            candidate = generate_candidate(loader,static_models(),vehicle_models(),seed,count=3,allow_three=True)
            geometry = validate_candidate(candidate,loader)
            report = candidate["placement"]["geometry"]
            self.assertEqual(report,geometry["placement_geometry"])
            self.assertEqual(report["actor_count"],3)
            self.assertEqual(report["all_pair_obb_check_count"],3)
            self.assertEqual(report["length_budget"]["required_center_span_m"],36.)
            self.assertEqual(report["length_budget"]["center_window_slack_m"],19.)
            self.assertGreaterEqual(report["minimum_center_gap_m"],18.-1e-9)
            self.assertGreater(report["length_budget"]["before_first_corner_m"],0.)
            self.assertGreater(report["length_budget"]["after_last_corner_m"],0.)
            for pair in report["adjacent_pairs"]:
                first,last = (candidate["obstacles"][pair[key]] for key in ("first_index","second_index"))
                expected = pair["center_gap_m"]-.5*(first["planned_bbox"]["length_m"]+last["planned_bbox"]["length_m"])
                self.assertAlmostEqual(pair["longitudinal_surface_gap_m"],expected,places=8)
            for actor in report["actors_in_station_order"]:
                self.assertEqual(len(actor["corners_world_xy"]),4)
                self.assertGreater(actor["corner_station_min_m"],.01)
                self.assertLess(actor["corner_station_max_m"],geometry["length_m"]-.01)

    def test_three_actor_infeasible_gap_range_is_rejected_without_random_retries(self):
        with self.assertRaisesRegex(ValueError,"2 center gaps"):
            self.candidate(count=3,allow_three=True,station_max_m=55.)
        candidate = self.candidate(count=3,allow_three=True,station_max_m=56.,actor_kind="obstacle")
        self.assertEqual([item["station_m"] for item in candidate["obstacles"]],[20.,38.,56.])
        self.assertEqual(candidate["placement"]["geometry"]["length_budget"]["center_window_slack_m"],0.)

    def test_three_actor_validation_checks_nonadjacent_obb_overlap(self):
        candidate = self.candidate(count=3,allow_three=True,actor_kind="obstacle")
        candidate["obstacles"][2] = copy.deepcopy(candidate["obstacles"][0])
        candidate["placement"].pop("geometry")
        with self.assertRaisesRegex(ValueError,"overlap: 0 and 2"):
            validate_candidate(candidate,sample_map())

    def test_third_actor_corners_must_fit_tunnel_end_and_lane(self):
        original = self.candidate(count=3,allow_three=True,actor_kind="obstacle")
        at_end = copy.deepcopy(original)
        at_end["placement"].pop("geometry")
        third = at_end["obstacles"][2]
        third["position"]["x"] += 119.6-third["station_m"]
        third["station_m"] = 119.6
        with self.assertRaisesRegex(ValueError,"whole footprint"):
            validate_candidate(at_end,sample_map())
        at_edge = copy.deepcopy(original)
        at_edge["placement"].pop("geometry")
        third = at_edge["obstacles"][2]
        third["position"]["y"] += 1.1-third["lateral_m"]
        third["lateral_m"] = 1.1
        with self.assertRaisesRegex(ValueError,"whole footprint"):
            validate_candidate(at_edge,sample_map())

    def test_declared_center_gap_and_spacing_metadata_cannot_hide_bad_geometry(self):
        candidate = self.candidate(count=3,allow_three=True,station_max_m=56.,actor_kind="obstacle")
        stale = copy.deepcopy(candidate)
        stale["placement"]["geometry"]["minimum_center_gap_m"] = 100.
        with self.assertRaisesRegex(ValueError,"metadata disagrees"):
            validate_candidate(stale,sample_map())
        close = copy.deepcopy(candidate)
        close["placement"].pop("geometry")
        second = close["obstacles"][1]
        second["position"]["x"] -= 8.
        second["station_m"] -= 8.
        with self.assertRaisesRegex(ValueError,"center gap"):
            validate_candidate(close,sample_map())

    def test_old_one_and_two_actor_candidates_without_new_metadata_remain_valid(self):
        for count in (1,2):
            candidate = self.candidate(count=count)
            placement = candidate["placement"]
            self.assertEqual(placement["station_min_m"],20.)
            self.assertEqual(placement["station_max_m"],75.)
            self.assertEqual(placement["min_pair_gap_m"],18.)
            placement.pop("geometry")
            self.assertEqual(validate_candidate(candidate,sample_map())["placement_geometry"]["actor_count"],count)


def connected_sample_map():
    links = copy.deepcopy(sample_map().links)
    links[TUNNEL_LINK].update(from_node_idx="own_entry",to_node_idx="own_exit")
    links[OPPOSING_LINK].update(from_node_idx="opp_entry",to_node_idx="opp_exit")
    for identity,first,last,y,start,end in (
        ("own_before",-150.,0.,0.,"own_before","own_entry"),
        ("own_after",120.,240.,0.,"own_exit","own_after"),
        ("opp_before",240.,120.,3.5,"opp_before","opp_entry"),
        ("opp_after",0.,-150.,3.5,"opp_exit","opp_after")):
        links[identity] = {"idx":identity,"points":[[first,y,28.],[last,y,28.]],
                          "width_start":3.5,"width_end":3.5,"road_type":"1",
                          "from_node_idx":start,"to_node_idx":end}
    return SimpleNamespace(get_link=lambda identity:links[identity],links=links,link_set=links)


class ExpandedTunnelLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ..scenarios.full_loop_parts.tunnel_static_parts.layout import expanded_road_boundary
            expanded_road_boundary(connected_sample_map())
        except (ImportError,ValueError) as error:
            raise unittest.SkipTest("Expanded road dependencies unavailable: "+str(error))

    def setUp(self):
        self.loader = connected_sample_map()
        self.statics = copy.deepcopy(static_models()["models"])
        for name,row in self.statics.items():
            row.update(geometry_verified=True,geometry_source={"type":"unit_test_native_footprint","model":name})
        self.vehicles = {"VerifiedCar":copy.deepcopy(vehicle_models()["VerifiedCar"])}
        self.vehicles["VerifiedCar"]["geometry_source"] = {"type":"unit_test_vehicle_geometry"}

    def candidate(self,**kwargs):
        arguments = {"footprint_domain":"two_lane_physical_road","pattern":"short_detour",
                     "count":1,"actor_kind":"obstacle","station_window_m":[20.,105.],
                     "max_width_m":6.7,"lateral_limit_m":0.}
        arguments.update(kwargs)
        return generate_candidate(self.loader,self.statics,self.vehicles,42,**arguments)

    def test_opposing_vehicle_follows_native_reverse_traffic_with_pivot_compensation(self):
        c = self.candidate(actor_kind="vehicle",lateral_regions=["opposing_lane"])
        row = c["obstacles"][0]
        self.assertEqual(row["placement_lane_id"],OPPOSING_LINK)
        self.assertAlmostEqual(row["native_forward_yaw_deg"],180.)
        self.assertAlmostEqual(row["yaw_deg"]+row["planned_bbox"]["heading_offset_deg"],180.)
        box = bbox_world(row)
        self.assertAlmostEqual(box["center_xyz"][0],row["station_m"])
        self.assertAlmostEqual(box["center_xyz"][1],3.5)
        self.assertNotAlmostEqual(row["position"]["x"],row["station_m"])
        geom = validate_candidate(c,self.loader)
        self.assertIn("road_boundary",geom)
        bad = copy.deepcopy(c)
        bad["obstacles"][0]["yaw_deg"] -= 180.
        with self.assertRaisesRegex(ValueError,"yaw disagrees"):
            validate_candidate(bad,self.loader)
        bad = copy.deepcopy(c)
        bad["obstacles"][0]["placement_lane_id"] = TUNNEL_LINK
        with self.assertRaisesRegex(ValueError,"lane identity"):
            validate_candidate(bad,self.loader)

    def test_expanded_other_lane_is_allowed_only_with_the_explicit_marker(self):
        c = self.candidate(actor_kind="vehicle",lateral_regions="opposing_lane")
        self.assertEqual(validate_candidate(c,self.loader)["placement_domain"],"two_lane_physical_road")
        c.pop("experimental_placement_domain")
        with self.assertRaisesRegex(ValueError,"whole footprint"):
            validate_candidate(c,self.loader)
        with self.assertRaisesRegex(ValueError,"explicit footprint_domain"):
            generate_candidate(self.loader,self.statics,self.vehicles,42,pattern="short_detour")

    def test_whole_polygon_and_portal_footprints_remain_hard_constraints(self):
        c = self.candidate(actor_kind="vehicle",lateral_regions="opposing_lane")
        wide = copy.deepcopy(c)
        wide["placement"].pop("geometry")
        wide["obstacles"][0]["planned_bbox"]["width_m"] = 7.
        with self.assertRaisesRegex(ValueError,"full OBB"):
            validate_candidate(wide,self.loader)
        portal = copy.deepcopy(c)
        portal["placement"].pop("geometry")
        row = portal["obstacles"][0]
        row["position"]["x"] += .1-row["station_m"]
        row["station_m"] = .1
        with self.assertRaises(ValueError):
            validate_candidate(portal,self.loader)

    def test_rotated_eligibility_keeps_native_dimensions_and_spawn_offset(self):
        steel = {"Steel_Barricade":{"width_m":3.01,"length_m":1.08,
            "center_offset_spawn_xy":[.4,-.12],"heading_offset_deg":1.347,
            "geometry_verified":True,"geometry_source":{"type":"unit_native"}}}
        before = copy.deepcopy(steel)
        c = generate_candidate(self.loader,steel,{},42,count=1,actor_kind="obstacle",
            footprint_domain="two_lane_physical_road",pattern="short_detour",station_window_m=[35.,45.],
            static_yaw_offsets_deg={"Steel_Barricade":[90.]},max_width_m=1.1,lateral_limit_m=0.)
        row = c["obstacles"][0]
        self.assertEqual(row["planned_bbox"]["width_m"],3.01)
        self.assertEqual(row["planned_bbox"]["length_m"],1.08)
        self.assertEqual(row["planned_bbox"]["center_offset_spawn_xy"],[.4,-.12])
        self.assertAlmostEqual(bbox_world(row)["heading_deg"],90.)
        self.assertAlmostEqual(bbox_world(row)["center_xyz"][0],row["station_m"])
        self.assertAlmostEqual(bbox_world(row)["center_xyz"][1],0.)
        self.assertEqual(steel,before)
        with self.assertRaisesRegex(ValueError,"No verified native"):
            generate_candidate(self.loader,steel,{},42,count=1,actor_kind="obstacle",
                footprint_domain="two_lane_physical_road",static_yaw_offsets_deg={"Steel_Barricade":[]})

    def test_unknown_unverified_or_scaled_geometry_never_gets_guessed(self):
        for row in ({"width_m":1.,"length_m":1.},
                    {"width_m":1.,"length_m":1.,"geometry_verified":True},
                    {"width_m":1.,"length_m":1.,"geometry_verified":True,
                     "geometry_source":{"type":"native"},"scale_xyz":[2.,1.,1.]}):
            with self.subTest(row=row),self.assertRaisesRegex(ValueError,"No verified native"):
                generate_candidate(self.loader,{"Unknown":row},{},42,count=1,
                                   footprint_domain="two_lane_physical_road",actor_kind="obstacle")

    def test_campaign_eligibility_uses_rotation_and_each_required_region(self):
        native = {"width_m":3.01,"length_m":1.08,
                  "center_offset_spawn_xy":[.4,-.12],"heading_offset_deg":1.347}
        self.assertEqual(projected_footprint_dimensions(native),{"width_m":3.01,"length_m":1.08})
        rotated = projected_footprint_dimensions(native,90.)
        self.assertAlmostEqual(rotated["width_m"],1.08)
        self.assertAlmostEqual(rotated["length_m"],3.01)
        outer = expanded_region_width_limit("outer_own")
        self.assertAlmostEqual(outer["nominal_projected_width_limit_m"],1.5)
        self.assertAlmostEqual(outer["best_case_projected_width_limit_m"],1.6)
        self.assertTrue(outer["full_polygon_validation_required"])
        # A normal car can pass generic two-lane eligibility but cannot be
        # repeatedly assigned a narrow wall-side slot in this pattern.
        with self.assertRaisesRegex(ValueError,"pattern region: outer_own"):
            self.candidate(pattern="window_bypass",count=2,actor_kind="vehicle")
        mixed = self.candidate(pattern="window_bypass",count=3,allow_three=True,actor_kind="mixed")
        self.assertTrue(all(row["actor_kind"] == "obstacle" for row in mixed["obstacles"][:2]))
        for region in ("own_lane","opposing_lane","near_divider","outer_own","outer_opposing"):
            self.assertGreater(expanded_region_width_limit(region)["best_case_projected_width_limit_m"],0.)
        for value in (float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                projected_footprint_dimensions(native,value)

    def test_expanded_rotation_boundary_and_spacing_contracts_are_immutable(self):
        c = self.candidate()
        mutations = (("outer_road_margin_m",.1,"fixed physical-road"),
                     ("road_boundary_source_sha256","wrong","current common boundary"),
                     ("enforce_adjacent_center_gap",False,"gap policy"))
        for field,value,message in mutations:
            bad = copy.deepcopy(c)
            bad["placement"][field] = value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,message):
                validate_candidate(bad,self.loader)
        bad = copy.deepcopy(c)
        bad["placement"]["static_yaw_offsets_deg"] = {bad["obstacles"][0]["model"]:[90.]}
        with self.assertRaisesRegex(ValueError,"explicit per-model yaw"):
            validate_candidate(bad,self.loader)

    def test_side_by_side_window_allows_equal_station_but_still_checks_every_obb_pair(self):
        c = self.candidate(pattern="window_bypass",count=2)
        self.assertEqual(c["obstacles"][0]["station_m"],c["obstacles"][1]["station_m"])
        self.assertFalse(c["placement"]["enforce_adjacent_center_gap"])
        self.assertIsNone(c["placement"]["geometry"]["length_budget"]["required_center_span_m"])
        self.assertEqual(c["placement"]["geometry"]["minimum_center_gap_m"],0.)
        validate_candidate(c,self.loader)
        bad = copy.deepcopy(c)
        bad["placement"].pop("geometry")
        bad["obstacles"][1] = copy.deepcopy(bad["obstacles"][0])
        with self.assertRaisesRegex(ValueError,"overlap"):
            validate_candidate(bad,self.loader)

    def test_chokepoint_is_geometry_only_and_requires_real_jamming_validation(self):
        wide = {"Wide":{"width_m":4.,"length_m":1.,"geometry_verified":True,
                        "geometry_source":{"type":"unit_native"}}}
        c = generate_candidate(self.loader,wide,{},42,count=1,actor_kind="obstacle",
            footprint_domain="two_lane_physical_road",lateral_regions="near_divider",max_width_m=6.7,
            station_window_m=[45.,45.],lateral_limit_m=0.)
        validate_candidate(c,self.loader)
        self.assertTrue(c["placement"]["actual_jamming_path_required"])
        self.assertFalse(c["expert_verified"])
        self.assertFalse(c["live_avoidance_verified"])
        self.assertIsNone(c["placement"]["realized_pattern"])

    def test_counts_and_template_body_spacing_are_not_silently_compressed(self):
        with self.assertRaisesRegex(ValueError,"explicit count"):
            self.candidate(pattern="alternating_three",count=None,allow_three=True)
        with self.assertRaises(ValueError):
            self.candidate(pattern="alternating_three",count=3)
        c = self.candidate(pattern="alternating_three",count=3,allow_three=True)
        self.assertTrue(c["experimental_three_actor_trial"])
        self.assertEqual([row["placement_lane_id"] for row in c["obstacles"]],
                         [TUNNEL_LINK,OPPOSING_LINK,TUNNEL_LINK])
        self.assertEqual(c["placement"]["requested_pattern"],"alternating_three")
        self.assertEqual(c["placement"]["realized_layout_pattern"],"alternating_three")
        with self.assertRaisesRegex(ValueError,"too short"):
            self.candidate(pattern="independent_detours",count=2,station_window_m=[25.,65.])
        separated = self.candidate(pattern="independent_detours",count=2)
        self.assertAlmostEqual(separated["obstacles"][1]["station_m"]-separated["obstacles"][0]["station_m"],50.)


if __name__ == "__main__":
    unittest.main()
