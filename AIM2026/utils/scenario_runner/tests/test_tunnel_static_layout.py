"""Tunnel regression checks: geometry."""

import copy, json, math, tempfile, unittest, numpy as np, hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ..scenarios.full_loop_parts.tunnel_static_parts.layout import OPPOSING_LINK, TUNNEL_LINK, bbox_world, boxes_overlap, footprint_corners, generate_candidate, normalized_geometry, project_polyline, validate_candidate, projected_footprint_dimensions, expanded_region_width_limit
from ..scenarios.full_loop_parts.tunnel_static_parts.survey import APPROACH_LINK, DEPARTURE_LINK, analyze_clouds, cloud_to_world, ground_crosssection, project_cloud, rotation_rpy, sensor_calibration, station_pose, structure_extent, survey_route, validate_exposure
from ..scenarios.full_loop_parts.tunnel_static_parts import diversity


# Stationary survey provenance, full rotations and conservative authority.

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


# Path novelty is measured behavior, separate from model/role coverage.

def geometry(departure=True, opposite_left=True):
    separation = 3.5 if opposite_left else -3.5
    result = {"own_points":[[0.,0.,0.],[120.,0.,0.]],
              "opposing_points":[[0.,separation,0.],[120.,separation,0.]],
              "own_lane_width_m":3.5, "length_m":120., "source_sha256":"mapped-test-geometry"}
    if departure:
        result["departure_points"] = [[120.,0.,0.],[150.,0.,0.]]
    return result


def path(knots, noise=0.):
    result = []
    for step in range(561):
        station = step*.25
        lateral = diversity._interpolate([row[0] for row in knots], [row[1] for row in knots], station)
        result.append([station, lateral+noise*math.sin(station*2.7)])
    return result


ONE = [(0.,0.),(12.,0.),(30.,3.),(62.,3.),(92.,0.),(140.,0.)]
TWO = [(0.,0.),(10.,0.),(22.,3.),(35.,3.),(46.,0.),(61.,0.),(74.,3.),(86.,3.),(98.,0.),(140.,0.)]


def candidate(count=1, model="Box", station=30., role="short_detour"):
    return {"count":count, "requested_pattern":role,
            "source_geometry":{"sha256":"mapped-test-geometry"},
            "obstacles":[{"actor_kind":"obstacle", "model":model,
                          "position":{"x":station+i*10., "y":0., "z":0.},
                          "yaw_deg":0., "scale":[1.,1.,1.],
                          "planned_bbox":{"width_m":1.,"length_m":2.}}
                         for i in range(count)]}


def screened(identity, count=1, model="Box", station=30., knots=ONE, role="short_detour"):
    return {"id":identity, "candidate":candidate(count,model,station,role),
            "screen":{"status":"nominal_path_found", "path_found":True, "path":path(knots)}}


class PathDiversityTests(unittest.TestCase):
    def features(self, knots=ONE, **kwargs):
        return diversity.path_features(path(knots), geometry(), **kwargs)

    def test_same_path_matches_despite_model_record_or_requested_role(self):
        first = screened("first",model="A",role="short_detour")
        second = screened("second",model="B",role="slalom")
        first["screen"]["selected_routes"] = ["jamming_01.csv"]
        second["screen"]["selected_routes"] = ["jamming_07.csv"]
        a = diversity.path_features(first["screen"]["path"],geometry())
        b = diversity.path_features(second["screen"]["path"],geometry())
        self.assertEqual(a["fingerprint_sha256"],b["fingerprint_sha256"])
        self.assertTrue(diversity.compare_paths(a,b)["duplicate"])

    def test_small_tracking_noise_is_a_duplicate_in_same_actual_nominal_frame(self):
        nominal = self.features()
        actual = diversity.path_features(path(ONE,noise=.035),geometry(),evidence_kind="completed_actual_trace")
        comparison = diversity.compare_paths(nominal,actual)
        self.assertTrue(comparison["duplicate"], comparison)
        self.assertLess(comparison["mean_lateral_discrepancy_m"],.04)
        self.assertEqual(nominal["stations_m"],actual["stations_m"])

    def test_translation_and_later_return_change_actual_path(self):
        moved = [(s+15. if 0. < s < 140. else s,v) for s,v in ONE]
        comparison = diversity.compare_paths(self.features(),self.features(moved))
        self.assertFalse(comparison["duplicate"],comparison)
        self.assertGreater(comparison["return_station_difference_m"],6.)
        self.assertIn("return_station_differs",comparison["reasons"])

    def test_excursion_count_order_and_shape_are_observed(self):
        one = self.features(evidence_kind="completed_actual_trace")
        two = self.features(TWO,evidence_kind="completed_actual_trace")
        reverse = [(s,-v) for s,v in ONE]
        right = self.features(reverse,evidence_kind="completed_actual_trace")
        self.assertEqual(one["completed_opposing_excursions"],1)
        self.assertEqual(two["completed_opposing_excursions"],2)
        self.assertEqual(two["excursion_direction_sequence"],["left","left"])
        self.assertTrue(two["actual_s_maneuver_observed"])
        self.assertFalse(one["actual_s_maneuver_observed"])
        self.assertFalse(self.features(TWO)["actual_s_maneuver_observed"])
        self.assertFalse(diversity.compare_paths(one,two)["duplicate"])
        self.assertFalse(diversity.compare_paths(one,right)["duplicate"])
        self.assertEqual(right["excursion_direction_sequence"],["right"])

    def test_opposing_lane_direction_comes_from_geometry(self):
        own_opposing = diversity.path_features(path([(s,-v) for s,v in TWO]),geometry(opposite_left=False),
                                              evidence_kind="completed_actual_trace")
        self.assertEqual(own_opposing["completed_opposing_excursions"],2)
        self.assertEqual(own_opposing["excursion_direction_sequence"],["right","right"])

    def test_fingerprint_reproducible_and_station_location_retained(self):
        a = self.features()
        b = self.features()
        self.assertEqual(a,b)
        self.assertEqual(a["expected_length_m"],140.)
        self.assertTrue(a["departure_return_coverage"])
        self.assertGreater(a["normalized_station"][-1],1.)
        no_departure = diversity.path_features(path(ONE),geometry(departure=False))
        self.assertFalse(no_departure["departure_return_coverage"])
        self.assertEqual(no_departure["expected_length_m"],120.)

    def test_invalid_nan_short_reversed_and_missing_coverage_fail_unknown(self):
        cases = [path(ONE)[:50],list(reversed(path(ONE))),
                 [[0.,0.],[1.,float("nan")]]+path(ONE),
                 path(ONE)[:200]+path(ONE)[260:]]
        for rows in cases:
            feature = diversity.path_features(rows,geometry())
            self.assertFalse(feature["valid"],feature)
            comparison = diversity.compare_paths(self.features(),feature)
            self.assertEqual(comparison["status"],"unknown")
            self.assertIsNone(comparison["duplicate"])
            self.assertIsNone(comparison["distance"])

    def test_common_coverage_cannot_invent_similarity(self):
        first = self.features()
        second = copy.deepcopy(first)
        second["stations_m"] = second["stations_m"][:50]
        second["lateral_m"] = second["lateral_m"][:50]
        result = diversity.compare_paths(first,second)
        self.assertEqual(result["status"],"unknown")

    def test_tampered_feature_nan_or_reverse_is_unknown(self):
        original = self.features()
        for field in ("stations_m", "lateral_m", "maximum_width_m", "return_station_m"):
            broken = copy.deepcopy(original)
            if isinstance(broken[field], list):
                broken[field][20] = float("nan")
            else:
                broken[field] = float("nan")
            result = diversity.compare_paths(original,broken)
            self.assertEqual(result["status"],"unknown",result)
            self.assertIsNone(result["duplicate"])

    def test_invalid_opposing_geometry_is_unknown(self):
        broken = geometry()
        broken["opposing_points"][1][0] = float("nan")
        result = diversity.path_features(path(ONE),broken)
        self.assertEqual(result["status"],"unknown")
        self.assertFalse(result["valid"])

    def test_selector_is_deterministic_balanced_and_keeps_non_novel_model_representatives(self):
        rows = [screened(str(i), count=(i%3)+1,model="Model"+str(i),station=10.+i)
                for i in range(12)]
        baseline = [{"candidate":candidate(model="Historical",station=90.), "actual_features":self.features()}]
        first = diversity.select_diverse_candidates(rows,geometry(),existing_paths=baseline,limit=6,count_quotas={1:2,2:2,3:2})
        second = diversity.select_diverse_candidates(list(reversed(rows)),geometry(),existing_paths=baseline,limit=6,count_quotas={1:2,2:2,3:2})
        self.assertEqual(first["selected_ids"],second["selected_ids"])
        self.assertEqual(first["selected_counts"],{1:2,2:2,3:2})
        self.assertTrue(all(row["actual_approved"] is False for row in first["selected"]))
        self.assertTrue(all(row["rationale"]["path_novelty_predicted"] is False for row in first["selected"]))
        self.assertTrue(all(row["rationale"]["new_coverage"] for row in first["selected"]))
        self.assertEqual(len(first["feature_records"]),len(rows))

    def test_selector_rejects_exact_layout_even_if_requested_role_differs_and_unknown_screen(self):
        a = screened("a",role="one")
        b = copy.deepcopy(a)
        b["id"] = "b"
        b["candidate"]["requested_pattern"] = "two"
        bad = screened("invalid")
        bad["screen"]["path"] = list(reversed(bad["screen"]["path"]))
        result = diversity.select_diverse_candidates([a,b,bad],geometry(),limit=3)
        self.assertEqual(result["selected_count"],1)
        self.assertIn("exact_layout_duplicate",[row["reason"] for row in result["rejected"]])
        self.assertIn("nonmonotonic_station_coverage",[row["reason"] for row in result["rejected"]])

    def test_selector_limit_and_remaining_quotas(self):
        rows = [screened(str(i),count=3,model=str(i),station=i) for i in range(5)]
        result = diversity.select_diverse_candidates(rows,geometry(),limit=2,count_quotas={1:0,2:0,3:2})
        self.assertEqual(result["selected_counts"][3],2)
        with self.assertRaises(ValueError):
            diversity.select_diverse_candidates(rows,geometry(),limit=101)

    def test_duplicate_path_without_semantic_or_count_deficit_is_rejected(self):
        baseline = [{"candidate":candidate(station=90.), "actual_features":self.features()}]
        row = screened("same-behavior",station=80.)
        result = diversity.select_diverse_candidates([row],geometry(),existing_paths=baseline,limit=1,
                                                    count_quotas={1:0,2:0,3:0})
        self.assertEqual(result["selected_count"],0)
        self.assertEqual(result["rejected"][0]["reason"],"similar_path_without_coverage_deficit")

    def test_cluster_count_counts_paths_not_model_names(self):
        rows = [self.features(),self.features(),self.features(TWO)]
        result = diversity.cluster_paths(rows)
        self.assertEqual(result["cluster_count"],2)
        self.assertEqual(result["valid_path_count"],3)


class BaselineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.trial = self.directory/"trial"
        self.trial.mkdir()
        self.provenance = {"recorded_library_sha256":{"recording":"abc"}, "original_route_sha256":"route",
                           "screening_policy_sha256":"policy", "screening_policy":{"body":"current"},
                           "planner_source_sha256":{"expert/planner.py":"expert-hash", "scenario_runner/layout.py":"new-layout"}}
        old = copy.deepcopy(self.provenance)
        old["planner_source_sha256"]["scenario_runner/layout.py"] = "old-layout"
        body_policy = {"body":{"width_m":2.36},"source_sha256":"road-source"}
        screen = dict(old,status="nominal_path_found",path_found=True,path=path(ONE),
                      road_boundary_policy=body_policy,road_boundary_policy_sha256="body-policy")
        drive = dict(status="passed", completion_conditions={key:True for key in diversity.STRICT_CONDITIONS},
                     collision_objects=[], failed_completion_conditions=[], road_boundary_policy=body_policy,
                     road_boundary_policy_sha256="body-policy", runtime_source_sha256={"runtime.py":"old-runtime"},
                     trajectory=[{"position_xyz":point+[0.]} for point in path(ONE)])
        drive.update({key:True for key in diversity.STRICT_FLAGS})
        self.approval = {"fixture_cleanup_verified":True,"recovery_verified":True}
        for name, document in (("candidate",candidate()),("preview_report",{"status":"valid_driving_trial","cleanup":{"verified":True},"measured_route_screen":screen}),
                               ("drive_report",drive)):
            target = self.trial/("drive_trial.json" if name == "drive_report" else name+".json")
            target.write_text(json.dumps(document))
            self.approval[name+"_path"] = str(target)
            self.approval[name+"_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (self.trial/"manifest.json").write_text(json.dumps({"status":"passed","expert_verified":True,"live_avoidance_verified":True}))
        (self.trial/"planner_diagnostics.json").write_text(json.dumps({"validated_path_seen":True,"road_boundary_policy_sha256":"body-policy"}))
        self.trace_path = self.trial/"ego_trace.csv"
        self.trace_path.write_text("elapsed_s,x,y,speed\n"+"".join(f"{i*.1},{p[0]},{p[1]},2.\n" for i,p in enumerate(path(ONE))))
        self.approved_path = self.directory/"approved_candidates.json"
        self.write_approval()
        (self.directory/"batch_manifest.json").write_text(json.dumps({"attempts":[{"trial_dir":str(self.trial),"committed":True,"outcome":{"strict_pass":True}}]}))

    def write_approval(self):
        self.approved_path.write_text(json.dumps({"candidates":[self.approval]}))

    def load(self):
        return diversity.load_approved_baseline(self.approved_path,geometry(),current_provenance=self.provenance,
                                               current_runtime_sources={"runtime.py":"new-runtime"})

    def test_historical_success_kept_for_dedup_despite_preparation_and_runtime_changes(self):
        result = self.load()
        self.assertEqual(result["loaded_count"],1,result["rejected"])
        record = result["records"][0]
        self.assertFalse(record["runtime_sources_current"])
        self.assertFalse(record["all_screen_sources_current"])
        self.assertFalse(record["actual_approved"])
        self.assertTrue(record["dedup_usable"])
        self.assertIn("computed_at_load",record["csv_sha256_seal_status"])

    def test_planner_or_recorded_library_changes_reject_historical_baseline(self):
        for key in ("planner_source_sha256","recorded_library_sha256"):
            original = copy.deepcopy(self.provenance)
            if key == "planner_source_sha256":
                self.provenance[key]["expert/planner.py"] = "changed"
            else:
                self.provenance[key]["recording"] = "changed"
            result = self.load()
            self.assertEqual(result["loaded_count"],0)
            self.assertIn("current_planning_inputs_incompatible",result["rejected"][0]["reason"])
            self.provenance = original

    def test_hashed_report_tampering_missing_completion_and_trace_tampering_reject(self):
        target = Path(self.approval["drive_report_path"])
        raw = target.read_text()
        target.write_text(raw+" ")
        self.assertIn("SHA256 mismatch",self.load()["rejected"][0]["reason"])
        target.write_text(raw)
        drive = json.loads(raw)
        drive["completion_conditions"]["mapped_exit_observed"] = False
        target.write_text(json.dumps(drive))
        self.approval["drive_report_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.write_approval()
        self.assertEqual(self.load()["loaded_count"],0)
        target.write_text(raw)
        self.approval["drive_report_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.write_approval()
        self.trace_path.write_text("elapsed_s,x,y\n"+"".join(f"{i},{p[0]},{p[1]}\n" for i,p in enumerate(path(TWO))))
        result = self.load()
        self.assertEqual(result["loaded_count"],0)
        self.assertIn("csv_trace_does_not_match",result["rejected"][0]["reason"])


# Candidate generation must preserve map, full geometry and reproducibility.

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
