"""Path novelty is measured behavior, separate from model/role coverage."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from ..scenarios.full_loop_parts.tunnel_static_parts import diversity


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


if __name__ == "__main__":
    unittest.main()
