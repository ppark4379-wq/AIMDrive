"""Random replay selection preserves actual evidence and fresh-guard barriers."""

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..tools import replay_tunnel_static_library as replay
from ..scenarios.full_loop_parts.tunnel_static_parts import catalog, diversity
from .test_tunnel_path_diversity import geometry, path, ONE


LAYOUT = "scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/layout.py"
LIVE = replay.LIVE_PREPARATION_SOURCE


def reviewed_live_hash_pair():
    if len(replay.REVIEWED_LIVE_PREPARATION_TRANSITIONS) != 1:
        raise AssertionError("Exactly one reviewed preparation upgrade must be sealed")
    return next(iter(replay.REVIEWED_LIVE_PREPARATION_TRANSITIONS))


class ReplayEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.trial = self.directory/"trial"
        self.trial.mkdir()
        self.provenance = {"planner_source_sha256":{LAYOUT:"layout-current","expert/planner.py":"expert-current",
                            "scenario_runner/screening.py":"screen-current","scenario_runner/live.py":"live-current"},
                           "recorded_library_sha256":{"recording":"current"},"screening_policy_sha256":"current",
                           "screening_policy":{"body":"current"},"original_route_sha256":"route"}
        self.runtime = {"runtime.py":"current-runtime"}
        self.profile = {"width_m":1.,"length_m":2.,"center_offset_spawn_xy":[0.,0.],"heading_offset_deg":1.347,
                        "geometry_source":{"type":"current-native-profile"}}
        self.candidate = {"count":1,"source_geometry":{"sha256":"mapped-test-geometry"},
                          "obstacles":[{"actor_kind":"obstacle","model":"Box","station_m":30.,"lateral_m":0.,
                                        "scale":[1.,1.,1.],"yaw_deg":0.,"position":{"x":30.,"y":0.,"z":0.},
                                        "geometry_source":self.profile["geometry_source"],
                                        "planned_bbox":{key:self.profile[key] for key in ("width_m","length_m","center_offset_spawn_xy","heading_offset_deg")}}]}
        self.screen = dict(copy.deepcopy(self.provenance),status="nominal_path_found",path_found=True,
                           path=path(ONE),geometry_checks={"passed":True})
        self.preview = {"status":"valid_driving_trial","cleanup":{"verified":True},"measured_route_screen":self.screen}
        self.drive = {"status":"passed","completion_conditions":{key:True for key in diversity.STRICT_CONDITIONS},
                      "runtime_source_sha256":self.runtime,"collision_objects":[],"failed_completion_conditions":[],
                      "trajectory":[{"position_xyz":xy+[0.]} for xy in path(ONE)]}
        self.drive.update({key:True for key in diversity.STRICT_FLAGS})
        self.manifest = {"status":"passed","expert_verified":True,"live_avoidance_verified":True,
                         "fixture_result":{"status":"valid_driving_trial","cleanup":{"verified":True}}}
        self.planner = {"calls":50,"validated_path_seen":True}
        self.row = {"fixture_cleanup_verified":True,"recovery_verified":True}
        for name,value in (("candidate",self.candidate),("preview_report",self.preview),("drive_report",self.drive),
                           ("trial_manifest",self.manifest),("planner_diagnostics",self.planner)):
            self.put(name,value)
        self.trace = self.trial/"ego_trace.csv"
        self.trace.write_text("elapsed_s,x,y,speed\n"+"".join(f"{i*.1},{xy[0]},{xy[1]},2.\n" for i,xy in enumerate(path(ONE))))
        self.row.update(ego_trace_path=str(self.trace),ego_trace_sha256=replay.digest(self.trace))
        self.library_path = self.directory/"approved_catalog.json"
        self.library = {"schema":replay.LIBRARY_SCHEMA,"candidates":[self.row]}
        self.save_library()
        self.gate = {"contracts":{("obstacle","Box"):self.profile},"file_sha256":{},"accepted_model_count":64}

    def put(self,name,value):
        target = self.trial/(name+".json")
        target.write_text(json.dumps(value))
        self.row[name+"_path"] = str(target)
        self.row[name+"_sha256"] = replay.digest(target)

    def save_library(self):
        self.library_path.write_text(json.dumps(self.library))

    def verify(self):
        return replay.verify_library(self.library_path,geometry(),{},provenance=self.provenance,
                                     runtime_sources=self.runtime,road_body_check=lambda *args:None,model_gate=self.gate)

    def test_only_actual_strict_success_is_eligible_and_defaults_to_fresh_measurement(self):
        verified = self.verify()
        self.assertEqual(verified["eligible_count"],1,verified["rejected"])
        row = verified["eligible"][0]
        self.assertTrue(row["replay_requires_fresh_measurement"])
        self.assertFalse(row["historical_screening_provenance"])
        self.assertEqual(row["strata"],{"count":1,"longitudinal_third":0,"actor_kind":"obstacle",
                                         "actual_behavior":"single_opposing_lane_excursion"})
        self.drive["completed"] = False
        self.put("drive_report",self.drive)
        self.save_library()
        self.assertEqual(self.verify()["eligible_count"],0)

    def test_layout_only_mismatch_is_explicit_historical_eligibility_with_fresh_replay(self):
        self.screen["planner_source_sha256"][LAYOUT] = "historical-layout"
        self.put("preview_report",self.preview)
        self.save_library()
        result = self.verify()
        self.assertEqual(result["eligible_count"],1,result["rejected"])
        row = result["eligible"][0]
        self.assertTrue(row["historical_screening_provenance"])
        self.assertEqual(row["preparation_layout_source_hashes"],{"original":"historical-layout","current":"layout-current"})
        self.assertTrue(row["replay_requires_fresh_measurement"])

    def test_expert_live_screen_policy_or_recording_changes_are_fatal(self):
        for change in ("expert/planner.py","scenario_runner/live.py","scenario_runner/screening.py","policy","recording"):
            with self.subTest(change=change):
                self.screen = dict(copy.deepcopy(self.provenance),status="nominal_path_found",path_found=True,
                                   path=path(ONE),geometry_checks={"passed":True})
                if change == "policy":
                    self.screen["screening_policy_sha256"] = "old-policy"
                elif change == "recording":
                    self.screen["recorded_library_sha256"]["recording"] = "old-recording"
                else:
                    self.screen["planner_source_sha256"][change] = "old-source"
                self.preview["measured_route_screen"] = self.screen
                self.put("preview_report",self.preview)
                self.save_library()
                self.assertEqual(self.verify()["eligible_count"],0)

    def test_exact_reviewed_spawn_preparation_upgrade_preserves_raw_historical_proof(self):
        old,new = reviewed_live_hash_pair()
        self.provenance["planner_source_sha256"][LIVE] = new
        self.screen["planner_source_sha256"][LIVE] = old
        self.put("preview_report",self.preview)
        self.save_library()
        original = Path(self.row["preview_report_path"]).read_bytes()
        result = self.verify()
        self.assertEqual(result["eligible_count"],1,result["rejected"])
        row = result["eligible"][0]
        self.assertTrue(row["historical_screening_provenance"])
        self.assertTrue(row["replay_requires_fresh_measurement"])
        transition = row["reviewed_live_preparation_transition"]
        self.assertEqual((transition["original_sha256"],transition["current_sha256"]),(old,new))
        self.assertFalse(transition["historical_evidence_rewritten"])
        self.assertTrue(transition["fresh_guarded_replay_required"])
        self.assertEqual(Path(self.row["preview_report_path"]).read_bytes(),original)

    def test_unknown_reverse_or_additional_source_changes_remain_ineligible(self):
        old,new = reviewed_live_hash_pair()
        cases = [("unknown-old",new,None),(old,"unknown-new",None),(new,old,None),
                 ("c801358db80d339d0804240cd48ea83b8b5167ca1e6a118a138053c14f933b73",new,None),
                 (old,new,"expert/planner.py"),(old,new,"scenario_runner/screening.py")]
        for before,after,additional in cases:
            with self.subTest(before=before,after=after,additional=additional):
                self.provenance["planner_source_sha256"][LIVE] = after
                self.screen = dict(copy.deepcopy(self.provenance),status="nominal_path_found",path_found=True,
                                   path=path(ONE),geometry_checks={"passed":True})
                self.screen["planner_source_sha256"][LIVE] = before
                if additional:
                    self.screen["planner_source_sha256"][additional] = "unknown-source"
                self.preview["measured_route_screen"] = self.screen
                self.put("preview_report",self.preview)
                self.save_library()
                self.assertEqual(self.verify()["eligible_count"],0)

    def test_reviewed_preparation_does_not_relax_policy_or_recorded_routes(self):
        old,new = reviewed_live_hash_pair()
        for key in ("screening_policy","screening_policy_sha256","original_route_sha256","recorded_library_sha256"):
            with self.subTest(key=key):
                current = copy.deepcopy(self.provenance)
                current["planner_source_sha256"][LIVE] = new
                historical = copy.deepcopy(current)
                historical["planner_source_sha256"][LIVE] = old
                historical[key] = "unknown-policy-or-recording"
                with self.assertRaisesRegex(ValueError,"sources or policy changed"):
                    replay._screen_compatibility(historical,current)

    def test_reviewed_target_is_the_exact_current_preparation_source(self):
        old,new = reviewed_live_hash_pair()
        self.assertEqual(old,"c36eabf578c319e7a7a1d1cfbee045abffbacdf3e8a9a45d1e6bdcee6e9cc82a")
        self.assertEqual(new,replay.digest(replay.PROJECT_ROOT/LIVE))

    def test_runtime_hash_mismatch_never_gets_refreshed(self):
        original = copy.deepcopy(self.drive)
        self.drive["runtime_source_sha256"] = {"runtime.py":"original-hash"}
        self.put("drive_report",self.drive)
        self.save_library()
        before = Path(self.row["drive_report_path"]).read_bytes()
        result = self.verify()
        self.assertEqual(result["eligible_count"],0)
        self.assertIn("Runtime source identity/hash changed",result["rejected"][0]["reason"])
        self.assertEqual(Path(self.row["drive_report_path"]).read_bytes(),before)
        self.assertNotEqual(original["runtime_source_sha256"],self.drive["runtime_source_sha256"])

    def test_candidate_preview_drive_and_trace_hashes_are_all_required(self):
        for name in ("candidate","preview_report","drive_report","ego_trace"):
            with self.subTest(name=name):
                original = self.row[name+"_sha256"]
                self.row[name+"_sha256"] = "0"*64
                self.save_library()
                self.assertEqual(self.verify()["eligible_count"],0)
                self.row[name+"_sha256"] = original
        del self.row["ego_trace_sha256"]
        self.save_library()
        result = self.verify()
        self.assertEqual(result["eligible_count"],0)
        self.assertIn("Missing sealed ego_trace",result["rejected"][0]["reason"])

    def test_current_native_footprint_and_unit_yaw_must_match_saved_candidate(self):
        self.candidate["obstacles"][0]["planned_bbox"]["heading_offset_deg"] += 1.
        self.put("candidate",self.candidate)
        self.save_library()
        result = self.verify()
        self.assertEqual(result["eligible_count"],0)
        self.assertIn("current accepted native footprint/unit-yaw",result["rejected"][0]["reason"])

    def test_actual_trace_unknown_is_excluded_not_counted_as_novel(self):
        self.trace.write_text("elapsed_s,x,y,speed\n0,0,0,0\n1,1,nan,2\n2,140,0,2\n")
        self.row["ego_trace_sha256"] = replay.digest(self.trace)
        self.save_library()
        result = self.verify()
        self.assertEqual(result["eligible_count"],0)
        self.assertIn("nonfinite",result["rejected"][0]["reason"])

    def make_args(self,live=False):
        map_root = self.directory/"map"
        map_root.mkdir(exist_ok=True)
        (map_root/"link_set.json").write_text("[]")
        args = replay.build_parser().parse_args(["--library",str(self.library_path),"--seed","4",
                                                "--output",str(self.directory/"selection.json"),"--mgeo-root",str(map_root)])
        args.live = live
        own = {"road_type":"2","width_start":3.5,"width_end":3.5,"points":[[0.,0.,0.],[120.,0.,0.]]}
        opposite = dict(own,points=[[120.,3.5,0.],[0.,3.5,0.]])
        links = {"A2256W000126":own,"A2256W000131":opposite,"A2256W000128":{"points":[[120.,0.,0.],[150.,0.,0.]]}}
        return args,SimpleNamespace(link_set=links,get_link=lambda identity:links[identity])

    def test_manifest_only_mode_never_invokes_live_or_mutates_library_and_history(self):
        args,loader = self.make_args()
        verified = self.verify()
        before = self.library_path.read_bytes()
        live = Mock(side_effect=AssertionError("live must never be called"))
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            result = replay.run(args,verify_fn=lambda *a,**k:verified,live_fn=live)
        live.assert_not_called()
        self.assertFalse(result["saved_trajectory_reused_for_control"])
        self.assertFalse(result["replay_actual_pass"])
        self.assertEqual(result["status"],"selected")
        self.assertEqual(self.library_path.read_bytes(),before)
        history = self.directory/"history.json"
        history.write_text(json.dumps([result]))
        args.history = history
        history_before = history.read_bytes()
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            replay.run(args,verify_fn=lambda *a,**k:verified,live_fn=live)
        self.assertEqual(history.read_bytes(),history_before)

    def test_live_only_passes_saved_candidate_and_requires_existing_guarded_wrapper(self):
        args,loader = self.make_args(live=True)
        verified = self.verify()
        live = Mock(return_value={"outcome":{"strict_pass":True},"scene_restored":True,"cleanup_verified":True})
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            result = replay.run(args,verify_fn=lambda *a,**k:verified,live_fn=live)
        self.assertTrue(result["replay_actual_pass"])
        self.assertEqual(live.call_args.args[0],Path(self.row["candidate_path"]))
        self.assertIn("require_sources_fn",live.call_args.kwargs)
        self.assertNotIn("path",live.call_args.kwargs)
        self.assertNotIn("trajectory",live.call_args.kwargs)

    def test_fresh_pass_requires_exact_scene_restore_and_cleanup(self):
        args,loader = self.make_args(live=True)
        verified = self.verify()
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            result = replay.run(args,verify_fn=lambda *a,**k:verified,
                                live_fn=Mock(return_value={"outcome":{"strict_pass":True},"scene_restored":False,"cleanup_verified":True}))
        self.assertFalse(result["replay_actual_pass"])
        self.assertEqual(result["status"],"failed")

    def test_live_failure_keeps_durable_failed_selection(self):
        args,loader = self.make_args(live=True)
        verified = self.verify()
        with patch.object(replay,"MGeoMapLoader",return_value=loader),self.assertRaisesRegex(RuntimeError,"guard failed"):
            replay.run(args,verify_fn=lambda *a,**k:verified,live_fn=Mock(side_effect=RuntimeError("guard failed")))
        saved = json.loads(args.output.read_text())
        self.assertEqual(saved["status"],"failed")
        self.assertFalse(saved["replay_actual_pass"])

    def test_source_pin_check_rejects_changed_library_without_refresh(self):
        args,loader = self.make_args()
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            selected = replay.run(args,verify_fn=lambda *a,**k:self.verify())
        self.library_path.write_text(self.library_path.read_text()+" ")
        with self.assertRaisesRegex(RuntimeError,"library changed"):
            replay.require_selection_sources(selected)

    def test_replay_verifier_and_transition_registry_are_pinned_after_selection(self):
        args,loader = self.make_args()
        with patch.object(replay,"MGeoMapLoader",return_value=loader):
            selected = replay.run(args,verify_fn=lambda *a,**k:self.verify())
        selected["replay_verifier_sha256"] = "0"*64
        with patch.object(replay,"current_runtime_sources",return_value=self.runtime), \
                patch.object(replay.screening,"screening_provenance",return_value=self.provenance), \
                self.assertRaisesRegex(RuntimeError,"Replay verifier or preparation compatibility changed"):
            replay.require_selection_sources(selected)

    def test_output_cannot_overwrite_original_library_or_trial_proof(self):
        args,loader = self.make_args()
        for target in (self.library_path,Path(self.row["drive_report_path"])):
            args.output = target
            before = target.read_bytes()
            with patch.object(replay,"MGeoMapLoader",return_value=loader),self.assertRaisesRegex(ValueError,"cannot overwrite"):
                replay.run(args,verify_fn=lambda *a,**k:self.verify())
            self.assertEqual(target.read_bytes(),before)


class BalancedChoiceTests(unittest.TestCase):
    def row(self,identity,count=1,third=0,kind="obstacle",behavior="within_original_lane"):
        return {"id":identity,"strata":{"count":count,"longitudinal_third":third,"actor_kind":kind,"actual_behavior":behavior}}

    def test_seeded_choice_is_order_independent(self):
        rows = [self.row(str(i),count=(i%3)+1,third=i%3,kind="mixed" if i%2 else "obstacle") for i in range(30)]
        a = replay.balanced_choice(rows,123)
        b = replay.balanced_choice(list(reversed(rows)),123)
        self.assertEqual(a,b)

    def test_large_model_pool_does_not_dominate_balanced_count_and_kind_branches(self):
        rows = [self.row("large-"+str(i),count=1,kind="obstacle") for i in range(200)]
        rows += [self.row("one-mixed",count=1,kind="mixed"),self.row("two",count=2),self.row("three",count=3)]
        choices = [replay.balanced_choice(rows,seed)[0] for seed in range(2000)]
        count = Counter(row["strata"]["count"] for row in choices)
        self.assertTrue(500 < count[1] < 700,count)
        self.assertTrue(700 < count[2] < 900,count)
        kinds = Counter(row["strata"]["actor_kind"] for row in choices if row["strata"]["count"] == 1)
        self.assertLess(abs(kinds["mixed"]-kinds["obstacle"]),100,kinds)

    def test_history_prioritizes_less_used_record_and_remains_read_only(self):
        rows = [self.row("first"),self.row("second")]
        history = [{"schema":replay.SCHEMA,"selected":rows[0]}]
        before = copy.deepcopy(history)
        self.assertEqual(replay.balanced_choice(rows,1,history)[0]["id"],"second")
        self.assertEqual(history,before)


class FreshModelGateTests(unittest.TestCase):
    def test_current_parked_preview_sources_are_required_even_when_historical_drive_layout_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {"width_m":1.,"length_m":2.,"center_offset_spawn_xy":[0.,0.],"heading_offset_deg":1.347,
                       "geometry_source":{"type":"native"}}
            placement = {"model":"Box","actor_kind":"obstacle","scale":[1.,1.,1.],
                         "geometry_source":profile["geometry_source"],"planned_bbox":profile}
            current = {"planner_source_sha256":{LAYOUT:"new-layout",LIVE:"live-current","expert/source.py":"expert-current"}}
            preview = {"status":"valid_preview","cleanup":{"verified":True},"hold_seconds":5.,"stationary_observed_seconds":5.,
                       "route_screened":True,"expert_verified":False,"live_avoidance_verified":False,
                       "planned":{"source_geometry":{"sha256":"mapped-test-geometry"},"obstacles":[placement]},
                       "observed":[{"model":"Box","actor_kind":"obstacle"}],
                       "measured_route_screen":dict(copy.deepcopy(current),path_found=True,geometry_checks={"passed":True})}
            preview_path = root/"preview.json"
            preview_path.write_text(json.dumps(preview))
            entry = {"model":"Box","actor_kind":"obstacle","accepted":True,"scale":[1.,1.,1.],
                     "geometry_source":profile["geometry_source"],"preview_path":str(preview_path),"preview_sha256":replay.digest(preview_path)}
            acceptance = {"schema":"aimdrive_tunnel_model_acceptance_v1","map":"R_KR_PR_K-city_2025",
                          "source_geometry_sha256":"mapped-test-geometry","models":[entry]}
            acceptance_path,runtime_path = root/"acceptance.json",root/"runtime.json"
            acceptance_path.write_text(json.dumps(acceptance))
            runtime_path.write_text('{"models":[]}')
            with patch.object(catalog,"load_static_models",return_value={"Box":profile}),patch.object(catalog,"load_vehicle_models",return_value={}):
                result = replay.verify_model_acceptance(acceptance_path,runtime_path,geometry(),current,expected_count=1)
                self.assertEqual(result["accepted_model_count"],1)
                preview["measured_route_screen"]["planner_source_sha256"][LAYOUT] = "old-layout"
                preview_path.write_text(json.dumps(preview))
                entry["preview_sha256"] = replay.digest(preview_path)
                acceptance_path.write_text(json.dumps(acceptance))
                with self.assertRaisesRegex(ValueError,"parked native model proof incomplete"):
                    replay.verify_model_acceptance(acceptance_path,runtime_path,geometry(),current,expected_count=1)
                old,new = reviewed_live_hash_pair()
                current["planner_source_sha256"][LIVE] = new
                preview["measured_route_screen"]["planner_source_sha256"][LAYOUT] = "new-layout"
                preview["measured_route_screen"]["planner_source_sha256"][LIVE] = old
                preview_path.write_text(json.dumps(preview))
                entry["preview_sha256"] = replay.digest(preview_path)
                acceptance_path.write_text(json.dumps(acceptance))
                original_preview = preview_path.read_bytes()
                original_acceptance = acceptance_path.read_bytes()
                compatible = replay.verify_model_acceptance(acceptance_path,runtime_path,geometry(),current,expected_count=1)
                self.assertEqual(compatible["accepted_model_count"],1)
                self.assertEqual(len(compatible["reviewed_live_preparation_transitions"]),1)
                self.assertEqual(compatible["reviewed_live_preparation_transitions"][0]["original_sha256"],old)
                self.assertEqual(preview_path.read_bytes(),original_preview)
                self.assertEqual(acceptance_path.read_bytes(),original_acceptance)
                for source,value in ((LIVE,"unknown-live"),("expert/source.py","unknown-expert"),(LAYOUT,"old-layout")):
                    with self.subTest(source=source):
                        preview["measured_route_screen"]["planner_source_sha256"] = dict(current["planner_source_sha256"])
                        preview["measured_route_screen"]["planner_source_sha256"][LIVE] = old
                        preview["measured_route_screen"]["planner_source_sha256"][source] = value
                        preview_path.write_text(json.dumps(preview))
                        entry["preview_sha256"] = replay.digest(preview_path)
                        acceptance_path.write_text(json.dumps(acceptance))
                        with self.assertRaisesRegex(ValueError,"parked native model proof incomplete"):
                            replay.verify_model_acceptance(acceptance_path,runtime_path,geometry(),current,expected_count=1)


if __name__ == "__main__":
    unittest.main()
