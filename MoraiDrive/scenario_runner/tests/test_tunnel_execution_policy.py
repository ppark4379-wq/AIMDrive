"""Only fresh physical success under the current inherited policy is approved."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..scenarios.full_loop_parts.tunnel_static_parts import diversity, execution_policy as policy_module


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trace_geometry={"own_points":[[0.,0.,28.],[20.,0.,28.]],
                             "opposing_points":[[0.,3.5,28.],[20.,3.5,28.]],
                             "own_lane_width_m":3.5,"length_m":20.,"source_sha256":"isolated-trace-map"}
        geometry_patch=patch.object(policy_module,"trace_geometry",return_value=self.trace_geometry)
        geometry_patch.start()
        self.addCleanup(geometry_patch.stop)
        self.expert = self.root/"fake_expert"
        original = self.expert/"src/config/config.json"
        original.parent.mkdir(parents=True)
        self.write(original,{"map":{},"planning":{"velocity_profile":{"max_velocity":37.,"window_size":50}},
                             "control":{"gain":"unchanged"}})
        self.library = self.root/"library.json"
        self.write(self.library,{"schema":"isolated-test-layout-source","candidates":[]})
        self.policy = policy_module.describe({"paths":{"expert_root":str(self.expert)}},self.library)
        self.store = self.root/"current_approvals.json"
        self.selection_path = self.root/"selection.json"
        self.live = self.root/"selection_live"
        self.trial = self.live/"attempts/attempt_01/trial"
        self.trial.mkdir(parents=True)
        self.candidate = {"count":1,"obstacles":[{"actor_kind":"obstacle","model":"Box",
                          "position":{"x":50.,"y":3.5,"z":28.},"yaw_deg":0.,"scale":[1.,1.,1.]}]}
        self.write(self.trial/"candidate.json",self.candidate)
        self.candidate_id = policy_module.digest(self.trial/"candidate.json")
        self.historical_candidate = self.root/"immutable_layout.json"
        self.historical_candidate.write_bytes((self.trial/"candidate.json").read_bytes())
        self.reference = {"raw_actor_id":"7","actor_kind":"obstacle","model":"Box",
                          "position_xyz":[50.,3.5,28.],"center_xyz":[50.,3.5,28.],
                          "heading_deg":0.,"width_m":1.,"length_m":1.}
        self.road = {"schema":"aimdrive_tunnel_road_boundary_v1","body":{"rear_m":.85,"front_m":3.85,"width_m":2.17},
                     "planner_model":{"rear_m":0.,"front_m":4.64,"width_m":2.17},"source_sha256":"isolated-mapped-road",
                     "planning_road_margin_m":.35,"runtime_road_margin_m":.15}
        self.road_hash = policy_module.identity(self.road)
        self.scene_hash = policy_module.identity([self.reference])
        self.screen = dict(copy.deepcopy(self.policy["screening"]),status="nominal_path_found",path_found=True,
                           geometry_checks={"passed":True,"runtime_path_road_passed":True,"planning_road_envelope_passed":True},
                           road_boundary_policy=self.road,road_boundary_policy_sha256=self.road_hash,
                           measured_ego_body=self.road["body"],obstacle_input=[self.reference])
        self.drive = {"status":"passed","completion_conditions":{key:True for key in diversity.STRICT_CONDITIONS},
                      "collision_objects":[],"failed_completion_conditions":[],
                      "runtime_source_sha256":copy.deepcopy(self.policy["runtime_files"]),
                      "road_boundary_policy":self.road,"road_boundary_policy_sha256":self.road_hash,
                      "owned_actor_geometry":[self.reference],"fixture_raw_actor_ids":["7"],"observed_object_ids":["7"],
                      "trajectory_checks":22,"trajectory":[{"elapsed_sec":float(i),"position_xyz":[float(i),0.,28.],
                                                             "yaw_deg":0.} for i in range(22)]}
        self.drive.update({key:True for key in diversity.STRICT_FLAGS})
        self.preview = {"status":"valid_driving_trial","expert_verified":True,"live_avoidance_verified":True,
                        "cleanup":{"verified":True},"route_screened":True,"measured_route_screen":self.screen,
                        "planned":self.candidate,"observed":[self.reference],"owned_actor_geometry":[self.reference]}
        self.manifest = {"status":"passed","expert_verified":True,"live_avoidance_verified":True,
                         "fixture_result":self.preview}
        self.planner = {"calls":50,"validated_path_seen":True,"selected_routes":["jamming_01.csv"],
                        "scene_geometry_sha256":self.scene_hash,"road_boundary_policy":self.road,
                        "road_boundary_policy_sha256":self.road_hash}
        self.config = {"planning":{"velocity_profile":{"max_velocity":37.,"window_size":15},
                        "jamming_scene_geometry":[self.reference],"jamming_trial_contract":{"mode":"full_loop_tunnel_static",
                        "road_boundary_policy":self.road,"road_boundary_policy_sha256":self.road_hash}}}
        self.entry = {"committed":True,"recovery_verified":True,"trial_dir":str(self.trial),
                      "process":{"returncode":0,"timed_out":False,"forced_kill":False,"interrupted":False},
                      "outcome":{"actual_driving":True,"strict_pass":True}}
        self.state = {"status":"completed","actual_drives":1,"actual_passes":1,"actual_failures":0,
                      "scene_restored":True,"cleanup_verified":True,"attempts":[self.entry]}
        self.result = {"status":"completed","scene_restored":True,"cleanup_verified":True,
                       "outcome":{"actual_driving":True,"strict_pass":True}}
        self.selection = {"schema":"aimdrive_tunnel_static_replay_selection_v1","status":"passed",
                          "live_requested":True,"replay_actual_pass":True,"execution_policy":self.policy,
                          "seed":42,"incoming_environment_history":None,
                          "environment_profile":{"run_seed":42,"weather":"SUNNY","time_hour":11},
                          "saved_trajectory_reused_for_control":False,"selected":{"id":self.candidate_id,
                          "paths":{"candidate":str(self.historical_candidate)},
                          "file_sha256":{"candidate":self.candidate_id}},
                          "live_result":dict(self.result,output_dir=str(self.live))}
        self.save_fixture()

    def write(self,path,document):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(document,allow_nan=True))

    def save_fixture(self):
        for name,document in (("drive_trial.json",self.drive),("preview.json",self.preview),
                               ("manifest.json",self.manifest),("planner_diagnostics.json",self.planner),
                               ("runner_runtime/expert/config.json",self.config)):
            self.write(self.trial/name,document)
        (self.trial/"ego_trace.csv").write_text("elapsed_s,x,y,yaw_deg,speed\n"+
            "".join(f"{i},{float(i)},0.,0.,{0. if i==0 else 1.}\n" for i in range(22)))
        self.write(self.live/"search_manifest.json",self.state)
        self.write(self.live/"replay_result.json",self.result)
        self.write(self.selection_path,self.selection)

    def approve(self):
        return policy_module.approve(self.store,self.policy,self.selection_path)

    def assert_rejected(self):
        with self.assertRaises((ValueError,RuntimeError,OSError)):
            self.approve()
        self.assertFalse(self.store.exists())

    def test_describe_reads_fake_original_37_and_does_not_modify_it(self):
        original=self.expert/"src/config/config.json"
        before=original.read_bytes()
        described=policy_module.describe({"paths":{"expert_root":str(self.expert)}},self.library)
        self.assertEqual(described["expert_speed"]["max_velocity_kmh"],37.)
        self.assertTrue(described["expert_speed"]["inherited"])
        self.assertEqual(described["expert_speed"]["expert_config_sha256"],policy_module.digest(original))
        self.assertEqual(original.read_bytes(),before)

    def test_fresh_same_policy_physical_success_is_approved_and_sealed(self):
        before=self.historical_candidate.read_bytes()
        self.assertEqual(self.approve(),self.candidate_id)
        self.assertEqual(policy_module.approved_ids(self.store,self.policy),{self.candidate_id})
        ledger=json.loads(self.store.read_text())
        proof=ledger["policies"][policy_module.identity(self.policy)]["layouts"][0]
        mandatory={str(self.selection_path),str(self.live/"search_manifest.json"),str(self.live/"replay_result.json"),
                   str(self.historical_candidate),str(self.trial/"candidate.json"),str(self.trial/"drive_trial.json"),
                   str(self.trial/"preview.json"),str(self.trial/"manifest.json"),str(self.trial/"planner_diagnostics.json"),
                   str(self.trial/"runner_runtime/expert/config.json"),str(self.trial/"ego_trace.csv")}
        self.assertTrue(mandatory.issubset(proof["proof_sha256"]))
        self.assertEqual(self.historical_candidate.read_bytes(),before)

    def test_old_10_policy_and_runtime_cannot_approve_inherited_37(self):
        old=copy.deepcopy(self.policy)
        old["expert_speed"]["max_velocity_kmh"]=10.
        self.selection["execution_policy"]=old
        self.config["planning"]["velocity_profile"]["max_velocity"]=10.
        self.save_fixture()
        self.assert_rejected()

    def test_current_policy_label_cannot_mask_an_old_10_runtime_config(self):
        self.config["planning"]["velocity_profile"]["max_velocity"]=10.
        self.save_fixture()
        self.assert_rejected()

    def test_selected_raw_historical_or_failed_status_never_approves(self):
        for status in ("selected","failed","historical_strict_success"):
            self.selection["status"]=status
            self.save_fixture()
            self.assert_rejected()
        self.selection["status"]="passed"
        self.selection["live_requested"]=False
        self.save_fixture()
        self.assert_rejected()

    def test_each_strict_completion_flag_and_condition_is_required(self):
        original_drive=copy.deepcopy(self.drive)
        for key in diversity.STRICT_FLAGS:
            with self.subTest(flag=key):
                self.drive=copy.deepcopy(original_drive)
                self.drive[key]=False
                self.save_fixture()
                self.assert_rejected()
        for key in diversity.STRICT_CONDITIONS:
            with self.subTest(condition=key):
                self.drive=copy.deepcopy(original_drive)
                del self.drive["completion_conditions"][key]
                self.save_fixture()
                self.assert_rejected()

    def test_collision_and_missing_physical_motion_are_rejected(self):
        self.drive["collision_objects"]=["7"]
        self.save_fixture()
        self.assert_rejected()
        self.drive["collision_objects"]=[]
        self.drive["trajectory"]=[{"position_xyz":[0.,0.,28.]} for _ in range(4)]
        self.save_fixture()
        self.assert_rejected()

    def test_completion_summary_without_committed_or_recovered_drive_is_rejected(self):
        for key in ("committed","recovery_verified"):
            self.entry[key]=False
            self.save_fixture()
            self.assert_rejected()
            self.entry[key]=True

    def test_cleanup_and_restore_are_required_in_both_state_and_result(self):
        for document in (self.state,self.result):
            for key in ("cleanup_verified","scene_restored"):
                with self.subTest(document="state" if document is self.state else "result",key=key):
                    document[key]=False
                    self.save_fixture()
                    self.assert_rejected()
                    document[key]=True

    def test_failed_or_timed_out_process_cannot_use_stale_strict_summary(self):
        for key,value in (("returncode",1),("timed_out",True),("forced_kill",True),("interrupted",True)):
            with self.subTest(key=key):
                original=self.entry["process"][key]
                self.entry["process"][key]=value
                self.save_fixture()
                self.assert_rejected()
                self.entry["process"][key]=original

    def test_fresh_manifest_and_measured_geometry_must_be_valid(self):
        self.manifest["status"]="failed"
        self.save_fixture()
        self.assert_rejected()
        self.manifest["status"]="passed"
        self.preview["status"]="failed"
        self.save_fixture()
        self.assert_rejected()
        self.preview["status"]="valid_driving_trial"
        self.screen["geometry_checks"]["passed"]=False
        self.save_fixture()
        self.assert_rejected()

    def test_scene_and_road_hashes_must_match_installed_runtime(self):
        self.planner["scene_geometry_sha256"]="wrong-scene"
        self.save_fixture()
        self.assert_rejected()
        self.planner["scene_geometry_sha256"]=self.scene_hash
        self.config["planning"]["jamming_trial_contract"]["road_boundary_policy_sha256"]="wrong-policy"
        self.save_fixture()
        self.assert_rejected()

    def test_complete_csv_is_finite_and_matches_drive_trajectory(self):
        normal=(self.trial/"ego_trace.csv").read_text()
        changed="elapsed_s,x,y,speed\n"+"".join(f"{i},{float(i)},5.,1.\n" for i in range(22))
        for contents in (normal+"22,22.,nan,0.,1.\n",changed):
            (self.trial/"ego_trace.csv").write_text(contents)
            self.assert_rejected()

    def test_tail_timing_difference_outside_mapped_tunnel_keeps_original_duplicate_criterion(self):
        rows=(self.trial/"ego_trace.csv").read_text().splitlines()
        rows[-1]="21,21.55,0.,0.,1."
        (self.trial/"ego_trace.csv").write_text("\n".join(rows)+"\n")
        self.assertGreater(21.55-21.,.25)
        self.assertGreater(21.,self.trace_geometry["length_m"])
        self.assertEqual(self.approve(),self.candidate_id)
        self.assertEqual(policy_module.approved_ids(self.store,self.policy),{self.candidate_id})

    def test_insufficient_genuine_mapped_coverage_is_not_a_valid_duplicate(self):
        (self.trial/"ego_trace.csv").write_text("elapsed_s,x,y,speed\n0,0,0,0\n1,1,0,1\n2,2,0,1\n")
        self.assert_rejected()

    def test_sealed_approval_rejects_later_proof_tampering(self):
        self.approve()
        path=self.trial/"drive_trial.json"
        path.write_text(path.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"proof changed"):
            policy_module.approved_ids(self.store,self.policy)

    def test_changed_policy_never_reuses_existing_approvals(self):
        self.approve()
        changed=copy.deepcopy(self.policy)
        changed["expert_speed"]["max_velocity_kmh"]=43.
        self.assertEqual(policy_module.approved_ids(self.store,changed),set())
        changed=copy.deepcopy(self.policy)
        changed["library_sha256"]="different-curated-library"
        self.assertEqual(policy_module.approved_ids(self.store,changed),set())

    def test_multiple_physical_drives_do_not_collapse_into_one_approval(self):
        self.state["attempts"].append(copy.deepcopy(self.entry))
        self.save_fixture()
        self.assert_rejected()

    def test_completed_pre_drive_retry_does_not_replace_latest_actual_proof(self):
        retry={"committed":True,"recovery_verified":True,"trial_dir":str(self.live/"attempts/failed_preflight/trial"),
               "outcome":{"actual_driving":False,"strict_pass":False}}
        self.state["attempts"].insert(0,retry)
        self.save_fixture()
        self.assertEqual(self.approve(),self.candidate_id)

    def test_additional_actual_failure_cannot_be_hidden_by_passed_summary(self):
        failed={"committed":True,"recovery_verified":True,"trial_dir":str(self.live/"attempts/actual_failed/trial"),
                "outcome":{"actual_driving":True,"strict_pass":False}}
        self.state["attempts"].insert(0,failed)
        self.save_fixture()
        self.assert_rejected()

    def test_fresh_candidate_formatting_can_change_without_changing_immutable_layout(self):
        (self.trial/"candidate.json").write_text(json.dumps(self.candidate,indent=3)+"\n")
        self.assertNotEqual(policy_module.digest(self.trial/"candidate.json"),self.candidate_id)
        self.assertEqual(self.approve(),self.candidate_id)
        self.assertEqual(policy_module.approved_ids(self.store,self.policy),{self.candidate_id})

    def test_semantically_changed_fresh_candidate_is_rejected(self):
        changed=copy.deepcopy(self.candidate)
        changed["obstacles"][0]["position"]["y"]=4.
        self.write(self.trial/"candidate.json",changed)
        self.assert_rejected()

    def test_environment_seed_must_match_selection_seed(self):
        self.selection["environment_profile"]["run_seed"]=43
        self.save_fixture()
        self.assert_rejected()

    def test_unified_environment_must_change_previous_weather(self):
        self.selection["incoming_environment_history"]={"run_seed":41,"weather":"SUNNY","time_hour":13}
        self.save_fixture()
        self.assert_rejected()

    def test_unified_environment_must_change_previous_time(self):
        self.selection["incoming_environment_history"]={"run_seed":41,"weather":"FOGGY","time_hour":11}
        self.save_fixture()
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
