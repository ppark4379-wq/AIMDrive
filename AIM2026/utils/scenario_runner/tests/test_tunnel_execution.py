"""Tunnel regression checks: execution batch."""

import copy, json, tempfile, unittest, os, signal
from pathlib import Path
from unittest.mock import patch, Mock
from ..scenarios.full_loop_parts.tunnel_static_parts import diversity, execution_policy as policy_module
from ..tools import run_tunnel_static_batch as batch


# Only fresh physical success under the current inherited policy is approved.

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


# Batch accounting, fixed sources, recovery barriers and process ownership.

def trial_files(directory, *, passed=True, moved=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    proof = {"status": "passed" if passed else "failed", "completed": passed,
             "collision_free": True, "returned_to_original_lane": passed, "fixture_stationary": True,
             "completion_conditions": {"whole_body_in_original_lane": passed}, "ego_stopped_and_parked": True,
             "trajectory_checks": 10, "failure_reason": "" if passed else "bounded test failure"}
    batch.atomic_json(directory/"drive_trial.json", proof)
    batch.atomic_json(directory/"planner_diagnostics.json", {"calls": 20, "validated_path_seen": passed})
    batch.atomic_json(directory/"manifest.json", {"status": "passed" if passed else "failed",
        "expert_verified": passed, "live_avoidance_verified": passed,
        "fixture_result": {"status": "valid_driving_trial" if passed else "failed", "cleanup": {"verified": True}}})
    values = "0,0,0,0,0\n1,0.01,0,0,.051\n2,1,0,0,10\n" if moved else "0,0,0,0,0\n1,.001,0,0,.051\n2,.002,0,0,.02\n"
    (directory/"ego_trace.csv").write_text("elapsed_s,x,y,yaw_deg,speed\n"+values)


class FakeBatch(batch.Batch):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.events = []
        self.reject_first = False
        self.fail_recovery = False

    def helper(self, action, *, directory=None):
        self.events.append(action)
        if action == "recover" and directory and Path(directory).name == "recovery" and self.fail_recovery:
            raise RuntimeError("recovery failed")
        self.state["recovery_checks"] += int(action == "recover")
        self.save()
        return {"returncode": 0}

    def candidate(self, case, attempt):
        return {"seed": 42+attempt, "candidate_index": attempt}

    def process(self, command, log_path, timeout, **kwargs):
        self.events.append("trial")
        target = Path(command[command.index("--output")+1])
        if self.reject_first and len(self.state["attempts"]) == 1:
            target.mkdir(parents=True, exist_ok=True)
            batch.atomic_json(target/"manifest.json", {"status": "failed", "failure_reason": "screening rejection"})
        else:
            trial_files(target)
        return {"returncode": 0, "timed_out": False, "forced_kill": False}


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = self.root/"helper.sh"
        self.helper.write_text("#!/bin/sh\nexit 0\n")
        self.golden = {}
        for kind in batch.KINDS:
            self.golden[kind] = self.root/(kind+".json")
            self.golden[kind].write_text('{"seed":42}')
        self.snapshot = {"source": "fixed"}

    def args(self, output=None, **kwargs):
        argv = ["--output", str(output or self.root/"batch"), "--recovery-helper", str(self.helper), "--actual-runs", "3"]
        for kind, path in self.golden.items():
            argv.extend(("--"+kind+"-candidate", str(path)))
        result = batch.build_parser().parse_args(argv)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result

    def make(self, args=None):
        return FakeBatch(args or self.args(), snapshot_fn=lambda *args: copy.deepcopy(self.snapshot),
                         global_lock_path=self.root/"global.lock")

    def test_default24_plan_has_eight_of_each_and_three_known_starting_cases(self):
        plan = batch.make_plan(24)
        self.assertEqual({kind: sum(row["kind"] == kind for row in plan) for kind in batch.KINDS},
                         {"woodbox": 8, "sedan": 8, "mixed": 8})
        self.assertEqual(sum(row["golden"] for row in plan), 3)

    def test_existing_output_with_foreign_baseline_and_no_manifest_is_first_run(self):
        output = self.root/"batch"
        output.mkdir()
        (output/"foreign_actor_inventory.json").write_text('{"actor_ids":["Ego","2"]}')
        worker = self.make()
        result = worker.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(worker.events, ["recover", "trial", "recover", "trial", "recover", "trial", "recover", "recover"])

    def test_screen_rejection_does_not_count_and_is_recovered_before_next_trial(self):
        worker = self.make()
        worker.reject_first = True
        result = worker.run()
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(result["preflight_rejects"], 1)
        self.assertEqual(len(result["attempts"]), 4)
        self.assertTrue(all(row["recovery_verified"] for row in result["attempts"]))

    def test_failed_recovery_blocks_next_trial_and_never_commits_pending_case(self):
        worker = self.make()
        worker.fail_recovery = True
        with self.assertRaisesRegex(RuntimeError, "recovery failed"):
            worker.run()
        self.assertEqual(worker.events.count("trial"), 1)
        self.assertEqual(worker.state["actual_drives"], 0)
        self.assertFalse(worker.state["attempts"][0]["committed"])

    def test_source_mismatch_resume_never_executes_recovery_or_scene_actions(self):
        original = self.make()
        original.acquire()
        original.release()
        self.snapshot = {"source": "changed"}
        resumed = self.make(self.args(resume=True))
        with self.assertRaisesRegex(RuntimeError, "Frozen batch sources"):
            resumed.run()
        self.assertEqual(resumed.events, [])

    def test_global_lock_prevents_a_second_output_from_controlling_same_simulator(self):
        first = self.make()
        first.acquire()
        self.addCleanup(first.release)
        second = self.make(self.args(output=self.root/"other"))
        with self.assertRaisesRegex(RuntimeError, "global MORAI"):
            second.run()
        self.assertEqual(second.events, [])

    def test_suspension_noise_cannot_count_as_actual_driving(self):
        target = self.root/"noise"
        trial_files(target, passed=False, moved=False)
        result = batch.classify_trial(target)
        self.assertTrue(result["expert_entered"])
        self.assertFalse(result["actual_driving"])
        self.assertFalse(result["strict_pass"])

    def test_strict_pass_requires_cleanup_and_all_completion_fields(self):
        target = self.root/"pass"
        trial_files(target)
        self.assertTrue(batch.classify_trial(target)["strict_pass"])
        manifest = batch.read_json(target/"manifest.json")
        manifest["fixture_result"]["cleanup"]["verified"] = False
        batch.atomic_json(target/"manifest.json", manifest)
        self.assertFalse(batch.classify_trial(target)["strict_pass"])

    def test_resumed_pending_attempt_is_recovered_and_counted_exactly_once(self):
        worker = self.make()
        worker.acquire()
        target = self.root/"pending"
        trial_files(target)
        worker.state["attempts"] = [{"case_index": 0, "kind": "woodbox", "trial_dir": str(target), "committed": False}]
        worker.save()
        worker.release()
        resumed = self.make(self.args(resume=True))
        result = resumed.run()
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(sum(entry["case_index"] == 0 for entry in result["attempts"]), 1)

    def test_unknown_resumed_pid_identity_is_not_signaled_or_recovered(self):
        worker = self.make()
        worker.acquire()
        worker.state["inflight"] = {"stage": "trial", "process_pid": 123, "process_pgid": 123, "process_start_ticks": "old"}
        worker.save()
        worker.release()
        resumed = self.make(self.args(resume=True))
        with patch.object(resumed, "process_identity", return_value={"root": "unrelated", "pgid": 123,
                "command": ["other"], "start_ticks": "new"}), patch.object(batch.os, "killpg") as kill, self.assertRaisesRegex(RuntimeError, "identity is uncertain"):
            resumed.run()
        kill.assert_not_called()
        self.assertEqual(resumed.events, [])

    def test_process_save_exception_still_interrupts_and_waits_for_child_before_recovery(self):
        worker = self.make()
        worker.state = {"inflight": {}}
        process = Mock(pid=123, returncode=130)
        process.poll.side_effect = [None, None, 130]
        with patch.object(batch.subprocess, "Popen", return_value=process), patch.object(worker, "save", side_effect=OSError("disk full")), patch.object(
                worker, "send_group") as send, self.assertRaisesRegex(OSError, "disk full"):
            batch.Batch.process(worker, ["script"], self.root/"child.log", 240.)
        send.assert_called_with(process, signal.SIGINT)
        process.wait.assert_called()

    def test_expanded_default30_preserves_legacy24_and_explicit12_in_either_order(self):
        base = ["--output", str(self.root), "--recovery-helper", str(self.helper)]
        self.assertEqual(batch.build_parser().parse_args(base).actual_runs, 24)
        self.assertEqual(batch.build_parser().parse_args(base+["--expanded-models"]).actual_runs, 30)
        for flags in (["--expanded-models", "--actual-runs", "12"], ["--actual-runs", "12", "--expanded-models"]):
            self.assertEqual(batch.build_parser().parse_args(base+flags).actual_runs, 12)

    def test_expanded_balanced30_rotates_every_20_static_and12_vehicle_model(self):
        statics = {"static_%02d" %i: {} for i in range(20)}
        vehicles = {"vehicle_%02d" %i: {} for i in range(12)}
        plan, counts = batch.expanded_plan(30, statics, vehicles, 7)
        self.assertEqual({kind: sum(row["kind"] == kind for row in plan) for kind in batch.EXPANDED_KINDS},
                         {kind: 6 for kind in batch.EXPANDED_KINDS})
        self.assertTrue(all(value > 0 for rows in counts.values() for value in rows.values()))
        self.assertEqual(plan, batch.expanded_plan(30, statics, vehicles, 7)[0])
        for row in plan:
            if row["kind"] in ("static_two", "vehicle_two"):
                self.assertEqual(len({item["model"] for item in row["models"]}), 2)

    def test_fixed_triples_run_last_and_duplicate_species_do_not_inflate_coverage_capacity(self):
        triples = [{"kind": "static_three", "path": "triple_%d.json" %i,
                    "models": [{"actor_kind": "obstacle", "model": "S0"}]*3} for i in range(3)]
        plan, _ = batch.expanded_plan(30, {"S%d" %i: {} for i in range(22)}, {"V": {}}, 7, triples)
        self.assertTrue(all("fixed_candidate" in row for row in plan[-3:]))
        self.assertTrue(all("fixed_candidate" not in row for row in plan[:-3]))
        with self.assertRaisesRegex(ValueError, "coverage exceeds"):
            batch.expanded_plan(30, {"S%d" %i: {} for i in range(23)}, {"V": {}}, 7, triples)

    def expanded_fixture(self, *, changed_dimensions=False):
        acceptance = self.root/"acceptance.json"
        footprint = {"width_m": 1., "length_m": 2., "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 1.347}
        entries = []
        for kind, name in (("obstacle", "S"), ("vehicle", "V")):
            path = self.root/(name+"_preview.json")
            planned = dict(footprint, width_m=1.1 if changed_dimensions else 1.)
            batch.atomic_json(path, {"planned": {"obstacles": [{"planned_bbox": planned}]}})
            entries.append({"actor_kind": kind, "model": name, "preview_path": str(path)})
        batch.atomic_json(acceptance, {"models": entries})
        args = batch.build_parser().parse_args(["--output", str(self.root/"expanded"), "--recovery-helper", str(self.helper),
            "--expanded-models", "--model-acceptance", str(acceptance)])
        pools = ({"S": footprint}, {"V": footprint})
        return args, pools

    def expanded_load(self, args, pools):
        return patch.multiple(batch, load_configuration=Mock(return_value=({"paths": {"mgeo_root": "map"}}, None, None)),
            MGeoMapLoader=Mock(return_value="loader"), load_static_models=Mock(return_value=pools[0]),
            load_vehicle_models=Mock(return_value=pools[1]), tunnel_geometry=Mock(return_value={"source_sha256": "current-map"}),
            accepted_model_pools=Mock(return_value=pools))

    def test_expanded_inputs_validate_acceptance_provenance_and_pin_model_bytes(self):
        args, pools = self.expanded_fixture()
        with self.expanded_load(args, pools):
            models, contract = batch.load_expanded_inputs(args)
            batch.accepted_model_pools.assert_called_once_with(args.model_acceptance, pools[0], pools[1], source_geometry_sha256="current-map")
        self.assertEqual(len(contract["plan"]), 30)
        self.assertEqual(contract["acceptance_sha256"], batch.digest(args.model_acceptance))
        self.assertEqual(contract["model_sha256"]["obstacle"]["S"], batch.fingerprint(pools[0]["S"]))
        self.assertEqual(contract["rotation_fingerprint"], batch.fingerprint(contract["plan"]))

    def test_changed_accepted_model_dimensions_are_rejected_before_scene_helper(self):
        args, pools = self.expanded_fixture(changed_dimensions=True)
        with self.expanded_load(args, pools), self.assertRaisesRegex(ValueError, "dimensions changed"):
            batch.load_expanded_inputs(args)

    def test_expanded_requires_acceptance_and_unaccepted_selection_is_rejected(self):
        args = batch.build_parser().parse_args(["--output", str(self.root), "--recovery-helper", str(self.helper), "--expanded-models"])
        with self.assertRaisesRegex(ValueError, "requires --model-acceptance"):
            batch.validate_args(args)
        args, pools = self.expanded_fixture()
        args.static_model = ["unaccepted"]
        with self.expanded_load(args, pools), self.assertRaisesRegex(ValueError, "Unaccepted selected"):
            batch.load_expanded_inputs(args)

    def test_three_candidate_cannot_enter_plan_with_stale_or_failed_pre_screening(self):
        args, pools = self.expanded_fixture()
        path = self.root/"triple.json"
        batch.atomic_json(path, {"count": 3, "experimental_three_actor_trial": True,
            "obstacles": [{"actor_kind": "obstacle", "model": "S"}]*3})
        args.three_candidate = [path]
        batch.atomic_json(path.with_name("screening.json"), {"status": "nominal_path_found", "path_found": True,
                                                         "geometry_checks": {"passed": True}})
        with self.expanded_load(args, pools), patch.object(batch, "provenance_matches", return_value=False), self.assertRaisesRegex(ValueError, "current passed screening"):
            batch.load_expanded_inputs(args)

    def test_expanded_resume_rejects_changed_acceptance_bytes_before_recovery(self):
        args, pools = self.expanded_fixture()
        with self.expanded_load(args, pools):
            original = self.make(args)
            original.acquire()
            original.release()
        document = batch.read_json(args.model_acceptance)
        document["source_revision_changed"] = True
        batch.atomic_json(args.model_acceptance, document)
        args.resume = True
        with self.expanded_load(args, pools):
            resumed = self.make(args)
            with self.assertRaisesRegex(RuntimeError, "Resume configuration differs"):
                resumed.run()
        self.assertEqual(resumed.events, [])

    def bound_candidate(self):
        return {"seed": 42, "candidate_index": 0, "count": 3, "experimental_three_actor_trial": True,
                "obstacles": [{"position": {"x": x, "y": 0., "z": 28.}, "yaw_deg": 12.,
                    "planned_bbox": {"width_m": 2., "length_m": 5., "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 0.}}
                    for x in (25., 50., 75.)]}

    def test_bound_conservative_triple_envelopes_are_accepted_without_rewriting_catalogue(self):
        candidate = self.bound_candidate()
        original = copy.deepcopy(candidate)
        boxes = [dict(batch.bbox_world(row), width_m=2.2, length_m=6.3) for row in candidate["obstacles"]]
        screen = {"candidate_seed": 42, "candidate_index": 0, "candidate_document_sha256": batch.fingerprint(candidate),
                  "obstacle_input": boxes}
        self.assertTrue(batch.screening_binds_candidate(candidate, screen))
        self.assertEqual(candidate, original)
        screen.pop("candidate_document_sha256")
        self.assertFalse(batch.screening_binds_candidate(candidate, screen))

    def test_bound_triple_rejects_shrunk_unrelated_or_different_document_geometry(self):
        candidate = self.bound_candidate()
        good = {"candidate_seed": 42, "candidate_index": 0, "candidate_document_sha256": batch.fingerprint(candidate),
                "obstacle_input": [dict(batch.bbox_world(row), width_m=2.2, length_m=6.3) for row in candidate["obstacles"]]}
        for variation in ("shrink", "relocate", "wrong_document", "reordered_rows"):
            with self.subTest(variation=variation):
                screen = copy.deepcopy(good)
                if variation == "shrink":
                    screen["obstacle_input"][0]["width_m"] = 1.9
                elif variation == "relocate":
                    screen["obstacle_input"][0]["center_xyz"] = [500., 500., 28.]
                elif variation == "wrong_document":
                    screen["candidate_document_sha256"] = "0"*64
                else:
                    screen["obstacle_input"] = list(reversed(screen["obstacle_input"]))
                self.assertFalse(batch.screening_binds_candidate(candidate, screen))

    def expanded_policy(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts.advanced_patterns import PATTERN_COUNT_CAPS
        return {"selected_pattern": "short_detour", "footprint_domain": "two_lane_physical_road",
            "station_window_m": [20., 105.], "static_yaw_offsets_deg": {"Steel_Barricade": [90.]},
            "layout": {"min_pair_gap_m": 18., "max_width_m": 6.7},
            "campaign": {"patterns": list(PATTERN_COUNT_CAPS)},
            "actor_count_caps": {name: list(counts) for name, counts in PATTERN_COUNT_CAPS.items()}}

    def test_selected_campaign_covers_every_pattern_count_and_model_deterministically(self):
        policy = self.expanded_policy()
        statics = {"S%d" %i: {"width_m": .8, "length_m": 1.2} for i in range(11)}
        vehicles = {"V%d" %i: {"width_m": 2.2, "length_m": 4.7} for i in range(24)}
        plan, counts = batch.campaign_plan(54, statics, vehicles, 17, policy)
        expected = {(name, count) for name, caps in policy["actor_count_caps"].items() for count in caps}
        self.assertEqual({(row["pattern"], row["count"]) for row in plan}, expected)
        self.assertTrue(all(n>0 for pool in counts.values() for n in pool.values()))
        self.assertEqual((plan, counts), batch.campaign_plan(54, statics, vehicles, 17, policy))
        self.assertTrue(all(row["requested_pattern"] == row["pattern"] for row in plan))
        self.assertTrue(all(item["actor_kind"]=="obstacle" for row in plan if row["pattern"]=="window_bypass" for item in row["models"]))
        self.assertTrue(all(row["pattern"]=="long_group_bypass" for row in plan if any(item["actor_kind"]=="vehicle" for item in row["models"])))

    def test_selected_campaign_refuses_insufficient_layout_or_species_slots(self):
        policy = self.expanded_policy()
        with self.assertRaisesRegex(ValueError, "pattern/count"):
            batch.campaign_plan(2, {"S": {}}, {}, 1, policy)
        with self.assertRaisesRegex(ValueError, "coverage exceeds"):
            batch.campaign_plan(9, {"S%d" %i: {"width_m": .8, "length_m": 1.2} for i in range(100)}, {}, 1, policy)

    def test_incompatible_recipe_is_recorded_without_forcing_wide_vehicle_slots(self):
        policy=self.expanded_policy(); policy["campaign"]["patterns"]=["window_bypass", "long_group_bypass"]
        plan, counts=batch.campaign_plan(8, {}, {"WideCar": {"width_m": 2.2, "length_m": 4.7}}, 1, policy)
        self.assertTrue(all(row["pattern_supported"] is False for row in plan if row["pattern"]=="window_bypass"))
        self.assertGreater(counts["vehicle"]["WideCar"], 0)

    def test_batch_passes_exact_selected_pattern_domain_yaw_and_three_actor_cap(self):
        worker = self.make(); worker.models = (Mock(), {"Steel_Barricade": {}}, {})
        worker.expanded_contract = {"selected_layout_policy": self.expanded_policy()}
        case = {"index": 0, "kind": "alternating_three_3", "golden": False, "count": 3,
                "pattern": "alternating_three", "models": [{"actor_kind": "obstacle", "model": "Steel_Barricade"}]*3}
        value = {"obstacles": copy.deepcopy(case["models"])}
        with patch.object(batch, "generate_candidate", return_value=value) as generate:
            self.assertEqual(batch.Batch.candidate(worker, case, 0), value)
        options = generate.call_args.kwargs
        self.assertEqual(options["pattern"], "alternating_three")
        self.assertEqual(options["footprint_domain"], "two_lane_physical_road")
        self.assertEqual(options["static_yaw_offsets_deg"], {"Steel_Barricade": [90.]})
        self.assertEqual(options["station_window_m"], [20., 105.])
        self.assertIs(options["allow_three"], True)

    def test_trial_requested_layout_never_substitutes_for_actual_trajectory_pattern(self):
        target = self.root/"patterns"; trial_files(target)
        document = batch.read_json(target/"drive_trial.json")
        document.update(requested_pattern="alternating_three", realized_pattern="single_opposing_lane_excursion",
                        pattern_classification={"s_avoidance_verified": False})
        batch.atomic_json(target/"drive_trial.json", document)
        result = batch.classify_trial(target)
        self.assertEqual(result["requested_pattern"], "alternating_three")
        self.assertEqual(result["realized_pattern"], "single_opposing_lane_excursion")
        self.assertFalse(result["pattern_classification"]["s_avoidance_verified"])

    def test_selected_config_sha_and_policy_validation_are_explicit(self):
        import yaml
        policy = self.expanded_policy(); policy.update(schema="aimdrive_tunnel_static_config_v1", map="R_KR_PR_K-city_2025")
        path = self.root/"expanded.yaml"; path.write_text(yaml.safe_dump(policy))
        _, contract = batch.campaign_options(path)
        self.assertEqual(contract["config_sha256"], batch.digest(path))
        policy["actor_count_caps"]["alternating_three"] = [2]
        path.write_text(yaml.safe_dump(policy))
        with self.assertRaisesRegex(ValueError, "count caps"): batch.campaign_options(path)

    def prepare_pattern_worker(self, *, geometry=True):
        worker=self.make()
        plan=[dict(index=i,kind="short_detour_1",golden=False,count=1,
            requested_pattern="short_detour",pattern="short_detour",pattern_supported=True,
            models=[{"actor_kind":"obstacle","model":"WoodBox"}]) for i in range(2)]
        worker.expanded_contract={"selected_layout_policy": self.expanded_policy(),
            "model_sha256": {"obstacle": {"WoodBox": "fixed"}, "vehicle": {}}, "plan": plan}
        worker.snapshot=lambda: copy.deepcopy(self.snapshot)
        worker.args.max_attempts_per_case=2
        worker.acquire()
        worker.state["plan"]=plan
        worker.state["goal_actual_drives"]=2; worker.args.actual_runs=2
        worker.candidate=Mock(side_effect=[batch.NoCandidateGeometry("road body rejected")]*2+
                              [{"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]}])
        return worker

    def test_advanced_geometry_exhaustion_recovers_excludes_then_runs_next_case(self):
        worker=self.prepare_pattern_worker()
        self.addCleanup(worker.release)
        worker.execute()
        self.assertEqual(len(worker.state["excluded_placements"]),1)
        self.assertEqual(worker.state["actual_drives"],1)
        self.assertEqual(worker.state["status"],"completed_with_exclusions")
        self.assertTrue(all(row.get("recovery_verified") for row in worker.state["attempts"]))
        self.assertEqual(worker.state["actual_model_occurrences"]["obstacle"]["WoodBox"],1)
        self.assertEqual(worker.state["strict_pass_model_occurrences"]["obstacle"]["WoodBox"],1)

    def test_advanced_transport_or_protocol_preflight_never_continues_as_geometry(self):
        worker=self.prepare_pattern_worker()
        self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def failed(command, log_path, timeout, **kwargs):
            target=Path(command[command.index("--output")+1]); target.mkdir(parents=True)
            batch.atomic_json(target/"manifest.json",{"status":"failed","failure_reason":"StatusCode.UNAVAILABLE"})
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=failed
        with self.assertRaisesRegex(RuntimeError,"outside bounded geometric rejection"): worker.execute()
        self.assertEqual(worker.candidate.call_count,1)
        self.assertFalse(worker.state.get("excluded_placements"))

    def metadata_ordering_files(self, target, *, changed_publisher=False):
        target=Path(target); target.mkdir(parents=True,exist_ok=True)
        failure={"schema":"aimdrive_native_static_metadata_failure_v1","stage":"preflight_static_binding",
            "actor_kind":"obstacle","reason":"publisher_changed" if changed_publisher else "source_sequence_regression",
            "previous_publisher":"/native_bridge","current_publisher":"/replacement" if changed_publisher else "/native_bridge",
            "previous_source_sequence":100,"current_source_sequence":99,
            "previous_source_stamp_ns":1000,"current_source_stamp_ns":1100,
            "retryable_same_publisher_ordering":not changed_publisher}
        batch.atomic_json(target/"manifest.json",{"status":"failed","failure_reason":"Native static metadata ordering",
            "fixture_result":{"cleanup":{"verified":True},"native_metadata_failure":failure}})
        batch.atomic_json(target/"planner_diagnostics.json",{"calls":0})
        return failure

    def test_ordering_classifier_requires_static_same_publisher_preflight_and_cleanup(self):
        target=self.root/"ordering"; self.metadata_ordering_files(target)
        self.assertTrue(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        for key,value in (("actor_kind","vehicle"),("current_publisher","/replacement"),
                          ("stage","runtime"),("current_source_sequence",101)):
            self.metadata_ordering_files(target)
            document=batch.read_json(target/"manifest.json")
            document["fixture_result"]["native_metadata_failure"][key]=value
            batch.atomic_json(target/"manifest.json",document)
            self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        self.metadata_ordering_files(target)
        batch.atomic_json(target/"planner_diagnostics.json",{"calls":1})
        self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        self.metadata_ordering_files(target)
        document=batch.read_json(target/"manifest.json");document["fixture_result"]["cleanup"]["verified"]=False
        batch.atomic_json(target/"manifest.json",document)
        self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])

    def test_advanced_ordering_retries_only_after_recovery_and_then_counts_actual_pass(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        attempts=[]
        def process(command, log, timeout, **kwargs):
            target=Path(command[command.index("--output")+1]);attempts.append(target);worker.events.append("trial")
            if len(attempts)==1:self.metadata_ordering_files(target)
            else:trial_files(target)
            return {"returncode":2 if len(attempts)==1 else 0,"timed_out":False,"forced_kill":False}
        worker.process=process;worker.execute()
        self.assertEqual(worker.state["actual_drives"],2)
        self.assertEqual(worker.state["preflight_rejects"],1)
        self.assertFalse(worker.state.get("excluded_placements"))
        self.assertEqual(worker.events[:5],["recover","trial","recover","trial","recover"])

    def test_ordering_retry_exhaustion_stops_and_never_becomes_geometry_exclusion(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def process(command, log, timeout, **kwargs):
            self.metadata_ordering_files(Path(command[command.index("--output")+1]))
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=process
        with self.assertRaisesRegex(RuntimeError,"bounded attempts"):worker.execute()
        self.assertEqual(worker.candidate.call_count,2)
        self.assertFalse(worker.state.get("excluded_placements"))
        self.assertTrue(all(row["recovery_verified"] for row in worker.state["attempts"]))

    def test_resumed_ordering_is_recovered_once_and_remains_same_pending_case(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        target=self.root/"resume_ordering";self.metadata_ordering_files(target)
        worker.state["attempts"]=[{"case_index":0,"kind":"short_detour_1","trial_dir":str(target),"committed":False,
            "process":{"returncode":2,"timed_out":False,"forced_kill":False}}]
        worker.reconcile_resume()
        self.assertEqual(worker.events,["recover"])
        self.assertEqual(worker.state["preflight_rejects"],1)
        self.assertEqual(worker.state["next_case_index"],0)
        worker.reconcile_resume()
        self.assertEqual(worker.state["preflight_rejects"],1)

    def test_advanced_publisher_replacement_stops_after_first_verified_recovery(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def process(command, log, timeout, **kwargs):
            self.metadata_ordering_files(Path(command[command.index("--output")+1]),changed_publisher=True)
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=process
        with self.assertRaisesRegex(RuntimeError,"outside bounded geometric rejection"):worker.execute()
        self.assertEqual(worker.candidate.call_count,1)
        self.assertFalse(worker.state.get("excluded_placements"))


if __name__ == "__main__":
    unittest.main()
