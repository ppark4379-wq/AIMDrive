"""Collection catalog/storage changes cannot replace fresh tunnel proof."""
import copy
from contextlib import ExitStack, nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT/"data_collection/internal_sensor_record"
sys.path[:0]=[str(PROJECT),str(INTERNAL),str(INTERNAL/"tools")]
from async_capture.scenario_catalog import COLLECTION_SCENARIOS, scenario_zone_contract, validate_tunnel_static_collection
from dataset_layout import scenario_folder as local_folder, collecting_campaign, final_campaign
from nas_publish import scenario_folder as nas_folder
from build_sensor_record_training_dataset import required_training_coverage
from scenario_runner.scenarios.segment_modes.modes import PRIMARY_SCENARIOS
from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import diversity
from async_capture import scenario_catalog as catalog
from async_capture.journal import RawWriter
from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import source_compatibility as compatibility


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


class TunnelCollectionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.episode=self.root/"campaign/run/episode_000"
        self.guards=self.episode/"tunnel_static"
        self.guards.mkdir(parents=True)
        source=self.root/"immutable_runtime.py"
        source.write_text("# isolated immutable producer\n")
        self.sources={str(source):digest(source)}
        ref={"raw_actor_id":"7","actor_kind":"obstacle","model":"Box","center_xyz":[10.,3.5,28.],
             "position_xyz":[10.,3.5,28.],"heading_deg":0.,"length_m":1.,"width_m":1.}
        boundary={"body":{"rear_m":.85,"front_m":3.85,"width_m":2.17},"source_sha256":"isolated-map"}
        boundary_sha=identity(boundary)
        self.candidate={"count":1,"obstacles":[{"actor_kind":"obstacle","model":"Box","position":{"x":10.,"y":3.5,"z":28.}}]}
        screen={"status":"nominal_path_found","path_found":True,"geometry_checks":{"passed":True},
                "road_boundary_policy":boundary,"road_boundary_policy_sha256":boundary_sha}
        self.documents={
            "candidate":self.candidate,
            "episode":{"scenario_name":"full_loop_tunnel_static","global_config":{"paths":{"mgeo_root":"isolated-map"}}},
            "outcome":{"status":"mission_completed","full_mission_completed":True},
            "raw_seal":{"status":"completed"},
            "scene_restore":{"status":"restored"},
            "preview_report":{"status":"valid_driving_trial","cleanup":{"verified":True},"observed":[ref],"measured_route_screen":screen},
            "drive_report":{"status":"passed","collision_objects":[],"failed_completion_conditions":[],
                "completion_conditions":{key:True for key in diversity.STRICT_CONDITIONS},"runtime_source_sha256":self.sources,
                "owned_actor_geometry":[ref],"observed_object_ids":["7"],"road_boundary_policy":boundary,
                "road_boundary_policy_sha256":boundary_sha,"trajectory":[{"position_xyz":[float(x),0.,28.]} for x in range(22)]},
            "planner_diagnostics":{"calls":50,"validated_path_seen":True,"selected_routes":["jamming_01.csv"],
                "scene_geometry_sha256":identity([ref]),"road_boundary_policy":boundary,"road_boundary_policy_sha256":boundary_sha}}
        self.documents["drive_report"].update({key:True for key in diversity.STRICT_FLAGS})
        names={"candidate":"candidate.json","preview_report":"preview.json","drive_report":"drive_trial.json",
               "planner_diagnostics":"planner_diagnostics.json","scene_restore":"scene_restore.json"}
        self.paths={role:self.guards/name for role,name in names.items()}
        self.paths.update(episode=self.episode/"episode.json",outcome=self.episode/"outcome.json",raw_seal=self.episode/"RAW_COMPLETE.json",ego_trace=self.guards/"ego_trace.csv")
        for role,document in self.documents.items():self.paths[role].write_text(json.dumps(document))
        self.paths["ego_trace"].write_text("elapsed_s,x,y,speed\n"+"".join(f"{x},{x},0.,1.\n" for x in range(22)))
        self.proof={"schema":"aimdrive_tunnel_static_collection_complete_v1","scenario":"full_loop_tunnel_static","status":"passed",
                    "selected_candidate_sha256":digest(self.paths["candidate"]),"library_sha256":"a"*64,
                    "fixture_cleanup_verified":True,"scene_restored":True,"raw_record_complete":True,
                    "source_files":self.sources,"runtime_source_sha256":self.sources,
                    "files":{role:{"path":str(path),"sha256":digest(path)} for role,path in self.paths.items()}}
        self.proof_path=self.guards/"COLLECTION_TUNNEL_COMPLETE.json"
        self.save_proof()
        own={"road_type":"2","width_start":3.5,"width_end":3.5,"points":[[0.,0.,28.],[20.,0.,28.]]}
        links={"A2256W000126":own,"A2256W000131":dict(own,points=[[20.,3.5,28.],[0.,3.5,28.]])}
        self.loader=SimpleNamespace(get_link=lambda key:links[key])

    def save_proof(self):
        self.proof_path.write_text(json.dumps(self.proof))

    def update(self,role):
        self.paths[role].write_text(json.dumps(self.documents[role]))
        self.proof["files"][role]["sha256"]=digest(self.paths[role])
        self.save_proof()

    def validate(self,episode=None):
        with patch("scenario_runner.utils.map_loader.MGeoMapLoader",return_value=self.loader):
            return validate_tunnel_static_collection(episode or self.episode,"full_loop_tunnel_static")

    def test_catalog_and_storage_have_dedicated_tunnel_without_changing_primary_modes(self):
        self.assertNotIn("full_loop_tunnel_static",PRIMARY_SCENARIOS)
        self.assertIn("full_loop_tunnel_static",COLLECTION_SCENARIOS)
        self.assertEqual(local_folder("full_loop_tunnel_static"),"tunnel_static_obstacle")
        self.assertEqual(nas_folder({"scenario_name":"full_loop_tunnel_static"}),"tunnel_static_obstacle")
        self.assertEqual({k for k,v in scenario_zone_contract("full_loop_tunnel_static").items() if v},{"tunnel_static"})
        self.assertEqual(required_training_coverage("full_loop_tunnel_static"),("waypoint_frames","object_positive_frames"))
        self.assertEqual(collecting_campaign(self.root,"full_loop_tunnel_static","t1"),self.root/"dataset/.collecting/tunnel_static_obstacle/t1")
        self.assertEqual(final_campaign(self.root,"full_loop_tunnel_static","t1"),self.root/"dataset/tunnel_static_obstacle/t1")

    def test_fresh_guard_capture_cleanup_restore_proof_is_required(self):
        valid=self.validate()
        self.assertTrue(valid["valid"],valid)
        self.proof_path.unlink()
        result=self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(result["required"])

    def test_current_guard_failures_cannot_be_replaced_by_historical_success(self):
        for key in ("fixture_stationary","returned_to_original_lane","ego_stopped_and_parked"):
            self.documents["drive_report"][key]=False
            self.update("drive_report")
            self.assertFalse(self.validate()["valid"])
            self.documents["drive_report"][key]=True
        self.documents["drive_report"]["completion_conditions"]["whole_body_in_original_lane"]=False
        self.update("drive_report")
        self.assertFalse(self.validate()["valid"])

    def test_source_scene_policy_or_raw_capture_mismatch_is_rejected(self):
        self.documents["planner_diagnostics"]["scene_geometry_sha256"]="wrong-scene"
        self.update("planner_diagnostics")
        self.assertFalse(self.validate()["valid"])
        self.documents["planner_diagnostics"]["scene_geometry_sha256"]=identity(self.documents["drive_report"]["owned_actor_geometry"])
        self.update("planner_diagnostics")
        self.documents["raw_seal"]["status"]="failed"
        self.update("raw_seal")
        self.assertFalse(self.validate()["valid"])

    def test_current_source_change_cannot_reuse_sealed_success(self):
        Path(next(iter(self.sources))).write_text("# different current producer\n")
        result=self.validate()
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["source_pins"])
        self.assertFalse(result["checks"]["runtime_source_pins"])

    def prepare_reviewed_goal_stop_capture(self):
        runtime = {}
        changed = (
            "scenario_runner/core/full_loop_drive_base.py",
            "scenario_runner/scenarios/segment_modes/scenario.py",
            "scenario_runner/scenarios/segment_modes/tunnel_static_trial.py",
        )
        unchanged = (
            "scenario_runner/adapters/expert_controller.py",
            "scenario_runner/adapters/expert_process.py",
            "scenario_runner/scenarios/segment_modes/expert.py",
            "scenario_runner/scenarios/segment_modes/tunnel_static_startup.py",
            "expert/src/expert/src/network/ros_manager.py",
        )
        for relative in changed + unchanged:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# captured producer: ' + relative + '\n')
            runtime[str(path.resolve())] = digest(path)
        registry = self.root / "scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/source_compatibility.py"
        registry.parent.mkdir(parents=True)
        registry.write_text('# original exact pins\n')
        registry_old = digest(registry)
        self.proof['runtime_source_sha256'] = dict(runtime)
        self.proof['source_files'] = dict(runtime, **{str(registry.resolve()): registry_old})
        self.documents['drive_report']['runtime_source_sha256'] = dict(runtime)
        self.documents['episode']['expert_root'] = str(self.root/'expert')
        self.documents['episode']['tunnel_static_trial'] = {
            'runtime_source_sha256': dict(runtime),
            'selected_candidate_sha256': self.proof['selected_candidate_sha256'],
        }
        self.update('episode')
        self.update('drive_report')
        writer = RawWriter(self.episode/'raw', {'episode': copy.deepcopy(self.documents['episode'])})
        self.assertTrue(writer.submit({'role': 'ego'}, b'isolated source binding record'))
        writer.close()
        self.paths['raw_seal'] = self.episode/'raw/RAW_COMPLETE.json'
        self.proof['files']['raw_seal'] = {'path': str(self.paths['raw_seal']),
                                         'sha256': digest(self.paths['raw_seal'])}
        for relative in changed:
            (self.root/relative).write_text('# reviewed normal-goal stop: '+relative+'\n')
        endpoint = {path: digest(path) for path in runtime}
        registry.write_text('# registry of exactly the reviewed stop hashes\n')
        for patcher in (
            patch.object(catalog, '_PROJECT_ROOT', self.root),
            patch.object(catalog, '_GOAL_STOP_REGISTRY_PAIR', (registry_old, digest(registry))),
            patch.object(compatibility, 'PROJECT_ROOT', self.root),
            patch.object(compatibility, 'PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256', runtime),
            patch.object(compatibility, 'PRE_MAIN_RUNTIME_SOURCE_SHA256', endpoint),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.save_proof()
        return runtime

    def test_exact_reviewed_stop_change_preserves_closed_producer_and_all_guards(self):
        runtime = self.prepare_reviewed_goal_stop_capture()
        recorded = (self.episode/'raw/session.json').read_bytes()
        result = self.validate()
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['reviewed_goal_stop_source_transition']['producer_source_pins_verified'])
        self.assertEqual(self.proof['runtime_source_sha256'], runtime)
        self.assertEqual((self.episode/'raw/session.json').read_bytes(), recorded)

    def test_reviewed_stop_change_cannot_admit_an_unreviewed_expert_or_registry(self):
        self.prepare_reviewed_goal_stop_capture()
        path = self.root/'expert/src/expert/src/network/ros_manager.py'
        original = path.read_bytes()
        path.write_text('# unrelated expert modification\n')
        self.assertFalse(self.validate()['valid'])
        path.write_bytes(original)
        registry = self.root/'scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/source_compatibility.py'
        registry.write_text('# another unreviewed registry\n')
        self.assertFalse(self.validate()['checks']['source_pins'])

    def test_reviewed_stop_change_requires_the_exact_raw_session_episode_binding(self):
        self.prepare_reviewed_goal_stop_capture()
        self.documents['episode']['seed'] = 999
        self.update('episode')
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertFalse(result['checks']['source_pins'])

    def test_reviewed_stop_change_keeps_failed_physical_conditions_rejected(self):
        self.prepare_reviewed_goal_stop_capture()
        self.documents['drive_report']['completion_conditions']['whole_body_in_original_lane'] = False
        self.update('drive_report')
        result = self.validate()
        self.assertTrue(result['checks']['source_pins'])
        self.assertFalse(result['checks']['actual_strict_trial'])
        self.assertFalse(result['valid'])

    def archive_reviewed_goal_stop_endpoint(self):
        runtime = self.prepare_reviewed_goal_stop_capture()
        endpoint = dict(compatibility.PRE_MAIN_RUNTIME_SOURCE_SHA256)
        registry = self.root/'scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/source_compatibility.py'
        endpoint[str(registry)] = digest(registry)
        archive = {}
        for path, sha in endpoint.items():
            blob = self.root/'archive'/(sha+'.source')
            blob.parent.mkdir(exist_ok=True)
            blob.write_bytes(Path(path).read_bytes())
            archive[(str(path), sha)] = blob
            Path(path).write_text('# changed current main source; do not execute archived code\n')
        def resolve(path, sha):
            blob = archive.get((str(path), sha))
            return blob if blob is not None and blob.is_file() and digest(blob) == sha else None
        patcher = patch.object(compatibility, 'archived_source_path', side_effect=resolve)
        patcher.start()
        self.addCleanup(patcher.stop)
        return runtime, archive

    def test_archived_reviewed_stop_endpoint_preserves_capture_not_new_expert_approval(self):
        runtime, _ = self.archive_reviewed_goal_stop_endpoint()
        raw = (self.episode/'raw/session.json').read_bytes()
        result = self.validate()
        self.assertTrue(result['valid'], result)
        self.assertFalse(result['current_expert_approval_transferred'])
        self.assertEqual(result['captured_source_archives'][next(iter(runtime))]['captured_sha256'],
                         runtime[next(iter(runtime))])
        self.assertEqual((self.episode/'raw/session.json').read_bytes(), raw)

    def test_missing_or_changed_archived_endpoint_keeps_capture_rejected(self):
        _, archive = self.archive_reviewed_goal_stop_endpoint()
        blob = next(iter(archive.values()))
        original = blob.read_bytes()
        blob.write_text('# archive bytes changed\n')
        self.assertFalse(self.validate()['valid'])
        blob.write_bytes(original)
        blob.unlink()
        self.assertFalse(self.validate()['valid'])

    def test_archived_sources_do_not_relax_body_collision_or_selected_layout(self):
        self.archive_reviewed_goal_stop_endpoint()
        self.documents['drive_report']['collision_objects'] = ['actual_contact']
        self.update('drive_report')
        result = self.validate()
        self.assertTrue(result['checks']['source_pins'])
        self.assertFalse(result['checks']['actual_strict_trial'])
        self.assertFalse(result['valid'])

    def test_archive_does_not_accept_unknown_captured_source_or_mixed_raw_producer(self):
        self.archive_reviewed_goal_stop_endpoint()
        self.proof['source_files'][str(self.root/'unknown.py')] = 'f'*64
        self.save_proof()
        self.assertFalse(self.validate()['checks']['source_pins'])
        del self.proof['source_files'][str(self.root/'unknown.py')]
        self.documents['episode']['seed'] = 12345
        self.update('episode')
        self.assertFalse(self.validate()['valid'])

    def test_owned_actor_and_body_policy_must_match_current_preview_and_planner(self):
        self.documents["drive_report"]["observed_object_ids"]=["8"]
        self.update("drive_report")
        self.assertFalse(self.validate()["checks"]["owned_actor_identity"])
        self.documents["drive_report"]["observed_object_ids"]=["7"]
        self.documents["drive_report"]["road_boundary_policy"]["body"]["width_m"]=1.
        self.update("drive_report")
        result=self.validate()
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["road_body_policy"])

    def test_finite_moving_csv_must_match_current_guarded_path(self):
        self.paths["ego_trace"].write_text("elapsed_s,x,y,speed\n"+"".join(f"{x},{x},5.,1.\n" for x in range(22)))
        self.proof["files"]["ego_trace"]["sha256"]=digest(self.paths["ego_trace"])
        self.save_proof()
        result=self.validate()
        self.assertTrue(result["checks"]["finite_csv_motion"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["csv_matches_guard_trajectory"])

    def test_unknown_or_nonfinite_csv_cannot_supply_actual_evidence(self):
        for body in ("0,0.,0.,0.\n1,1.,0.,0.\n", "".join(f"{x},{x},nan,1.\n" for x in range(22))):
            with self.subTest(body=body[:20]):
                self.paths["ego_trace"].write_text("elapsed_s,x,y,speed\n"+body)
                self.proof["files"]["ego_trace"]["sha256"]=digest(self.paths["ego_trace"])
                self.save_proof()
                result=self.validate()
                self.assertFalse(result["valid"])
                self.assertFalse(result["checks"]["finite_csv_motion"])

    def test_unsealed_file_change_is_rejected(self):
        path=self.paths["drive_report"]
        path.write_text(path.read_text()+" ")
        self.assertFalse(self.validate()["valid"])

    def test_no_foreign_actors_requires_explicit_parked_restore_evidence(self):
        report={"status":"nothing_to_restore","actors":[],"scene_was_moved":False,
                "parking":{"parked":True,"external_input_disabled":True}}
        self.documents["scene_restore"]=report
        self.update("scene_restore")
        result=self.validate()
        self.assertTrue(result["valid"],result)
        for key,value in (("actors",[{"actor_id":"2"}]),("scene_was_moved",True)):
            with self.subTest(key=key):
                self.documents["scene_restore"]=dict(report,**{key:value})
                self.update("scene_restore")
                self.assertFalse(self.validate()["valid"])
        for key in ("parked","external_input_disabled"):
            with self.subTest(parking=key):
                self.documents["scene_restore"]=dict(report,parking={"parked":True,"external_input_disabled":True,key:False})
                self.update("scene_restore")
                self.assertFalse(self.validate()["valid"])
        self.documents["scene_restore"]={"status":"nothing_to_restore"}
        self.update("scene_restore")
        self.assertFalse(self.validate()["valid"])

    def test_no_foreign_actors_accepts_only_verified_common_auto_d_brake_hold(self):
        parking={"parked":True,"external_input_disabled":False,"brake_hold_verified":True,
                 "control_publishers_absent":True,"gear_mode":4,"control_mode":3,
                 "brake":1.,"throttle":0.,"speed_norm_raw":0.}
        report={"status":"nothing_to_restore","actors":[],"scene_was_moved":False,"parking":parking}
        self.documents["scene_restore"]=report
        self.update("scene_restore")
        result=self.validate()
        self.assertTrue(result["valid"],result)
        for key,value in (("parked",False),("brake_hold_verified",False),("control_publishers_absent",False),
                          ("external_input_disabled",None),("gear_mode",1),("control_mode",1),
                          ("brake",.98),("throttle",.02),("speed_norm_raw",.051)):
            with self.subTest(key=key,value=value):
                self.documents["scene_restore"]=dict(report,parking=dict(parking,**{key:value}))
                self.update("scene_restore")
                rejected=self.validate()
                self.assertFalse(rejected["valid"],rejected)
                self.assertFalse(rejected["checks"]["foreign_scene_restored"])

    def test_staging_rename_resolves_only_same_episode_exact_hash_copies(self):
        destination=self.root/"final/run/episode_000"
        destination.parent.mkdir(parents=True)
        self.episode.rename(destination)
        result=self.validate(destination)
        self.assertTrue(result["valid"],result)

    def produce_collection_proof(self,fail_stage=None):
        from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import trial as owner, replay, runtime, safety, live
        from scenario_runner.scenarios.segment_modes import tunnel_static_trial as drive, tunnel_static_replay as facade
        from scenario_runner.scenarios.segment_modes.modes import TUNNEL_NORMAL_START
        source_candidate=self.root/"saved_candidate.json"
        source_candidate.write_bytes(self.paths["candidate"].read_bytes())
        library=self.root/"active_library.json"
        library.write_text("{}")
        selected={"id":digest(source_candidate),"paths":{"candidate":str(source_candidate)},
                  "file_sha256":{"candidate":digest(source_candidate)},"strata":{"count":"1"}}
        verified={"eligible":[selected],"eligible_count":121,"library_sha256":digest(library)}
        profile=self.root/"_campaign/full_loop_tunnel_static_profile.json"
        events=[]
        setup_calls=[]
        harness=self
        self.loader.link_set={}
        preview=copy.deepcopy(self.documents["preview_report"])
        preview["planned"]=self.candidate
        measured_body=copy.deepcopy(preview["measured_route_screen"]["road_boundary_policy"]["body"])
        preview["measured_route_screen"].update(measured_ego_body=measured_body)
        boundary_factory=Mock(return_value=SimpleNamespace())
        preview["measured_route_screen"]["road_boundary_policy"].update(
            approach_m=120.,departure_m=70.,planning_road_margin_m=.35,runtime_road_margin_m=.15)
        policy=preview["measured_route_screen"]["road_boundary_policy"]
        policy_sha=identity(policy)
        preview["measured_route_screen"]["road_boundary_policy_sha256"]=policy_sha
        for role in ("drive_report","planner_diagnostics"):
            self.documents[role]["road_boundary_policy"]=policy
            self.documents[role]["road_boundary_policy_sha256"]=policy_sha
        class Context:
            def __init__(self,candidate,observed,gc,output,boundary,monitor):
                self.references={str(row["raw_actor_id"]):row for row in observed["observed"]}
                self.report=copy.deepcopy(harness.documents["drive_report"])
                self.output_dir=Path(output)
                self.fail=Mock()
        class Scenario:
            def __init__(self,client,loader,gc,sc,*,output):
                self.grpc,self.cfg,self.global_cfg=client,sc,gc
                self.output=Path(output)
                self.output.mkdir(parents=True,exist_ok=False)
                harness.assertEqual(Path(sc["profile_state_path"]),profile)
            def run(self):
                self.grpc.start_world(SimpleNamespace(location=SimpleNamespace(
                    **dict(zip("xyz",TUNNEL_NORMAL_START.xyz)))))
                folder=self.tunnel_static_trial_context.output_dir
                for role,name in (("drive_report","drive_trial.json"),("planner_diagnostics","planner_diagnostics.json")):
                    (folder/name).write_text(json.dumps(harness.documents[role]))
                (folder/"ego_trace.csv").write_text("elapsed_s,x,y,speed\n"+"".join(f"{x},{x},0.,1.\n" for x in range(22)))
                (self.output/"episode.json").write_text(json.dumps(harness.documents["episode"]))
                (self.output/"raw").mkdir()
                (self.output/"raw/RAW_COMPLETE.json").write_text(json.dumps(harness.documents["raw_seal"]))
                events.append("raw_sealed")
                return copy.deepcopy(harness.documents["outcome"])
        class FixtureClient(SimpleNamespace):
            def start_world(self,_transform):
                pass
        grpc=FixtureClient(client_key="fixture_owner",world=object(),
            resume_world=Mock(),reset_actors=Mock(return_value=True),
            get_ego=lambda:SimpleNamespace(),make_transform=lambda *args:args)
        def preview_drive(_grpc,_loader,_candidate,**kwargs):
            kwargs["drive_fn"](preview,[{"raw_id":"7"}])
            harness.assertEqual(events,["raw_sealed"])
            events.append("fixture_cleaned")
            return preview
        def restore(_grpc,_ego,_output,_ids,action):
            if action=="restore":
                events.append("scene_restored")
                return {"status":"nothing_to_restore","parking":{"parked":True,"external_input_disabled":False,
                    "brake_hold_verified":True,"control_publishers_absent":True,"gear_mode":4,"control_mode":3,
                    "brake":1.,"throttle":0.,"speed_norm_raw":0.}}
            return {"status":"nothing_to_move"}
        def prepare_common(_grpc,evidence,**kwargs):
            setup_calls.append("common_ready")
            _grpc._tunnel_static_common_start=True
            observed={"position_xyz":[0.,0.,28.],"speed_norm_raw":0.,"gear_mode":4,
                      "control_mode":3,"brake":1.,"throttle":0.,"brake_hold_verified":True}
            evidence.update(status="ready",samples=7,stable_samples=7,stable_duration_sec=.6,
                            last_readback=observed,actual_start_pose=observed,
                            stop_method="common_auto_d_brake")
            if fail_stage in ("settling","settling_cleanup"):
                evidence.update(status="failed",last_failure="isolated asynchronous suspension readback")
                raise RuntimeError("isolated asynchronous settling failure")
            if fail_stage=="start_xy":
                evidence.update(status="failed",last_failure="isolated measured START XY mismatch")
                raise RuntimeError("isolated measured START XY mismatch")
            return observed
        shutil.rmtree(self.episode)
        def final_park(*args,**kwargs):
            setup_calls.append("strict_park")
            if fail_stage=="settling_cleanup":
                raise RuntimeError("isolated cleanup brake failure")
        with ExitStack() as patches:
            # The common helper still requires actual evidence. Do not permit
            # a missing argument or the old forced-P/alignment path to hide.
            patches.enter_context(patch.object(owner,"prepare_common_start",autospec=True,side_effect=prepare_common))
            patches.enter_context(patch.object(owner,"block_and_park",autospec=True,
                                              side_effect=AssertionError("Collection must not force P")))
            patches.enter_context(patch.object(owner,"align_parked_ego",autospec=True,
                                              side_effect=AssertionError("Collection must not force pose alignment")))
            patches.enter_context(patch.object(live,"_park",autospec=True,side_effect=final_park))
            replacements=[(facade,"DEFAULT_LIBRARY",library),(facade,"_curation_pins",lambda _:{}),
                (replay,"verify_library",lambda *args:verified),(replay,"balanced_choice",lambda *args:(selected,{})),
                (replay,"current_runtime_sources",lambda:self.sources),(runtime,"source_snapshot",lambda *args:{"production":{"runtime_files":self.sources}}),
                (runtime,"require_sources",Mock()),(runtime,"ExecutionLock",nullcontext),
                (owner,"_controller_publishers",lambda:[]),
                (owner,"preview_candidate",preview_drive),(live,"_inventory",lambda *args:{}),
                (safety,"relocate",restore),(drive,"TrialContext",Context),(drive,"TrialRosEvidence",lambda:SimpleNamespace(start=Mock(),close=Mock())),
                (drive.screening,"_expert_search",lambda:(None,SimpleNamespace(from_map_dir=boundary_factory),None,None,None)),
                (drive,"park",Mock())]
            for module,key,value in replacements:patches.enter_context(patch.object(module,key,value))
            result=owner.run_collection_episode(grpc,self.loader,self.documents["episode"]["global_config"],
                {"profile_state_path":str(profile)},scenario_factory=Scenario,scenario_kwargs={"output":self.episode},
                output_dir=self.root/"preparation",seed=42,history_path=self.root/"_campaign/tunnel_static_selection_history.json")
        if fail_stage is None:
            boundary_factory.assert_called_once_with("isolated-map",approach_m=120.,departure_m=70.,
                planning_margin_m=.35,runtime_margin_m=.15,body_policy=measured_body)
            self.assertEqual(measured_body,{"rear_m":.85,"front_m":3.85,"width_m":2.17})
        return result,events,setup_calls

    def test_collection_owner_and_real_callback_produce_gate_compatible_final_proof(self):
        result,events,setup_calls=self.produce_collection_proof()
        self.assertEqual(result["status"],"mission_completed",result)
        self.assertEqual(events,["raw_sealed","fixture_cleaned","scene_restored"])
        self.assertEqual(setup_calls,["common_ready","strict_park"])
        startup=json.loads((self.root/"preparation/startup.json").read_text())
        self.assertEqual(startup["status"],"ready")
        self.assertEqual(startup["startup_policy"],"common_auto_d_brake")
        self.assertEqual(startup["settling"]["samples"],7)
        self.assertEqual(startup["settling"]["actual_start_pose"]["gear_mode"],4)
        self.assertNotIn("alignment",startup)
        # The production owner wrote and hashed this proof. The gate is neither
        # mocked nor given any patched proof, SHA, completion or policy decision.
        accepted=self.validate()
        self.assertTrue(accepted["valid"],accepted)
        self.assertEqual(len(json.loads(self.proof_path.read_text())["files"]),9)

    def assert_early_setup_failure_is_preserved(self,stage):
        result,events,setup_calls=self.produce_collection_proof(fail_stage=stage)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(events,[])
        self.assertFalse(self.episode.exists())
        self.assertFalse(self.proof_path.exists())
        selection=json.loads((self.root/"preparation/selection.json").read_text())
        startup=json.loads((self.root/"preparation/startup.json").read_text())
        self.assertEqual(selection["status"],"failed")
        self.assertFalse(selection["replay_actual_pass"])
        self.assertEqual(selection["failure_reason"],result["reason"])
        self.assertEqual(startup["status"],"failed")
        self.assertEqual(startup["failure_reason"],result["reason"])
        self.assertEqual(startup["settling"]["samples"],7)
        self.assertEqual(startup["settling"]["last_readback"]["position_xyz"],[0.,0.,28.])
        self.assertFalse(self.validate()["valid"])
        return startup,setup_calls

    def test_failed_asynchronous_settling_keeps_evidence_without_starting_capture(self):
        startup,calls=self.assert_early_setup_failure_is_preserved("settling")
        self.assertEqual(startup["settling"]["status"],"failed")
        self.assertEqual(startup["settling"]["last_failure"],"isolated asynchronous suspension readback")
        self.assertEqual(calls,["common_ready","strict_park"])
        self.assertNotIn("alignment",startup)

    def test_failed_measured_start_keeps_readbacks_without_forcing_pose_or_creating_approval(self):
        startup,calls=self.assert_early_setup_failure_is_preserved("start_xy")
        self.assertEqual(startup["settling"]["status"],"failed")
        self.assertEqual(startup["settling"]["last_failure"],"isolated measured START XY mismatch")
        self.assertEqual(calls,["common_ready","strict_park"])
        self.assertNotIn("alignment",startup)

    def test_cleanup_failure_does_not_hide_the_original_pre_capture_failure(self):
        result,events,calls=self.produce_collection_proof(fail_stage="settling_cleanup")
        self.assertEqual(result["status"],"failed")
        self.assertIn("isolated asynchronous settling failure",result["reason"])
        self.assertNotIn("isolated cleanup brake failure",result["reason"])
        self.assertIn("isolated cleanup brake failure",result["cleanup_failure_reason"])
        self.assertEqual(events,[])
        self.assertFalse(self.episode.exists())
        startup=json.loads((self.root/"preparation/startup.json").read_text())
        self.assertIn("isolated asynchronous settling failure",startup["failure_reason"])
        self.assertEqual(calls,["common_ready","strict_park"])

    def test_parking_reassertion_remains_opt_in_for_other_trial_callers(self):
        from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import trial
        parameter=inspect.signature(trial.settle_initial_ego).parameters["reassert_park"]
        self.assertIs(parameter.default,False)
        self.assertEqual(parameter.kind,inspect.Parameter.KEYWORD_ONLY)

    def test_normal_static_needs_no_tunnel_manifest_and_retains_legacy_contract(self):
        self.assertTrue(validate_tunnel_static_collection(self.root/"absent","full_loop_static")["valid"])
        self.assertNotIn("tunnel_static",scenario_zone_contract("full_loop_static"))
        self.assertEqual(required_training_coverage("full_loop_static"),("waypoint_frames","object_positive_frames"))


if __name__=="__main__":unittest.main()
