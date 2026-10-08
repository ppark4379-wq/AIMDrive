"""Tunnel scale recovery revisions are additive and shared by admission/build."""
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

PROJECT=Path(__file__).resolve().parents[2]
INTERNAL=PROJECT/"data_collection/internal_sensor_record"
sys.path[:0]=[str(INTERNAL/"tools"),str(INTERNAL),str(PROJECT)]
import recover_sensor_record_bev_geometry as recovery
import validate_sensor_record_campaign as validator


class TunnelScaleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/"run_lap_001"
        self.episode=self.run/"episode_000"
        (self.episode/"raw").mkdir(parents=True)
        self.session={"episode":{"scenario_name":"full_loop_tunnel_static","static_obstacle_preset":None}}
        (self.episode/"raw/session.json").write_text(json.dumps(self.session))
        self.source=self.root/"scale_resolver.py";self.source.write_text("# reviewed scale resolver A\n")
        self.registry=self.root/"registry.json";self.registry.write_text('{"revision":"A"}')
        self.addCleanup(patch.stopall)
        patch.object(recovery,"SCALE_RESOLVER_SOURCE",self.source).start()
        patch.object(recovery,"DEFAULT_GEOMETRY_REGISTRY",self.registry).start()
        self.raw_verifier=patch.object(recovery,"verify_run",return_value=({},self.session)).start()
        patch.object(recovery,"verify_sealed_directory",return_value={"valid":True}).start()
        patch.object(validator,"verify_sealed_directory",return_value={"valid":True}).start()
        patch.object(recovery,"require_ssd").start()
        self.original={"raw_manifest_sha256":"captured_raw","source":"original_unknown_scale"}
        folder=self.episode/"admission_v1";folder.mkdir()
        (folder/"report.json").write_text(json.dumps(self.original))
        (folder/"COMPLETE.json").write_text('{"status":"completed"}')
        self.stage_commands=[]

    def stage(self,command):
        self.stage_commands.append(command)
        output=Path(command[command.index("--output")+1])
        output.mkdir()
        tool=Path(command[1]).name
        if tool.startswith("audit_sensor_record_object_geometry"):
            document={"schema":"morai_sensor_record_object_geometry_audit_v5","structural_valid":True,
                      "raw_manifest_sha256":"captured_raw","training_bev":{},
                      "geometry_registry":{"sha256":recovery.digest(self.registry)}}
        elif tool=="admit_sensor_record_collection.py":
            geometry=self.episode/command[command.index("--object-geometry-dir")+1]
            document={"raw_manifest_sha256":"captured_raw","capture_contract_valid":True,
                      "runtime_streams_valid":True,"source_timestamps_modified":False,"training_visible":False,
                      "input_seals":{"object_geometry_audit":recovery.digest(geometry/"COMPLETE.json")},
                      "source":"current_tunnel_scale"}
        else:document={"status":"completed"}
        (output/"report.json").write_text(json.dumps(document))
        (output/"COMPLETE.json").write_text(json.dumps({"status":"completed","files":{"report.json":recovery.digest(output/"report.json")}}))

    def produce(self):
        with patch.object(sys,"argv",["recover","--run",str(self.run)]), \
                patch.object(recovery.subprocess,"check_call",side_effect=self.stage) as command, \
                redirect_stdout(io.StringIO()):
            recovery.main()
        return command

    def test_current_tunnel_pointer_and_dirs_leave_legacy_unknown_scale_evidence_untouched(self):
        legacy=self.run/"ADMISSION_RECOVERY.json"
        legacy.write_text('{"source":"sealed_old_unknown_scale"}')
        old=legacy.read_bytes()
        self.produce()
        selected=recovery.recovery_selection(self.run)
        self.assertNotEqual(selected["path"],legacy)
        self.assertTrue(selected["path"].is_file())
        self.assertEqual(legacy.read_bytes(),old)
        self.assertEqual(len(self.stage_commands),3)
        for command in self.stage_commands:
            self.assertIn(selected["suffix"],command[command.index("--output")+1])

    def test_resolver_or_registry_revision_selects_an_additive_new_pointer(self):
        first=recovery.recovery_selection(self.run)
        self.source.write_text("# reviewed scale resolver B\n")
        second=recovery.recovery_selection(self.run)
        self.assertNotEqual(first["path"],second["path"])
        self.registry.write_text('{"revision":"B"}')
        third=recovery.recovery_selection(self.run)
        self.assertNotEqual(second["path"],third["path"])
        self.assertNotEqual(second["suffix"],third["suffix"])

    def test_non_tunnel_default_and_explicit_legacy_names_are_preserved(self):
        self.session["episode"]["scenario_name"]="full_loop_static"
        (self.episode/"raw/session.json").write_text(json.dumps(self.session))
        self.raw_verifier.reset_mock()
        self.assertEqual(recovery.recovery_selection(self.run)["path"],self.run/"ADMISSION_RECOVERY.json")
        self.raw_verifier.assert_not_called()
        self.assertEqual(recovery.recovery_selection(self.run,geometry_version=3)["path"],self.run/"ADMISSION_RECOVERY_SCALE_V1.json")
        self.assertEqual(recovery.recovery_selection(self.run,geometry_version=5,geometry_registry=self.registry)["path"],
                         self.run/("ADMISSION_RECOVERY_FUSED_%s.json" % recovery.registry_tag(self.registry)))

    def test_tunnel_type_is_not_authorized_by_unsealed_raw_session(self):
        self.raw_verifier.side_effect=ValueError("Raw metadata hash mismatch")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            recovery.recovery_selection(self.run)

    def test_existing_current_revision_reuses_only_its_verified_pointer(self):
        self.produce()
        before=len(self.stage_commands)
        command=self.produce()
        command.assert_not_called()
        self.assertEqual(len(self.stage_commands),before)

    def test_tampered_revision_cannot_reuse_a_matching_filename(self):
        self.produce()
        selected=recovery.recovery_selection(self.run)
        record=json.loads(selected["path"].read_text())
        record["tunnel_scale_revision"]["scale_resolver_sha256"]="0"*64
        selected["path"].write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"revision differs"):
            recovery.validate_recovery(self.run,selected["path"],selection=selected)

    def test_current_pointer_cannot_redirect_to_old_sealed_output_directories(self):
        self.produce()
        selected=recovery.recovery_selection(self.run)
        payload=json.loads(selected["path"].read_text())
        current=self.run/payload["corrected_review"]["path"]
        old=self.episode/"review_closeout_v5_legacy"
        current.parent.rename(old)
        payload["corrected_review"]["path"]=str((old/"report.json").relative_to(self.run))
        selected["path"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,"current revision"):
            recovery.validate_recovery(self.run,selected["path"],selection=selected)

    def test_effective_admission_and_builder_share_current_revision(self):
        self.run.joinpath("ADMISSION_RECOVERY.json").write_text('{"source":"stale"}')
        self.produce()
        selected=recovery.recovery_selection(self.run)
        errors,seals=[],{}
        admission=validator.effective_admission(self.run,self.episode,errors,seals)
        self.assertFalse(errors)
        self.assertEqual(admission["source"],"current_tunnel_scale")
        args=validator.admission_recovery_build_arguments([self.run])
        self.assertEqual(args,["--admission-recovery-name",selected["path"].name])
        self.assertEqual(seals["admission_recovery"],recovery.digest(selected["path"]))

    def test_builder_rejects_missing_current_pointer_instead_of_stale_legacy(self):
        self.run.joinpath("ADMISSION_RECOVERY.json").write_text('{"source":"stale"}')
        with self.assertRaisesRegex(ValueError,"missing before dataset"):
            validator.admission_recovery_build_arguments([self.run])

    def test_builder_rejects_mixed_resolver_revisions(self):
        first=recovery.recovery_selection(self.run)
        self.source.write_text("# different resolver revision\n")
        second=recovery.recovery_selection(self.run)
        with patch.object(recovery,"recovery_selection",side_effect=[first,second]):
            with self.assertRaisesRegex(ValueError,"one current scale recovery revision"):
                validator.admission_recovery_build_arguments([self.run,self.run])

    def test_changed_corrected_bytes_cannot_be_effective_admission_or_build_input(self):
        self.produce()
        selected=recovery.recovery_selection(self.run)
        pointer=json.loads(selected["path"].read_text())
        corrected=self.run/pointer["corrected_admission"]["path"]
        corrected.write_text(corrected.read_text()+" ")
        errors,seals=[],{}
        self.assertEqual(validator.effective_admission(self.run,self.episode,errors,seals),self.original)
        self.assertTrue(errors)
        with self.assertRaisesRegex(ValueError,"Changed BEV"):
            validator.admission_recovery_build_arguments([self.run])

    def test_non_tunnel_legacy_null_geometry_still_skips_recovery_and_builder_defaults(self):
        self.session["episode"]["scenario_name"]="full_loop_dynamic"
        (self.episode/"raw/session.json").write_text(json.dumps(self.session))
        self.run.joinpath("ADMISSION_RECOVERY.json").write_text('{"corrected_object_geometry":null}')
        with patch.object(validator.subprocess,"Popen") as process,redirect_stdout(io.StringIO()):
            validator.complete_bev_geometry_recovery(self.run,analysis_python="python",env={},log=io.StringIO())
        process.assert_not_called()
        self.assertEqual(validator.admission_recovery_build_arguments([self.run]),[])

    def test_complete_recovery_does_not_skip_tunnel_for_legacy_geometry_pointer(self):
        self.run.joinpath("ADMISSION_RECOVERY.json").write_text('{"corrected_object_geometry":null}')
        selected=recovery.recovery_selection(self.run)
        def launch(*args,**kwargs):
            self.produce()
            return Mock(poll=lambda:0,returncode=0)
        with patch.object(validator.subprocess,"Popen",side_effect=launch) as process:
            validator.complete_bev_geometry_recovery(self.run,analysis_python="python",env={},log=io.StringIO())
        process.assert_called_once()


if __name__=="__main__":unittest.main()
