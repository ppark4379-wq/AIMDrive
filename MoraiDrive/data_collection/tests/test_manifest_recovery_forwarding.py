"""Manifest building verifies the exact source recovery before selecting output."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/"data_collection/internal_sensor_record"),str(PROJECT)]
from async_capture.journal import digest
from morai_pretraining import sensor_record_manifest as manifests


class ManifestRecoveryForwardingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)/"run"
        self.episode=self.run/"episode_000"
        self.native=self.run/"native";self.native.mkdir(parents=True)
        self.named="ADMISSION_RECOVERY_TUNNEL_SCALE_reviewed_resolver_registry.json"
        self.assets={}
        for role in manifests.REQUIRED_ROLES:
            path=self.native/(role+".bin");path.write_bytes(b"native fixture bytes")
            self.assets[role]={"relative_path":path.name,"sha256":digest(path),"size_bytes":path.stat().st_size}
        frame={"index":0,"dataset_build_candidate":True,"lidar_source_time_ns":500_000_000,
               "native_filename_time_ns":500_000_000,"assets":self.assets}
        self.frames=[frame]
        self.raw_hash="sealed_same_raw_manifest"
        active=[{"role":"ego","source_timestamp_ns":i*20_000_000,
                 "signal":{"position":[i*.1,0.,0.],"yaw":0.,"speed":5.}} for i in range(176)]
        self.seals={}
        stages={
            "quality":("quality",{"active_signals.json":active},"QUALITY_COMPLETE.json"),
            "native_audit":("native_audit",{"report.json":{"source_root":str(self.native)}},"COMPLETE.json"),
            "gt_timeline":("gt_timeline_v1",{"frames.json":[{"index":0,"lidar_source_ns":500_000_000,"object_gt":{"objects":[]}}]},"COMPLETE.json"),
            "object_geometry":("object_geometry_audit_v1",{"report.json":{"raw_manifest_sha256":self.raw_hash},"frames.json":[{"index":0,"objects":[]}]},"COMPLETE.json"),
            "route_signal":("route_signal_audit_v1",{"report.json":{},"frames.json":[{"index":0,"valid":False}]},"COMPLETE.json"),
            "motion_time":("motion_time_v1",{"report.json":{"status":"passed"}},"COMPLETE.json"),
            "admission":("admission_v1",{"report.json":{"raw_manifest_sha256":self.raw_hash},"frames.json":self.frames},"COMPLETE.json")}
        for key,(folder,files,seal_name) in stages.items():
            path=self.seal(self.episode/folder,files,seal_name)
            self.seals[key]=digest(path/seal_name)
        (self.run/"sensor_contract.json").write_text('{}')
        self.old=self.recovery("ADMISSION_RECOVERY.json","legacy")
        self.new=self.recovery(self.named,"current_tunnel")
        progress=NS(route_index=0,target_x=25.,target_y=0.,target_route_index=1,distance_to_route_m=0.)
        self.route=patch.object(manifests,"official_route_tracker",return_value=(NS(update=lambda *args:progress),{"source":"isolated official-route fixture"}))
        self.route.start();self.addCleanup(self.route.stop)

    def seal(self,folder,files,seal_name="COMPLETE.json"):
        folder.mkdir(parents=True)
        for name,value in files.items():
            (folder/name).write_text(json.dumps(value))
        (folder/seal_name).write_text(json.dumps({"status":"completed","files":{name:digest(folder/name) for name in files}}))
        return folder

    def recovery(self,name,tag):
        geometry=self.seal(self.episode/("geometry_"+tag),{
            "report.json":{"schema":"morai_sensor_record_object_geometry_audit_v5","raw_manifest_sha256":self.raw_hash,"structural_valid":True,"revision":tag},
            "frames.json":[{"index":0,"objects":[],"revision":tag}]})
        admission=self.seal(self.episode/("admission_"+tag),{
            "report.json":{"raw_manifest_sha256":self.raw_hash,"capture_contract_valid":True,
                "runtime_streams_valid":True,"sensor_capture_status":"admitted","teacher_transport_status":"admitted",
                "source_timestamps_modified":False,"training_visible":False,
                "input_seals":{"object_geometry_audit":digest(geometry/"COMPLETE.json")},"revision":tag},
            "frames.json":self.frames})
        payload={"schema":"morai_sensor_record_admission_recovery_v1","status":"completed",
                 "recovery_is_additive":True,"source_timestamps_modified":False,"training_visible":False,
                 "original_admission":{"sha256":digest(self.episode/"admission_v1/report.json")},
                 "corrected_admission":{"path":str((admission/"report.json").relative_to(self.run)),
                     "sha256":digest(admission/"report.json"),"seal_sha256":digest(admission/"COMPLETE.json")},
                 "corrected_object_geometry":{"path":str((geometry/"report.json").relative_to(self.run)),
                     "sha256":digest(geometry/"report.json"),"seal_sha256":digest(geometry/"COMPLETE.json")}}
        path=self.run/name;path.write_text(json.dumps(payload))
        expected=dict(self.seals,admission_recovery=digest(path),effective_admission=digest(admission/"COMPLETE.json"),effective_object_geometry=digest(geometry/"COMPLETE.json"))
        return {"pointer":path,"admission":admission,"geometry":geometry,"expected":expected}

    def build(self,expected=None,name=None):
        return manifests.build_episode_manifest(self.episode,campaign_validation_sha256="sealed_validation",
            split="train",expected_evidence=expected or self.new["expected"],recovery_name=name or self.named)

    def test_current_named_source_builds_full_manifest_using_matching_sealed_named_recovery(self):
        result=self.build()
        self.assertEqual(result["admission_recovery_path"],self.named)
        self.assertEqual(result["evidence_seal_sha256"],self.new["expected"])
        self.assertEqual(result["report"]["counts"]["waypoint_frames"],1)
        self.assertEqual(len(result["entries"]),1)

    def test_original_default_source_can_reprocess_to_requested_additive_geometry(self):
        old_bytes=self.old["pointer"].read_bytes()
        result=self.build(expected=self.old["expected"])
        self.assertEqual(result["evidence_seal_sha256"],self.new["expected"])
        self.assertEqual(result["admission_recovery_path"],self.named)
        self.assertEqual(self.old["pointer"].read_bytes(),old_bytes)

    def test_old_default_source_mismatch_is_not_bypassed_by_valid_requested_recovery(self):
        expected=copy.deepcopy(self.old["expected"])
        expected["effective_admission"]="0"*64
        with self.assertRaisesRegex(ValueError,"recovery seal changed"):
            self.build(expected=expected)

    def test_wrong_requested_name_cannot_substitute_default_evidence_for_named_source(self):
        with self.assertRaisesRegex(ValueError,"recovery seal changed"):
            self.build(name="ADMISSION_RECOVERY_WRONG.json")

    def test_tampered_named_pointer_cannot_fall_back_to_old_default(self):
        self.new["pointer"].write_text(self.new["pointer"].read_text()+" ")
        with self.assertRaisesRegex(ValueError,"recovery seal changed"):
            self.build()

    def test_named_corrected_payload_tamper_is_rejected_even_with_original_pointer_hash(self):
        path=self.new["admission"]/"frames.json"
        path.write_text(path.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"Changed or unsafe sealed"):
            self.build()

    def test_changed_original_stage_seal_is_rejected_before_additive_selection(self):
        path=self.episode/"motion_time_v1/report.json"
        path.write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError,"Changed or unsafe sealed"):
            self.build()


if __name__=="__main__":unittest.main()
