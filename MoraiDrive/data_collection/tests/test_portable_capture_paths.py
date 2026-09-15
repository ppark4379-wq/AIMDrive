import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

INTERNAL=Path(__file__).resolve().parents[1]/'internal_sensor_record'
sys.path.insert(0,str(INTERNAL))
from async_capture.portable_paths import recorded_episode_directory
from async_capture.journal import digest
from async_capture.sensor_record_admission import (
    _verify_pinned_file,
    _verify_producer_snapshot,
    validate_native_session_snapshot,
)
from morai_pretraining.sensor_record_manifest import (
    effective_admission_directory,
    resolve_audited_map_root,
)


class PortableCapturePathTests(unittest.TestCase):
    def make_episode(self,tmp):
        episode=Path(tmp)/'sensor_record_auto_v1/copied_lap_001/episode_000'
        raw=episode/'raw';raw.mkdir(parents=True)
        manifest=raw/'raw_manifest.json';manifest.write_text('{"run_id":"x"}\n')
        sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        (raw/'RAW_COMPLETE.json').write_text(json.dumps({'status':'completed','manifest_sha256':sha}))
        return episode,raw

    def test_cross_pc_raw_path_resolves_without_rewriting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,raw=self.make_episode(tmp)
            old='/mnt/c/MoraiDriveData/sensor_record_auto_v1/copied_lap_001/episode_000/raw'
            self.assertEqual(recorded_episode_directory(old,episode,'raw',tmp),raw.resolve())

    def test_wrong_episode_or_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,raw=self.make_episode(tmp)
            with self.assertRaisesRegex(ValueError,'identity differs'):
                recorded_episode_directory('/old/other/episode_000/raw',episode,'raw',tmp)
            (raw/'raw_manifest.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'seal does not match'):
                recorded_episode_directory('/old/copied_lap_001/episode_000/raw',episode,'raw',tmp)

    def make_native_provenance(self, tmp):
        run=Path(tmp)/'copied_lap_001';run.mkdir()
        files={
            'session_plan':run/'copied_lap_001_native_session_plan.json',
            'session_apply':run/'copied_lap_001_native_session_apply.json',
            'session_record':run/'SESSION.json',
        }
        for key,path in files.items():path.write_text(key)
        snapshot={
            'prepared_before_capture':True,
            'session_name':run.name,
            'native_root':'/old/native_sessions/'+run.name,
        }
        for key,path in files.items():
            snapshot[key]={
                'path':'/old/'+path.name,
                'sha256':digest(path),
            }
        return run,files,snapshot

    def test_exact_run_local_native_provenance_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,_,snapshot=self.make_native_provenance(tmp)
            result=validate_native_session_snapshot(snapshot,transferred_run=run)
            self.assertTrue(result['valid'])
            self.assertEqual(
                result['resolutions']['session_plan']['resolution'],
                'run_local_exact_sha256_copy')

    def test_changed_or_wrong_run_native_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,files,snapshot=self.make_native_provenance(tmp)
            files['session_apply'].write_text('changed')
            self.assertFalse(validate_native_session_snapshot(
                snapshot,transferred_run=run)['valid'])
            files['session_apply'].write_text('session_apply')
            wrong=run.parent/'wrong';wrong.mkdir()
            self.assertFalse(validate_native_session_snapshot(
                snapshot,transferred_run=wrong)['valid'])

    def test_symlinked_local_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            outside=Path(tmp)/'sensor_contract.json';outside.write_text('contract')
            linked=run/'sensor_contract.json';linked.symlink_to(outside)
            result=_verify_pinned_file(
                {'path':'/old/sensor_contract.json','sha256':digest(outside)},
                local_copy=linked,local_root=run)
            self.assertFalse(result['valid'])

    def test_producer_snapshot_requires_every_exact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'copied_lap_001';root=run/'software_snapshot'
            (root/'async_capture').mkdir(parents=True)
            source=root/'async_capture/source.py';source.write_text('source')
            producer={'async_capture/source.py':digest(source)}
            snapshot={'root':'/old/software_snapshot','files':producer}
            result=_verify_producer_snapshot(snapshot,producer,transferred_run=run)
            self.assertTrue(result['valid'])
            self.assertEqual(result['resolution'],'run_local_exact_sha256_copy')
            source.write_text('changed')
            self.assertFalse(_verify_producer_snapshot(
                snapshot,producer,transferred_run=run)['valid'])

    def test_additive_admission_recovery_selects_exact_sealed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'copied_lap_001';episode=run/'episode_000'
            original=episode/'admission_v1';corrected=episode/'admission_portable_v2'
            for folder,report in (
                (original,{'raw_manifest_sha256':'raw','capture_contract_valid':False}),
                (corrected,{
                    'raw_manifest_sha256':'raw','capture_contract_valid':True,
                    'runtime_streams_valid':True,'sensor_capture_status':'admitted',
                    'teacher_transport_status':'admitted','source_timestamps_modified':False,
                    'training_visible':False,
                }),
            ):
                folder.mkdir(parents=True)
                (folder/'report.json').write_text(json.dumps(report))
                (folder/'frames.json').write_text('[]')
                (folder/'COMPLETE.json').write_text(json.dumps({
                    'files':{
                        'report.json':digest(folder/'report.json'),
                        'frames.json':digest(folder/'frames.json'),
                    }}))
            recovery={
                'schema':'morai_sensor_record_admission_recovery_v1',
                'status':'completed','recovery_is_additive':True,
                'source_timestamps_modified':False,'training_visible':False,
                'original_admission':{'sha256':digest(original/'report.json')},
                'corrected_admission':{
                    'path':'episode_000/admission_portable_v2/report.json',
                    'sha256':digest(corrected/'report.json'),
                    'seal_sha256':digest(corrected/'COMPLETE.json'),
                },
            }
            (run/'ADMISSION_RECOVERY.json').write_text(json.dumps(recovery))
            selected,seals=effective_admission_directory(episode)
            self.assertEqual(selected,corrected.resolve())
            self.assertIn('effective_admission',seals)
            (corrected/'frames.json').write_text('[1]')
            with self.assertRaisesRegex(ValueError,'Changed or unsafe sealed evidence'):
                effective_admission_directory(episode)

    def test_audited_map_accepts_only_recorded_lf_hash_equivalent_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario=Path(tmp)/'scenario_runner';map_root=scenario/'data/map'
            config=scenario/'config';map_root.mkdir(parents=True);config.mkdir()
            content={
                'link_set.json':b'{\n"links": []\n}\n',
                'node_set.json':b'{\n"nodes": []\n}\n',
                'full_loop_links.yaml':b'route:\n  - L1\n',
            }
            for name,value in content.items():
                target=(config/name if name.endswith('.yaml') else map_root/name)
                target.write_bytes(value.replace(b'\n',b'\r\n'))
            global_path=map_root/'2026_molit_comp_global_path.txt'
            global_path.write_bytes(b'0 0\n1 1\n')
            hashes={
                'link_set':hashlib.sha256(content['link_set.json']).hexdigest(),
                'node_set':hashlib.sha256(content['node_set.json']).hexdigest(),
                'route_definition':hashlib.sha256(content['full_loop_links.yaml']).hexdigest(),
            }
            selected,relocated,resolution=resolve_audited_map_root(
                Path('/missing/scenario_runner/data/map'),scenario,
                hashlib.sha256(global_path.read_bytes()).hexdigest(),hashes)
            self.assertEqual(selected,map_root)
            self.assertTrue(relocated)
            self.assertEqual(resolution,'recorded_lf_sha256_equivalent_checkout')
            (map_root/'link_set.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'recorded LF hash'):
                resolve_audited_map_root(
                    Path('/missing/scenario_runner/data/map'),scenario,
                    hashlib.sha256(global_path.read_bytes()).hexdigest(),hashes)


if __name__=='__main__':unittest.main()
