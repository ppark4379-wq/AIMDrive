import json
from pathlib import Path
import sys
import tempfile
import unittest

INTERNAL=Path(__file__).resolve().parents[1]/'internal_sensor_record'
sys.path.insert(0,str(INTERNAL))
from async_capture.native_session import recorded_native_root,same_recorded_native_source
from async_capture.sensor_record import ROLES
from async_capture.journal import digest
from async_capture.sensor_record_admission import validate_native_session_snapshot


class TransferredNativeSessionTests(unittest.TestCase):
    def test_promoted_direct_session_uses_exact_campaign_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / 'dataset/static_obstacle/static_lap10_20260920'
            run = campaign / (campaign.name + '_lap_001')
            run.mkdir(parents=True)
            runtime = campaign / '_campaign/runtime' / run.name
            runtime.mkdir(parents=True)
            snapshot = dict(prepared_before_capture=True, session_name=run.name,
                            native_root='/old/native')
            for key, filename in [('session_plan', 'native_session_plan.json'),
                                  ('session_apply', 'native_session_apply.json'),
                                  ('session_record', 'SESSION.json')]:
                path = (run if key == 'session_record' else runtime) / filename
                path.write_text('{}\n')
                old = Path('/gone/.collecting/static_obstacle') / campaign.name
                old = (old / run.name if key == 'session_record'
                       else old / '_campaign/runtime' / run.name) / filename
                snapshot[key] = dict(path=str(old), sha256=digest(path))
            result = validate_native_session_snapshot(snapshot, transferred_run=run)
            self.assertTrue(result['valid'])
            self.assertEqual(result['resolutions']['session_plan']['resolution'],
                             'campaign_runtime_exact_sha256_copy')
            plan = runtime / 'native_session_plan.json'
            plan.write_text('{"tampered":true}\n')
            self.assertFalse(validate_native_session_snapshot(
                snapshot, transferred_run=run)['valid'])
            plan.write_text('{}\n')
            pinned_path = snapshot['session_plan']['path']
            snapshot['session_plan']['path'] = pinned_path.replace(
                '/runtime/' + run.name + '/', '/runtime/other_lap/')
            self.assertFalse(validate_native_session_snapshot(
                snapshot, transferred_run=run)['valid'])
            snapshot['session_plan']['path'] = pinned_path
            plan.unlink()
            plan.symlink_to(runtime / 'native_session_apply.json')
            self.assertFalse(validate_native_session_snapshot(
                snapshot, transferred_run=run)['valid'])

    def make_run(self,root,passed=True,session_name='copied_lap_001'):
        base=Path(root)/'sensor_record_auto_v1'
        run=base/'copied_lap_001';run.mkdir(parents=True)
        for role in ROLES:(run/role).mkdir()
        (run/'native_setup.json').write_text(json.dumps({
            'native_root':'/mnt/c/MoraiDriveData/sensor_record_auto_v1/native_sessions/copied_lap_001'}))
        (run/'SESSION.json').write_text(json.dumps({'plan':{'session':session_name}}))
        (run/'NATIVE_CAPTURE_RESULT.json').write_text(json.dumps({
            'error':None,'quick_native_capture':{'passed':passed}}))
        return base,run

    def test_completed_cross_pc_run_uses_local_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,run=self.make_run(tmp)
            self.assertEqual(recorded_native_root(run,tmp),run)
            self.assertEqual(recorded_native_root(run/'episode_000',tmp),run)
            original='/mnt/c/MoraiDriveData/sensor_record_auto_v1/native_sessions/copied_lap_001'
            self.assertTrue(same_recorded_native_source(run/'episode_000',run,original,tmp))

    def test_unverified_cross_pc_mirror_is_rejected(self):
        for passed,session in ((False,'copied_lap_001'),(True,'wrong_session')):
            with tempfile.TemporaryDirectory() as tmp:
                _,run=self.make_run(tmp,passed=passed,session_name=session)
                with self.assertRaisesRegex(ValueError,'no verified local mirror'):
                    recorded_native_root(run,tmp)

    def test_canonical_scenario_campaign_run_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=(root/'dataset/dynamic_obstacle/scenario_campaign_20261001_123456/'
                 'scenario_campaign_20261001_123456_lap_001')
            run.mkdir(parents=True)
            for role in ROLES:(run/role).mkdir()
            original=('/mnt/c/MoraiDriveData/sensor_record_auto_v1/native_sessions/'
                      +run.name)
            (run/'native_setup.json').write_text(json.dumps({'native_root':original}))
            (run/'SESSION.json').write_text(json.dumps({'plan':{'session':run.name}}))
            (run/'NATIVE_CAPTURE_RESULT.json').write_text(json.dumps({
                'error':None,'quick_native_capture':{'passed':True}}))
            (run/'.LOCAL_DATASET_COMPLETE.json').write_text(json.dumps({
                'schema':'morai_local_dataset_materialization_v1','kind':'run',
                'run_name':run.name,'hardlinked':True}))
            self.assertEqual(recorded_native_root(run,root),run)
            self.assertEqual(recorded_native_root(run/'episode_000',root),run)
            self.assertTrue(same_recorded_native_source(
                run/'episode_000',run,original,root))


if __name__=='__main__':unittest.main()
