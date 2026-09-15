import json
from pathlib import Path
import sys
import tempfile
import unittest

INTERNAL=Path(__file__).resolve().parents[1]/'internal_sensor_record'
sys.path.insert(0,str(INTERNAL))
from async_capture.native_session import recorded_native_root,same_recorded_native_source
from async_capture.sensor_record import ROLES


class TransferredNativeSessionTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
