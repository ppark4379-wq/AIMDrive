"""Tools enforce recording chronology and prevent evaluation data entering replay."""
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import replay_localization as replay
import localization_metrics as metrics
import render_localization_report as report
import run_localization as observer


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        (self.root/'plan.json').write_text(json.dumps(dict(map_basis={},simulate_zone=None)))

    def write_rows(self,name,values):
        (self.root/name).write_text(''.join(json.dumps(value)+'\n' for value in values))

    def test_only_allowed_values_reach_estimator(self):
        row=dict(sensor='status',value=dict(signed_speed=36.,sim_time=1.,yaw=999,x=900,
                 velocity_xy=[100,100]),evaluation_only=dict(packet_position_xyz=[1000,2000,3000]))
        self.assertEqual(replay.allowed_value(row),dict(signed_speed=36.,sim_time=1.))

    def test_replay_receipt_chronology_stream_ids_duplicates_and_no_future_gps(self):
        records=[dict(sensor='gps',received_ns=30,sample_id=1,value=dict(latitude=37,longitude=126)),
                 dict(sensor='status',received_ns=10,sample_id=7,value=dict(signed_speed=36.,sim_time=1.,x=9999)),
                 dict(sensor='imu',received_ns=20,sample_id=3,value=dict(yaw=0.,sim_time=1.)),
                 dict(sensor='gps',received_ns=40,sample_id=1,value=dict(latitude=0,longitude=0))]
        self.write_rows('sensor_audit.jsonl',records)
        self.write_rows('poses.jsonl',[dict(monotonic_ns=tick) for tick in (15,25,35,45)])
        calls=[]
        class Estimator:
            def __init__(self,gps,status,*args,**kwargs):self.gps,self.status=gps,status
            def latest_snapshot(self):
                fix,age,identity=self.gps.latest_snapshot()
                calls.append((fix,identity,[dict(i) for _,i in self.status.speed.history],
                              [dict(v) for v,_ in self.status.speed.history]))
                return None,float('inf'),dict(position_mode='unavailable')
        with patch.object(replay,'AlignedGpsBlackoutHold',Estimator):
            result,_=replay.replay(self.root,self.root/'replay')
        self.assertIsNone(calls[0][0]);self.assertIsNone(calls[1][0])
        self.assertEqual(calls[2][0],dict(latitude=37,longitude=126))
        self.assertEqual(calls[3][0],calls[2][0])
        self.assertEqual(calls[2][1]['sample_id'],1)
        self.assertEqual(calls[0][2][0]['sample_id'],7)
        self.assertNotIn('x',calls[0][3][0])
        self.assertEqual(result['assessment'],'not_assessed_no_application_tolerance')
        with self.assertRaises(FileExistsError):
            replay.replay(self.root,self.root/'replay')

    def test_batch_replay_cannot_leak_late_observed_gps_into_earlier_poll(self):
        records=[dict(sensor='gps',received_ns=12,sample_id=1,observed_poll_id=2,
                      value=dict(latitude=37,longitude=126)),
                 dict(sensor='status',received_ns=10,sample_id=1,observed_poll_id=1,
                      value=dict(signed_speed=36.,sim_time=1.)),
                 dict(sensor='imu',received_ns=11,sample_id=1,observed_poll_id=1,
                      value=dict(yaw=0.,sim_time=1.))]
        self.write_rows('sensor_audit.jsonl',records)
        self.write_rows('poses.jsonl',[dict(monotonic_ns=20,poll_id=1),dict(monotonic_ns=30,poll_id=2)])
        observations=[]
        class Estimator:
            def __init__(self,gps,*args,**kwargs):self.gps=gps
            def latest_snapshot(self):
                observations.append(self.gps.latest_snapshot()[0])
                return None,float('inf'),dict(position_mode='unavailable')
        with patch.object(replay,'AlignedGpsBlackoutHold',Estimator):
            replay.replay(self.root,self.root/'replay')
        self.assertIsNone(observations[0])
        self.assertEqual(observations[1],dict(latitude=37,longitude=126))
        plan=json.loads((self.root/'replay/plan.json').read_text())
        self.assertEqual(plan['replay_visibility'],'observed_poll_and_receipt')
        self.assertEqual([r['poll_id'] for r in metrics.rows(self.root/'replay/poses.jsonl')],[1,2])

    def test_batch_replay_requires_both_receipt_and_observation_visibility(self):
        self.write_rows('sensor_audit.jsonl',[dict(sensor='gps',received_ns=25,sample_id=1,observed_poll_id=1,
                        value=dict(latitude=37,longitude=126))])
        self.write_rows('poses.jsonl',[dict(monotonic_ns=20,poll_id=1),dict(monotonic_ns=30,poll_id=2)])
        observations=[]
        class Estimator:
            def __init__(self,gps,*args,**kwargs):self.gps=gps
            def latest_snapshot(self):
                observations.append(self.gps.latest_snapshot()[0])
                return None,float('inf'),dict(position_mode='unavailable')
        with patch.object(replay,'AlignedGpsBlackoutHold',Estimator):
            replay.replay(self.root,self.root/'replay')
        self.assertIsNone(observations[0])
        self.assertIsNotNone(observations[1])

    def test_batch_recording_requires_complete_positive_monotonic_poll_ids(self):
        sensor=dict(sensor='gps',received_ns=10,sample_id=1,observed_poll_id=1,value=dict(latitude=37,longitude=126))
        for poses in ([dict(monotonic_ns=20)],
                      [dict(monotonic_ns=20,poll_id=0)],
                      [dict(monotonic_ns=20,poll_id=2),dict(monotonic_ns=30,poll_id=1)],
                      [dict(monotonic_ns=20,poll_id=1),dict(monotonic_ns=30,poll_id=1)]):
            with self.subTest(poses=poses):
                self.write_rows('sensor_audit.jsonl',[sensor])
                self.write_rows('poses.jsonl',poses)
                with self.assertRaisesRegex(ValueError,'poll'):
                    replay.replay(self.root,self.root/'bad')

    def test_reversed_output_ticks_are_rejected(self):
        self.write_rows('sensor_audit.jsonl',[])
        self.write_rows('poses.jsonl',[dict(monotonic_ns=20),dict(monotonic_ns=10)])
        with self.assertRaisesRegex(ValueError,'monotonic'):
            replay.replay(self.root,self.root/'out')

    def test_negative_receipt_timestamps_are_rejected(self):
        self.write_rows('sensor_audit.jsonl',[dict(sensor='gps',received_ns=-1,value={})])
        with self.assertRaisesRegex(ValueError,'receipt'):
            replay.replay(self.root,self.root/'out')

    def test_metrics_deduplicate_gps_and_keep_reference_separate(self):
        comparison=dict(gps_received_ns=10,dr_received_ns=10,real_gps_xy=[0.,0.],estimated_xy=[3.,4.],
                        yaw_grid_rad=0.,time_basis='receipt_monotonic',alignment_method='interpolated_dr_history')
        row=dict(monotonic_ns=20,position=dict(x=3.,y=4.,yaw_grid_rad=0.),selected_samples=dict(gps=dict(
            position_mode='dead_reckoning',source_stamp_ns=1_000_000_000,simulated_comparisons=[comparison])))
        self.write_rows('poses.jsonl',[row,row])
        self.write_rows('sensor_audit.jsonl',[dict(sensor='status',received_ns=1,value=dict(sim_time=1.),
                        evaluation_only=dict(position_provided=True,packet_position_xyz=[3.,4.,0.]))])
        result=metrics.analyze(self.root)
        self.assertEqual(result['unique_gps_comparisons'],1)
        self.assertEqual(result['metrics']['error_2d_m']['max'],5.)
        self.assertEqual(result['evaluation_reference']['unique_samples'],1)
        self.assertEqual(result['evaluation_reference']['error_2d_m']['max'],0.)
        self.assertFalse(result['evaluation_reference']['used_by_estimator'])

    def test_mismatched_times_and_invalid_values_are_not_counted(self):
        comparison=dict(gps_received_ns=10,dr_received_ns=11,real_gps_xy=[0.,0.],estimated_xy=[3.,4.],
                        yaw_grid_rad=0.,time_basis='receipt_monotonic',alignment_method='interpolated_dr_history')
        self.write_rows('poses.jsonl',[dict(selected_samples=dict(gps=dict(simulated_comparisons=[comparison])))])
        self.assertEqual(metrics.analyze(self.root)['unique_gps_comparisons'],0)
        comparison['dr_received_ns']=10;comparison['yaw_grid_rad']=None
        self.write_rows('poses.jsonl',[dict(selected_samples=dict(gps=dict(simulated_comparisons=[comparison])))])
        self.assertEqual(metrics.analyze(self.root)['unique_gps_comparisons'],0)

    def test_empty_report_remains_undecided_and_embedded_json_is_safe(self):
        self.write_rows('poses.jsonl',[])
        result=metrics.analyze(self.root)
        attack='</script><script>alert(1)</script>'
        result['source']=attack
        source=self.root/'analysis.json';source.write_text(json.dumps(result))
        output=report.render(source)
        self.assertNotIn(attack,output.read_text())
        self.assertIn('판정 미정',output.read_text())
        self.assertEqual(json.loads(report.encoded(dict(source=attack)))['source'],attack)

    def test_audit_reports_gaps_duplicates_and_separates_source_clock_epochs(self):
        values=[dict(sensor='status',received_ns=10,sample_id=5,clock_epoch=0),
                dict(sensor='status',received_ns=20,sample_id=7,clock_epoch=0),
                dict(sensor='status',received_ns=30,sample_id=7,clock_epoch=0),
                dict(sensor='status',received_ns=40,sample_id=1,clock_epoch=1)]
        result=metrics.audit_integrity(values)
        self.assertEqual(result['missing_sample_ids'],1)
        self.assertEqual(result['duplicate_sample_ids'],1)
        self.assertEqual(result['stream_epochs']['status'],[0,1])

    def test_observer_check_subprocess_opens_no_sockets_loads_no_model_or_record_dir(self):
        target=self.root/'must_not_be_created'
        cli=ROOT/'tools/run_localization.py'
        program="""import runpy,sys

def audit(event,args):
    if event=='socket.__new__':raise AssertionError('network is forbidden during --check')
sys.addaudithook(audit)
sys.argv=[sys.argv[1],'--check','--config','AIM2026/config.yaml','--record-dir',sys.argv[2]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:
    if exc.code:raise
assert 'torch' not in sys.modules
"""
        result=subprocess.run([sys.executable,'-B','-c',program,str(cli),str(target)],
                              cwd=self.root,text=True,capture_output=True,check=True)
        value=json.loads(result.stdout)
        self.assertEqual(value['status'],'CHECKED')
        self.assertEqual(value['opened_sockets'],0)
        self.assertFalse(value['command_sender_enabled'])
        self.assertEqual(set(value['sensors']),{'gps','imu','vehicle_status'})
        self.assertFalse(target.exists())

    def test_observer_copy_disables_senders_and_preserves_source_config(self):
        import yaml
        config_path=ROOT.parents[0]/'config.yaml'
        before=yaml.safe_load(config_path.read_text())
        copied,_=observer.load_configuration(config_path,self.root/'record')
        self.assertEqual(copied['Detection'],{})
        self.assertIsNone(copied['Ego']['cmd_port'])
        self.assertEqual(copied['Localization']['record_dir'],str((self.root/'record').resolve()))
        self.assertEqual(yaml.safe_load(config_path.read_text()),before)
        manager=observer.location_manager(copied)
        self.assertEqual(set(manager.sensor_config),{'gps','imu','vehicle_status'})
        self.assertIsNone(manager.cmd)
        self.assertIsNone(manager.detection)
        manager.close()

    def test_observer_rejects_disabled_localization(self):
        import yaml
        path=self.root/'disabled.yaml'
        path.write_text(yaml.safe_dump(dict(Localization=dict(enabled=False))))
        with self.assertRaisesRegex(ValueError,'enabled'):
            observer.load_configuration(path)

    @unittest.skipUnless(os.environ.get('GPS_JAMMING_REGRESSION_DRIVER'),'external recorded sensors not bundled')
    def test_real_recorded_input_matches_original_gps_metrics(self):
        source=Path(os.environ['GPS_JAMMING_REGRESSION_DRIVER'])
        result,_=replay.replay(source,self.root/'regression','aligned')
        self.assertEqual(result['unique_gps_comparisons'],244)
        for key,expected in dict(max=.19624427141028514,median=.07984366675438319,p95=.1575456275746673).items():
            self.assertAlmostEqual(result['metrics']['error_2d_m'][key],expected,places=12)


if __name__=='__main__':unittest.main()
