"""Tunnel capture keeps the shared recorder and waits for verified Record ON."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'tools'),str(ROOT.parents[1])]
from async_capture.scenario_simulation import wait_for_native_capture_armed
from async_capture.scenario_catalog import COLLECTION_SCENARIOS, TUNNEL_STATIC_SCENARIO
from scenario_collection import tunnel_collection_evidence


class NativeArmedBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.recorder = SimpleNamespace(poll=lambda:None)
        self.clock = 0.

    def sleep(self,seconds):
        self.clock += seconds

    def wait(self,timeout=.1,sleep=None):
        return wait_for_native_capture_armed(self.path,self.recorder,timeout=timeout,
                                             monotonic=lambda:self.clock,sleep=sleep or self.sleep)

    def arm(self,**changes):
        marker=dict(status='armed',record_on=True,stopwatch_ticks=100)
        marker.update(changes)
        (self.path/'recording_armed.json').write_text(json.dumps(marker),encoding='utf-8-sig')

    def test_waits_for_actual_owner_acknowledgement_while_raw_is_alive(self):
        def delayed(seconds):
            self.sleep(seconds)
            self.arm()
        result=self.wait(sleep=delayed)
        self.assertGreater(result['elapsed_wall_seconds'],0.)
        self.assertTrue(result['owner_evidence']['record_on'])

    def test_native_stop_blocks_drive_even_if_an_armed_marker_remains(self):
        self.arm()
        (self.path/'recording_stopped.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'stopped before Ego'):
            self.wait()

    def test_raw_exit_blocks_drive_even_with_a_valid_native_marker(self):
        self.arm()
        self.recorder=SimpleNamespace(poll=lambda:0)
        with self.assertRaisesRegex(RuntimeError,'Raw recorder exited'):
            self.wait()

    def test_unverified_or_malformed_native_marker_is_rejected(self):
        for changes in ({'status':'pending'},{'record_on':False},{'error':'owner failed'}):
            with self.subTest(changes=changes):
                self.arm(**changes)
                with self.assertRaisesRegex(RuntimeError,'ON was not verified'):
                    self.wait()
        (self.path/'recording_armed.json').write_text('invalid json')
        with self.assertRaisesRegex(RuntimeError,'marker is invalid'):
            self.wait()

    def test_missing_acknowledgement_has_a_bounded_deadline(self):
        with self.assertRaises(TimeoutError):
            self.wait()
        self.assertLessEqual(self.clock,.1)


class SharedRecorderHookTests(unittest.TestCase):
    def exercise_setup(self,tunnel):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder=Path(temporary.name)
        output=folder/'episode'
        output.mkdir()
        config_path=output/'runtime.json'
        config_path.write_text(json.dumps({'planning':{'velocity_profile':{
            'max_velocity':73. if tunnel else 20.,'speed_limit_zones':[{'max_velocity':100.}]}}}))
        route=output/'route.csv';route.write_text('x,y\n0,0\n')
        global_path=folder/'global_path.txt';global_path.write_text('0 0\n')
        contract=folder/'contract.json';contract.write_text('{}')
        events=[]
        class Base:
            def setup_expert_controller(self):
                events.append('runtime_exported_before_raw')
                self.expert_controller=SimpleNamespace(runtime_config_path=str(config_path),
                                                       runtime_path_csv=str(route),expert_root=str(folder/'expert'))
        source=ROOT/'tools/run_proven_three_lap_scenario.py'
        tree=ast.parse(source.read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RawScenario')
        cls.body=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='setup_expert_controller']
        commands=[]
        def start(command,**options):
            events.append('shared_raw_recorder_started')
            commands.append(command)
            (output/'ready.json').write_text('{}')
            return SimpleNamespace(pid=42,poll=lambda:None)
        barrier=Mock(side_effect=lambda *a,**k:{'owner_evidence':{'record_on':True}})
        write=lambda path,value:Path(path).write_text(json.dumps(value))
        namespace=dict(FullLoopScenario=Base,Path=Path,json=json,ROOT=ROOT,os=__import__('os'),
                       time=__import__('time'),subprocess=SimpleNamespace(Popen=start,STDOUT=-2),
                       digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                       write_json=write,wait_for_native_capture_armed=barrier,
                       tunnel_collection_evidence=tunnel_collection_evidence)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
        scenario=namespace['RawScenario']()
        scenario.collector=None
        scenario.output=output
        scenario.cfg={'global_path':str(global_path),'expert_max_velocity_kmh':None if tunnel else 20.,
                      '_tunnel_static_collection':tunnel,'_tunnel_static_candidate_sha256':'a'*64}
        scenario.global_cfg={}
        scenario.scenario_name=TUNNEL_STATIC_SCENARIO if tunnel else 'full_loop_static'
        scenario.run_seed=123;scenario.profile='SUNNY';scenario.scenario_hour=11
        scenario.route_length_m=180.;scenario.clock=SimpleNamespace(domain='grpc_simulation_timestamp')
        scenario.collision_policy='abort';scenario.time_mode='simulation_free_run';scenario.expert_clock='ros_clock'
        scenario.contract_path=contract;scenario.seconds=100.;scenario.record_rosbag=False
        scenario.sensor_record_coordination=folder/'coordination'
        scenario.event=lambda event,**fields:events.append(event)
        scenario.tunnel_static_trial_context=SimpleNamespace(
            output_dir=folder/'guard',runtime_sources={'unchanged.py':'b'*64},
            boundary=SimpleNamespace(policy_sha256='c'*64))
        scenario.setup_expert_controller()
        self.addCleanup(scenario.capture_log.close)
        return scenario,commands,events,barrier,json.loads((output/'episode.json').read_text()),json.loads(config_path.read_text())

    def test_tunnel_uses_shared_recorder_after_measured_inputs_and_before_native_ack(self):
        scenario,commands,events,barrier,episode,runtime=self.exercise_setup(True)
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0][2],str(ROOT/'tools/collect_async_raw_proven.py'))
        self.assertEqual(commands[0][commands[0].index('--seconds')+1],'220.0')
        self.assertLess(events.index('runtime_exported_before_raw'),events.index('shared_raw_recorder_started'))
        self.assertLess(events.index('raw_ready_before_expert'),events.index('native_recording_ready_before_expert'))
        barrier.assert_called_once_with(scenario.sensor_record_coordination,scenario.collector,timeout=65.)
        self.assertEqual(runtime['planning']['velocity_profile']['max_velocity'],73.)
        self.assertEqual(episode['tunnel_static_trial']['selected_candidate_sha256'],'a'*64)
        self.assertFalse(episode['tunnel_static_trial']['finalized'])

    def test_regular_static_capture_keeps_its_original_ready_and_time_budget(self):
        scenario,commands,events,barrier,episode,runtime=self.exercise_setup(False)
        # Any coordinated Sensor Record capture waits for the UI owner,
        # including the regular static mode exercised by this fixture.
        barrier.assert_called_once_with(scenario.sensor_record_coordination,scenario.collector,timeout=65.)
        self.assertEqual(commands[0][commands[0].index('--seconds')+1],'160.0')
        self.assertNotIn('tunnel_static_trial',episode)
        self.assertEqual(runtime['planning']['velocity_profile']['max_velocity'],20.)

    def test_tunnel_is_an_explicit_addition_to_shared_collection_choices(self):
        self.assertIn(TUNNEL_STATIC_SCENARIO,COLLECTION_SCENARIOS)
        self.assertIn('full_loop_static',COLLECTION_SCENARIOS)
        self.assertIn('full_loop_dynamic',COLLECTION_SCENARIOS)


if __name__=='__main__':
    unittest.main()
