"""Existing runner with VE-only object input; no simulator or model needed."""
import copy
import json
import io
import socket
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scenario_runner.adapters.ve_objects import configure, configure_expert, verify_topology, VeObjectInput
from scenario_runner.main_runner import load_configuration
from scenario_runner.core.full_loop_drive_base import FullLoopDriveBase
from MoraiDrive.expert.tools.ve_udp_session import PerceptionSession


class WiringTests(unittest.TestCase):
    def test_config_changes_only_object_transport_and_checkpoint(self):
        _, _, base = load_configuration()
        base['sync_runtime']['enabled'] = False
        for scenario in ('full_loop_roundabout', 'full_loop_highway'):
            cfg = copy.deepcopy(base)
            configure(cfg, scenario, 'E3')
            self.assertEqual(cfg.pop('ve_objects_checkpoint'), 'E3')
            self.assertEqual(cfg.pop('expert_ros_remaps'), ['/Object_topic:=/ve/Object_topic'])
            self.assertEqual(cfg, {k: v for k, v in base.items() if k != 'expert_ros_remaps'})
        with self.assertRaises(ValueError):
            configure({'sync_runtime': {'enabled': True}}, 'full_loop_highway', 'E3')

    def test_runtime_expert_has_no_gt_fallback_or_geometry_injection(self):
        doc = dict(common=dict(perception_source='ve', sync_grpc_inputs=dict(enabled=True)),
                   planning=dict(max_velocity=50, jamming_scene_geometry=[1],
                                 jamming_trial_contract={}, jamming_trial_diagnostics_path='/old'),
                   map=dict(traffic_light_control=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            path.write_text(json.dumps(doc))
            configure_expert(path)
            result = json.loads(path.read_text())
        self.assertEqual(result['planning'], dict(max_velocity=50))
        self.assertEqual(result['map'], doc['map'])
        self.assertEqual(result['common'], dict(perception_source='ros', sync_grpc_inputs=dict(enabled=False)))

    def test_topic_check_rejects_gt_subscription_and_other_controller(self):
        node = '/morai_standard_42_123'
        state = [[('/ve/Object_topic', ['/adapter']), ('/ctrl_cmd', [node])],
                 [('/ve/Object_topic', [node])], []]
        self.assertTrue(verify_topology(state, '/adapter', 42))
        observing = copy.deepcopy(state)
        observing[1].append(('/Object_topic', ['/ve_gt_observer']))
        self.assertTrue(verify_topology(observing, '/adapter', 42))
        leaked = copy.deepcopy(state)
        leaked[1].append(('/Object_topic', [node]))
        with self.assertRaisesRegex(RuntimeError, 'GT'):
            verify_topology(leaked, '/adapter', 42)
        state[0][1][1].append('/other')
        with self.assertRaisesRegex(RuntimeError, '다른'):
            verify_topology(state, '/adapter', 42)

    def test_no_gt_stream_is_required_in_scenario_mode(self):
        session = object.__new__(PerceptionSession)
        session.observe_gt, session.error = False, None
        session.sender = Mock()
        session.sender.poll.return_value = None
        data = dict(gt_fresh=False, ve_fresh=True, ego_age_s=.1, ego=dict(x=1., y=2.),
                    packet=dict(sensor_received_ns=10_000_000_000, ego_pose=dict(x=1., y=2.)))
        session.snapshot = lambda: data
        self.assertIs(session.require_ready(since=9.), data)
        data['ve_fresh'] = False
        with self.assertRaises(RuntimeError):
            session.require_ready()

    def test_transport_does_not_subscribe_to_gt_in_scenario_mode(self):
        session = object.__new__(PerceptionSession)
        session.observe_gt = False
        session.camera_receiver_mode = 'process'
        session.ros = Mock()
        session.checkpoint = dict(path='/model', sha256='hash')
        session.generation = 'generation'
        with tempfile.TemporaryDirectory() as tmp, \
             patch('MoraiDrive.expert.tools.ve_udp_session.socket.socket'), \
             patch('MoraiDrive.expert.tools.ve_udp_session.build_analysis_environment', return_value={}), \
             patch('MoraiDrive.expert.tools.ve_udp_session.subprocess.Popen'), \
             patch('MoraiDrive.expert.tools.ve_udp_session.threading.Thread'):
            session.output = Path(tmp)
            session.start()
            config = json.loads((session.output / 'publisher_config.json').read_text())
            self.assertEqual(config['runtime']['camera_receiver_mode'], 'process')
            topics = [c.args[0] for c in session.ros.Subscriber.call_args_list]
            self.assertEqual(topics, ['/Ego_topic'])
            session.trace.close()
            session.log.close()
            session.transport_log.close()

    def test_transport_audit_measures_publish_and_records_rejection_without_gt(self):
        from inference.ve_objects_udp import SCHEMA, encode_packet
        import rospy
        session = PerceptionSession(Path('/unused'), 'E3', observe_gt=False)
        session.trace, session.transport_log = io.StringIO(), io.StringIO()
        session.publisher, session.sock = Mock(), Mock()
        session.ros = Mock()
        session.ros.Time.now.return_value = rospy.Time(123, 456)
        session.tracker = Mock()
        session.tracker.update.side_effect = lambda p: p
        now = time.monotonic_ns()
        base = dict(schema=SCHEMA, generation=session.generation,
            checkpoint_sha256=session.checkpoint['sha256'],output_frame='projected_grid',
            scene_assumption='unknown',sent_ns=now,sensor_received_ns=now-100_000,
            boxes=[],ego_pose=dict(x=1.,y=2.,yaw_raw_deg=0.))
        items = iter([encode_packet(dict(base,sequence=3,datagram_sequence=1,ready=True)),
                      encode_packet(dict(base,sequence=8,datagram_sequence=2,ready=False,reason='missing:lidar'))])
        def recv(_):
            try:
                return next(items), ('127.0.0.1',1)
            except StopIteration:
                session.stop.set()
                raise socket.timeout()
        session.sock.recvfrom.side_effect = recv
        session.receive()
        events = [json.loads(v) for v in session.transport_log.getvalue().splitlines()]
        accepted = json.loads(session.trace.getvalue())
        session.publisher.publish.assert_called_once()
        self.assertEqual([e['datagram_sequence'] for e in events],[1,2])
        self.assertEqual(events[1]['reason'],'VE not ready: missing:lidar')
        self.assertEqual(accepted['delivery_audit']['ros_stamp_ns'],123_000_000_456)
        self.assertGreaterEqual(accepted['delivery_audit']['published_ns'], accepted['delivery_audit']['received_ns'])
        self.assertIsNone(session.packet)

    def test_runner_starts_one_expert_after_ve_ready_and_stops_it_first(self):
        runner = object.__new__(FullLoopDriveBase)
        events = []
        runner.cfg = {}
        runner.grpc = Mock()
        runner.grpc.set_ego_control_mode_auto.return_value = True
        runner.expert_controller = NS(start_process=lambda: events.append('expert_start'))
        runner.setup_expert_controller = lambda: events.append('scenario_export')
        runner.ve_object_input = NS(before_drive=lambda: events.append('ve_ready'),
                                   after_start=lambda: events.append('remap_verified'),
                                   close=lambda: events.append('ve_close'))
        runner.activate_expert_drive()
        self.assertEqual(events, ['scenario_export', 've_ready', 'expert_start', 'remap_verified'])
        runner.stop_expert_controller = lambda: events.append('expert_stop')
        runner.cleanup()
        self.assertEqual(events[-2:], ['expert_stop', 've_close'])

    def test_stale_ve_interrupts_instead_of_using_gt_or_empty_scene(self):
        adapter = object.__new__(VeObjectInput)
        adapter.session = Mock(error=None)
        adapter.session.sender.poll.return_value = None
        adapter.session.snapshot.return_value = dict(ve_fresh=False, ego_age_s=.1)
        adapter.unhealthy_since = 1.
        adapter.output = Path('/tmp/log')
        with patch('scenario_runner.adapters.ve_objects.time.monotonic', return_value=4.), \
             self.assertRaisesRegex(RuntimeError, '끊겼습니다'):
            adapter.inspect()


if __name__ == '__main__':
    unittest.main()
