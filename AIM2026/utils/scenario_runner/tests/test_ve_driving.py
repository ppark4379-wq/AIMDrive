# Direct GPS+VE integration keeps the original planner's object contract.
from pathlib import Path
from scenario_runner.project_paths import REPO_ROOT, RUNNER_ROOT, SDK_PROTO
import json
import struct
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = REPO_ROOT
sys.path.insert(0, str(RUNNER_ROOT/'legacy_expert'))
from run_ve_gps_expert import planner_inputs, configure_udp_steering, place_route_start, encode_planner_control
from gt_udp_protocol import pack_control


class GpsExpertInputTests(unittest.TestCase):
    def test_planner_radians_are_converted_to_normalized_wire_and_pedals_preserved(self):
        for degrees, expected in ((0.,0.), (10.,.25), (-20.,-.5), (40.,1.), (80.,1.)):
            control = SimpleNamespace(accel=.3,brake=0.,front_steer=math.radians(degrees),rear_steer=0.)
            wire = encode_planner_control(control,math.radians(40.))
            self.assertEqual(len(wire),55)
            self.assertAlmostEqual(struct.unpack_from('<f',wire,49)[0],expected)
            self.assertAlmostEqual(struct.unpack_from('<f',wire,41)[0],.3)
            self.assertEqual(struct.unpack_from('<f',wire,45)[0],0.)
        with self.assertRaises(ValueError):
            encode_planner_control(control,0.)
    def test_start_placement_sends_only_bounded_setup_writes(self):
        sys.path.insert(0, str(SDK_PROTO))
        from morai.common.object_info_pb2 import ObjectInfo
        from morai.common.object_identifier_pb2 import ObjectIdentifier
        from morai.common.enum_pb2 import STATUS_CODE_SUCCESS
        stub = SimpleNamespace(**{name: Mock(return_value=SimpleNamespace(status=STATUS_CODE_SUCCESS))
            for name in ('SetVehicleEgoCruise','SetVelocity','SetTransform')})
        actor = ObjectInfo(id=ObjectIdentifier(value='Ego'),object_type=1)
        place_route_start(stub,actor,(-131.,-428.,28.,61.))
        self.assertFalse(stub.SetVehicleEgoCruise.call_args.args[0].cruise_on)
        self.assertEqual(stub.SetVelocity.call_count,2)
        for call in stub.SetVelocity.call_args_list:
            self.assertEqual(call.args[0].velocity,0.)
            self.assertEqual(call.kwargs,dict(timeout=1.))
        request = stub.SetTransform.call_args.args[0]
        self.assertEqual((request.transform.location.x,request.transform.location.y,
                          request.transform.location.z,request.transform.rotation.z),(-131.,-428.,28.,61.))
        stub.SetTransform.reset_mock()
        stub.SetVelocity.return_value = SimpleNamespace(status=-1,description='denied')
        with self.assertRaisesRegex(RuntimeError,'velocity_zero'):
            place_route_start(stub,actor,(-131.,-428.,28.,61.))
        stub.SetTransform.assert_not_called()

    def test_left_and_right_path_errors_produce_corrective_udp_steering(self):
        from AIM2026.Expert.src.control.path_controller import PathController
        from AIM2026.Expert.src.vehicle_state import VehicleState
        from AIM2026.Expert.src.interfaces.geometry import Point
        config = json.loads((ROOT/'AIM2026/Expert/src/config/config.json').read_text())
        for profile, physical_left_sign in (('normal', -1.), ('competition', 1.)):
            configure_udp_steering(config, profile)
            for y, direction in ((1., -1), (-1., 1)):
                with self.subTest(profile=profile, y=y):
                    controller = PathController(config)
                    result = SimpleNamespace(mode='normal', active=False, reason='', debug_details={}, highway_state='')
                    control = controller.control_path([Point(x, 0.) for x in range(30)],
                        VehicleState(0.,y,0.,3.), 3., result, sync_mode=False, control_dt=1./30)
                    # PurePursuit -> per-build transport -> physical direction.
                    wire = encode_planner_control(control,controller.max_steering)
                    self.assertGreater(direction*physical_left_sign*struct.unpack_from('<f',wire,49)[0], 0)
        configure_udp_steering(config)
        self.assertEqual(config['common']['realtime_steering_sign'], -1.)
        with self.assertRaises(KeyError):
            configure_udp_steering(config, 'unverified')

    def test_grid_heading_speed_and_local_motion_tracks_are_preserved(self):
        packet = dict(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading',
            ego_pose=dict(x=-60., y=-130., yaw_raw_deg=90., yaw_grid_rad=1.58, speed_mps=10.),
            boxes=[dict(center_world=[-58.,-110.], planning_speed_mps=0., motion_state='stationary',
                length_m=3., width_m=2., planning_heading_deg=91., track_id=42),
                dict(center_world=[-60.,-105.], planning_speed_mps=4., motion_state='moving',
                length_m=4., width_m=2., planning_heading_deg=89., track_id=43)])
        vehicle, objects = planner_inputs(packet)
        self.assertEqual((vehicle.position.x,vehicle.position.y,vehicle.yaw,vehicle.velocity),
                         (-60.,-130.,1.58,10.))
        self.assertEqual([o.velocity for o in objects], [0.,4.])
        self.assertEqual([o.is_static for o in objects], [True,False])
        self.assertEqual([o.unique_id for o in objects], [42,43])
        self.assertEqual([o.type for o in objects], [-1,-1])
        latest_pose = dict(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading',
            ego_pose=dict(x=-61., y=-120., yaw_raw_deg=80., yaw_grid_rad=1.4, speed_mps=12.))
        vehicle, objects = planner_inputs(packet, latest_pose)
        self.assertEqual((vehicle.position.x,vehicle.position.y,vehicle.yaw,vehicle.velocity),
                         (-61.,-120.,1.4,12.))
        # Changing control time must never re-transform the old world boxes.
        self.assertEqual((objects[0].position.x,objects[0].position.y),(-58.,-110.))
        self.assertEqual(packet['ego_pose']['x'],-60.)
        latest_pose['ego_position_source'] = '/Ego_topic'
        with self.assertRaises(ValueError):
            planner_inputs(packet, latest_pose)
        packet['ego_position_source'] = '/Ego_topic'
        with self.assertRaises(ValueError):
            planner_inputs(packet)


# Scenario lifecycle must never switch the GPS driver back to simulator GT.
import copy
import contextlib
import io
import tempfile
from unittest.mock import Mock, patch

from MoraiDrive.scenario_runner.tools.run_ve_gps_full_route import configure_scenario, GpsController
from run_ve_gps_expert import prepare
from MoraiDrive.scenario_runner.tools import run_ve_gps_full_route as route_runner


class GpsFullRouteTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(speed_cap_kmh=60., seed=123, stop_before_tunnel=False,
            static_preset=None, ego_trace=None, sim_profile='normal', signal_head=Path('/unused/head.pt'),
            bind_ip='0.0.0.0',sim_ip='127.0.0.1',status_port=8000,gps_port=10506,imu_port=10507,
            control_port=7601,control_source_port=7600,
            mission='',tunnel_dynamic_speed=None,tunnel_candidate_id=None,tunnel_layout=None,gt_expert=False,
            tunnel_vehicle_behavior=None,tunnel_vehicle_random=False,
            moving_carry_s=0.,slow_window_s=0.,stationary_hold_s=0.)

    def test_previous_ros_config_cannot_start_parallel_ve_or_gt_inputs(self):
        base = dict(ve_objects_checkpoint='E3',ve_signal_head='old',ve_full_route_signals=True,
            expert_ros_remaps=['/Object_topic:=/Object_topic'], sync_runtime={'enabled':True},
            highway_merge_static_vehicle={'enabled':True},traffic_light_wait_override={'enabled':True},
            dynamic_obstacle={'pre_signal_guard':{'enabled':True}},roundabout={'enabled':True})
        original = copy.deepcopy(base)
        cfg = configure_scenario(base, self.args())
        self.assertEqual(base, original)
        self.assertNotIn('ve_objects_checkpoint',cfg)
        self.assertFalse(cfg['sync_runtime']['enabled'])
        self.assertFalse(cfg['highway_merge_static_vehicle']['enabled'])
        self.assertFalse(cfg['traffic_light_wait_override']['enabled'])
        self.assertTrue(cfg['dynamic_obstacle']['pre_signal_guard']['enabled'])
        self.assertTrue(cfg['roundabout']['enabled'])
        self.assertEqual(cfg['expert_ros_remaps'], [])
        self.assertTrue(cfg['static_verified_pair'])
        self.assertEqual(cfg['max_laps'],1)

    def test_driver_uses_same_route_snapshot_and_no_second_teleport_or_gt_ports(self):
        with tempfile.TemporaryDirectory() as d:
            context = SimpleNamespace(gps_output=Path(d),gps_args=self.args(),
                global_cfg={'paths':{}},cfg={'ve_stop_before_tunnel':True},route_points=[(1.,2.,3.),(4.,5.,6.)])
            controller = GpsController(Mock(),context)
            command = controller.command()
            self.assertIn('--drive',command)
            for disallowed in ('--from-start','--object-port','--signal-port','--signal-source'):
                self.assertNotIn(disallowed,command)
            self.assertIn('run_ve_gps_expert.py', ' '.join(command))
            saved = json.loads(controller.context_path.read_text())
            self.assertEqual(saved['route_points'], [[1.,2.,3.],[4.,5.,6.]])
            self.assertTrue(saved['scenario_config']['ve_stop_before_tunnel'])
            self.assertNotIn('--drive',controller.command(check=True))

    def test_aim_backend_forwarding_changes_only_the_explicit_backend_option(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args()
            context=SimpleNamespace(gps_output=Path(directory),gps_args=args,
                global_cfg={'paths':{}},cfg={},route_points=[(1.,2.,3.),(4.,5.,6.)])
            controller=GpsController(Mock(),context)
            before=controller.command()
            args.aim2026_config=Path(directory)/'backend with spaces.yaml'
            after=controller.command()
            index=after.index('--aim2026-config')
            self.assertEqual(after[index+1],str(args.aim2026_config))
            self.assertEqual(after[:index]+after[index+2:],before)
            checked=controller.command(check=True)
            self.assertIn('--aim2026-config',checked)
            self.assertIn('--check',checked)

    def test_aim_backend_guards_run_before_scenario_setup_or_model_checks(self):
        base=['--check','--aim2026-config',str(route_runner.REPO/'AIM2026/config.yaml'),
              '--mission','full_loop_tunnel_normal']
        for extra in ([],['--gps-blackout-hold','--simulate-gps-blackout'],
                      ['--gps-blackout-hold','--gps-latency-s','.1'],
                      ['--gps-blackout-hold','--gt-expert']):
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(route_runner,'load_configuration') as setup:
                with self.assertRaises(SystemExit):route_runner.main(base+extra)
                setup.assert_not_called()

    def test_child_refuses_a_route_different_from_the_scene_before_planner_setup(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args()
            args.scenario_context = Path(d)/'context.json'
            args.scenario_context.write_text(json.dumps(dict(schema='ve_gps_scenario_context_v1',
                global_config={'paths':{'mgeo_root':'unused'}},scenario_config={},route_points=[[9.,9.,9.]])))
            context = Mock(route_points=[(1.,2.,3.)])
            with patch('scenario_runner.scenarios.full_loop_scenario.FullLoopScenario',return_value=context), \
                 patch('scenario_runner.utils.map_loader.MGeoMapLoader'):
                with self.assertRaisesRegex(ValueError,'경로가 다릅니다'):
                    prepare(args,Path(d))
            context.setup_expert_controller.assert_not_called()


    def test_stop_terminates_actual_driver_not_unused_ros_controller(self):
        context = SimpleNamespace(process=Mock(),log=Mock())
        process, log = context.process, context.log
        with patch('MoraiDrive.scenario_runner.tools.run_ve_gps_full_route.stop_child') as stop:
            GpsController.stop_process(context)
        stop.assert_called_once_with(process)
        log.close.assert_called_once()
        self.assertIsNone(context.process)


# Existing runner with VE-only object input; no simulator or model needed.
import socket
import time
from types import SimpleNamespace as NS

from scenario_runner.adapters.ve_objects import configure, configure_expert, verify_topology, VeObjectInput
from scenario_runner.main_runner import load_configuration
from scenario_runner.core.full_loop_drive_base import FullLoopDriveBase
from MoraiDrive.scenario_runner.legacy_expert.ve_udp_session import PerceptionSession


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
             patch('MoraiDrive.scenario_runner.legacy_expert.ve_udp_session.socket.socket'), \
             patch('MoraiDrive.scenario_runner.legacy_expert.ve_udp_session.build_analysis_environment', return_value={}), \
             patch('MoraiDrive.scenario_runner.legacy_expert.ve_udp_session.subprocess.Popen'), \
             patch('MoraiDrive.scenario_runner.legacy_expert.ve_udp_session.threading.Thread'):
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
        from AIM2026.UDP.Detection import SCHEMA, encode_legacy_packet as encode_packet
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


class LegacyProcessGuardTests(unittest.TestCase):
    def test_scan_skips_exited_process_and_returns_first_match_without_extra_reads(self):
        from scenario_runner.legacy_expert import _find_running_command
        gone = Mock()
        gone.read_bytes.side_effect = FileNotFoundError('process exited')
        benign = Mock()
        benign.read_bytes.return_value = b'python\0unrelated.py'
        first = Mock()
        first.read_bytes.return_value = b'python\0collect_scenario_laps.sh\0--x\xff'
        later = Mock()
        result = _find_running_command(iter([gone, benign, first, later]), ('collect_scenario_laps.sh',))
        self.assertEqual(result, 'python collect_scenario_laps.sh --x\ufffd')
        later.read_bytes.assert_not_called()

    def test_scan_distinguishes_empty_match_from_absence(self):
        from scenario_runner.legacy_expert import _find_running_command
        empty = Mock()
        empty.read_bytes.return_value = b''
        self.assertEqual(_find_running_command([empty], ('',)), '')
        self.assertIsNone(_find_running_command([empty], ('collector',)))
        self.assertIsNone(_find_running_command([], ('',)))

    def test_scan_propagates_unexpected_read_errors(self):
        from scenario_runner.legacy_expert import _find_running_command
        broken = Mock()
        broken.read_bytes.side_effect = ValueError('unexpected read failure')
        with self.assertRaisesRegex(ValueError, 'unexpected read failure'):
            _find_running_command([broken], ('collector',))

    def test_capture_guard_rejects_runner_while_manual_collector_guard_allows_it(self):
        from scenario_runner.legacy_expert import record_ve_manual, run_ve_udp_expert
        process = Mock()
        process.read_bytes.return_value = b'python\0-m\0scenario_runner.main_runner'
        for module, function, expected in (
            (run_ve_udp_expert, run_ve_udp_expert.capture_guard, RuntimeError),
            (record_ve_manual, record_ve_manual._collector_guard, None),
        ):
            with self.subTest(module=module.__name__), patch.object(module, 'Path') as path:
                path.return_value.glob.return_value = iter([process])
                if expected:
                    with self.assertRaisesRegex(expected, '^수집/시나리오 러너가 실행 중입니다. 종료한 뒤 실행하세요.$'):
                        function()
                else:
                    self.assertIsNone(function())
                path.assert_called_once_with('/proc')
                path.return_value.glob.assert_called_once_with('[0-9]*/cmdline')

    def test_both_guards_reject_each_collector_with_existing_error_policy(self):
        from scenario_runner.legacy_expert import record_ve_manual, run_ve_udp_expert
        for token in ('collect_scenario_laps.sh', 'run_sensor_record_scenario.py'):
            command = 'python ' + token + ' ' + ('긴' * 120)
            for module, function, error, message in (
                (run_ve_udp_expert, run_ve_udp_expert.capture_guard, RuntimeError,
                 '수집/시나리오 러너가 실행 중입니다. 종료한 뒤 실행하세요.'),
                (record_ve_manual, record_ve_manual._collector_guard, SystemExit,
                 '센서 수집기가 실행 중입니다. 수집과 동시에 VE를 돌리지 않습니다: ' + command[:100]),
            ):
                with self.subTest(token=token, module=module.__name__), patch.object(module, 'Path') as path:
                    process = Mock()
                    process.read_bytes.return_value = command.encode()
                    path.return_value.glob.return_value = iter([process])
                    with self.assertRaises(error) as caught:
                        function()
                    self.assertEqual(str(caught.exception), message)


if __name__ == "__main__":
    unittest.main()
