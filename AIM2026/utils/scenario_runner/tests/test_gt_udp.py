# GT filming must reject pre-existing contacts before starting UDP control.
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..tools.run_gt_udp_full_route import GtUdpFullLoop, UdpController
from ..tools import run_gt_udp_full_route as runner
from ..scenarios.segment_modes.modes import ROUNDABOUT_END, HIGHWAY_END


class GtUdpStartupTests(unittest.TestCase):
    def test_highway_probe_needs_no_traffic_light_input(self):
        with patch('run_gt_udp_expert.main', return_value=0) as probe:
            self.assertEqual(runner.main(['--scenario', 'full_loop_highway', '--probe']), 0)
        arguments = probe.call_args.args[0]
        self.assertEqual(arguments[arguments.index('--signal-source') + 1], 'none')

    def test_full_route_cannot_disable_traffic_light_input(self):
        with self.assertRaises(SystemExit):
            runner.main(['--signal-source', 'none', '--check'])

    def test_highway_prepares_segment_spawn_and_udp_expert(self):
        args = SimpleNamespace(scenario='full_loop_highway', speed_cap_kmh=60.,
                               stop_before_tunnel=False, seed=None, signal_source='grpc')
        global_config, _, base = runner.load_configuration()
        with tempfile.TemporaryDirectory() as directory:
            global_config['paths']['runtime_root'] = directory
            config = runner.configure_scenario(base, args)
            self.assertEqual(config['expert_speed_cap_kmh'], 60.)
            context = runner.prepare_controller(global_config, config, args, Path(directory))
            effective = json.loads(Path(context.expert_controller.runtime_config_path).read_text())
            self.assertLessEqual(effective['planning']['velocity_profile']['max_velocity'], 60.)
            self.assertEqual(context.start_tf, (*ROUNDABOUT_END.xyz, ROUNDABOUT_END.yaw_deg))
            self.assertEqual(context.mission_route.first_index, ROUNDABOUT_END.index)
            self.assertEqual(context.mission_route.last_index, HIGHWAY_END.index)
            self.assertTrue(context.mission_mode.zones.highway)
            self.assertFalse(context.mission_mode.zones.roundabout)
            self.assertFalse(context.mission_mode.zones.tunnel)
            self.assertIsInstance(context.expert_controller, UdpController)
            self.assertFalse(context.cfg['ve_stop_before_tunnel'])

    def test_repeated_runs_use_fresh_scenarios_and_configs(self):
        args = SimpleNamespace(scenario='full_loop_highway', repeat=True, max_runs=3)
        factory = Mock()
        instances = [Mock() for _ in range(3)]
        factory.side_effect = instances
        config = {'random_seed': None, 'nested': {'value': 1}}
        global_config = {'paths': {'mgeo_root': 'map', 'runtime_root': 'old'}}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, 'scenario_class', return_value=factory
        ), patch.object(runner, 'MGeoMapLoader'), patch.object(runner.time, 'sleep'):
            runner.run_scenarios(Mock(), global_config, config, args, Path(directory))
        self.assertEqual(factory.call_count, 3)
        for instance in instances:
            instance.run.assert_called_once_with()
        paths = [call.kwargs['global_cfg']['paths']['runtime_root'] for call in factory.call_args_list]
        self.assertEqual(len(set(paths)), 3)
        factory.call_args_list[0].kwargs['scenario_cfg']['nested']['value'] = 2
        self.assertEqual(factory.call_args_list[1].kwargs['scenario_cfg']['nested']['value'], 1)
        self.assertEqual(config['nested']['value'], 1)

    def test_repeat_stops_on_interrupt_or_failure(self):
        for error in (KeyboardInterrupt(), RuntimeError('failed')):
            factory = Mock(return_value=Mock(run=Mock(side_effect=error)))
            args = SimpleNamespace(scenario='full_loop_highway', repeat=True, max_runs=None)
            with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, 'scenario_class', return_value=factory
            ), patch.object(runner, 'MGeoMapLoader'):
                with self.assertRaises(type(error)):
                    runner.run_scenarios(Mock(), {'paths': {'mgeo_root': 'map'}}, {}, args, Path(directory))
            self.assertEqual(factory.call_count, 1)

    def test_clean_start_records_raw_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = dict(x=1., y=2., collision_objects=[])
            context = SimpleNamespace(gt_output=Path(directory),
                                      get_ego_motion_state=lambda: state)
            GtUdpFullLoop.check_startup_contact(context)
            record = json.loads((context.gt_output / 'startup_contact.json').read_text())
            self.assertEqual(record['raw_state'], state)
            self.assertFalse(record['control_backend_started'])

    def test_self_and_foreign_contacts_are_not_silently_filtered(self):
        for contacts in (['Ego'], ['vehicle_2'], ['Ego', 'vehicle_2']):
            with self.subTest(contacts=contacts), tempfile.TemporaryDirectory() as directory:
                context = SimpleNamespace(gt_output=Path(directory),
                    get_ego_motion_state=lambda: dict(collision_objects=contacts))
                with self.assertRaisesRegex(RuntimeError, '출발 전부터'):
                    GtUdpFullLoop.check_startup_contact(context)
                self.assertEqual(json.loads((context.gt_output / 'startup_contact.json').read_text())
                                 ['collision_objects'], sorted(contacts))

    def test_contact_failure_does_not_launch_backend(self):
        context = SimpleNamespace(is_running=lambda: False,
            before_start=Mock(side_effect=RuntimeError('contact')))
        # No process, log, or output directory access may occur after rejection.
        with self.assertRaisesRegex(RuntimeError, 'contact'):
            UdpController.start_process(context)


# Wire-format and transport tests without a running MORAI or ROS master.
import math
from scenario_runner.project_paths import REPO_ROOT, RUNNER_ROOT, SDK_PROTO
import signal
import struct
import sys
from types import SimpleNamespace as NS

ROOT = REPO_ROOT
sys.path.insert(0, str(RUNNER_ROOT / 'legacy_expert'))
from gt_udp_protocol import parse_ego, parse_objects, parse_signal, pack_control
from run_gt_udp_expert import expert_inputs


def datagram(header, body):
    return header + struct.pack('<4i', len(body), 0, 0, 0) + body + b'\r\n'


def ego_packet(size=245, stamp=100):
    body = bytearray(size-29)
    struct.pack_into('<ii', body, 0, stamp, 123456789)
    struct.pack_into('<bbf', body, 8, 2, 4, 36.)
    struct.pack_into('<3f', body, 50, -60., -130., 28.)
    struct.pack_into('<f', body, 70, 90.)
    return datagram(b'#MoraiInfo$', body)


def object_packet(stride=106, objects=None):
    body = bytearray(8 + 20*stride)
    struct.pack_into('<ii', body, 0, 100, 0)
    for index, (actor_id, kind, speed) in enumerate(objects or []):
        struct.pack_into('<hh16f', body, 8+index*stride, actor_id, kind,
            -60., -110., 28., 90., 4., 2., 1.5, .9, 3., .7, speed, 0., 0., 0., 0., 0.)
    return datagram(b'#MoraiObjInfo$', body)


class ProtocolTests(unittest.TestCase):
    def test_full_ego_versions_keep_si_speed_and_gt_pose(self):
        for size in (229, 245):
            parsed = parse_ego(ego_packet(size))
            self.assertEqual(parsed['position'], (-60., -130., 28.))
            self.assertEqual(parsed['heading_deg'], 90.)
            self.assertEqual(parsed['speed_mps'], 10.)
            self.assertEqual(parsed['stamp_ns'], 100123456789)

    def test_ego_control_readback_offsets(self):
        packet = bytearray(ego_packet(229))
        struct.pack_into('<2f', packet, 45, .3, .7)
        struct.pack_into('<f', packet, 137, -.12)
        parsed = parse_ego(packet)
        for key, value in (('accel', .3), ('brake', .7), ('steering', -.12)):
            self.assertAlmostEqual(parsed[key], value)

    def test_competition_status_is_not_silently_used_as_gt_pose(self):
        with self.assertRaises(ValueError):
            parse_ego(ego_packet(181))

    def test_parked_competition_build_mode_zero_is_valid_gt_telemetry(self):
        packet = bytearray(ego_packet(229))
        struct.pack_into('<bb', packet, 35, 0, 1)
        result = parse_ego(packet)
        self.assertEqual((result['ctrl_mode'], result['gear']), (0, 1))
        self.assertEqual(result['position'], (-60., -130., 28.))

    def test_object_versions_empty_packets_and_full_identity(self):
        for stride in (68, 106):
            self.assertEqual(parse_objects(object_packet(stride))['objects'], [])
            parsed = parse_objects(object_packet(stride, [(7, 1, 36.), (8, 2, 0.), (9, 0, 5.)]))
            self.assertEqual([o['type'] for o in parsed['objects']], [1, 2, 0])
            self.assertEqual([o['unique_id'] for o in parsed['objects']], [7, 8, 9])
            self.assertEqual(parsed['objects'][0]['velocity_kmh'], (36., 0., 0.))
            self.assertEqual(parsed['objects'][0]['size'], (4., 2., 1.5))

    def test_signal_preserves_exact_id_left_arrow_and_combination_bits(self):
        for status in (1, 16, 32, 33, 48):
            packet = datagram(b'#TrafficLight$', b'C1256W000052'+struct.pack('<hh', 2, status))
            value = parse_signal(packet)
            self.assertEqual(value, dict(signal_id='C1256W000052', type=2, status=status))

    def test_bad_length_tail_nonfinite_and_duplicate_id_are_rejected(self):
        packet = ego_packet()
        for broken in (packet[:-1], packet+b'\0', b'X'+packet[1:]):
            with self.assertRaises(ValueError):
                parse_ego(broken)
        packet = bytearray(ego_packet())
        struct.pack_into('<f', packet, 77, float('nan'))
        with self.assertRaises(ValueError):
            parse_ego(packet)
        with self.assertRaises(ValueError):
            parse_objects(object_packet(objects=[(7, 1, 36.), (7, 1, 36.)]))

    def test_default_control_matches_applied_competition_55_byte_layout(self):
        packet = pack_control(.3, .1, .12)
        self.assertEqual(len(packet), 55)
        self.assertEqual(packet[:14], b'#MoraiCtrlCmd$')
        self.assertEqual(struct.unpack_from('<i', packet, 14)[0], 23)
        self.assertEqual(struct.unpack_from('<3b', packet, 30), (2, 4, 1))
        self.assertEqual(struct.unpack_from('<2f', packet, 33), (0., 0.))
        for offset, value in ((41, .3), (45, .1), (49, .12)):
            self.assertAlmostEqual(struct.unpack_from('<f', packet, offset)[0], value)
        self.assertEqual(packet[-2:], b'\r\n')
        with self.assertRaises(ValueError):
            pack_control(0., 1., .12, .02)
        with self.assertRaises(ValueError):
            pack_control(0., 1., layout='guess')

    def test_optional_front_rear_control_keeps_explicit_59_byte_layout(self):
        packet = pack_control(.3, 0., .12, -.02, layout='front_rear59')
        self.assertEqual(len(packet), 59)
        self.assertEqual(packet[:14], b'#MoraiCtrlCmd$')
        self.assertEqual(struct.unpack_from('<i', packet, 14)[0], 27)
        self.assertEqual(struct.unpack_from('<3b', packet, 30), (2, 4, 1))
        self.assertAlmostEqual(struct.unpack_from('<f', packet, 49)[0], .12)
        self.assertAlmostEqual(struct.unpack_from('<f', packet, 53)[0], -.02)
        self.assertEqual(packet[-2:], b'\r\n')

    def test_invalid_controls_are_rejected(self):
        for values in ((1.1, 0., 0.), (0., -.1, 0.), (0., 0., float('nan'))):
            with self.assertRaises(ValueError):
                pack_control(*values)

    def test_adapter_matches_existing_expert_units_categories_and_freshness(self):
        vehicle_type = lambda x, y, yaw, speed: NS(x=x, y=y, yaw=yaw, speed=speed)
        object_type = lambda x, y, speed, kind, **kw: NS(x=x, y=y, speed=speed, kind=kind, **kw)
        data = dict(ego=parse_ego(ego_packet()),
                    objects=parse_objects(object_packet(objects=[(7, 1, 36.), (8, 2, 0.)])),
                    signal=dict(signal_id='C1256W000052', type=2, status=33))
        vehicle, objects, light = expert_inputs(data, dict(signal=123.), vehicle_type, object_type)
        self.assertAlmostEqual(vehicle.yaw, math.pi/2)
        self.assertEqual(vehicle.speed, 10.)
        self.assertEqual(objects[0].speed, 10.)
        self.assertFalse(objects[0].is_static)
        self.assertTrue(objects[1].is_static)
        self.assertEqual(objects[0].unique_id, 7)
        self.assertEqual(light, ['C1256W000052', 33, 2, 123.])


if __name__ == "__main__":
    unittest.main()
