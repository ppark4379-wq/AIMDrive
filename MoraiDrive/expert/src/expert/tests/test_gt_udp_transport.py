"""Wire-format and transport tests without a running MORAI or ROS master."""
import json
import math
from pathlib import Path
import signal
import struct
import sys
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'MoraiDrive/expert/tools'))
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


if __name__ == '__main__':
    unittest.main()
