import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from inference.ve_objects_udp import (SCHEMA, decode_packet, encode_packet,
    make_object_message)
from inference.ve_object_tracking import VeObjectTracker

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('udp_expert_launcher',
    REPO / 'MoraiDrive/expert/tools/run_ve_udp_expert.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def sample():
    return dict(schema=SCHEMA, generation='test', checkpoint_sha256='a' * 64,
        output_frame='projected_grid', scene_assumption='unknown', sequence=4,
        ready=True, sent_ns=1200000000, sensor_received_ns=1000000000,
        ego_pose=dict(x=-60., y=-160., yaw_raw_deg=90.),
        boxes=[dict(center_world=[-60., -145.], heading_deg=90., length_m=1.1,
                    width_m=.3, speed_mps=2.5)])


def msg():
    return NS(header=NS(), obstacle_list=[], npc_list=[], pedestrian_list=[])


def obj():
    return NS(position=NS(), size=NS(), velocity=NS())


class UdpObjectsTests(unittest.TestCase):
    def decode(self, packet, now=1300000000):
        return decode_packet(encode_packet(packet), 'test', 'a' * 64, now)

    def test_roundtrip_preserves_physical_geometry_and_scalar_speed(self):
        packet = self.decode(sample())
        # Transport preserves the speed selected by tracking; warm-up speed is
        # now explicitly unconfirmed instead of copied from the model.
        tracker = VeObjectTracker()
        for i in range(5):
            packet['sent_ns'] = 1200000000 + i*100000000
            packet['boxes'][0]['center_world'][1] = -146. + i*.25
            tracked = tracker.update(packet)
        wire = make_object_message(tracked, msg, obj, 'publication_stamp')
        self.assertEqual(wire.header.stamp, 'publication_stamp')
        self.assertEqual(wire.header.frame_id, 'map')
        self.assertEqual((wire.num_of_obstacle, wire.num_of_npcs, wire.num_of_pedestrian), (0, 1, 0))
        item = wire.npc_list[0]
        self.assertEqual((item.position.x, item.position.y), (-60., -145.))
        self.assertEqual((item.size.x, item.size.y, item.heading), (1.1, .3, 90.))
        self.assertEqual(item.velocity.x / 3.6, 2.5)
        self.assertEqual((item.type, item.unique_id), (-1, 1))

    def test_unconfirmed_nonzero_model_speed_keeps_obstacle_without_claiming_motion(self):
        tracked = VeObjectTracker().update(self.decode(sample()))
        self.assertEqual(tracked['boxes'][0]['speed_mps'], 2.5)
        wire = make_object_message(tracked, msg, obj, 1)
        self.assertEqual(wire.npc_list[0].name, 've_tracked_unknown:uncertain')
        self.assertEqual(wire.npc_list[0].velocity.x, 0.)
        self.assertEqual(wire.num_of_npcs, 1)
        self.assertEqual(wire.num_of_obstacle, 0)

    def test_valid_empty_frame_is_distinct_from_missing_sensor(self):
        packet = sample(); packet['boxes'] = []
        self.assertEqual(self.decode(packet)['boxes'], [])
        packet.update(ready=False, reason='missing:lidar')
        with self.assertRaisesRegex(ValueError, 'not ready'):
            self.decode(packet)

    def test_unknown_speed_keeps_a_tagged_uncertain_box(self):
        packet = sample(); packet['boxes'][0]['speed_mps'] = None
        wire = make_object_message(VeObjectTracker().update(self.decode(packet)), msg, obj, 1)
        self.assertEqual(wire.npc_list[0].velocity.x, 0.)
        self.assertEqual(wire.npc_list[0].unique_id, 1)
        self.assertEqual(wire.npc_list[0].name, 've_tracked_unknown:uncertain')

    def test_other_run_checkpoint_and_coordinate_basis_rejected(self):
        for key, value in (('generation', 'other'), ('checkpoint_sha256', 'b' * 64),
                           ('output_frame', 'ego'), ('scene_assumption', 'dynamic')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.decode(dict(sample(), **{key: value}))

    def test_stale_and_future_packets_rejected(self):
        for now in (1199999999, 1700000001):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.decode(sample(), now=now)

    def test_fresh_send_cannot_hide_stale_sensor_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'snapshot expired'):
            self.decode(sample(), now=1500000001)

    def test_invalid_geometry_speed_and_timestamp_rejected(self):
        for key, value in (('length_m', 0), ('width_m', -1), ('speed_mps', -1),
                           ('center_world', [2]), ('heading_deg', '90')):
            packet = sample(); packet['boxes'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.decode(packet)
        with self.assertRaises(ValueError):
            self.decode(dict(sample(), sensor_received_ns=1400000000))

    def test_nan_inf_and_datagram_overflow_rejected(self):
        for value in (float('nan'), float('inf')):
            packet = sample(); packet['boxes'][0]['heading_deg'] = value
            with self.assertRaises(ValueError):
                encode_packet(packet)
        with self.assertRaises(ValueError):
            encode_packet(dict(sample(), extra='x' * 60000))
        # JSON NaN accepted by the stdlib reader is rejected by finite checks.
        raw = encode_packet(sample()).replace(b'"heading_deg":90.0', b'"heading_deg":NaN')
        with self.assertRaises(ValueError):
            decode_packet(raw, 'test', 'a' * 64, 1300000000)

    def test_object_input_requires_live_packet_and_ego(self):
        packet, now = sample(), 1300000000
        ego = dict(x=-60., y=-160., heading=90., received_ns=1200000000)
        self.assertEqual(launcher.input_status(packet, ego, now), (True, 'ready'))
        for values in ((None, ego, now), (packet, ego, 1600000000),
                       (packet, None, now), (packet, dict(ego, received_ns=1), now)):
            with self.subTest(values=values):
                self.assertFalse(launcher.input_status(*values)[0])

    def test_no_trial_region_or_current_pose_gate_blocks_original_expert(self):
        packet = sample()
        ego = dict(x=-60., y=-160., heading=90., received_ns=1200000000)
        for position in ((-131., -427.), (-60., -80.), (100., 100.)):
            self.assertTrue(launcher.input_status(packet, dict(ego, x=position[0], y=position[1]), 1300000000)[0])

    def test_expert_uses_original_planning_and_control_with_object_remap(self):
        original = json.loads((REPO / 'MoraiDrive/expert/src/expert/src/config/config.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, 'is_file', return_value=True):
            config_path, path = launcher.prepare_expert(Path(tmp))
            config = json.loads(config_path.read_text())
            expected = copy.deepcopy(original)
            expected['common']['perception_source'] = 'ros'
            expected['common']['sync_grpc_inputs']['enabled'] = False
            # Whole-config equality prevents accidental changes to speed,
            # steering, signals, map binding or any planning parameters.
            self.assertEqual(config, expected)
            self.assertEqual(path, REPO / 'MoraiDrive/expert/.runtime/path.csv')
        command = launcher.expert_command(drive=True)
        self.assertIn('/Object_topic:=/ve/Object_topic', command)
        self.assertFalse(any('/ctrl_cmd:=' in word for word in command))
        self.assertFalse(any('GetTrafficLightStatus:=' in word for word in command))
        shadow = launcher.expert_command(drive=False)
        self.assertIn('/ctrl_cmd:=/ve/shadow_ctrl_cmd', shadow)


if __name__ == '__main__':
    unittest.main()
