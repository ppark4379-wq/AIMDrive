"""Exercise the recorded byte stream, seals, and separate timing domains."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from MoraiDrive.data_collection.situation_labeling.expert_publications import (
    CTRL_FIELDS, MD5, PublicationReader, attach_expert_publications)


NS = 1_000_000_000
ANCHOR = 10_000_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def header(frame='map'):
    return struct.pack('<IIII', 1, 0, 0, len(frame)) + frame.encode()


def path_bytes(frame='map', count=2):
    return header(frame) + struct.pack('<I', count) + b''.join(
        header('') + struct.pack('<7d', float(i), 0., 0., 0., 0., 0., 1.)
        for i in range(count))


class RecordedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.episode = Path(self.temp.name)
        for part in ('quality', 'raw/chunks', 'native_associations'):
            (self.episode / part).mkdir(parents=True)
        self.frame = {'index': 7, 'lidar_source_ns': NS, 'zone_id': 'highway'}
        self.rows = [dict(role='lidar', record_id=1, source_timestamp_ns=NS,
                          receive_monotonic_ns=ANCHOR)]
        self.blob = bytearray(b'journal!')
        self.add('expert_control', struct.pack('<i6d', 1, .2, 0., 0., 0., 0., 0.), 2)
        self.add('expert_target_velocity', struct.pack('<f', 4.), 3)
        self.add('expert_local_path', path_bytes(), 4)
        self.rows.append(dict(role='ego', record_id=5, source_timestamp_ns=NS,
            receive_monotonic_ns=ANCHOR - 154_000_000, preceding_received_clock_ns=NS,
            chunk_path='chunks/chunk.bin', payload_offset=8, payload_bytes=0,
            payload_sha256=digest(b''), ros_type='morai_msgs/EgoVehicleStatus',
            signal=dict(position=[1., 2., 3.], speed=2., yaw=.1)))

    def add(self, role, data, record, receive=None, clock=None):
        typ = {'expert_control': 'morai_msgs/CtrlCmd',
               'expert_target_velocity': 'std_msgs/Float32',
               'expert_local_path': 'nav_msgs/Path'}[role]
        row = dict(role=role, ros_type=typ, md5sum=MD5[typ], record_id=record,
            source_timestamp_ns=None, receive_monotonic_ns=receive if receive is not None else ANCHOR-10_000_000,
            preceding_received_clock_ns=clock if clock is not None else NS-10_000_000,
            payload_offset=len(self.blob), payload_bytes=len(data), payload_sha256=digest(data),
            chunk_path='chunks/chunk.bin', caller_id='/captured_expert')
        self.blob.extend(data)
        self.rows.append(row)
        return row

    def write(self, name, data):
        raw = json.dumps(data).encode()
        (self.episode / name).write_bytes(raw)
        return digest(raw)

    def seal(self):
        (self.episode / 'raw/chunks/chunk.bin').write_bytes(self.blob)
        manifest_sha = self.write('raw/raw_manifest.json', dict(run_id='run1',
            chunks=[dict(path='chunks/chunk.bin', bytes=len(self.blob))],
            extra=dict(topic_state=dict(definitions={typ: dict(md5sum=md5,
                definition='\n'.join(CTRL_FIELDS) if typ == 'morai_msgs/CtrlCmd' else '')
                for typ, md5 in MD5.items()}))))
        self.write('raw/RAW_COMPLETE.json', dict(status='completed', manifest_sha256=manifest_sha))
        signals_sha = self.write('quality/active_signals.json', self.rows)
        self.write('quality/QUALITY_COMPLETE.json', dict(raw_run_id='run1', files={'active_signals.json': signals_sha}))
        assoc_sha = self.write('native_associations/associations.json', [dict(index=7,
            scene_time_hypothesis_ns=NS, lidar_ros_record_id=1)])
        self.write('native_associations/COMPLETE.json', dict(raw_manifest_sha256=manifest_sha,
            files={'associations.json': assoc_sha}))

    def read(self):
        self.seal()
        reader = PublicationReader(self.episode)
        self.addCleanup(reader.close)
        return reader, reader.for_frame(self.frame)

    def test_real_payloads_and_source_teacher_use_distinct_clocks(self):
        _, result = self.read()
        self.assertEqual(result['control']['accel'], .2)
        self.assertEqual(result['local_path']['world_xyz'], [[0., 0., 0.], [1., 0., 0.]])
        self.assertEqual(result['ego']['receive_age_ms'], 154.)
        self.assertEqual(result['ego']['ego_pose_source_age_ms'], 0.)
        self.assertFalse(result['physical_sensor_synchrony_certified'])
        self.assertIsNone(result['target_velocity']['source_timestamp_ns'])
        self.assertFalse(result['target_velocity']['source_timestamp_fabricated'])

    def test_same_clock_uses_latest_preceding_receipt_and_ignores_future(self):
        self.add('expert_target_velocity', struct.pack('<f', .24), 6,
                 receive=ANCHOR-1, clock=NS)
        self.add('expert_target_velocity', struct.pack('<f', 99.), 7,
                 receive=ANCHOR+1, clock=NS)
        _, result = self.read()
        self.assertEqual(result['target_velocity']['record_id'], 6)
        self.assertAlmostEqual(result['target_velocity']['value_mps'], .24)

    def test_stale_receive_or_sim_clock_cannot_be_current_command(self):
        self.rows[1]['receive_monotonic_ns'] = ANCHOR-101_000_000
        self.rows[2]['preceding_received_clock_ns'] = NS-101_000_000
        _, result = self.read()
        self.assertIsNone(result['control'])
        self.assertIsNone(result['target_velocity'])

    def test_future_sim_clock_is_excluded_even_with_prior_receive(self):
        self.rows[1]['preceding_received_clock_ns'] = NS+1
        _, result = self.read()
        self.assertIsNone(result['control'])

    def test_historical_zero_target_has_separate_bounded_history(self):
        self.add('expert_target_velocity', struct.pack('<f', 0.), 6,
                 receive=ANCHOR-1_200_000_000, clock=NS-400_000_000)
        _, result = self.read()
        prior = result['target_velocity']['prior_zero_target']
        self.assertEqual(prior['record_id'], 6)
        self.assertEqual(prior['purpose'], 'prior_stop_context_not_current_command')
        self.assertEqual(prior['history_receive_bound_ms'], 5000)

    def test_historical_zero_older_than_sim_bound_is_not_used(self):
        self.add('expert_target_velocity', struct.pack('<f', 0.), 6,
                 receive=ANCHOR-500_000_000, clock=NS-1_000_000_001)
        _, result = self.read()
        self.assertNotIn('prior_zero_target', result['target_velocity'])

    def test_negative_current_target_is_audited_without_clamp_or_older_fallback(self):
        raw = struct.pack('<f', -.024160197004675865)
        row = self.add('expert_target_velocity', raw, 6, receive=ANCHOR-1)
        _, result = self.read()
        self.assertIsNone(result['target_velocity'])
        self.assertIsNotNone(result['control'])
        self.assertIsNotNone(result['local_path'])
        self.assertIn('unsupported_current_expert_target_velocity', result['reasons'])
        receipt = result['unsupported_publications'][0]
        self.assertEqual(receipt['record_id'], row['record_id'])
        self.assertEqual(receipt['payload_sha256'], digest(raw))
        self.assertEqual(receipt['raw_value_mps'], struct.unpack('<f', raw)[0])
        self.assertIsNone(receipt['source_timestamp_ns'])
        self.assertFalse(receipt['source_timestamp_fabricated'])

    def test_negative_history_blocks_prior_zero_search_without_aborting_current_frame(self):
        self.add('expert_target_velocity', struct.pack('<f', 0.), 6,
                 receive=ANCHOR-700_000_000, clock=NS-700_000_000)
        self.add('expert_target_velocity', struct.pack('<f', -.003930388018488884), 7,
                 receive=ANCHOR-500_000_000, clock=NS-500_000_000)
        _, result = self.read()
        self.assertEqual(result['target_velocity']['value_mps'], 4.)
        self.assertNotIn('prior_zero_target', result['target_velocity'])
        self.assertIn('unsupported_historical_expert_target_velocity', result['reasons'])
        receipt = result['unsupported_publications'][0]
        self.assertEqual(receipt['record_id'], 7)
        self.assertLess(receipt['raw_value_mps'], 0.)
        self.assertEqual(receipt['purpose'], 'prior_stop_context_search_barrier')

    def test_zero_more_recent_than_negative_history_is_still_valid_stop_context(self):
        self.add('expert_target_velocity', struct.pack('<f', -.003930388018488884), 6,
                 receive=ANCHOR-700_000_000, clock=NS-700_000_000)
        self.add('expert_target_velocity', struct.pack('<f', 0.), 7,
                 receive=ANCHOR-500_000_000, clock=NS-500_000_000)
        _, result = self.read()
        self.assertEqual(result['target_velocity']['prior_zero_target']['record_id'], 7)
        self.assertEqual(result['unsupported_publications'], [])

    def test_nonfinite_target_is_not_swallowed_as_unsupported(self):
        self.add('expert_target_velocity', struct.pack('<f', float('nan')), 6,
                 receive=ANCHOR-1)
        self.seal()
        reader = PublicationReader(self.episode)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, 'Invalid Expert target velocity'):
            reader.for_frame(self.frame)

    def test_corrupt_negative_target_remains_integrity_failure(self):
        row = self.add('expert_target_velocity', struct.pack('<f', -.024160197), 6,
                       receive=ANCHOR-1)
        self.seal()
        path = self.episode / 'raw/chunks/chunk.bin'
        blob = bytearray(path.read_bytes())
        blob[row['payload_offset']] ^= 1
        path.write_bytes(blob)
        reader = PublicationReader(self.episode)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, 'differs from sealed'):
            reader.for_frame(self.frame)

    def test_empty_latest_path_does_not_resurrect_older_plan(self):
        self.add('expert_local_path', path_bytes(count=0), 6, receive=ANCHOR-1)
        _, result = self.read()
        self.assertIsNone(result['local_path'])
        self.assertIn('unsupported_current_expert_local_path', result['reasons'])

    def test_other_coordinate_frame_is_unavailable_without_guess_transform(self):
        self.add('expert_local_path', path_bytes(frame='odom'), 6, receive=ANCHOR-1)
        _, result = self.read()
        self.assertIsNone(result['local_path'])

    def test_duplicate_exact_source_teacher_does_not_pick_arbitrary_value(self):
        second = copy.deepcopy(self.rows[-1])
        second.update(record_id=6, receive_monotonic_ns=ANCHOR-1)
        second['signal']['speed'] = 20.
        self.rows.append(second)
        _, result = self.read()
        self.assertIsNone(result['ego'])
        self.assertIn('ambiguous_same_source_ego_teacher', result['reasons'])

    def test_payload_corruption_is_integrity_failure(self):
        self.seal()
        path = self.episode / 'raw/chunks/chunk.bin'
        blob = bytearray(path.read_bytes())
        blob[self.rows[1]['payload_offset']+5] ^= 1
        path.write_bytes(blob)
        reader = PublicationReader(self.episode)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, 'differs from sealed'):
            reader.for_frame(self.frame)

    def test_mutated_sealed_index_is_integrity_failure(self):
        self.seal()
        self.write('quality/active_signals.json', [])
        with self.assertRaisesRegex(ValueError, 'input changed'):
            PublicationReader(self.episode)

    def test_raw_path_traversal_is_integrity_failure(self):
        self.rows[1]['chunk_path'] = '../outside.bin'
        self.seal()
        reader = PublicationReader(self.episode)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            reader.for_frame(self.frame)

    def test_wrong_native_frame_cannot_bind_publications(self):
        reader, _ = self.read()
        result = reader.for_frame(dict(self.frame, lidar_source_ns=NS+1))
        self.assertEqual(result['status'], 'NO_VERIFIED_NATIVE_FRAME_ANCHOR')

    def test_dynamic_frames_receive_the_same_sealed_current_command_context(self):
        self.frame['zone_id'] = 'dynamic_obstacle'
        self.seal()
        report = attach_expert_publications([self.frame], self.episode)
        context = self.frame['expert_publications']
        self.assertEqual(report['frames'], 1)
        self.assertEqual(report['sensor_payloads_read'], 0)
        self.assertEqual(context['status'], 'RECEIVE_BOUND_CURRENT_CONTEXT')
        self.assertEqual(context['control']['accel'], .2)
        self.assertEqual(context['target_velocity']['value_mps'], 4.)
        self.assertEqual(context['sensor_ns'], NS)


if __name__ == '__main__':
    unittest.main()
