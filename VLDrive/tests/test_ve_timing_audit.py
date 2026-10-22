import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from inference.ve_timing_audit import summarize
from inference.sensors.morai_udp import CameraReceiver, CAMERA_PREFIX, _Receiver


class TimingTests(unittest.TestCase):
    def write(self, root, name, values):
        (root/name).write_text(''.join(json.dumps(v)+'\n' for v in values))

    def test_loop_sequence_gaps_are_not_network_loss_and_real_drops_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = [dict(sequence=seq, sent_ns=stamp, sensor_received_ns=stamp-10,
                            model_inference_ms=50.) for seq,stamp in ((1,1_000_000_000),(5,1_100_000_000))]
            self.write(root, 'adapter.jsonl', packets)
            self.write(root, 'performance.jsonl', [dict(event='sent',datagram_sequence=i) for i in range(1,5)])
            self.write(root, 'transport.jsonl', [dict(identity_matched=True,datagram_sequence=i,status='published') for i in (1,3)])
            report = summarize(root)
            self.assertEqual(report['sender_output_rate']['hz'], 10.)
            self.assertEqual(report['transport_delivery']['unreceived_within_observed_window'], [2])
            self.assertEqual(report['transport_delivery']['unreceived_at_start_or_end'], [4])
            self.assertFalse(report['all_inputs_lossless_verified'])
            self.assertIsNone(report['expert_callback_rate'])

    def test_selected_skips_and_expert_missed_delivery_have_different_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = [dict(sent_ns=i*100_000_000, sensor_received_ns=1,
                audit=dict(selected_samples=dict(lidar=dict(sample_id=sample)), gate_lidar_id=sample),
                delivery_audit=dict(received_ns=i*100_000_000+1, published_ns=i*100_000_000+2, ros_stamp_ns=i))
                for i,sample in ((1,10),(2,12),(3,12))]
            self.write(root, 'adapter.jsonl', packets)
            self.write(root, 'expert_input.jsonl', [dict(received_ns=i*100_000_000+3,ros_stamp_ns=i) for i in (1,3)])
            report = summarize(root)
            self.assertEqual(report['input_selection']['lidar']['completed_samples_not_selected_between_outputs'], 1)
            self.assertEqual(report['input_selection']['lidar']['reused_selected_sample'], 1)
            self.assertEqual(report['expert_delivery']['published_stamps_not_seen_inside_callback_window'], 1)
            self.assertEqual(report['expert_delivery']['unique_stamp_matches'], 2)

    def test_rates_use_elapsed_timestamps_and_duplicate_ros_stamps_are_not_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'performance.jsonl', [dict(event='streams',monotonic_ns=t,
                streams=dict(lidar=dict(samples=count,packets=count*75)))
                for t,count in ((1_000_000_000,10),(4_000_000_000,40))])
            report = summarize(root)
            self.assertEqual(report['sensors']['lidar']['completed_sample_hz'], 10.)
            self.assertIsNone(report['sender_output_rate'])

    def test_receiver_snapshot_keeps_sample_and_counter_together(self):
        receiver = object.__new__(_Receiver)
        receiver._lock = threading.Lock()
        receiver._latest, receiver.samples = b'first', 7
        receiver._stamp, receiver._received_ns = time.time(), 123
        receiver._source_stamp_ns = 456
        receiver._sample_metadata = dict(source_stamp_us_within_hour=789)
        sample, age, identity = receiver.latest_snapshot()
        receiver._latest, receiver.samples = b'second', 8
        receiver._sample_metadata['source_stamp_us_within_hour'] = 999
        self.assertEqual((sample,identity), (b'first',dict(sample_id=7,received_ns=123,
            source_stamp_ns=456,source_stamp_us_within_hour=789)))
        self.assertGreaterEqual(age, 0.)

    def parser(self):
        receiver = object.__new__(CameraReceiver)
        receiver._buffer, receiver._intact, receiver._expect_index = bytearray(),False,0
        receiver.torn_images = 0
        return receiver

    def fragment(self, index, data, end=False, stamp=10):
        return CAMERA_PREFIX.pack(b'MOR',stamp,0,index,len(data))+data+(b'EI' if end else b'AI')

    def test_camera_records_missing_fragment_and_unfinished_frame(self):
        receiver = self.parser()
        receiver._consume(self.fragment(0,b'\xff\xd8'))
        self.assertIsNone(receiver._consume(self.fragment(2,b'\xff\xd9',True)))
        receiver._consume(self.fragment(0,b'first'))
        receiver._consume(self.fragment(0,b'second'))
        self.assertEqual(receiver.audit_counts['discarded_torn_images'], 1)
        self.assertEqual(receiver.audit_counts['fragment_index_discontinuities'], 1)
        self.assertEqual(receiver.audit_counts['unfinished_frames_replaced'], 1)

    def test_camera_rejects_mixed_timestamps_and_recovers_at_next_frame(self):
        receiver = self.parser()
        receiver._consume(self.fragment(0,b'\xff\xd8',stamp=10))
        image = receiver._consume(self.fragment(1,b'\xff\xd9',True,stamp=11))
        self.assertIsNone(image)
        self.assertEqual(receiver.audit_counts['fragment_stamp_mismatches'], 1)
        self.assertEqual(receiver.audit_counts['discarded_torn_images'], 1)
        good = b'\xff\xd8next\xff\xd9'
        self.assertEqual(receiver._consume(self.fragment(0,good,True,stamp=12)),good)

    def test_camera_valid_padded_fragments_preserve_exact_bytes(self):
        receiver = self.parser()
        first = b'\xff\xd8abc'
        wire = CAMERA_PREFIX.pack(b'MOR',10,0,0,len(first))+first+b'\0'*50+b'AI'
        self.assertIsNone(receiver._consume(wire))
        self.assertEqual(receiver._consume(self.fragment(1,b'def\xff\xd9',True)),
                         b'\xff\xd8abcdef\xff\xd9')

    def test_real_multifragment_jpeg_round_trip(self):
        import cv2
        import numpy as np
        from inference.sensors.morai_udp import CAMERA_JPEG_CAPACITY
        rng = np.random.RandomState(9)
        ok, encoded = cv2.imencode('.jpg',rng.randint(0,256,(720,1280,3),dtype=np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        receiver = self.parser()
        image = None
        for index, offset in enumerate(range(0,len(jpeg),CAMERA_JPEG_CAPACITY)):
            part = jpeg[offset:offset+CAMERA_JPEG_CAPACITY]
            image = receiver._consume(self.fragment(index,part,offset+len(part)==len(jpeg)))
        self.assertEqual(image,jpeg)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(image,np.uint8),1),cv2.imdecode(encoded,1))

    def test_camera_invalid_length_or_envelope_is_not_emitted(self):
        receiver = self.parser()
        self.assertIsNone(receiver._consume(CAMERA_PREFIX.pack(b'MOR',10,0,0,500)+b'xEI'))
        self.assertIsNone(receiver._consume(self.fragment(0,b'broken',True)))
        good = b'\xff\xd8good\xff\xd9'
        self.assertEqual(receiver._consume(self.fragment(0,good,True,stamp=11)),good)

    def test_camera_bounded_buffer_and_reordered_fragments(self):
        receiver = self.parser()
        receiver.MAX_IMAGE_BYTES = 6
        receiver._consume(self.fragment(0,b'\xff\xd8abc'))
        self.assertIsNone(receiver._consume(self.fragment(1,b'def\xff\xd9',True)))
        self.assertEqual(receiver.audit_counts['oversize_images'],1)
        self.assertEqual(len(receiver._buffer),0)
        self.assertIsNone(receiver._consume(self.fragment(1,b'\xff\xd9',True,stamp=11)))
        self.assertEqual(receiver._consume(self.fragment(0,b'\xff\xd8\xff\xd9',True,stamp=12)),b'\xff\xd8\xff\xd9')


if __name__ == '__main__':
    unittest.main()
