import socket
import struct
import unittest
from unittest.mock import patch

from camera_mot.streams import CameraBuffers
from camera_mot.udp_source import UdpCameraSource

CAMERAS = {key: {'frame_id': 'Camera-' + str(i), 'udp_port': 0}
           for i, key in enumerate(('front', 'left', 'right'), 1)}
JPEG = b'\xff\xd8test\xff\xd9'


def packet(payload, stamp, index=0, final=True):
    return b'MOR' + struct.pack('<IIII', *stamp, index, len(payload)) + payload + (b'EI' if final else b'AI')


class BufferTests(unittest.TestCase):
    def test_latest_only_and_round_robin(self):
        source = CameraBuffers(CAMERAS)
        for t in range(20):
            source.put('Camera-1', t, 123456789, JPEG)
        source.put('Camera-2', 7, 1, JPEG)
        source.put('Camera-3', 7, 1, JPEG)
        front = source.get(0)
        self.assertEqual(front.stamp, (19, 123456789))
        source.put('Camera-1', 20, 1, JPEG)
        self.assertEqual(source.get(0).camera_id, 'left')
        self.assertEqual(source.get(0).camera_id, 'right')
        self.assertEqual(source.get(0).camera_id, 'front')
        self.assertIsNone(source.get(0))
        self.assertEqual(source.stats()['front']['overwritten_frames'], 19)

    def test_rear_invalid_duplicate_and_backward_rejected(self):
        source = CameraBuffers(CAMERAS)
        self.assertFalse(source.put('Camera-4', 1, 0, JPEG))
        self.assertFalse(source.put('Camera-1', 1, 1_000_000_000, JPEG))
        self.assertFalse(source.put('Camera-1', 1, 0, b'bad'))
        self.assertTrue(source.put('Camera-1', 10, 1, JPEG))
        self.assertFalse(source.put('Camera-1', 10, 1, JPEG))
        self.assertFalse(source.put('Camera-1', 10, 0, JPEG))
        stats = source.stats()['front']
        self.assertEqual(stats['invalid_frames'], 2)
        self.assertEqual(stats['stale_frames'], 2)
        self.assertEqual(stats['backward_timestamps'], 1)
        self.assertEqual(source.ignored_frames, 1)

    def test_one_camera_reset_does_not_change_other_camera(self):
        source = CameraBuffers(CAMERAS)
        source.put('Camera-1', 100, 0, JPEG)
        source.put('Camera-2', 100, 0, JPEG)
        source.reset('front')
        self.assertTrue(source.put('Camera-1', 1, 0, JPEG))
        front = source.get(0)
        left = source.get(0)
        self.assertEqual((front.camera_id, front.stream_epoch, front.stamp), ('front', 1, (1, 0)))
        self.assertEqual((left.camera_id, left.stream_epoch, left.stamp), ('left', 0, (100, 0)))
        self.assertFalse(source.put('Camera-2', 1, 0, JPEG))

    def test_missing_camera_does_not_block_others_or_repeat_frames(self):
        source = CameraBuffers(CAMERAS)
        for t in range(5):
            source.put('Camera-3', t, 0, JPEG)
            self.assertEqual(source.get(0).camera_id, 'right')
            self.assertIsNone(source.get(0))
        source.close()
        self.assertFalse(source.put('Camera-1', 10, 0, JPEG))


class MultiUdpTests(unittest.TestCase):
    def test_worker_failure_reaches_consumer_instead_of_looking_like_eof(self):
        source = UdpCameraSource(CAMERAS, '127.0.0.1')
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with patch.object(source.assemblers['front'], 'feed', side_effect=OSError('receive failure')):
                sender.sendto(packet(JPEG, (10, 0)), source.addresses['front'])
                source.thread.join(timeout=2)
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(source.closed)
            with self.assertRaisesRegex(RuntimeError, 'UDP receive worker failed'):
                source.get(0)
        finally:
            sender.close()
            source.close()

    def test_same_timestamp_on_three_ports_reordered_fragments_and_reset(self):
        source = UdpCameraSource(CAMERAS, '127.0.0.1')
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for key in CAMERAS:
                data = b'\xff\xd8' + key.encode() + b'\xff\xd9'
                sender.sendto(packet(data[4:], (100, 123456789), 1), source.addresses[key])
                sender.sendto(packet(data[:4], (100, 123456789), 0, False), source.addresses[key])
            found = [source.get(2) for _ in CAMERAS]
            self.assertTrue(all(found))
            self.assertEqual({f.camera_id for f in found}, set(CAMERAS))
            for frame in found:
                self.assertEqual(frame.jpeg, b'\xff\xd8' + frame.camera_id.encode() + b'\xff\xd9')
                self.assertEqual(frame.stamp, (100, 123456789))
                self.assertEqual(frame.transport, 'udp')
            source.reset('front')
            sender.sendto(packet(JPEG, (1, 0)), source.addresses['front'])
            frame = source.get(2)
            self.assertEqual((frame.camera_id, frame.stream_epoch, frame.stamp), ('front', 1, (1, 0)))
            sender.sendto(packet(JPEG, (101, 0)), source.addresses['right'])
            frame = source.get(2)
            self.assertEqual((frame.camera_id, frame.stream_epoch), ('right', 0))
        finally:
            sender.close()
            source.close()
        self.assertFalse(source.thread.is_alive())

    def test_incomplete_front_frame_does_not_stall_right(self):
        source = UdpCameraSource(CAMERAS, '127.0.0.1')
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(packet(JPEG[:4], (10, 0), final=False), source.addresses['front'])
            sender.sendto(packet(JPEG, (10, 0)), source.addresses['right'])
            self.assertEqual(source.get(2).camera_id, 'right')
            self.assertIsNone(source.get(0.01))
        finally:
            sender.close()
            source.close()


if __name__ == '__main__':
    unittest.main()
