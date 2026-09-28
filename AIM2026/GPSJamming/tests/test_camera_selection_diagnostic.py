"""Receiver metadata distinguishes frame loss from unconsumed frames."""
from pathlib import Path
import sys
import socket
import struct
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'BoTSORT'))
from tools.morai_udp import UdpCamera, CameraFrame


class CameraDiagnosticTests(unittest.TestCase):
    def test_buffer_selects_nearest_and_keeps_future_without_rewind(self):
        receiver = UdpCamera('127.0.0.1', 0, frame_buffer_size=8)
        try:
            now = time.monotonic()
            with receiver.condition:
                receiver.frame_buffer.extend(CameraFrame(b'', (10, n), now, i)
                    for i, n in enumerate((0, 50_000_000, 100_000_000), 1))
            frame = receiver.get_nearest(10_051_000_000)
            self.assertEqual(frame.sequence, 2)
            self.assertEqual([f.sequence for f in receiver.frame_buffer], [3])
            self.assertIsNone(receiver.get_nearest(10_000_000_000, after_stamp=frame.stamp))
            future = receiver.get_nearest(10_101_000_000, after_stamp=frame.stamp)
            self.assertEqual(future.sequence, 3)
            with receiver.condition:
                receiver.frame_buffer.append(CameraFrame(b'', (10, 150_000_000), now, 4))
            with patch('tools.morai_udp.time.monotonic', return_value=now+1):
                self.assertIsNone(receiver.get_nearest(10_150_000_000))
            self.assertEqual(len(receiver.frame_buffer), 0)
        finally:
            receiver.close()

    def test_overwritten_pending_processing_and_processed(self):
        receiver = UdpCamera('127.0.0.1', 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def send(nsec):
            jpeg = b'\xff\xd8sample\xff\xd9'
            sender.sendto(b'MOR' + struct.pack('<IIII', 10, nsec, 0, len(jpeg))
                          + jpeg + b'EI', receiver.address)
        def wait(count):
            deadline = time.monotonic()+2
            while receiver.received_frames < count and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(receiver.received_frames, count)
        def nearest(stamp, eligible=()):
            return receiver.diagnose_selection(stamp, None, set(eligible), 10**9,
                                               50_000_000)['nearest_received']
        try:
            send(0)
            wait(1)
            send(50_000_000)
            wait(2)
            lost = nearest(10_000_000_000)
            self.assertEqual(lost['state'], 'overwritten')
            self.assertTrue(lost['within_tolerance'])
            self.assertFalse(lost['in_sync_buffer'])
            self.assertEqual(nearest(10_050_000_000)['state'], 'pending')
            frame = receiver.get()
            self.assertEqual(nearest(10_050_000_000)['state'], 'processing')
            receiver.mark_processed(frame.sequence)
            ready = nearest(10_050_000_000, [frame.sequence])
            self.assertEqual(ready['state'], 'processed')
            self.assertTrue(ready['in_sync_buffer'])
            self.assertIsNone(receiver.diagnose_selection(None, None, set(), 10**9,
                                                        50_000_000)['nearest_received'])
        finally:
            receiver.close()
            sender.close()
