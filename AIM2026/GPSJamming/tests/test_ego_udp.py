"""MORAI Ego packet layout, unit conversion and receiver lifecycle."""
from pathlib import Path
import sys
import socket
import struct
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.interfaces.ego_udp import decode_ego_packet, EgoUDPReceiver


def packet(extended=True, sec=10):
    # Full sequential wire layout, independent of decoder offsets.
    data = struct.pack('<iiBBfi', sec, 25, 1, 4, 36., 42)
    data += struct.pack('<8f', .1, .2, 4, 2, 1.5, .8, 2.7, .9)
    data += struct.pack('<3f', 100, 200, 3)  # position
    data += struct.pack('<3f', 1, 2, 90)     # rotation degrees
    data += struct.pack('<3f', 36, -18, 0)   # velocity km/h
    data += struct.pack('<3f', 0, 0, 180)    # angular velocity deg/s
    data += struct.pack('<3f', 1, 2, 3)     # acceleration
    data += struct.pack('<f38s', .5, b'link')
    if extended:
        data += struct.pack('<12f', *range(12))
    return b'#MoraiInfo$' + struct.pack('<i3i', len(data), 0, 0, 0) + data + b'\r\n'


class EgoTests(unittest.TestCase):
    def test_both_layouts_and_units(self):
        for extended, size in [(False, 181), (True, 229)]:
            raw = packet(extended)
            self.assertEqual(len(raw), size)
            state = decode_ego_packet(raw, 123)
            self.assertEqual(state.position_m, (100, 200, 3))
            self.assertEqual(state.rotation_rpy_deg, (1, 2, 90))
            self.assertEqual(state.velocity_mps, (10, -5, 0))
            self.assertEqual(state.signed_speed_mps, 10)
            self.assertAlmostEqual(state.angular_velocity_radps[2], 3.141592653589793)
            self.assertEqual(state.timestamp_ns, 10_000_000_025)
            self.assertEqual(state.receive_monotonic_ns, 123)
            self.assertEqual(state.map_id, 42)

    def test_malformed(self):
        valid = packet()
        invalid = [valid[:-1], b'x' + valid[1:], valid[:-2] + b'xx']
        for offset, fmt, value in [(11, '<i', 1), (31, '<i', 10**9), (77, '<f', float('nan'))]:
            bad = bytearray(valid)
            struct.pack_into(fmt, bad, offset, value)
            invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError):
                decode_ego_packet(bad, 0)

    def test_loopback_filter_reset_and_close(self):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def wait_for(predicate):
            deadline = time.monotonic() + 2
            while not predicate() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(predicate())
        try:
            with EgoUDPReceiver(port=0, history_size=2) as receiver:
                self.assertEqual(receiver.snapshot()['status'], 'missing')
                sender.sendto(b'bad', receiver.address)
                sender.sendto(packet(), receiver.address)
                wait_for(lambda: receiver.received_packets == 1)
                self.assertEqual(receiver.snapshot()['status'], 'fresh')
                with patch('gps_jamming.interfaces.ego_udp.time.monotonic_ns',
                           return_value=receiver.history[-1].receive_monotonic_ns + 1):
                    self.assertEqual(receiver.snapshot(max_age_s=0)['status'], 'stale')
                other.sendto(packet(sec=11), receiver.address)
                sender.sendto(packet(sec=1), receiver.address)
                wait_for(lambda: receiver.clock_epoch == 1 and receiver.filtered_packets == 1)
                self.assertEqual(len(receiver.history), 1)
                self.assertEqual(receiver.snapshot()['state']['clock_epoch'], 1)
                self.assertEqual(receiver.invalid_packets, 1)
                snapshot = receiver.snapshot()
                self.assertEqual(snapshot['datagrams'], 4)
                self.assertEqual(snapshot['last_packet_size'], 229)
                self.assertEqual(snapshot['listen'], receiver.address)
            self.assertFalse(receiver.thread.is_alive())
        finally:
            sender.close()
            other.close()


if __name__ == '__main__':
    unittest.main()
