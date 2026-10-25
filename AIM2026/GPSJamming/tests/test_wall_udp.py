import json
import select
import time
import unittest
from src.UDP import WallUDPSender, WallUDPReceiver, line_edges


class WallTransportTest(unittest.TestCase):
    def setUp(self):
        self.receiver = WallUDPReceiver(port=0, max_age_s=0.05)
        self.sender = WallUDPSender(port=self.receiver.socket.getsockname()[1])
        self.display = dict(frame_id='base_link', frame_sequence=42,
            sensor_timestamp_ns=None, timestamp_status='unmapped',
            valid=dict(left=True, right=False),
            left=[dict(points=[[0.123456789, 3., -0.03], [10., 3.2, -0.03]]),
                  dict(points=[[12., 3.3, -0.03], [20., 3.5, -0.03]])],
            right=[dict(points=[[0., -3., -0.03], [10., -3.2, -0.03]],
                        inferred=True, source='opposite_wall_measured_boundary_width')])

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def receive(self):
        self.assertTrue(select.select([self.receiver.socket], [], [], 1.0)[0],
                        'UDP datagram did not arrive')
        return self.receiver.poll()

    def test_exact_display_coordinates_and_segment_breaks(self):
        self.sender.send(self.display)
        packet = self.receive()
        self.assertEqual(packet['boundaries'], self.display)
        self.assertEqual(line_edges(packet['boundaries'], 'left'),
                         [s['points'] for s in self.display['left']])
        self.assertFalse(packet['boundaries']['valid']['right'])
        self.assertTrue(packet['boundaries']['right'][0]['inferred'])

    def test_expiry_empty_frame_and_out_of_order(self):
        self.sender.send(self.display)
        old = self.receive()
        self.display['left'] = []
        self.sender.send(self.display)
        self.assertEqual(self.receive()['boundaries']['left'], [])
        self.sender.socket.sendto(json.dumps(old).encode(), self.sender.destination)
        self.assertEqual(self.receive()['sequence'], 1)
        self.assertEqual(self.receiver.rejected, 1)
        time.sleep(0.06)
        self.assertIsNone(self.receiver.poll())

    def test_restart_and_retired_session(self):
        self.sender.send(self.display)
        old = self.receive()
        self.sender.session = 'new-session'
        self.sender.sequence = 0
        self.sender.send(self.display)
        self.assertEqual(self.receive()['session'], 'new-session')
        self.sender.socket.sendto(json.dumps(old).encode(), self.sender.destination)
        self.assertEqual(self.receive()['session'], 'new-session')

    def test_invalid_and_oversized_datagrams(self):
        self.sender.socket.sendto(b'{}', self.sender.destination)
        self.assertIsNone(self.receive())
        self.assertEqual(self.receiver.rejected, 1)
        self.display['left'][0]['points'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            self.sender.send(self.display)
        self.display['left'] = [dict(points=[[1., 2., 3.]] * 10000)]
        with self.assertRaises(ValueError):
            self.sender.send(self.display)


if __name__ == '__main__':
    unittest.main()
