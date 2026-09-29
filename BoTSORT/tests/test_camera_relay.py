import unittest

from tools.morai_udp import JpegAssembler
from tools.morai_udp_relay import packetize_jpeg


class RelayTests(unittest.TestCase):
    def test_large_jpeg_and_timestamp_survive_repacketizing_reorder_and_duplicates(self):
        jpeg = b'\xff\xd8' + bytes(range(256)) * 500 + b'\xff\xd9'
        stamp = (123456, 987654321)
        packets = list(packetize_jpeg(jpeg, stamp))
        self.assertTrue(all(len(p) <= 1200 for p in packets))
        self.assertGreater(len(packets), 100)
        assembler = JpegAssembler()
        found = []
        for packet in reversed(packets):
            for _ in range(2):
                frame = assembler.feed(packet, now=1)
                if frame is not None:
                    found.append(frame)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].stamp, stamp)
        self.assertEqual(found[0].jpeg, jpeg)
        self.assertEqual(assembler.invalid_packets, 0)

    def test_missing_fragment_never_emits_partial_jpeg(self):
        packets = list(packetize_jpeg(b'\xff\xd8' + b'x' * 5000 + b'\xff\xd9', (1, 2)))
        assembler = JpegAssembler()
        for packet in packets[1:]:
            self.assertIsNone(assembler.feed(packet, now=1))
        assembler.expire(2)
        self.assertEqual(assembler.incomplete_frames, 1)

    def test_reject_bad_size_timestamp_and_jpeg(self):
        for jpeg, stamp, size in [(b'bad', (1, 0), 1200),
                                  (b'\xff\xd8\xff\xd9', (1, 10**9), 1200),
                                  (b'\xff\xd8\xff\xd9', (1, 0), 65000)]:
            with self.assertRaises(ValueError):
                list(packetize_jpeg(jpeg, stamp, size))


if __name__ == '__main__':
    unittest.main()
