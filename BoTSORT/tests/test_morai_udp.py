import socket
import struct
import unittest

from tools.morai_udp import JpegAssembler, UdpCamera
from tools.morai_udp_mot import track_records


def packet(data, index=0, final=True, stamp=(10, 50000000)):
    return b"MOR" + struct.pack("<IIII", *stamp, index, len(data)) + data + (b"EI" if final else b"AI")


class AssemblerTests(unittest.TestCase):
    def test_explicit_sender_port_filters_before_assembly(self):
        front = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        front.bind(('127.0.0.1', 0))
        receiver = UdpCamera('127.0.0.1', 0, sender_port=front.getsockname()[1])
        try:
            other.sendto(packet(b'\xff\xd8wrong\xff\xd9'), receiver.address)
            # Same timestamp from correct peer must still be accepted.
            front.sendto(packet(b'\xff\xd8front\xff\xd9'), receiver.address)
            result = receiver.get(timeout=2)
            self.assertIsNotNone(result)
            self.assertEqual(result.jpeg, b'\xff\xd8front\xff\xd9')
            self.assertIsNone(receiver.get(timeout=.05))
            self.assertEqual(receiver.filtered_packets, 1)
        finally:
            receiver.close()
            front.close()
            other.close()

    def test_front_filter_isolates_peers_and_preserves_latest(self):
        import cv2
        import numpy as np
        import time
        def jpeg(width, height):
            return cv2.imencode('.jpg', np.zeros((height, width, 3), np.uint8))[1].tobytes()
        receiver = UdpCamera('127.0.0.1', 0, expected_image_size=(32, 24))
        front = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        side = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Same acquisition timestamp across cameras must not suppress front.
            side.sendto(packet(jpeg(16, 12), stamp=(10, 0)), receiver.address)
            front.sendto(packet(jpeg(32, 24), stamp=(10, 0)), receiver.address)
            result = receiver.get(timeout=2)
            self.assertIsNotNone(result)
            self.assertEqual(result.stamp, (10, 0))
            self.assertEqual(receiver.selected_peer[1], front.getsockname()[1])
            # A different peer with even the SAME resolution cannot replace it.
            side.sendto(packet(jpeg(32, 24), stamp=(11, 0)), receiver.address)
            front.sendto(packet(jpeg(16, 12), stamp=(12, 0)), receiver.address)
            front.sendto(packet(jpeg(32, 24), stamp=(11, 0)), receiver.address)
            result = receiver.get(timeout=2)
            self.assertEqual(result.stamp, (11, 0))
            self.assertEqual(receiver.filtered_frames, 2)
            self.assertEqual(receiver.filtered_packets, 1)
            self.assertIsNone(receiver.get(timeout=.02))
        finally:
            front.close()
            side.close()
            receiver.close()

    def test_reordered_fragments_and_duplicate(self):
        a = JpegAssembler()
        end = packet(b"end\xff\xd9", 2)
        self.assertIsNone(a.feed(end, 0))
        self.assertIsNone(a.feed(end, 0.01))
        self.assertIsNone(a.feed(packet(b"\xff\xd8start", 0, False), 0.02))
        frame = a.feed(packet(b"middle", 1, False), 0.03)
        self.assertEqual(frame.jpeg, b"\xff\xd8startmiddleend\xff\xd9")
        self.assertEqual(frame.stamp, (10, 50000000))
        self.assertIsNone(a.feed(end, 0.04))

    def test_missing_fragment_does_not_mix_next_frame(self):
        a = JpegAssembler()
        a.feed(packet(b"\xff\xd8old", 0, False), 0)
        a.feed(packet(b"end\xff\xd9", 2), 0.01)
        frame = a.feed(packet(b"\xff\xd8new\xff\xd9", stamp=(11, 0)), 0.02)
        self.assertEqual(frame.jpeg, b"\xff\xd8new\xff\xd9")
        a.expire(0.4)
        self.assertEqual(a.incomplete_frames, 1)
        self.assertFalse(a.pending)

    def test_bad_size_header_tail_and_timestamp(self):
        a = JpegAssembler()
        valid = packet(b"\xff\xd8x\xff\xd9")
        invalid = [b"", b"BAD" + valid[3:], valid[:-2] + b"??",
                   valid[:15] + struct.pack("<I", 99) + valid[19:],
                   packet(b"x", stamp=(1, 1000000000))]
        for p in invalid:
            self.assertIsNone(a.feed(p))
        self.assertEqual(a.invalid_packets, len(invalid))

    def test_conflicting_duplicate_and_memory_limit(self):
        a = JpegAssembler(max_bytes=8)
        a.feed(packet(b"\xff\xd8abc", 0, False), 0)
        self.assertIsNone(a.feed(packet(b"different", 0, False), 0.01))
        self.assertFalse(a.pending)
        self.assertIsNone(a.feed(packet(b"\xff\xd8" + b"x" * 8 + b"\xff\xd9", stamp=(12, 0)), 0.02))
        self.assertFalse(a.pending)

    def test_padded_final_datagram(self):
        a = JpegAssembler()
        a.feed(packet(b"\xff\xd8start", 0, False), 0)
        final = packet(b"end\xff\xd9", 1)
        padded = final[:-2] + b"\x00" * (65000 - len(final)) + final[-2:]
        frame = a.feed(padded, 0.01)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.jpeg, b"\xff\xd8startend\xff\xd9")

    def test_one_based_index(self):
        a = JpegAssembler(index_base=1)
        self.assertIsNone(a.feed(packet(b"\xff\xd8x", 1, False), 0))
        self.assertEqual(a.feed(packet(b"y\xff\xd9", 2), 0.01).jpeg, b"\xff\xd8xy\xff\xd9")

    def test_udp_loopback_real_jpeg(self):
        import cv2
        import numpy as np
        image = np.full((24, 32, 3), 120, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        receiver = UdpCamera("127.0.0.1", 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            chunks = [jpeg[i:i + 200] for i in range(0, len(jpeg), 200)]
            for i in reversed(range(len(chunks))):
                sender.sendto(packet(chunks[i], i, i == len(chunks) - 1), receiver.address)
            result = receiver.get(timeout=2)
            self.assertIsNotNone(result)
            decoded = cv2.imdecode(np.frombuffer(result.jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, image.shape)
            self.assertLess(np.abs(decoded.astype(float) - image).max(), 3)
            self.assertIsNone(receiver.get(timeout=0.01))  # no repeat of a consumed frame
        finally:
            sender.close()
            receiver.close()
        self.assertFalse(receiver.thread.is_alive())

    def test_latest_frame_and_late_old_frame(self):
        receiver = UdpCamera("127.0.0.1", 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for second in (20, 19, 21):
                sender.sendto(packet(b"\xff\xd8x\xff\xd9", stamp=(second, 0)), receiver.address)
            with receiver.condition:
                self.assertTrue(receiver.condition.wait_for(
                    lambda: receiver.received_frames == 2, timeout=2))
            frame = receiver.get(timeout=1)
            self.assertEqual(frame.stamp, (21, 0))
            self.assertEqual(receiver.stale_frames, 1)
            self.assertEqual(receiver.overwritten_frames, 1)
        finally:
            sender.close()
            receiver.close()

    def test_track_output_coordinates_and_id(self):
        from types import SimpleNamespace
        tracks = [SimpleNamespace(tlbr=(-5, 10, 110, 90), track_id=7, cls=2,
                                  score=0.9, is_activated=True)]
        result = track_records(tracks, 100, 80, 10)
        self.assertEqual(result[0]["track_id"], 7)
        self.assertEqual(result[0]["bbox_xyxy"], [0, 10, 100, 80])
        self.assertEqual(track_records([], 100, 80, 10), [])


if __name__ == "__main__":
    unittest.main()
