import socket
import struct
import time
import unittest

from morai_udp import CAMERA_ORDER, CameraReceiver, Frame, JpegAssembler, select_frames


def packet(index, payload, end=False, stamp=b""):
    return b"MOR" + stamp + struct.pack("<II", index, len(payload)) + payload + (b"EI" if end else b"AI")


JPEG = b"\xff\xd8test-jpeg\xff\xd9"


class TransportTests(unittest.TestCase):
    def test_both_layouts(self):
        for stamp in (b"", struct.pack("<II", 123, 456)):
            assembler = JpegAssembler()
            self.assertIsNone(assembler.feed(packet(0, JPEG[:5], stamp=stamp), 0))
            frame = assembler.feed(packet(1, JPEG[5:], True, stamp), 0.1)
            self.assertEqual(frame.jpeg, JPEG)
            self.assertEqual(frame.timestamp, stamp)

    def test_fixed_size_single_packet(self):
        # Rear/side cameras: complete JPEG followed by zero padding and EI.
        for stamp in (b"", bytes.fromhex("d1a9b36a809af436")):
            data = packet(0, JPEG, True, stamp)
            data = data[:-2] + bytes(65000 - len(data)) + data[-2:]
            frame = JpegAssembler().feed(data)
            self.assertEqual(frame.jpeg, JPEG)

    def test_fixed_size_front_multiple_packets(self):
        # Front header captured from the user's simulator: 64979 bytes/block.
        stamp = bytes.fromhex("d1a9b36a40227735")
        jpeg = b"\xff\xd8" + b"x" * 140000 + b"\xff\xd9"
        assembler = JpegAssembler()
        chunks = [jpeg[i:i+64979] for i in range(0, len(jpeg), 64979)]
        for index, payload in enumerate(chunks):
            data = packet(index, payload, index == len(chunks)-1, stamp)
            data = data[:-2] + bytes(65000-len(data)) + data[-2:]
            frame = assembler.feed(data, index * .01)
            if index < len(chunks)-1:
                self.assertIsNone(frame)
        self.assertEqual(frame.jpeg, jpeg)

    def test_unused_buffer_bytes_and_invalid_size(self):
        data = packet(0, JPEG, True, b"12345678")
        padded = data[:-2] + bytes(65000-len(data)) + data[-2:]
        bad_padding = bytearray(padded)
        bad_padding[-3] = 1
        self.assertEqual(JpegAssembler().feed(bytes(bad_padding)).jpeg, JPEG)
        bad_size = bytearray(padded)
        struct.pack_into("<I", bad_size, 15, 65000)
        self.assertIsNone(JpegAssembler().feed(bytes(bad_size)))
        self.assertIsNone(JpegAssembler().feed(data[:-2] + bytes(20) + data[-2:]))

    def test_observed_front_two_block_layout(self):
        # Sizes and timestamp from the user's actual MORAI log.
        stamp = bytes.fromhex("02adb36ac05be80a")
        jpeg = b"\xff\xd8" + b"x" * (64979 + 53691 - 4) + b"\xff\xd9"
        assembler = JpegAssembler()
        first = packet(0, jpeg[:64979], stamp=stamp)
        self.assertEqual(len(first), 65000)
        self.assertIsNone(assembler.feed(first, 0))
        last = packet(1, jpeg[64979:], True, stamp)
        # Include stale marker-like bytes to verify size slicing, not EOI search.
        unused = (b"\xff\xd9\xff\xd8stale-data" * 1000)[:65000-len(last)]
        last = last[:-2] + unused + last[-2:]
        self.assertEqual(len(last), 65000)
        frame = assembler.feed(last, .01)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.jpeg, jpeg)

    def test_single_packet(self):
        self.assertEqual(JpegAssembler().feed(packet(0, JPEG, True)).jpeg, JPEG)

    def test_dropped_packet_and_recovery(self):
        assembler = JpegAssembler()
        assembler.feed(packet(0, JPEG[:3]), 0)
        self.assertIsNone(assembler.feed(packet(2, JPEG[3:], True), 0.1))
        self.assertEqual(assembler.feed(packet(0, JPEG, True), 0.2).jpeg, JPEG)

    def test_duplicate_and_reordering(self):
        for index in (1, 3):
            assembler = JpegAssembler()
            assembler.feed(packet(0, JPEG[:3]), 0)
            assembler.feed(packet(1, JPEG[3:5]), 0.1)
            self.assertIsNone(assembler.feed(packet(index, JPEG[5:], True), 0.2))

    def test_different_frame_timestamp(self):
        assembler = JpegAssembler()
        assembler.feed(packet(0, JPEG[:3], stamp=b"12345678"), 0)
        self.assertIsNone(assembler.feed(packet(1, JPEG[3:], True, b"abcdefgh"), 0.1))

    def test_timeout_and_size_limit(self):
        assembler = JpegAssembler(timeout=0.1)
        assembler.feed(packet(0, JPEG[:3]), 0)
        self.assertIsNone(assembler.feed(packet(1, JPEG[3:], True), 0.2))
        self.assertIsNone(JpegAssembler(max_bytes=4).feed(packet(0, JPEG, True)))

    def test_malformed(self):
        valid = packet(0, JPEG, True)
        for bad in (b"", b"MOR", valid[:-1], b"BAD" + valid[3:],
                    valid[:-2] + b"XX", packet(0, b"not-jpeg", True),
                    packet(0, b"\xff\xd8no-end", True)):
            self.assertIsNone(JpegAssembler().feed(bad))

    def test_protocol_selection(self):
        self.assertIsNone(JpegAssembler("legacy").feed(packet(0, JPEG, True, b"12345678")))
        self.assertIsNone(JpegAssembler("timestamped").feed(packet(0, JPEG, True)))

    def test_pairing_and_stale_rejection(self):
        queues = {name: [Frame(JPEG, 10 + i * .01, b"", 1)] for i, name in enumerate(CAMERA_ORDER)}
        self.assertIsNotNone(select_frames(queues, {}, .1, .5, 10.1))
        self.assertIsNone(select_frames(queues, {}, .01, .5, 10.1))
        self.assertIsNone(select_frames(queues, {}, .1, .5, 11))
        self.assertIsNone(select_frames(queues, {"front": 1}, .1, .5, 10.1))

    def test_four_udp_sockets(self):
        receivers = {}
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for name in CAMERA_ORDER:
                receivers[name] = CameraReceiver("127.0.0.1", 0)
            for receiver in receivers.values():
                address = receiver.socket.getsockname()
                sender.sendto(packet(0, JPEG[:4]), address)
                sender.sendto(packet(1, JPEG[4:], True), address)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                queues = {name: receiver.snapshot() for name, receiver in receivers.items()}
                batch = select_frames(queues, {}, .5, 2)
                if batch:
                    break
                time.sleep(.01)
            self.assertIsNotNone(batch)
            self.assertTrue(all(frame.jpeg == JPEG for frame in batch.values()))
        finally:
            sender.close()
            for receiver in receivers.values():
                receiver.close()


if __name__ == "__main__":
    unittest.main()
