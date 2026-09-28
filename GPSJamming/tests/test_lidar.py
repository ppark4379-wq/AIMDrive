"""Deterministic contracts, malformed/lost/reordered UDP, and scene tests."""
import json
from pathlib import Path
import socket
import struct
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.interfaces.lidar_udp import decode_packet, LidarAssembler, LidarUDPReceiver, LidarUDPStream
from gps_jamming.timing.timestamps import TimestampMapper
from GPSJamming.gps_jamming.lidar.ROI import preprocess
from gps_jamming.lidar.ground_removal import remove_ground
from gps_jamming.lidar.runtime import load_config, DEFAULT_CONFIG, frame_to_dict
from gps_jamming.pipeline import LidarPipeline
from gps_jamming.types import LidarFrame


def packet(stamp, start, step=3.0, distance=10.0):
    data = b"".join(struct.pack("<HH", 0xEEFF, round(((start + i * step) % 360) * 100)) +
                    struct.pack("<HB", round(distance / 0.002), 200) * 32 for i in range(12))
    return data + struct.pack("<IBB", stamp, 0x37, 0x22)


def scene():
    rng = np.random.default_rng(2)
    x, y = np.meshgrid(np.linspace(-10, 25, 50), np.linspace(-12, 12, 40))
    ground = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.48)))
    car = rng.uniform([8, -1, -1.1], [11, 1, 0.1], (700, 3))
    pedestrian = rng.uniform([15, 4, -1.1], [15.5, 4.5, 0.2], (200, 3))
    xyz = np.vstack((ground, car, pedestrian))
    return np.column_stack((xyz, np.full(len(xyz), 200))).astype(np.float32)


class LidarTests(unittest.TestCase):
    def assembler(self, step=3.0, **kwargs):
        return LidarAssembler(TimestampMapper(ns_per_tick=1000, source_anchor=0,
                              unix_anchor_ns=1_800_000_000_000_000_000),
                              azimuth_step_deg=step, **kwargs)

    def feed_sweep(self, assembler, stamp=100, order=None, step=3.0, now=0):
        total = round(360 / (12 * step))
        for i, index in enumerate(range(total) if order is None else order):
            assembler.feed(decode_packet(packet(stamp, index * 12 * step, step)), now + i)

    def test_decode_geometry_and_original_timestamp(self):
        p = decode_packet(packet(123456, 0))
        self.assertEqual(p.raw_timestamp, 123456)
        np.testing.assert_allclose(np.linalg.norm(p.points[:, :, :3], axis=-1), 10, atol=1e-5)
        self.assertTrue(np.all(p.points[:, :, 3] == 200))
        self.assertLess(p.points[1, 0, 1], 0)
        self.assertAlmostEqual(float(p.points[0, 0, 2]), 10 * np.sin(np.deg2rad(-15)), places=5)

    def test_reject_bad_packets(self):
        valid = packet(100, 0)
        for payload in (valid[:-1], b"\0" * 1206, valid[:1204] + b"\x39\x22"):
            with self.assertRaises(ValueError):
                decode_packet(payload)

    def test_reorder_duplicates_complete_without_fixed_packet_count(self):
        a = self.assembler()
        self.feed_sweep(a, order=[4, 1, 0, 9, 3, 2, 5, 7, 6, 8])
        a.feed(decode_packet(packet(100, 144)), 12)
        frame, = a.expire(150_000_000)
        self.assertTrue(frame.complete, frame.issues)
        self.assertEqual(frame.packet_count, 10)
        self.assertEqual(frame.duplicate_count, 1)
        self.assertEqual(len(frame.points), 3840)
        self.assertEqual(frame.raw_timestamp, 100)
        self.assertEqual(frame.timestamp, 1_800_000_000_000_100_000)
        self.assertEqual(frame.receive_monotonic_time, 0)
        self.assertEqual(a.feed(decode_packet(packet(100, 0)), 150_000_001), [])
        self.assertEqual(a.late_packets, 1)
        b = self.assembler(step=2.0)
        self.feed_sweep(b, step=2.0)
        other, = b.expire(150_000_000)
        self.assertTrue(other.complete)
        self.assertEqual(other.packet_count, 15)

    def test_missing_packet_and_unverified_step_are_incomplete(self):
        a = self.assembler()
        self.feed_sweep(a, order=[0, 1, 2, 4, 5, 6, 7, 8, 9])
        frame, = a.expire(150_000_000)
        self.assertFalse(frame.complete)
        self.assertIn("missing_or_irregular_azimuths", frame.issues)
        b = self.assembler(step=None)
        self.feed_sweep(b)
        self.assertIn("azimuth_step_unverified", b.expire(150_000_000)[0].issues)

    def test_distinct_timestamps_never_mix(self):
        a = self.assembler()
        for index in range(10):
            a.feed(decode_packet(packet(100, index * 36, distance=10)), index * 2)
            a.feed(decode_packet(packet(200, index * 36, distance=20)), index * 2 + 1)
        frames = a.expire(150_000_100)
        self.assertEqual([f.raw_timestamp for f in frames], [100, 200])
        for frame, distance in zip(frames, [10, 20]):
            self.assertTrue(frame.complete)
            np.testing.assert_allclose(np.linalg.norm(frame.points[:, :3], axis=1), distance, atol=1e-5)

    def test_conflicts_and_shutdown_are_visible(self):
        a = self.assembler()
        self.feed_sweep(a)
        a.feed(decode_packet(packet(100, 0, distance=12)), 20)
        frame, = a.flush()
        self.assertFalse(frame.complete)
        self.assertIn("conflicting_azimuth_block", frame.issues)
        self.assertEqual(frame.diagnostics["close_reason"], "stream_ended")

    def test_full_grid_can_finish_at_end_of_stream(self):
        a = self.assembler()
        self.feed_sweep(a)
        self.assertTrue(a.flush()[0].complete)

    def test_partial_start_and_packet_limit(self):
        a = self.assembler()
        self.feed_sweep(a, order=[5, 6, 7, 8, 9])
        self.assertFalse(a.flush()[0].complete)
        b = self.assembler(max_packets_per_frame=1)
        b.feed(decode_packet(packet(100, 0)), 0)
        frame, = b.feed(decode_packet(packet(100, 36)), 1)
        self.assertIn("packet_limit", frame.issues)

    def test_sloped_ground_keeps_elevated_objects(self):
        points = scene()
        points[:, 2] += 0.02 * points[:, 0]
        remaining, info = remove_ground(points)
        self.assertEqual(info["ground_points"], 2000)
        self.assertEqual(len(remaining), 900)

    def test_buffer_bound_and_empty_returns(self):
        a = self.assembler(max_pending_frames=1)
        a.feed(decode_packet(packet(1, 0, distance=0)), 0)
        frame, = a.feed(decode_packet(packet(2, 0)), 1)
        self.assertEqual(frame.points.shape, (0, 4))
        self.assertIn("buffer_limit", frame.issues)

    def test_clock_anchor_missing_jump_and_reset(self):
        self.assertEqual(TimestampMapper().map(100), (None, "unmapped", 0))
        mapper = TimestampMapper(1000, 0, 1_800_000_000_000_000_000, 10_000)
        mapper.map(100)
        self.assertEqual(mapper.map(90), (None, "clock_jump", 1))
        self.assertEqual(mapper.map(91), (None, "unmapped", 1))
        self.assertEqual(mapper.map(1000), (None, "clock_jump", 2))

    def test_roi_voxel_preserves_intensity_and_inputs(self):
        raw = np.array([[1, 1, 1, 200], [1.01, 1, 1, 100],
                        [99, 0, 0, 42], [np.nan, 0, 0, 5]], dtype=np.float32)
        result = preprocess(raw, [-2, -2, -2], [2, 2, 2], 0.2)
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result[0, 3], 150)
        self.assertEqual(raw[0, 3], 200)

    def test_scene_ground_candidates_features_metadata(self):
        config = load_config(DEFAULT_CONFIG)
        frame = LidarFrame(7, 123, 456, 789, "lidar", True, scene(), 900)
        result = LidarPipeline(config["lidar"]).process(frame)
        self.assertEqual(len(result.clusters), 2)
        self.assertGreater(result.diagnostics["ground_points"], 1900)
        self.assertEqual(result.raw_timestamp, 123)
        self.assertEqual(result.timestamp, 456)
        self.assertEqual(result.receive_monotonic_time, 789)
        self.assertEqual(result.coordinate_frame, "lidar")
        self.assertEqual(frame.clusters, [])
        centers = sorted([c.centroid for c in result.clusters], key=lambda c: c[0])
        np.testing.assert_allclose(centers[0], [9.5, 0, -0.5], atol=0.15)
        np.testing.assert_allclose(centers[1], [15.25, 4.25, -0.45], atol=0.15)
        for c in result.clusters:
            self.assertEqual(c.point_count, len(c.points))
            np.testing.assert_allclose(c.dimensions, np.ptp(c.points[:, :3], axis=0))
        json.dumps(frame_to_dict(result), allow_nan=False)

    def test_ground_missing_does_not_delete_objects(self):
        points = scene()[-200:]
        remaining, info = remove_ground(points)
        np.testing.assert_array_equal(remaining, points)
        self.assertEqual(info["ground_points"], 0)

    def test_loopback_udp_receiver(self):
        a = self.assembler()
        with LidarUDPReceiver(a, port=0, socket_timeout_ms=10) as receiver:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                address = receiver.socket.getsockname()
                sender.sendto(b"bad", address)
                receiver.poll()
                self.assertEqual(receiver.invalid_packets, 1)
                for i in range(10):
                    sender.sendto(packet(100, i * 36), address)
                    self.assertEqual(receiver.poll(), [])
            frame, = a.expire(a.groups[100].first + 150_000_000)
            self.assertTrue(frame.complete)
            self.assertEqual(frame.packet_count, 10)

    def test_other_sender_cannot_mix_into_frame(self):
        a = self.assembler()
        with LidarUDPReceiver(a, port=0) as receiver:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as first, \
                    socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as second:
                address = receiver.socket.getsockname()
                first.sendto(packet(100, 0), address)
                receiver.poll()
                second.sendto(packet(100, 36), address)
                receiver.poll()
                self.assertEqual(receiver.filtered_packets, 1)
                self.assertEqual(len(a.groups[100].packets), 1)

    def test_processing_queue_is_bounded_and_drops_whole_frames(self):
        class Receiver:
            calls = 0
            def poll(inner):
                inner.calls += 1
                if inner.calls == 1:
                    return list(range(8))
                stream.stop_event.set()
                return []
        stream = LidarUDPStream(Receiver(), queue_size=2)
        with stream:
            stream.thread.join(timeout=1)
            self.assertFalse(stream.thread.is_alive())
        self.assertEqual(stream.dropped_frames, 6)
        self.assertEqual(stream.drain(), [6, 7])

    def test_receiver_thread_errors_surface(self):
        class Receiver:
            def poll(inner):
                raise RuntimeError("synthetic receiver failure")
        with LidarUDPStream(Receiver()) as stream:
            stream.thread.join(timeout=1)
            with self.assertRaisesRegex(RuntimeError, "synthetic receiver failure"):
                stream.read()


if __name__ == "__main__":
    unittest.main()
