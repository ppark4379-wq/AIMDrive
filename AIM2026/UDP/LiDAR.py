import numpy as np
import math
import struct
from .Common import _Receiver, _Packet

LIDAR_PACKET_BYTES = 1206
HOUR_US = 3_600_000_000
FRAME_POLICIES = ('morai_cyclic_frame_timestamp', 'azimuth_wrap')


def resolve_hour_stamp(stamp_us, reference_s, max_delta_s=1.0):
    """MORAI LiDAR microseconds-within-hour -> seconds on a reference sensor clock.

    The reference must be a simulator source time (camera or vehicle status),
    never host receipt time. Adjacent hours handle a sweep across the boundary.
    """
    if not 0 <= stamp_us < HOUR_US or not math.isfinite(reference_s) or reference_s <= 0:
        raise ValueError('Invalid LiDAR hour timestamp or source-clock reference')
    base = math.floor(reference_s / 3600.0) * 3600.0
    candidates = [base + shift * 3600.0 + stamp_us / 1e6 for shift in (-1, 0, 1)]
    result = min(candidates, key=lambda value: abs(value - reference_s))
    if abs(result - reference_s) > max_delta_s:
        raise ValueError('LiDAR and reference source clocks differ by more than max_delta_s')
    return result


class LidarReceiver(_Receiver):
    """Velodyne-style packets -> one point cloud per complete MORAI frame.

    ``morai_cyclic_frame_timestamp`` (default, competition build): every packet
    of a frame shares one microsecond-of-hour stamp at byte 1200; a frame is
    900 blocks (75 packets) at 0.4 deg and may start at any azimuth. The cloud
    is emitted at its last block, and any missing, repeated or reordered block
    discards the frame. ``azimuth_wrap`` emits on the azimuth wrap instead and
    cannot detect lost packets.
    """
    VLP16_ELEVATIONS = [-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0]
    BLOCKS_PER_PACKET = 12
    CHANNELS_PER_BLOCK = 32
    BLOCK_BYTES = 100

    PACKET_DTYPE = np.dtype([
        ("flag", "<u2"),
        ("azimuth", "<u2"),
        ("returns", np.dtype([("distance", "<u2"), ("reflectivity", "u1")]), CHANNELS_PER_BLOCK),
    ])

    def __init__(self, port, elevations=None, distance_scale=0.002,
                 frame_policy='morai_cyclic_frame_timestamp', **kwargs):
        if frame_policy not in FRAME_POLICIES:
            raise ValueError('frame_policy must be one of ' + ', '.join(FRAME_POLICIES))
        self.frame_policy = frame_policy
        table = [math.radians(a) for a in (elevations or self.VLP16_ELEVATIONS)]
        repeats = -(-self.CHANNELS_PER_BLOCK // len(table))
        full = (table * repeats)[: self.CHANNELS_PER_BLOCK]
        self.elevations = table
        self._cos_elevation = np.cos(np.asarray(full, dtype=np.float32))
        self._sin_elevation = np.sin(np.asarray(full, dtype=np.float32))
        self.distance_scale = distance_scale
        self._sweep = []
        self._last_azimuth = None
        self._frame_stamp_us = None
        self._frame_first_azimuth = None
        self._frame_step = None
        self._frame_blocks = 0
        self._frame_packets = 0
        self._frame_chunks = []
        self._frame_valid = False
        self._frame_finished = False
        self._completed_metadata = None
        super().__init__(port, recv_bytes=2048, **kwargs)

    def sample_metadata(self, sample):
        return dict(self._completed_metadata) if self._completed_metadata else None

    def _consume(self, payload):
        if self.frame_policy == 'morai_cyclic_frame_timestamp':
            return self._consume_cyclic_frame(payload)
        return self._consume_azimuth_wrap(payload)

    def _decode_points(self, blocks):
        raw_azimuth = blocks["azimuth"].astype(np.int32)
        distance = blocks["returns"]["distance"].astype(np.float32) * self.distance_scale
        reflectivity = blocks["returns"]["reflectivity"].astype(np.float32)
        azimuth = np.radians(raw_azimuth.astype(np.float32) / 100.0)[:, None]
        horizontal = distance * self._cos_elevation[None, :]
        # Azimuth is clockwise from +y; output is +x forward, +y left.
        points = np.stack([
            horizontal * np.cos(azimuth),
            -horizontal * np.sin(azimuth),
            distance * self._sin_elevation[None, :],
            reflectivity,
        ], axis=-1)
        return points, distance > 0.0

    def _consume_azimuth_wrap(self, payload):
        if len(payload) < self.BLOCKS_PER_PACKET * self.BLOCK_BYTES:
            self._count('short_lidar_packets')
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE,
                               count=self.BLOCKS_PER_PACKET)
        valid = blocks["flag"] == 0xEEFF
        if not valid.all():
            self._count('invalid_lidar_blocks', int((~valid).sum()))
        if not valid.any():
            return None
        blocks = blocks[valid]

        finished = None
        raw_azimuth = blocks["azimuth"].astype(np.int32)
        wrap = np.flatnonzero(np.diff(raw_azimuth) < 0)
        first = int(wrap[0]) + 1 if wrap.size else None
        if self._last_azimuth is not None and raw_azimuth[0] < self._last_azimuth:
            first = 0
        self._last_azimuth = int(raw_azimuth[-1])

        points, keep = self._decode_points(blocks)

        if first is not None:
            head = points[:first][keep[:first]]
            if head.size:
                self._sweep.append(head)
            finished = np.concatenate(self._sweep) if self._sweep else None
            self._sweep = []
            tail = points[first:][keep[first:]]
            if tail.size:
                self._sweep.append(tail)
        else:
            chunk = points[keep]
            if chunk.size:
                self._sweep.append(chunk)
        self._completed_metadata = None
        return finished

    def _start_frame(self, stamp, azimuth):
        if self._frame_stamp_us is not None and not self._frame_finished:
            self._count('incomplete_lidar_frames')
        step = int(round(float(np.median(np.diff(azimuth) % 36000))))
        self._frame_stamp_us = stamp
        self._frame_first_azimuth = int(azimuth[0])
        self._frame_step = step
        self._frame_blocks = 0
        self._frame_packets = 0
        self._frame_chunks = []
        self._frame_finished = False
        self._frame_valid = bool(step > 0 and 36000 % step == 0
                                 and 36000 // step % self.BLOCKS_PER_PACKET == 0)

    def _consume_cyclic_frame(self, payload):
        if len(payload) != LIDAR_PACKET_BYTES:
            self._count('invalid_lidar_packet_size')
            self._frame_valid = False
            return None
        stamp = struct.unpack_from('<I', payload, self.BLOCKS_PER_PACKET * self.BLOCK_BYTES)[0]
        if stamp >= HOUR_US:
            self._count('invalid_lidar_hour_stamp')
            self._frame_valid = False
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE, count=self.BLOCKS_PER_PACKET)
        azimuth = blocks['azimuth'].astype(np.int32)
        if stamp != self._frame_stamp_us:
            if self._frame_stamp_us is not None and (stamp - self._frame_stamp_us) % HOUR_US > HOUR_US // 2:
                # A late packet of an older frame must not open a new one.
                self._count('old_lidar_frame_packets')
                return None
            self._start_frame(stamp, azimuth)
        if self._frame_finished:
            self._count('duplicate_completed_lidar_packets')
            return None
        if not self._frame_valid:
            return None
        expected_blocks = 36000 // self._frame_step
        expected = (self._frame_first_azimuth
                    + (self._frame_blocks + np.arange(self.BLOCKS_PER_PACKET)) * self._frame_step) % 36000
        # MORAI rounds 0.4 deg steps to 0.01 deg, so allow one centidegree.
        circular_error = (azimuth - expected + 18000) % 36000 - 18000
        if (not np.all(blocks['flag'] == 0xEEFF) or np.any(azimuth >= 36000)
                or np.any(np.abs(circular_error) > 1)
                or self._frame_blocks + self.BLOCKS_PER_PACKET > expected_blocks):
            self._count('lidar_frame_discontinuities')
            self._frame_valid = False
            self._frame_chunks = []
            return None
        points, keep = self._decode_points(blocks)
        self._frame_chunks.append(points[keep])
        self._frame_blocks += self.BLOCKS_PER_PACKET
        self._frame_packets += 1
        if self._frame_blocks != expected_blocks:
            return None
        self._frame_finished = True
        cloud = np.concatenate(self._frame_chunks)
        self._frame_chunks = []
        self._completed_metadata = dict(
            source_stamp_us_within_hour=stamp, frame_policy=self.frame_policy,
            frame_packets=self._frame_packets, start_azimuth_centidegrees=self._frame_first_azimuth,
            azimuth_step_centidegrees=self._frame_step)
        self._count('complete_lidar_frames')
        return cloud
