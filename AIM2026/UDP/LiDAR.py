import numpy as np
import math
import struct
from .Common import _Receiver, _Packet

class LidarReceiver(_Receiver):
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

    def __init__(self, port, elevations=None, distance_scale=0.002, **kwargs):
        table = [math.radians(a) for a in (elevations or self.VLP16_ELEVATIONS)]
        repeats = -(-self.CHANNELS_PER_BLOCK // len(table))
        full = (table * repeats)[: self.CHANNELS_PER_BLOCK]
        self.elevations = table
        self._cos_elevation = np.cos(np.asarray(full, dtype=np.float32))
        self._sin_elevation = np.sin(np.asarray(full, dtype=np.float32))
        self.distance_scale = distance_scale
        self._sweep = []
        self._last_azimuth = None
        super().__init__(port, recv_bytes=2048, **kwargs)

    def _consume(self, payload):
        if len(payload) < self.BLOCKS_PER_PACKET * self.BLOCK_BYTES:
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE,
                               count=self.BLOCKS_PER_PACKET)
        valid = blocks["flag"] == 0xEEFF
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

        distance = blocks["returns"]["distance"].astype(np.float32) * self.distance_scale
        reflectivity = blocks["returns"]["reflectivity"].astype(np.float32)
        azimuth = np.radians(raw_azimuth.astype(np.float32) / 100.0)[:, None]
        horizontal = distance * self._cos_elevation[None, :]
        points = np.stack([
            horizontal * np.cos(azimuth),
            -horizontal * np.sin(azimuth),
            distance * self._sin_elevation[None, :],
            reflectivity,
        ], axis=-1)
        keep = distance > 0.0

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
        return finished
