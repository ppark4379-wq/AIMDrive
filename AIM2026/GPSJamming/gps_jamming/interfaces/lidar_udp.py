"""MORAI grouped-timestamp Velodyne-style UDP frames.

Vectorized XYZI decoding is adapted from VLDrive/inference/morai_udp.py.
This decoder supports 1206-byte, single-return VLP-16-style packets only.
The uint32 trailer at offset 1200 is preserved without guessing its time unit.
This assembler requires one source timestamp per complete measurement group;
hardware per-packet timestamps require a different assembly policy.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import socket
import struct
import time
import threading
import queue
import numpy as np
from ..types import LidarFrame

VLP16_ELEVATIONS = [-15, 1, -13, 3, -11, 5, -9, 7,
                    -7, 9, -5, 11, -3, 13, -1, 15]
PACKET_DTYPE = np.dtype([
    ("flag", "<u2"), ("azimuth", "<u2"),
    ("returns", np.dtype([("distance", "<u2"), ("reflectivity", "u1")]), 32)])


@dataclass
class LidarPacket:
    raw_timestamp: int
    azimuths: np.ndarray  # Hundredths of a degree, one per block.
    points: np.ndarray  # 12 x 32 x 4; zero ranges removed only after assembly.
    valid_returns: np.ndarray
    block_bytes: tuple[bytes, ...]
    payload: bytes


def decode_packet(payload, elevations_deg=None, distance_scale=0.002):
    if len(payload) != 1206:
        raise ValueError(f"Expected 1206-byte VLP-16 packet, got {len(payload)}")
    # VLP-16 product ID 0x22, strongest/last return 0x37/0x38.
    if payload[1204] not in (0x37, 0x38) or payload[1205] != 0x22:
        raise ValueError("Unsupported return mode/model; dual return is not supported")
    blocks = np.frombuffer(payload, dtype=PACKET_DTYPE, count=12)
    if not np.all(blocks["flag"] == 0xEEFF) or np.any(blocks["azimuth"] >= 36000):
        raise ValueError("Invalid block header or azimuth")
    azimuths = blocks["azimuth"].astype(np.int32)
    progression = np.diff(azimuths) % 36000
    if np.any(progression == 0) or np.any(progression >= 18000):
        raise ValueError("Invalid within-packet azimuth progression")
    elevations = np.asarray(VLP16_ELEVATIONS if elevations_deg is None else elevations_deg)
    if elevations.shape != (16,) or not np.isfinite(elevations).all() or distance_scale <= 0:
        raise ValueError("Expected 16 finite beam elevations and a positive distance scale")
    angles = np.deg2rad(np.tile(elevations, 2))
    distance = blocks["returns"]["distance"].astype(np.float32) * distance_scale
    horizontal = distance * np.cos(angles)[None, :]
    azimuth = np.deg2rad(azimuths / 100.0)[:, None]
    # Same simulator convention as VLDrive: +x forward, +y left, +z up.
    points = np.stack((horizontal * np.cos(azimuth), -horizontal * np.sin(azimuth),
                       distance * np.sin(angles)[None, :],
                       blocks["returns"]["reflectivity"]), axis=-1).astype(np.float32)
    return LidarPacket(struct.unpack_from("<I", payload, 1200)[0], azimuths, points,
                       distance > 0, tuple(payload[i:i+100] for i in range(0, 1200, 100)), payload)


@dataclass
class _Group:
    first: int
    last: int
    blocks: dict = field(default_factory=dict)
    packets: set = field(default_factory=set)
    duplicates: int = 0
    issues: set = field(default_factory=set)


class LidarAssembler:
    """Bounded reorder buffer keyed by source timestamp, never by packet count.

    Groups close after a receipt-time deadline. Sorting block azimuths repairs
    reordering. A full circular grid with the configured step is required for
    complete=True. Without a verified step, completeness remains unverified.
    """
    def __init__(self, mapper, coordinate_frame="lidar", assembly_timeout_ms=150,
                 max_pending_frames=8, max_packets_per_frame=4096,
                 azimuth_step_deg=None, azimuth_tolerance_deg=0.02):
        if assembly_timeout_ms <= 0 or max_pending_frames < 1 or max_packets_per_frame < 1:
            raise ValueError("Invalid assembly limits")
        if azimuth_step_deg is not None and not 0 < azimuth_step_deg < 180:
            raise ValueError("azimuth_step_deg must be between 0 and 180")
        if azimuth_tolerance_deg < 0 or (azimuth_step_deg is not None and
                                       azimuth_tolerance_deg >= azimuth_step_deg / 2):
            raise ValueError("Azimuth tolerance must be less than half of the step")
        self.mapper = mapper
        self.coordinate_frame = coordinate_frame
        self.timeout = int(assembly_timeout_ms * 1_000_000)
        self.max_pending = max_pending_frames
        self.max_packets = max_packets_per_frame
        self.step = azimuth_step_deg
        self.tolerance = azimuth_tolerance_deg
        self.groups = OrderedDict()
        self.closed = deque(maxlen=256)
        self.next_id = 0
        self.late_packets = 0
        self.last_receipt = None

    def feed(self, packet, receive_monotonic_time):
        now = receive_monotonic_time
        if self.last_receipt is not None and now < self.last_receipt:
            raise ValueError("Receipt times must be monotonic even when source packets reorder")
        self.last_receipt = now
        output = self.expire(now)
        key = packet.raw_timestamp
        if key in self.closed:
            self.late_packets += 1
            return output
        if key not in self.groups:
            if len(self.groups) >= self.max_pending:
                output.append(self._finish(next(iter(self.groups)), "buffer_limit"))
            self.groups[key] = _Group(now, now)
        group = self.groups[key]
        group.last = now
        if packet.payload in group.packets:
            group.duplicates += 1
            return output
        if len(group.packets) >= self.max_packets:
            output.append(self._finish(key, "packet_limit"))
            return output
        group.packets.add(packet.payload)
        for i, angle in enumerate(packet.azimuths):
            angle = int(angle)
            if angle in group.blocks:
                if group.blocks[angle][0] != packet.block_bytes[i]:
                    group.issues.add("conflicting_azimuth_block")
                continue
            group.blocks[angle] = (packet.block_bytes[i], packet.points[i], packet.valid_returns[i])
        return output

    def expire(self, now):
        return [self._finish(key) for key, group in list(self.groups.items())
                if now - group.first >= self.timeout]

    def flush(self):
        return [self._finish(key, "stream_ended") for key in list(self.groups)]

    def _finish(self, key, reason=None):
        group = self.groups.pop(key)
        self.closed.append(key)
        angles = np.array(sorted(group.blocks)) / 100.0
        gaps = np.diff(np.r_[angles, angles[0] + 360.0])
        issues = set(group.issues)
        if self.step is None:
            issues.add("azimuth_step_unverified")
        elif not np.all(np.abs(gaps - self.step) <= self.tolerance):
            issues.add("missing_or_irregular_azimuths")
        if reason and reason != "stream_ended":
            issues.add(reason)
        points = np.concatenate([group.blocks[int(round(a * 100))][1][
            group.blocks[int(round(a * 100))][2]] for a in angles], axis=0)
        stamp, status, epoch = self.mapper.map(key)
        if status == "clock_jump":
            issues.add("clock_jump")
        frame = LidarFrame(self.next_id, key, stamp, group.first, self.coordinate_frame,
                           not issues, points, group.last, issues=sorted(issues),
                           packet_count=len(group.packets), duplicate_count=group.duplicates,
                           timestamp_status=status, clock_epoch=epoch,
                           diagnostics={"azimuth_blocks": len(angles),
                               "close_reason": reason or "assembly_deadline",
                               "max_azimuth_gap_deg": float(gaps.max()),
                               "late_packets_total": self.late_packets})
        self.next_id += 1
        return frame


class LidarUDPReceiver:
    """Context-managed socket; malformed packets are counted and skipped."""
    def __init__(self, assembler, bind="127.0.0.1", port=2368,
                 source_ip=None, decoder=None, socket_timeout_ms=50):
        self.assembler = assembler
        self.bind, self.port = bind, port
        self.source_ip = source_ip
        self.decoder = decoder or {}
        self.timeout = socket_timeout_ms / 1000
        self.socket = None
        self.invalid_packets = 0
        self.filtered_packets = 0
        self.last_error = None
        self.packet_sink = None
        self.peer = None

    def __enter__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.socket.bind((self.bind, self.port))
            self.socket.settimeout(self.timeout)
        except BaseException:
            self.socket.close()
            raise
        return self

    def poll(self):
        try:
            payload, peer = self.socket.recvfrom(65535)
        except socket.timeout:
            return self.assembler.expire(time.monotonic_ns())
        now = time.monotonic_ns()
        if self.source_ip and peer[0] != self.source_ip:
            self.filtered_packets += 1
            return self.assembler.expire(now)
        if self.peer is not None and peer != self.peer:
            self.filtered_packets += 1
            return self.assembler.expire(now)
        try:
            packet = decode_packet(payload, **self.decoder)
        except ValueError as exc:
            self.invalid_packets += 1
            self.last_error = str(exc)
            return self.assembler.expire(now)
        self.peer = peer
        if self.packet_sink:
            self.packet_sink(payload, now)
        return self.assembler.feed(packet, now)

    def __exit__(self, *_):
        self.socket.close()


class LidarUDPStream:
    """Receive continuously while clustering/rendering run on the caller thread.

    Full output queues drop the oldest *assembled frame*, never merge packets.
    The counter is surfaced by the runtime; all accepted packets can be recorded.
    """
    def __init__(self, receiver, queue_size=4):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.receiver = receiver
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.dropped_frames = 0
        self.error = None

    def _run(self):
        try:
            while not self.stop_event.is_set():
                for frame in self.receiver.poll():
                    try:
                        self.queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.queue.get_nowait()
                            self.dropped_frames += 1
                        except queue.Empty:
                            pass
                        self.queue.put_nowait(frame)
        except Exception as exc:
            self.error = exc

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, name="lidar-udp", daemon=True)
        self.thread.start()
        return self

    def read(self, timeout=0.1):
        if self.error:
            raise self.error
        try:
            return [self.queue.get(timeout=timeout)]
        except queue.Empty:
            return []

    def drain(self):
        frames = []
        while True:
            try:
                frames.append(self.queue.get_nowait())
            except queue.Empty:
                return frames

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
