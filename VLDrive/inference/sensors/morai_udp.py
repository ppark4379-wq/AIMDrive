"""UDP receivers for the MORAI streams the competition rules allow.

Rule 2-8 permits only these, and this module implements exactly the read-only
subset the vision encoder needs.  Nothing here ever sends to the simulator, so
no control command can leave by accident:

    CameraReceiver          [UDP] Camera        JPEG fragments, 'MOR' / 'EI'
    LidarReceiver           [UDP] 3D LiDAR      Velodyne-style packets
    GpsReceiver             [UDP] GPS           NMEA0183 GPRMC / GPGGA
    ImuReceiver             [UDP] IMU           107-byte '#IMUData$' block
    VehicleStatusReceiver   [UDP] Competition Vehicle Status

Packet layouts follow inference/third_party/morai_network_module (MORAI's own
reference, branch 24.R2.0) and the protocol pages it links to.  The differences
from that reference are deliberate:

* one daemon thread per socket instead of its multiprocessing.Process, which
  re-binds the socket in a child process and loses every packet on Windows;
* the camera reassembler honours the per-fragment ``size`` field rather than
  appending all 64979 payload bytes of every fragment, so a decoded image is
  exactly the JPEG the simulator sent;
* every receiver keeps only the newest complete sample, which is what a live
  loop wants: falling behind should drop frames, not queue them.

Competition Vehicle Status deliberately omits pos_x/pos_y/pos_z, so ego position
comes from GPS and heading from the status yaw (or the IMU quaternion).
"""
from __future__ import annotations

import ctypes
from collections import Counter
import math
import socket
import struct
import threading
import time

import numpy as np

# --------------------------------------------------------------------- packets


class _Packet(ctypes.Structure):
    _pack_ = 1


class ImuPacket(_Packet):
    """115-byte IMU block, the layout MORAI's own reference module declares.

    The protocol page instead documents 107 bytes: the same packet without the
    8-byte sec/nsec stamp.  The two disagree, so :class:`ImuReceiver` picks the
    layout from the datagram length rather than trusting either.
    """
    _fields_ = [
        ("header", ctypes.c_char * 9),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("sec", ctypes.c_int),
        ("nsec", ctypes.c_int),
        ("ori_w", ctypes.c_double),
        ("ori_x", ctypes.c_double),
        ("ori_y", ctypes.c_double),
        ("ori_z", ctypes.c_double),
        ("ang_vel_x", ctypes.c_double),
        ("ang_vel_y", ctypes.c_double),
        ("ang_vel_z", ctypes.c_double),
        ("lin_acc_x", ctypes.c_double),
        ("lin_acc_y", ctypes.c_double),
        ("lin_acc_z", ctypes.c_double),
        ("tail", ctypes.c_char * 2),
    ]


class ImuPacketNoStamp(_Packet):
    """The 107-byte variant from the protocol page: no sec/nsec."""
    _fields_ = [
        ("header", ctypes.c_char * 9),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("ori_w", ctypes.c_double),
        ("ori_x", ctypes.c_double),
        ("ori_y", ctypes.c_double),
        ("ori_z", ctypes.c_double),
        ("ang_vel_x", ctypes.c_double),
        ("ang_vel_y", ctypes.c_double),
        ("ang_vel_z", ctypes.c_double),
        ("lin_acc_x", ctypes.c_double),
        ("lin_acc_y", ctypes.c_double),
        ("lin_acc_z", ctypes.c_double),
        ("tail", ctypes.c_char * 2),
    ]


class VehicleStatusPacket(_Packet):
    """Competition Vehicle Status; same layout as EgoVehicleStatus with the
    fields rule 2-8 withholds left at zero."""
    _fields_ = [
        ("header", ctypes.c_char * 11),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("sec", ctypes.c_int),
        ("nsec", ctypes.c_int),
        ("ctrl_mode", ctypes.c_byte),
        ("gear", ctypes.c_byte),
        ("signed_vel", ctypes.c_float),
        ("map_data_id", ctypes.c_int),
        ("accel", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("size_x", ctypes.c_float),
        ("size_y", ctypes.c_float),
        ("size_z", ctypes.c_float),
        ("overhang", ctypes.c_float),
        ("wheelbase", ctypes.c_float),
        ("rear_overhang", ctypes.c_float),
        ("pos_x", ctypes.c_float),
        ("pos_y", ctypes.c_float),
        ("pos_z", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("vel_x", ctypes.c_float),
        ("vel_y", ctypes.c_float),
        ("vel_z", ctypes.c_float),
        ("ang_vel_x", ctypes.c_float),
        ("ang_vel_y", ctypes.c_float),
        ("ang_vel_z", ctypes.c_float),
        ("accel_x", ctypes.c_float),
        ("accel_y", ctypes.c_float),
        ("accel_z", ctypes.c_float),
        ("steer", ctypes.c_float),
        ("link_id", ctypes.c_char * 38),
        ("tire_lateral_force_fl", ctypes.c_float),
        ("tire_lateral_force_fr", ctypes.c_float),
        ("tire_lateral_force_rl", ctypes.c_float),
        ("tire_lateral_force_rr", ctypes.c_float),
        ("side_slip_angle_fl", ctypes.c_float),
        ("side_slip_angle_fr", ctypes.c_float),
        ("side_slip_angle_rl", ctypes.c_float),
        ("side_slip_angle_rr", ctypes.c_float),
        ("tire_cornering_stiffness_fl", ctypes.c_float),
        ("tire_cornering_stiffness_fr", ctypes.c_float),
        ("tire_cornering_stiffness_rl", ctypes.c_float),
        ("tire_cornering_stiffness_rr", ctypes.c_float),
        ("tail", ctypes.c_char * 2),
    ]


class CompetitionVehicleStatusPacket(_Packet):
    """181 bytes: the status minus the twelve floats rule 2-8 withholds.

    Confirmed against a running simulator, which publishes this variant and
    leaves pos_x/pos_y/pos_z at 0.0 exactly as the rules describe.
    """
    _fields_ = [(name, kind) for name, kind in VehicleStatusPacket._fields_
                if not name.startswith(("tire_", "side_slip_"))]


IMU_BYTES = ctypes.sizeof(ImuPacket)                  # 115, reference module
IMU_BYTES_NO_STAMP = ctypes.sizeof(ImuPacketNoStamp)  # 107, protocol page
VEHICLE_STATUS_BYTES = ctypes.sizeof(VehicleStatusPacket)              # 229, full
COMPETITION_STATUS_BYTES = ctypes.sizeof(CompetitionVehicleStatusPacket)  # 181

# Camera fragment: 'MOR' + sec + nsec + index + size + jpeg(64979) + tail(2).
CAMERA_HEADER = b"MOR"
CAMERA_BBOX_HEADER = b"BOX"
CAMERA_FRAGMENT_BYTES = 65000
CAMERA_JPEG_CAPACITY = 64979
CAMERA_PREFIX = struct.Struct("<3s4i")   # header, sec, nsec, index, size


# ------------------------------------------------------------------- receivers


class _Receiver:
    """A bound UDP socket drained by one daemon thread, keeping the latest sample."""

    def __init__(self, port, bind="0.0.0.0", recv_bytes=70000, name=None, diagnostics=False):
        self.port = int(port)
        self.name = name or self.__class__.__name__
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        except OSError:
            pass
        self._socket.bind((bind, self.port))
        self._socket.settimeout(0.5)
        self._rx_overflow_option = None
        if diagnostics and hasattr(self._socket, 'recvmsg'):
            # Linux reports socket-queue drops on the next received datagram.
            # This does not measure packets lost before this socket.
            try:
                self._socket.setsockopt(socket.SOL_SOCKET, 40, 1)  # SO_RXQ_OVFL
                self._rx_overflow_option = 40
            except OSError:
                pass
        self.audit_counts = Counter()
        self._received_ns = 0
        self._recv_bytes = recv_bytes
        self._lock = threading.Lock()
        self._latest = None
        self._stamp = 0.0
        self.packets = 0
        self.samples = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                if self._rx_overflow_option is not None:
                    payload, ancillary, flags, _ = self._socket.recvmsg(self._recv_bytes, 64)
                    for level, kind, value in ancillary:
                        if level == socket.SOL_SOCKET and kind == self._rx_overflow_option:
                            self.audit_counts['socket_queue_drops'] = struct.unpack('=I', value[:4])[0]
                    if flags & socket.MSG_TRUNC:
                        self.audit_counts['truncated_datagrams'] += 1
                else:
                    payload, _ = self._socket.recvfrom(self._recv_bytes)
            except socket.timeout:
                continue
            except OSError:
                break
            self.packets += 1
            try:
                sample = self._consume(payload)
            except Exception:
                self.audit_counts['parse_exceptions'] += 1
                continue
            if sample is not None:
                with self._lock:
                    self._latest = sample
                    self._stamp = time.time()
                    self._received_ns = time.monotonic_ns()
                    self._source_stamp_ns = getattr(self, '_completed_source_stamp_ns', 0)
                    self._sample_metadata = dict(getattr(self, '_completed_metadata', {}))
                    self.samples += 1

    def _consume(self, payload):
        raise NotImplementedError

    def latest(self):
        """(sample, age_seconds) or (None, inf) before anything has arrived."""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest, time.time() - self._stamp

    def close(self):
        self._running = False
        try:
            self._socket.close()
        except OSError:
            pass

    def latest_snapshot(self):
        """Sample identity and completion time, read under the same lock."""
        with self._lock:
            age = time.time() - self._stamp if self._latest is not None else float('inf')
            identity = dict(sample_id=self.samples, received_ns=self._received_ns,
                            source_stamp_ns=getattr(self, '_source_stamp_ns', 0))
            identity.update(getattr(self, '_sample_metadata', {}))
            return self._latest, age, identity

    def diagnostics(self):
        return dict(packets=self.packets, samples=self.samples,
                    socket_drop_counter_available=self._rx_overflow_option is not None,
                    counters=dict(self.audit_counts))


class CameraReceiver(_Receiver):
    """Reassembles one JPEG from the simulator's 'MOR' fragments.

    Each fragment carries an index, its own payload length in ``size`` and up to
    64979 JPEG bytes; the last one is tailed 'EI'.  Measured on a live 1280x720
    front camera, concatenating ``body[:size]`` reproduces the JPEG exactly:
    106766 bytes, SOI at the front, EOI at the very end, nothing trailing.

    A 65000-byte datagram is carried by about 45 IP fragments, so losing one
    loses the whole thing.  An image whose fragment indices are not 0,1,2,... is
    therefore discarded rather than handed on: emitting it would put a torn JPEG
    in front of the encoder, which decodes to None and stalls the live loop.
    """

    MAX_IMAGE_BYTES = 16 * 1024 * 1024

    def __init__(self, port, **kwargs):
        self._buffer = bytearray()
        self._expect_index = 0
        self._intact = False
        self._frame_stamp = None
        self.torn_images = 0
        super().__init__(port, recv_bytes=CAMERA_FRAGMENT_BYTES + 64, **kwargs)

    def _consume(self, payload):
        if not hasattr(self, 'audit_counts'):  # standalone protocol fixtures
            self.audit_counts = Counter()
        if len(payload) < CAMERA_PREFIX.size + 2:
            self.audit_counts['short_fragments'] += 1
            return None
        header, _sec, _nsec, index, size = CAMERA_PREFIX.unpack_from(payload, 0)
        if header != CAMERA_HEADER:
            return None  # 'BOX' bounding boxes are not used here
        tail = payload[-2:]
        body = payload[CAMERA_PREFIX.size:-2]
        # Never repair a truncated datagram by silently shortening its length.
        if index < 0 or size <= 0 or size > len(body):
            self.audit_counts['invalid_size_or_index_fields'] += 1
            self._buffer.clear()
            self._intact = False
            return None
        stamp = (_sec, _nsec)
        if index == 0:
            self.audit_counts['frames_started'] += 1
            if self._buffer:
                self.audit_counts['unfinished_frames_replaced'] += 1
            self._frame_stamp = stamp
            self._buffer = bytearray()
            self._intact = True
        else:
            if index != self._expect_index:
                self.audit_counts['fragment_index_discontinuities'] += 1
                self._intact = False
            if getattr(self, '_frame_stamp', None) != stamp:
                self.audit_counts['fragment_stamp_mismatches'] += 1
                self._intact = False
        if len(self._buffer) + size > self.MAX_IMAGE_BYTES:
            self.audit_counts['oversize_images'] += 1
            self._intact = False
        if self._intact:
            self._buffer += body[:size]
        else:
            self._buffer.clear()
        self._expect_index = index + 1
        if tail == b"EI":
            self.audit_counts['end_markers'] += 1
            image = bytes(self._buffer) if self._intact else None
            if image is None:
                self.torn_images += 1
                self.audit_counts['discarded_torn_images'] += 1
            elif not (image.startswith(b'\xff\xd8') and image.endswith(b'\xff\xd9')):
                self.audit_counts['jpeg_envelope_anomalies'] += 1
                image = None
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            if image is not None:
                self._completed_source_stamp_ns = _sec * 1_000_000_000 + _nsec
            return image
        return None


class GpsReceiver(_Receiver):
    """NMEA0183: keeps the newest fix seen in a GPRMC or GPGGA sentence."""

    def __init__(self, port, **kwargs):
        self._fix = {"latitude": None, "longitude": None, "altitude": None}
        super().__init__(port, recv_bytes=4096, **kwargs)

    @staticmethod
    def _degrees(value, hemisphere):
        if not value:
            return None
        dot = value.find(".")
        degrees = float(value[: dot - 2])
        minutes = float(value[dot - 2:])
        result = degrees + minutes / 60.0
        return -result if hemisphere in ("S", "W") else result

    def _consume(self, payload):
        text = payload.decode("ascii", "ignore")
        updated = False
        for sentence in text.split("$"):
            parts = sentence.strip().split(",")
            if len(parts) < 6:
                continue
            if parts[0].endswith("GGA") and len(parts) > 9:
                latitude = self._degrees(parts[2], parts[3])
                longitude = self._degrees(parts[4], parts[5])
                if latitude is not None and longitude is not None:
                    self._fix.update(latitude=latitude, longitude=longitude)
                    try:
                        self._fix["altitude"] = float(parts[9])
                    except ValueError:
                        pass
                    updated = True
            elif parts[0].endswith("RMC") and len(parts) > 6:
                latitude = self._degrees(parts[3], parts[4])
                longitude = self._degrees(parts[5], parts[6])
                if latitude is not None and longitude is not None:
                    self._fix.update(latitude=latitude, longitude=longitude)
                    updated = True
        if updated and self._fix["latitude"] is not None:
            return dict(self._fix)
        return None


class ImuReceiver(_Receiver):
    """107-byte IMU block; yaw is derived from the orientation quaternion."""

    def __init__(self, port, **kwargs):
        self.layout_bytes = None
        super().__init__(port, recv_bytes=512, **kwargs)

    def _consume(self, payload):
        """Choose the layout by datagram length; a unit quaternion confirms it."""
        if len(payload) >= IMU_BYTES:
            packet = ImuPacket.from_buffer_copy(payload[:IMU_BYTES])
            stamped = True
        elif len(payload) >= IMU_BYTES_NO_STAMP:
            packet = ImuPacketNoStamp.from_buffer_copy(payload[:IMU_BYTES_NO_STAMP])
            stamped = False
        else:
            return None
        if not packet.header.startswith(b"#IMU"):
            return None
        x, y, z, w = packet.ori_x, packet.ori_y, packet.ori_z, packet.ori_w
        if not 0.9 < math.sqrt(x * x + y * y + z * z + w * w) < 1.1:
            return None  # wrong layout: the quaternion landed on the wrong bytes
        self.layout_bytes = IMU_BYTES if stamped else IMU_BYTES_NO_STAMP
        return {
            "yaw": math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
            "quaternion": (x, y, z, w),
            "angular_velocity": (packet.ang_vel_x, packet.ang_vel_y, packet.ang_vel_z),
            "linear_acceleration": (packet.lin_acc_x, packet.lin_acc_y, packet.lin_acc_z),
            "sim_time": (packet.sec + packet.nsec / 1e9) if stamped else None,
            "layout_bytes": self.layout_bytes,
        }


class VehicleStatusReceiver(_Receiver):
    """Competition Vehicle Status; supplies the encoder's speed and heading."""

    def __init__(self, port, **kwargs):
        self.layout_bytes = None
        super().__init__(port, recv_bytes=1024, **kwargs)

    def _consume(self, payload):
        """Accept either the full status or the shorter competition variant."""
        if len(payload) >= VEHICLE_STATUS_BYTES:
            packet = VehicleStatusPacket.from_buffer_copy(payload[:VEHICLE_STATUS_BYTES])
            competition = False
        elif len(payload) >= COMPETITION_STATUS_BYTES:
            packet = CompetitionVehicleStatusPacket.from_buffer_copy(
                payload[:COMPETITION_STATUS_BYTES])
            competition = True
        else:
            return None
        if not packet.header.startswith(b"#"):
            return None
        self.layout_bytes = COMPETITION_STATUS_BYTES if competition else VEHICLE_STATUS_BYTES
        return {
            "competition_variant": competition,
            "layout_bytes": self.layout_bytes,
            "speed": math.hypot(float(packet.vel_x), float(packet.vel_y)),
            "velocity_xy": [float(packet.vel_x), float(packet.vel_y)],
            "signed_speed": float(packet.signed_vel),
            "yaw": math.radians(float(packet.yaw)),
            "yaw_degrees": float(packet.yaw),
            "accel": float(packet.accel),
            "brake": float(packet.brake),
            "steer": float(packet.steer),
            "gear": int(packet.gear),
            "sim_time": packet.sec + packet.nsec / 1e9,
        }


def resolve_lidar_hour_stamp(stamp_us, reference_ns, max_delta_ns=1_000_000_000):
    """Resolve MORAI microseconds within an hour against a sensor source clock.

    The reference must be a camera/Ego source header, never host wall time.
    Adjacent hours handle a sweep crossing the hour boundary.
    """
    if not 0 <= stamp_us < 3_600_000_000 or reference_ns <= 0:
        raise ValueError('Invalid LiDAR hour timestamp or source-clock reference')
    hour = 3_600_000_000_000
    base = reference_ns // hour * hour
    candidates = [base + shift * hour + stamp_us * 1000 for shift in (-1, 0, 1)]
    result = min(candidates, key=lambda value: abs(value-reference_ns))
    if abs(result-reference_ns) > max_delta_ns:
        raise ValueError('LiDAR and reference source clocks differ by more than one second')
    return result


class LidarReceiver(_Receiver):
    """Velodyne-style 3D LiDAR, accumulated into one revolution of points.

    A packet holds 12 blocks of 0xFFEE + azimuth + 32 channel returns
    (uint16 distance in 2 mm units, uint8 reflectivity).  A revolution ends when
    the azimuth wraps, and that sweep becomes the sample.

    ``elevations`` must match the LiDAR model configured in the simulator; the
    default is the VLP-16 table.  Pass the right one for a 32/64-channel sensor,
    or the point cloud will be geometrically wrong even though it decodes.
    """

    VLP16_ELEVATIONS = [-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0]
    BLOCKS_PER_PACKET = 12
    CHANNELS_PER_BLOCK = 32
    BLOCK_BYTES = 100
    # One packet read in a single frombuffer: flag, azimuth, then 32 returns.
    PACKET_DTYPE = np.dtype([
        ("flag", "<u2"),
        ("azimuth", "<u2"),
        ("returns", np.dtype([("distance", "<u2"), ("reflectivity", "u1")]), CHANNELS_PER_BLOCK),
    ])

    def __init__(self, port, elevations=None, distance_scale=0.002,
                 frame_policy='azimuth_wrap', **kwargs):
        if frame_policy not in ('azimuth_wrap', 'morai_frame_timestamp', 'morai_cyclic_frame_timestamp'):
            raise ValueError('Unknown LiDAR frame policy')
        self.frame_policy = frame_policy
        self._frame_timestamp_us = None
        self._frame_chunks = []
        self._frame_last = None
        self._frame_step = None
        self._frame_valid = False
        self._frame_finished = False
        self._frame_packets = 0
        table = [math.radians(a) for a in (elevations or self.VLP16_ELEVATIONS)]
        # Tiled to the 32 slots a block carries; VLP-16 fires its 16 lasers
        # twice per block, which a live capture confirmed (channels 0..15 and
        # 16..31 report the same ranges to within a few millimetres).
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
        """Whole packet in vectorised numpy; a python loop here cost the live
        loop about a quarter of its frame time."""
        if self.frame_policy == 'morai_frame_timestamp':
            return self._consume_timestamped(payload)
        if self.frame_policy == 'morai_cyclic_frame_timestamp':
            return self._consume_cyclic_timestamped(payload)
        if len(payload) < self.BLOCKS_PER_PACKET * self.BLOCK_BYTES:
            self.audit_counts['short_lidar_packets'] += 1
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE,
                               count=self.BLOCKS_PER_PACKET)
        valid = blocks["flag"] == 0xEEFF
        if not valid.all():
            self.audit_counts['invalid_lidar_blocks'] += int((~valid).sum())
        if not valid.any():
            return None
        blocks = blocks[valid]

        finished = None
        raw_azimuth = blocks["azimuth"].astype(np.int32)
        # A revolution ends where the azimuth wraps past 360 degrees.
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
        return finished

    def _decode_points(self, blocks):
        raw_azimuth = blocks['azimuth'].astype(np.int32)
        distance = blocks["returns"]["distance"].astype(np.float32) * self.distance_scale
        reflectivity = blocks["returns"]["reflectivity"].astype(np.float32)
        azimuth = np.radians(raw_azimuth.astype(np.float32) / 100.0)[:, None]
        horizontal = distance * self._cos_elevation[None, :]
        # MORAI/Velodyne azimuth is clockwise from +y; the encoder wants
        # +x forward, +y left, which is what the recordings also store.
        points = np.stack([
            horizontal * np.cos(azimuth),
            -horizontal * np.sin(azimuth),
            distance * self._sin_elevation[None, :],
            reflectivity,
        ], axis=-1)
        keep = distance > 0.0

        return points, keep

    def _consume_timestamped(self, payload):
        """MORAI full-frame packets share a microsecond-of-hour timestamp.

        Emit at the last angular block of that frame, not at the first packet
        of the next frame. Require complete, ordered angular coverage; never
        join returns from different frame timestamps. This policy is explicit
        because physical Velodyne per-packet timestamps have different semantics.
        """
        if len(payload) != 1206:
            self.audit_counts['invalid_timestamped_lidar_size'] += 1
            self._frame_valid = False
            return None
        stamp = struct.unpack_from('<I', payload, 1200)[0]
        if stamp >= 3_600_000_000:
            self.audit_counts['invalid_lidar_hour_timestamp'] += 1
            self._frame_valid = False
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE, count=12)
        az = blocks['azimuth'].astype(np.int32)
        if stamp != self._frame_timestamp_us:
            if self._frame_timestamp_us is not None:
                delta = (stamp - self._frame_timestamp_us) % 3_600_000_000
                if delta > 1_800_000_000:
                    self.audit_counts['old_lidar_frame_packets'] += 1
                    return None
                if not self._frame_finished:
                    self.audit_counts['incomplete_lidar_frames'] += 1
            self._frame_timestamp_us = stamp
            self._frame_chunks = []
            self._frame_last = None
            self._frame_step = None
            self._frame_valid = bool(az[0] == 0)
            self._frame_finished = False
            self._frame_packets = 0
        if self._frame_finished:
            self.audit_counts['duplicate_completed_lidar_packets'] += 1
            return None
        step = int(az[1] - az[0])
        if (not np.all(blocks['flag'] == 0xEEFF) or step <= 0
                or 36000 % step or not np.all(np.diff(az) == step)
                or az[-1] >= 36000
                or (self._frame_step is not None and step != self._frame_step)
                or (self._frame_last is not None and az[0] != self._frame_last + step)):
            self.audit_counts['lidar_frame_discontinuities'] += 1
            self._frame_valid = False
        self._frame_step, self._frame_last = step, int(az[-1])
        self._frame_packets += 1
        if self._frame_valid:
            points, keep = self._decode_points(blocks)
            self._frame_chunks.append(points[keep])
        if int(az[-1]) + step != 36000:
            return None
        self._frame_finished = True
        if not self._frame_valid:
            self.audit_counts['discarded_incomplete_lidar_frames'] += 1
            self._frame_chunks = []
            return None
        cloud = np.concatenate(self._frame_chunks)
        self._frame_chunks = []
        self._completed_metadata = dict(
            source_stamp_us_within_hour=stamp,
            frame_policy='morai_frame_timestamp',
            frame_packets=self._frame_packets, angular_coverage_complete=True)
        self.audit_counts['complete_timestamped_lidar_frames'] += 1
        return cloud

    def _consume_cyclic_timestamped(self, payload):
        """Competition frames: arbitrary start azimuth and 0.01deg rounding.

        A measured frame has 900 blocks (75 packets), 0.4deg spacing, one
        shared timestamp, and starts at 7.5deg instead of zero. Validate every
        block against its expected circular position. Missing, repeated, or
        reordered blocks cannot complete a frame; no timestamp mixing occurs.
        """
        if len(payload) != 1206:
            self.audit_counts['invalid_timestamped_lidar_size'] += 1
            self._frame_valid = False
            return None
        stamp = struct.unpack_from('<I', payload, 1200)[0]
        if stamp >= 3_600_000_000:
            self.audit_counts['invalid_lidar_hour_timestamp'] += 1
            self._frame_valid = False
            return None
        blocks = np.frombuffer(payload, dtype=self.PACKET_DTYPE, count=12)
        az = blocks['azimuth'].astype(np.int32)
        if stamp != self._frame_timestamp_us:
            if self._frame_timestamp_us is not None:
                delta = (stamp-self._frame_timestamp_us) % 3_600_000_000
                if delta > 1_800_000_000:
                    self.audit_counts['old_lidar_frame_packets'] += 1
                    return None
                if not self._frame_finished:
                    self.audit_counts['incomplete_lidar_frames'] += 1
            step = int(round(float(np.median(np.diff(az) % 36000))))
            self._frame_timestamp_us = stamp
            self._frame_first_azimuth = int(az[0])
            self._frame_step = step
            self._frame_block_count = 0
            self._frame_packets = 0
            self._frame_chunks = []
            self._frame_finished = False
            self._frame_valid = bool(step > 0 and 36000 % step == 0 and 36000//step % 12 == 0)
        if self._frame_finished:
            self.audit_counts['duplicate_completed_lidar_packets'] += 1
            return None
        if not self._frame_valid:
            return None
        expected_blocks = 36000//self._frame_step
        expected_az = (self._frame_first_azimuth +
                      (self._frame_block_count+np.arange(12))*self._frame_step) % 36000
        circular_error = (az-expected_az+18000) % 36000-18000
        if (not np.all(blocks['flag'] == 0xEEFF) or np.any(az >= 36000)
                or np.any(np.abs(circular_error) > 1)
                or self._frame_block_count+12 > expected_blocks):
            self.audit_counts['lidar_frame_discontinuities'] += 1
            self._frame_valid = False
            self._frame_chunks = []
            return None
        points, keep = self._decode_points(blocks)
        self._frame_chunks.append(points[keep])
        self._frame_block_count += 12
        self._frame_packets += 1
        if self._frame_block_count != expected_blocks:
            return None
        self._frame_finished = True
        cloud = np.concatenate(self._frame_chunks)
        self._frame_chunks = []
        self._completed_metadata = dict(source_stamp_us_within_hour=stamp,
            frame_policy='morai_cyclic_frame_timestamp', frame_packets=self._frame_packets,
            angular_coverage_complete=True, start_azimuth_centidegrees=self._frame_first_azimuth,
            azimuth_step_centidegrees=self._frame_step, rounding_tolerance_centidegrees=1)
        self.audit_counts['complete_timestamped_lidar_frames'] += 1
        return cloud


__all__ = [
    "CameraReceiver", "GpsReceiver", "ImuReceiver", "LidarReceiver",
    "VehicleStatusReceiver", "ImuPacket", "ImuPacketNoStamp", "VehicleStatusPacket",
    "CompetitionVehicleStatusPacket", "IMU_BYTES", "IMU_BYTES_NO_STAMP",
    "VEHICLE_STATUS_BYTES", "COMPETITION_STATUS_BYTES",
]
