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

    def __init__(self, port, bind="0.0.0.0", recv_bytes=70000, name=None):
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
                payload, _ = self._socket.recvfrom(self._recv_bytes)
            except socket.timeout:
                continue
            except OSError:
                break
            self.packets += 1
            try:
                sample = self._consume(payload)
            except Exception:
                continue
            if sample is not None:
                with self._lock:
                    self._latest = sample
                    self._stamp = time.time()
                    self._monotonic_stamp = time.monotonic()
                    self.samples += 1

    def _consume(self, payload):
        raise NotImplementedError

    def latest(self):
        """(sample, age_seconds) or (None, inf) before anything has arrived."""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest, time.time() - self._stamp

    def snapshot(self):
        """Atomic sample, age, sequence and monotonic receipt time (not sim time)."""
        with self._lock:
            if self._latest is None:
                return None, float('inf'), 0, None
            stamp = self._monotonic_stamp
            return self._latest, time.monotonic() - stamp, self.samples, stamp

    def close(self):
        self._running = False
        try:
            self._socket.close()
        except OSError:
            pass


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

    def __init__(self, port, **kwargs):
        self._buffer = bytearray()
        self._expect_index = 0
        self._intact = False
        self._frame_stamp = None
        self.torn_images = 0
        super().__init__(port, recv_bytes=CAMERA_FRAGMENT_BYTES + 64, **kwargs)

    def _consume(self, payload):
        if len(payload) < CAMERA_PREFIX.size + 2:
            return None
        header, sec, nsec, index, size = CAMERA_PREFIX.unpack_from(payload, 0)
        if header != CAMERA_HEADER:
            return None  # 'BOX' bounding boxes are not used here
        frame_stamp = (sec, nsec)
        if frame_stamp != self._frame_stamp:
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            self._frame_stamp = frame_stamp
        tail = payload[-2:]
        body = payload[CAMERA_PREFIX.size:-2]
        take = min(max(int(size), 0), len(body)) if size else len(body)
        if index == 0:
            self._buffer = bytearray()
            self._intact = True
        elif index != self._expect_index:
            self._intact = False   # a fragment went missing; this image is ruined
        self._buffer += body[:take]
        self._expect_index = index + 1
        if tail == b"EI":
            image = bytes(self._buffer) if self._intact else None
            if image is None:
                self.torn_images += 1
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
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
            # UDP signed velocity is km/h along the vehicle's direction.
            # PID/navigation consume m/s, not one velocity-vector component.
            "speed": float(packet.signed_vel) / 3.6,
            "signed_speed": float(packet.signed_vel),
            "yaw": math.radians(float(packet.yaw)),
            "yaw_degrees": float(packet.yaw),
            "accel": float(packet.accel),
            "brake": float(packet.brake),
            "steer": float(packet.steer),
            "gear": int(packet.gear),
            "sim_time": packet.sec + packet.nsec / 1e9,
        }


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

    def __init__(self, port, elevations=None, distance_scale=0.002, **kwargs):
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
        # A revolution ends where the azimuth wraps past 360 degrees.
        wrap = np.flatnonzero(np.diff(raw_azimuth) < 0)
        first = int(wrap[0]) + 1 if wrap.size else None
        if self._last_azimuth is not None and raw_azimuth[0] < self._last_azimuth:
            first = 0
        self._last_azimuth = int(raw_azimuth[-1])

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


__all__ = [
    "CameraReceiver", "GpsReceiver", "ImuReceiver", "LidarReceiver",
    "VehicleStatusReceiver", "ImuPacket", "ImuPacketNoStamp", "VehicleStatusPacket",
    "CompetitionVehicleStatusPacket", "IMU_BYTES", "IMU_BYTES_NO_STAMP",
    "VEHICLE_STATUS_BYTES", "COMPETITION_STATUS_BYTES",
]
