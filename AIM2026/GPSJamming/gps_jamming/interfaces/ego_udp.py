"""Receive MORAI EgoVehicleStatus, compatible with NetworkModule 24.R2.0.

Schema reference: MORAI-Autonomous/MORAI-NetworkModule, tag 24.R2.0,
lib/define/EgoVehicleStatus.py (packed little-endian fields).
Both the documented 181-byte base and 229-byte tire extension are supported.
No control commands are transmitted.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math
import socket
import struct
import threading
import time


@dataclass(frozen=True)
class EgoState:
    timestamp_ns: int  # Source clock: Unix in normal mode, simulation time in Sync Mode.
    receive_monotonic_ns: int
    position_m: tuple
    rotation_rpy_deg: tuple
    velocity_mps: tuple  # Protocol XYZ; frame convention still requires verification.
    angular_velocity_radps: tuple
    acceleration_mps2: tuple
    signed_speed_mps: float
    map_id: int
    control_mode: int
    gear: int
    packet_bytes: int
    clock_epoch: int = 0

    def to_dict(self):
        return {**asdict(self), 'clock_domain': 'morai_source_unverified',
                'pose_frame': 'morai_map', 'velocity_frame': 'unverified'}


def decode_ego_packet(payload, receive_monotonic_ns, clock_epoch=0):
    if len(payload) not in (181, 229):
        raise ValueError(f'Expected 181 or 229 EgoVehicleStatus bytes, got {len(payload)}')
    if payload[:11] != b'#MoraiInfo$' or payload[-2:] != b'\r\n':
        raise ValueError('Invalid EgoVehicleStatus header/tail')
    if struct.unpack_from('<i', payload, 11)[0] != len(payload) - 29:
        raise ValueError('Invalid EgoVehicleStatus data length')
    sec, nsec = struct.unpack_from('<ii', payload, 27)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise ValueError('Invalid Ego timestamp')
    signed_speed = struct.unpack_from('<f', payload, 37)[0]
    map_id = struct.unpack_from('<i', payload, 41)[0]
    position = struct.unpack_from('<3f', payload, 77)
    rotation = struct.unpack_from('<3f', payload, 89)
    velocity = struct.unpack_from('<3f', payload, 101)
    angular = struct.unpack_from('<3f', payload, 113)
    acceleration = struct.unpack_from('<3f', payload, 125)
    if not all(math.isfinite(v) for v in (*position, *rotation, *velocity, *angular,
                                         *acceleration, signed_speed)):
        raise ValueError('Nonfinite Ego measurement')
    return EgoState(sec * 1_000_000_000 + nsec, receive_monotonic_ns,
                    position, rotation, tuple(v / 3.6 for v in velocity),
                    tuple(math.radians(v) for v in angular), acceleration,
                    signed_speed / 3.6, map_id, payload[35], payload[36], len(payload), clock_epoch)


class EgoUDPReceiver:
    """Threaded bounded history. Missing/stale data is never replaced with zeros."""
    def __init__(self, bind='127.0.0.1', port=9094, source_ip=None, history_size=200):
        if history_size < 1:
            raise ValueError('history_size must be positive')
        self.bind, self.port, self.source_ip = bind, port, source_ip
        self.history = deque(maxlen=history_size)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.peer = None
        self.invalid_packets = self.filtered_packets = self.received_packets = 0
        self.clock_epoch = 0
        self.error = self.last_error = None
        self.datagrams = 0
        self.last_packet_peer = self.last_packet_size = None

    def __enter__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((self.bind, self.port))
            self.socket.settimeout(.1)
        except BaseException:
            self.socket.close()
            raise
        self.address = self.socket.getsockname()
        self.thread = threading.Thread(target=self._run, name='ego-udp', daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    payload, peer = self.socket.recvfrom(65535)
                except socket.timeout:
                    continue
                received = time.monotonic_ns()
                self.datagrams += 1
                self.last_packet_peer, self.last_packet_size = peer, len(payload)
                if ((self.source_ip and peer[0] != self.source_ip)
                        or (self.peer is not None and peer != self.peer)):
                    self.filtered_packets += 1
                    continue
                try:
                    state = decode_ego_packet(payload, received, self.clock_epoch)
                except ValueError as exc:
                    self.invalid_packets += 1
                    self.last_error = str(exc)
                    continue
                self.peer = peer
                with self.lock:
                    if self.history and state.timestamp_ns == self.history[-1].timestamp_ns:
                        continue
                    if self.history and state.timestamp_ns < self.history[-1].timestamp_ns:
                        # Conservative discontinuity: do not bridge resets/reordering.
                        self.clock_epoch += 1
                        self.history.clear()
                        state = decode_ego_packet(payload, received, self.clock_epoch)
                    self.history.append(state)
                    self.received_packets += 1
        except Exception as exc:
            self.error = exc

    def snapshot(self, max_age_s=.5):
        if self.error:
            raise self.error
        with self.lock:
            state = self.history[-1] if self.history else None
        age = (time.monotonic_ns() - state.receive_monotonic_ns) / 1e9 if state else None
        return {'status': 'missing' if state is None else 'fresh' if age <= max_age_s else 'stale',
                'age_s': age, 'state': state.to_dict() if state else None,
                'received_packets': self.received_packets, 'invalid_packets': self.invalid_packets,
                'filtered_packets': self.filtered_packets, 'peer': self.peer,
                'datagrams': self.datagrams, 'listen': self.address,
                'last_packet_peer': self.last_packet_peer, 'last_packet_size': self.last_packet_size,
                'last_error': self.last_error, 'synchronized_to_lidar': False}

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.socket.close()
