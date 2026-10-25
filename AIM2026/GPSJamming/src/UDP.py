"""LiDAR UDP transport and unchanged planning boundary UDP contract."""
import json
import math
import socket
import time
import uuid
import threading
import queue
from .LidarDecoder import decode_packet as decode_lidar_packet

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
            packet = decode_lidar_packet(payload, **self.decoder)
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
    """Receive continuously while wall detection/rendering run on the caller thread.

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

SCHEMA = 'gps_jamming_wall_display_v1'


PORT = 12051


MAX_BYTES = 60000


def line_edges(boundaries, side):
    """LINE_LIST pairs, preserving segment breaks and every original coordinate."""
    return [[a, b] for segment in boundaries[side]
            for a, b in zip(segment['points'], segment['points'][1:])]


def decode_packet(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Wall datagram too large')
    packet = json.loads(data)
    if packet['schema'] != SCHEMA or packet['units'] != 'm' or packet['axes'] != 'FLU':
        raise ValueError('Unsupported wall contract')
    if not isinstance(packet['session'], str) or not packet['session']:
        raise ValueError('Missing sender session')
    if type(packet['sequence']) is not int or packet['sequence'] < 0:
        raise ValueError('Invalid sequence')
    boundary = packet['boundaries']
    if boundary['frame_id'] != 'base_link':
        raise ValueError('Expected base_link geometry')
    for side in ('left', 'right'):
        for segment in boundary[side]:
            for point in segment['points']:
                if len(point) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in point):
                    raise ValueError('Expected finite XYZ')
    return packet


class WallUDPSender:
    def __init__(self, host='127.0.0.1', port=PORT):
        self.destination = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = str(uuid.uuid4())
        self.sequence = 0

    def send(self, display):
        packet = dict(schema=SCHEMA, session=self.session, sequence=self.sequence,
                      sent_unix_ns=time.time_ns(), units='m', axes='FLU',
                      geometry_source='rviz_wall_markers', boundaries=display)
        self.sequence += 1
        data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode('utf-8')
        decode_packet(data)
        self.socket.sendto(data, self.destination)

    def close(self):
        self.socket.close()


class WallUDPReceiver:
    """Call poll each planning cycle; None means absent/expired input, not free road.

    Freshness uses local arrival time; source scan timestamps remain in boundaries.
    This is not a timestamp synchronizer or a world-coordinate conversion.
    """
    def __init__(self, bind='127.0.0.1', port=PORT, max_age_s=0.5):
        if max_age_s <= 0:
            raise ValueError('max_age_s must be positive')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((bind, port))
        self.socket.setblocking(False)
        self.max_age_s = max_age_s
        self.packet = None
        self.received = 0.0
        self.retired_sessions = set()
        self.rejected = 0

    def poll(self):
        # Bound work per control cycle even if an input floods the socket.
        for _ in range(256):
            try:
                data, _ = self.socket.recvfrom(MAX_BYTES + 1)
            except BlockingIOError:
                break
            try:
                packet = decode_packet(data)
                if packet['session'] in self.retired_sessions:
                    raise ValueError('Retired sender')
                if self.packet is not None:
                    if packet['session'] == self.packet['session']:
                        if packet['sequence'] <= self.packet['sequence']:
                            raise ValueError('Duplicate or out-of-order frame')
                    else:
                        self.retired_sessions.add(self.packet['session'])
                self.packet, self.received = packet, time.monotonic()
            except (ValueError, KeyError, TypeError, OverflowError):
                self.rejected += 1
        if time.monotonic() - self.received > self.max_age_s:
            return None
        return self.packet

    def close(self):
        self.socket.close()
