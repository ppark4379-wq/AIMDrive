"""Bounded framed transport between Windows CUDA and the local ROS relay."""
import json
import socket
import struct
import threading
from types import SimpleNamespace

MAX_JSON = 1_000_000
MAX_RAW = 16_000_000


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespace(v) for v in value]
    return value


def send_frame(connection, metadata, raw=b''):
    encoded = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_JSON or len(raw) > MAX_RAW:
        raise ValueError('Oversize relay frame')
    connection.sendall(struct.pack('!II', len(encoded), len(raw)) + encoded + raw)


def frames(connection, stopped):
    """Keep partial headers/payloads across socket timeouts (TCP is a stream)."""
    buffer = bytearray()
    while not stopped.is_set():
        if len(buffer) >= 8:
            nj, nb = struct.unpack('!II', buffer[:8])
            if nj > MAX_JSON or nb > MAX_RAW:
                raise ValueError('Oversize relay frame')
            end = 8 + nj + nb
            if len(buffer) >= end:
                metadata = json.loads(buffer[8:8 + nj])
                raw = bytes(buffer[8 + nj:end])
                del buffer[:end]
                yield metadata, raw
                continue
        try:
            chunk = connection.recv(256 * 1024)
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError('Relay connection closed')
        buffer.extend(chunk)


class SensorClient:
    def __init__(self, source, host='127.0.0.1', port=12061):
        self.source, self.error = source, None
        self.counts = {}
        self.stopped = threading.Event()
        self.connection = socket.create_connection((host, port), timeout=5)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(.5)
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            for packet, raw in frames(self.connection, self.stopped):
                data, topic = packet['msg'], packet['topic']
                role = packet['role']
                self.counts[role] = self.counts.get(role, 0) + 1
                if topic in ('/image_jpeg/compressed', '/lidar3D'):
                    data['data'] = raw
                message = namespace(data)
                if topic == '/image_jpeg/compressed':
                    self.source._camera(message)
                elif topic == '/lidar3D':
                    self.source._lidar(message)
                elif topic == '/Ego_topic':
                    self.source._ego(message)
        except Exception as exc:
            if not self.stopped.is_set():
                self.error = repr(exc)

    def send(self, packet):
        send_frame(self.connection, packet)

    def close(self):
        self.stopped.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        self.thread.join(timeout=2)
