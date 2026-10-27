"""Bounded JSON/raw TCP frames for the optional Windows perception relay."""
import json
import socket
import struct
import threading
from types import SimpleNamespace

MAX_JSON = 1_000_000
MAX_RAW = 16_000_000
_HEADER = struct.Struct("!II")


def send_frame(connection, metadata, raw=b""):
    document = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(document) > MAX_JSON or len(raw) > MAX_RAW:
        raise ValueError("Oversize relay frame")
    connection.sendall(_HEADER.pack(len(document), len(raw)) + document + raw)


def frames(connection, stopped):
    pending = bytearray()
    while not stopped.is_set():
        if len(pending) >= _HEADER.size:
            json_size, raw_size = _HEADER.unpack_from(pending)
            if json_size > MAX_JSON or raw_size > MAX_RAW:
                raise ValueError("Oversize relay frame")
            total = _HEADER.size + json_size + raw_size
            if len(pending) >= total:
                split = _HEADER.size + json_size
                record = json.loads(pending[_HEADER.size:split])
                payload = bytes(pending[split:total])
                del pending[:total]
                yield record, payload
                continue
        try:
            chunk = connection.recv(256 * 1024)
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError("Relay connection closed")
        pending.extend(chunk)


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [namespace(item) for item in value]
    return value


class SensorClient:
    """Forward bounded relay messages to a native perception source's callbacks."""
    def __init__(self, source, host="127.0.0.1", port=12061):
        self.source, self.error, self.counts = source, None, {}
        self.stopped = threading.Event()
        self.connection = socket.create_connection((host, port), timeout=5)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(.5)
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        callbacks = {"/image_jpeg/compressed": "_camera", "/lidar3D": "_lidar", "/Ego_topic": "_ego"}
        try:
            for metadata, raw in frames(self.connection, self.stopped):
                topic, role = metadata["topic"], metadata["role"]
                self.counts[role] = self.counts.get(role, 0)+1
                message = metadata["msg"]
                if topic in ("/image_jpeg/compressed", "/lidar3D"):
                    message["data"] = raw
                callback = callbacks.get(topic)
                if callback is not None:
                    getattr(self.source, callback)(namespace(message))
        except Exception as error:
            if not self.stopped.is_set():
                self.error = repr(error)

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
