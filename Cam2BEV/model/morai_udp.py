"""ROS-free MORAI camera transport. Only complete, ordered JPEGs are published."""
import collections
import socket
import struct
import threading
import time
from dataclasses import dataclass

CAMERA_ORDER = ("front", "rear", "left", "right")


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    received_at: float
    timestamp: bytes  # Opaque simulator timestamp; format varies by release.
    sequence: int


class JpegAssembler:
    """Drop incomplete/out-of-order frames; restart at index zero.

    Supported layouts (little endian):
      legacy: MOR + uint32 index + uint32 size + JPEG + AI/EI
      timestamped: MOR + 8-byte timestamp + uint32 index + uint32 size + JPEG + AI/EI
    Fixed 65000-byte datagrams may contain arbitrary unused bytes before the tail.
    No frame identifier exists in legacy packets, so delayed fragments across
    frames cannot be distinguished reliably. JPEG decoding is checked downstream.
    """
    def __init__(self, protocol="auto", timeout=0.5, max_bytes=8 * 1024 * 1024):
        if protocol not in ("auto", "legacy", "timestamped"):
            raise ValueError("protocol must be auto, legacy or timestamped")
        self.protocol = protocol
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.sequence = 0
        self.reset()

    def reset(self):
        self.parts = []
        self.size = 0
        self.next_index = 0
        self.key = None
        self.started = None

    def feed(self, packet, now=None):
        now = time.monotonic() if now is None else now
        candidates = []
        if packet[:3] != b"MOR" or packet[-2:] not in (b"AI", b"EI"):
            self.reset()
            return None
        for layout, offset in (("legacy", 3), ("timestamped", 11)):
            if self.protocol not in ("auto", layout) or len(packet) < offset + 10:
                continue
            index, size = struct.unpack_from("<II", packet, offset)
            start = offset + 8
            end = start + size
            exact_size = len(packet) == end + 2
            # MORAI reuses fixed-size buffers: unused bytes after the declared
            # JPEG block can contain old nonzero data. Never append those bytes.
            padded_size = len(packet) == 65000 and end <= len(packet) - 2
            if 0 < size <= 65000 - start - 2 and (exact_size or padded_size):
                stamp = packet[3:11] if layout == "timestamped" else b""
                candidates.append((index, packet[start:end], (layout, stamp)))
        if len(candidates) != 1:
            self.reset()
            return None
        index, payload, key = candidates[0]
        if index == 0:
            self.reset()
            if not payload.startswith(b"\xff\xd8"):
                return None
            self.started, self.key = now, key
        if (self.started is None or now - self.started > self.timeout
                or key != self.key or index != self.next_index
                or self.size + len(payload) > self.max_bytes):
            self.reset()
            return None
        self.parts.append(payload)
        self.size += len(payload)
        self.next_index += 1
        if packet[-2:] != b"EI":
            return None
        jpeg = b"".join(self.parts)
        stamp = self.key[1]
        self.reset()
        if not jpeg.endswith(b"\xff\xd9"):
            return None
        self.sequence += 1
        return Frame(jpeg, now, stamp, self.sequence)


class CameraReceiver:
    def __init__(self, host, port, protocol="auto"):
        self.assembler = JpegAssembler(protocol)
        self.frames = collections.deque(maxlen=8)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.packet_count = 0
        self.byte_count = 0
        self.first_packet = None
        self.packet_samples = []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.socket.bind((host, port))
            self.socket.settimeout(0.1)
        except Exception:
            self.socket.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                packet, _ = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop_event.is_set():
                    self.error = exc
                return
            with self.lock:
                self.packet_count += 1
                self.byte_count += len(packet)
                if self.first_packet is None:
                    self.first_packet = {
                        "length": len(packet),
                        "head_hex": packet[:24].hex(),
                        "tail_hex": packet[-8:].hex(),
                        "jpeg_start_offset": packet.find(b"\xff\xd8"),
                    }
            with self.lock:
                if len(self.packet_samples) < 9:
                    sample = {"length": len(packet), "tail_hex": packet[-2:].hex(),
                              "head_hex": packet[:19].hex(),
                              "jpeg_start": packet.find(b"\xff\xd8"),
                              "jpeg_end": packet.rfind(b"\xff\xd9")}
                    if len(packet) >= 21:
                        index, size = struct.unpack_from("<II", packet, 11)
                        end = 19 + size
                        sample.update(index=index, size=size, timestamp_hex=packet[3:11].hex(),
                                      padding_bytes=max(0, len(packet)-2-end),
                                      padding_nonzero=(sum(b != 0 for b in packet[end:-2])
                                                       if end <= len(packet)-2 else None))
                    self.packet_samples.append(sample)
            frame = self.assembler.feed(packet)
            if frame is not None:
                with self.lock:
                    self.frames.append(frame)

    def snapshot(self):
        if self.error:
            raise RuntimeError("UDP receiver failed") from self.error
        with self.lock:
            return list(self.frames)

    def diagnostics(self):
        with self.lock:
            return {"packets": self.packet_count, "bytes": self.byte_count,
                    "first_packet": self.first_packet, "packet_samples": list(self.packet_samples)}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.socket.close()


def select_frames(queues, previous, max_skew, max_age, now=None):
    """Approximate pairing by host arrival time, NOT exposure synchronization."""
    now = time.monotonic() if now is None else now
    fresh = {
        name: [f for f in queues[name]
               if f.sequence > previous.get(name, 0) and 0 <= now - f.received_at <= max_age]
        for name in CAMERA_ORDER
    }
    if any(not items for items in fresh.values()):
        return None
    for anchor in reversed(fresh["front"]):
        batch = {"front": anchor}
        for name in CAMERA_ORDER[1:]:
            batch[name] = min(fresh[name], key=lambda f: abs(f.received_at - anchor.received_at))
        times = [f.received_at for f in batch.values()]
        if max(times) - min(times) <= max_skew:
            return batch
    return None
