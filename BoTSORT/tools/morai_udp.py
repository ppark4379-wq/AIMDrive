"""MORAI camera JPEG UDP receiver (timestamp header, 24.R1+).

Wire layout: MOR | uint32 seconds | uint32 nanoseconds | uint32 index |
uint32 payload size | JPEG fragment | AI/EI. Integers are little endian.
Reference: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35
"""
import socket
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass
class CameraFrame:
    jpeg: bytes
    stamp: tuple
    received_at: float
    sequence: int = 0


class JpegAssembler:
    """Bounded reassembly; discard incomplete frames instead of mixing images."""

    def __init__(self, timeout=0.3, index_base=0, max_bytes=8 * 1024 * 1024):
        self.timeout = timeout
        self.index_base = index_base
        self.max_bytes = max_bytes
        self.pending = OrderedDict()
        self.completed = deque(maxlen=64)
        self.invalid_packets = 0
        self.incomplete_frames = 0

    def expire(self, now):
        for stamp, state in list(self.pending.items()):
            if now - state["start"] > self.timeout:
                del self.pending[stamp]
                self.incomplete_frames += 1

    def feed(self, packet, now=None):
        now = time.monotonic() if now is None else now
        self.expire(now)
        if len(packet) < 21 or packet[:3] != b"MOR" or packet[-2:] not in (b"AI", b"EI"):
            self.invalid_packets += 1
            return None
        sec, nsec, index, size = struct.unpack_from("<IIII", packet, 3)
        if (nsec >= 1000000000 or size > len(packet) - 21 or size == 0
                or not self.index_base <= index < self.index_base + 4096):
            self.invalid_packets += 1
            return None
        stamp = (sec, nsec)
        if stamp in self.completed:
            return None
        if stamp not in self.pending:
            if len(self.pending) >= 4:
                self.pending.popitem(last=False)
                self.incomplete_frames += 1
            self.pending[stamp] = {"start": now, "parts": {}, "last": None, "bytes": 0}
        state = self.pending[stamp]
        payload = packet[19:19 + size]  # MORAI may pad the final datagram to 65000 bytes
        parts = state["parts"]
        final = packet[-2:] == b"EI"
        conflict = (
            (index in parts and parts[index] != payload)
            or (state["last"] is not None and index > state["last"])
            or (final and state["last"] is not None and state["last"] != index)
            or (final and any(i > index for i in parts))
            or (state["last"] == index and not final)
        )
        if conflict:
            self.invalid_packets += 1
            del self.pending[stamp]
            self.completed.append(stamp)
            return None
        if index not in parts:
            parts[index] = payload
            state["bytes"] += size
        if state["bytes"] > self.max_bytes:
            self.invalid_packets += 1
            del self.pending[stamp]
            self.completed.append(stamp)
            return None
        if final:
            state["last"] = index
        last = state["last"]
        if last is None or len(parts) != last - self.index_base + 1:
            return None
        jpeg = b"".join(parts[i] for i in range(self.index_base, last + 1))
        del self.pending[stamp]
        self.completed.append(stamp)
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            self.invalid_packets += 1
            return None
        return CameraFrame(jpeg, stamp, now)


class UdpCamera:
    """Receive independently of inference; keep only the latest complete frame."""

    def __init__(self, host, port, sender_ip=None, index_base=0):
        self.assembler = JpegAssembler(index_base=index_base)
        self.sender_ip = sender_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.sock.bind((host, port))
            self.sock.settimeout(0.1)
        except Exception:
            self.sock.close()
            raise
        self.address = self.sock.getsockname()
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest = None
        self.received_frames = 0
        self.overwritten_frames = 0
        self.error = None
        self.last_stamp = None
        self.stale_frames = 0
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stopped.is_set():
                try:
                    packet, sender = self.sock.recvfrom(65535)
                except socket.timeout:
                    self.assembler.expire(time.monotonic())
                    continue
                if self.sender_ip and sender[0] != self.sender_ip:
                    continue
                frame = self.assembler.feed(packet)
                if frame is not None:
                    if self.last_stamp is not None and frame.stamp <= self.last_stamp:
                        self.stale_frames += 1
                        continue
                    self.last_stamp = frame.stamp
                    with self.condition:
                        self.received_frames += 1
                        frame.sequence = self.received_frames
                        if self.latest is not None:
                            self.overwritten_frames += 1
                        self.latest = frame
                        self.condition.notify()
        except OSError as exc:
            if not self.stopped.is_set():
                self.error = exc
                with self.condition:
                    self.condition.notify_all()

    def get(self, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.latest is not None or self.error or self.stopped.is_set(), timeout
            )
            if self.error:
                raise self.error
            frame, self.latest = self.latest, None
            return frame

    def stats(self):
        return {
            "received_frames": self.received_frames,
            "overwritten_frames": self.overwritten_frames,
            "incomplete_frames": self.assembler.incomplete_frames,
            "invalid_packets": self.assembler.invalid_packets,
            "stale_frames": self.stale_frames,
        }

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1.0)
        self.sock.close()
        with self.condition:
            self.condition.notify_all()
