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

    def __init__(self, host, port, sender_ip=None, index_base=0, expected_image_size=None,
                 sender_port=None, frame_buffer_size=1):
        if not 1 <= frame_buffer_size <= 120:
            raise ValueError('frame_buffer_size must be in [1, 120]')
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = deque()
        self.buffer_dropped_frames = 0
        if sender_port is not None and not 1 <= sender_port <= 65535:
            raise ValueError('sender_port must be in [1, 65535]')
        self.sender_port = sender_port
        self.assembler = JpegAssembler(index_base=index_base)
        # Optional single-camera selection: reassemble peers separately, then lock
        # to the first peer producing the configured image size. Not a camera ID.
        self.expected_image_size = tuple(expected_image_size) if expected_image_size else None
        self.index_base = index_base
        self.peer_assemblers = OrderedDict()
        self.selected_peer = None
        self.filtered_frames = 0
        self.filtered_packets = 0
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
        # Metadata only: keep diagnostics bounded without retaining JPEG images.
        self.frame_history = deque(maxlen=240)
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
                    for assembler in self.peer_assemblers.values():
                        assembler.expire(time.monotonic())
                    continue
                if self.sender_ip and sender[0] != self.sender_ip:
                    continue
                if self.sender_port is not None and sender[1] != self.sender_port:
                    self.filtered_packets += 1
                    continue
                if self.expected_image_size:
                    if self.selected_peer is not None and sender != self.selected_peer:
                        self.filtered_packets += 1
                        continue
                    if sender not in self.peer_assemblers:
                        if len(self.peer_assemblers) >= 8:
                            self.peer_assemblers.popitem(last=False)
                        self.peer_assemblers[sender] = JpegAssembler(index_base=self.index_base)
                    assembler = self.peer_assemblers[sender]
                else:
                    assembler = self.assembler
                frame = assembler.feed(packet)
                if frame is not None:
                    if self.expected_image_size:
                        import cv2
                        import numpy as np
                        decoded = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None or (decoded.shape[1], decoded.shape[0]) != self.expected_image_size:
                            self.filtered_frames += 1
                            continue
                        self.selected_peer = sender
                        self.assembler = assembler
                    if self.last_stamp is not None and frame.stamp <= self.last_stamp:
                        self.stale_frames += 1
                        continue
                    self.last_stamp = frame.stamp
                    with self.condition:
                        self.received_frames += 1
                        frame.sequence = self.received_frames
                        if self.frame_buffer_size == 1 and self.latest is not None:
                            self.overwritten_frames += 1
                            self._set_frame_state(self.latest.sequence, 'overwritten')
                        self.frame_history.append({
                            'sequence': frame.sequence,
                            'timestamp_ns': frame.stamp[0] * 10**9 + frame.stamp[1],
                            'receive_ns': round(frame.received_at * 1e9),
                            'state': 'pending'})
                        self.latest = frame
                        if self.frame_buffer_size > 1:
                            if len(self.frame_buffer) >= self.frame_buffer_size:
                                dropped = self.frame_buffer.popleft()
                                self._set_frame_state(dropped.sequence, 'buffer_evicted')
                                self.buffer_dropped_frames += 1
                            self.frame_buffer.append(frame)
                        self.condition.notify()
        except Exception as exc:
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
            if frame is not None:
                self._set_frame_state(frame.sequence, 'processing')
            return frame

    def _set_frame_state(self, sequence, state):
        # Caller holds condition.
        for row in reversed(self.frame_history):
            if row['sequence'] == sequence:
                row['state'] = state
                break

    def latest_timestamp_ns(self):
        with self.condition:
            return (self.last_stamp[0] * 10**9 + self.last_stamp[1]
                    if self.last_stamp is not None else None)

    def get_nearest(self, timestamp_ns, after_stamp=None, max_age_ms=400,
                    max_delta_ms=50):
        """Nonblocking bounded selection; never rewind the tracker or await backlog."""
        with self.condition:
            if self.error:
                raise self.error
            now = time.monotonic()
            kept = deque()
            for frame in self.frame_buffer:
                if now-frame.received_at > max_age_ms/1000:
                    self._set_frame_state(frame.sequence, 'expired')
                elif after_stamp is not None and frame.stamp <= after_stamp:
                    self._set_frame_state(frame.sequence, 'skipped_before_tracker')
                else:
                    kept.append(frame)
            self.frame_buffer = kept
            candidates = [f for f in kept if abs(f.stamp[0]*10**9+f.stamp[1]-timestamp_ns) <= max_delta_ms*1e6]
            if not candidates:
                return None
            chosen = min(candidates, key=lambda f: (
                abs(f.stamp[0]*10**9+f.stamp[1]-timestamp_ns), -f.sequence))
            while self.frame_buffer and self.frame_buffer[0].sequence <= chosen.sequence:
                removed = self.frame_buffer.popleft()
                self._set_frame_state(removed.sequence, 'processing' if removed is chosen else 'skipped_for_newer_pair')
            self.latest = self.frame_buffer[-1] if self.frame_buffer else None
            return chosen

    def mark_processed(self, sequence, valid=True):
        with self.condition:
            self._set_frame_state(sequence, 'processed' if valid else 'decode_rejected')

    def diagnose_selection(self, lidar_ns, selected_sequence, eligible_sequences,
                           max_age_ns, tolerance_ns):
        """Snapshot complete accepted JPEG frames available at selection time.

        Cannot infer incomplete packets, future arrivals, or acquisition latency.
        """
        with self.condition:
            now = time.monotonic_ns()
            history = [dict(row) for row in self.frame_history]
        fresh = [r for r in history if 0 <= now-r['receive_ns'] <= max_age_ns]
        nearest = min(fresh, key=lambda r: abs(lidar_ns-r['timestamp_ns'])) if fresh and lidar_ns is not None else None
        if nearest is not None:
            nearest['delta_ms'] = (lidar_ns-nearest['timestamp_ns']) / 1e6
            nearest['within_tolerance'] = abs(lidar_ns-nearest['timestamp_ns']) <= tolerance_ns
            nearest['in_sync_buffer'] = nearest['sequence'] in eligible_sequences
        return {'snapshot_monotonic_ns': now, 'selected_sequence': selected_sequence,
                'nearest_received': nearest, 'retained_frames': len(history),
                'scope': 'complete accepted JPEGs received by selection; bounded history'}

    def stats(self):
        return {
            "received_frames": self.received_frames,
            "overwritten_frames": self.overwritten_frames,
            "buffer_dropped_frames": self.buffer_dropped_frames,
            "frame_buffer_size": self.frame_buffer_size,
            "incomplete_frames": self.assembler.incomplete_frames,
            "invalid_packets": self.assembler.invalid_packets,
            "stale_frames": self.stale_frames,
            "selected_peer": self.selected_peer,
            "filtered_frames": self.filtered_frames,
            "filtered_packets": self.filtered_packets,
        }

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1.0)
        self.sock.close()
        with self.condition:
            self.condition.notify_all()
