import ctypes
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivedSample:
    value: object
    received_at: float
    sequence: int
    source_time: object

class _Packet(ctypes.Structure):
    _pack_ = 1

class _Receiver:
    def __init__(self, port, bind="0.0.0.0", recv_bytes=70000, name=None, history_size=32):
        if type(history_size) is not int or history_size < 1:
            raise ValueError('history_size must be a positive integer')
        self._history = deque(maxlen=history_size)
        self.port = int(port)
        self.name = name or self.__class__.__name__
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        except OSError:
            pass
        try:
            self._socket.bind((bind, self.port))
        except BaseException:
            self._socket.close()
            raise
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
                    self._history.append(ReceivedSample(sample, self._monotonic_stamp,
                                                        self.samples, self.source_time(sample)))

    def source_time(self, sample):
        return sample.get('sim_time') if isinstance(sample, dict) else None

    def recent(self):
        """Bounded immutable receipt records for causal sensor alignment."""
        with self._lock:
            return tuple(self._history)

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

        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

Receiver = _Receiver
Packet = _Packet
