import ctypes
import queue
import socket
import struct
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivedSample:
    value: object
    received_at: float
    sequence: int
    source_time: object
    meta: object = None
    received_ns: object = None  # Exact monotonic receipt, before packet parsing.
    source_stamp_ns: object = None  # Accepted source clock; raw integer remains in value.
    clock_epoch: int = 0

class _Packet(ctypes.Structure):
    _pack_ = 1

# Linux reports socket-queue drops on the next datagram (SO_RXQ_OVFL).
SO_RXQ_OVFL = 40
WSAEMSGSIZE = 10040
# A source clock that goes back is a simulator reset only once this many samples in a row stay behind the last
# accepted one. MORAI sends the odd IMU stamp one second back (sec not yet advanced at a second boundary: 542.000
# between 542.998 and 543.030; 5 of 2.8 M IMU samples, 10-25..10-29). Treated as a reset, it emptied the IMU history
# and the tunnel dead reckoning failed for good (Ego stopped in the GPS blackout, twice).
RESET_CONFIRM_SAMPLES = 2


class _Receiver:
    def __init__(self, port, bind="0.0.0.0", recv_bytes=70000, name=None, history_size=32,
                 drop_audit=False, shared_port=None, source_ip=None, source_port=None):
        if type(history_size) is not int or history_size < 1:
            raise ValueError('history_size must be a positive integer')
        self._history = deque(maxlen=history_size)
        self.port = int(port)
        self.name = name or self.__class__.__name__
        self.audit_counts = Counter()
        self._counts_lock = threading.Lock()
        self._shared_port = shared_port
        self._packet_received_ns = None
        self._channel = (shared_port.subscribe(source_ip, source_port, self._count)
                         if shared_port is not None else None)
        self._socket = None
        if shared_port is None:
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
        self._rx_overflow_option = None
        if drop_audit and sys.platform.startswith('linux') and hasattr(self._socket, 'recvmsg'):
            try:
                self._socket.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
                self._rx_overflow_option = SO_RXQ_OVFL
            except OSError:
                pass
        self._recv_bytes = recv_bytes
        self._lock = threading.Lock()
        self._latest = None
        self._stamp = 0.0
        self.packets = 0
        self.samples = 0
        self.clock_epoch = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()

    def _receive(self):
        """One datagram, or None when it must not be parsed (truncated)."""
        if self._channel is not None:
            try:
                payload, self._packet_received_ns = self._channel.get(timeout=.5)
            except queue.Empty:
                raise socket.timeout from None
            if len(payload) > self._recv_bytes:
                self._count('truncated_datagrams')
                return None
            return payload
        if self._rx_overflow_option is None:
            payload, _ = self._socket.recvfrom(self._recv_bytes)
            self._packet_received_ns = time.monotonic_ns()
            return payload
        payload, ancillary, flags, _ = self._socket.recvmsg(self._recv_bytes, 64)
        self._packet_received_ns = time.monotonic_ns()
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == self._rx_overflow_option:
                self._set_count('socket_queue_drops', struct.unpack('=I', value[:4])[0])
        if flags & socket.MSG_TRUNC:
            self._count('truncated_datagrams')
            return None
        return payload

    def _run(self):
        while self._running:
            try:
                payload = self._receive()
            except socket.timeout:
                continue
            except OSError as exc:
                # Windows raises instead of truncating an oversized datagram.
                if getattr(exc, 'winerror', None) == WSAEMSGSIZE:
                    self.packets += 1
                    self._count('truncated_datagrams')
                    continue
                break
            self.packets += 1
            if payload is None:
                continue
            try:
                sample = self._consume(payload)
            except Exception:
                self._count('parse_exceptions')
                continue
            if sample is not None:
                self._store(sample, received_ns=self._packet_received_ns)

    def _store(self, sample, received_ns=None):
        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        source_time = self.source_time(sample)
        # Keep the original DR HistoryMixin arithmetic during migration. Parsers
        # also preserve exact protocol ns in sample['source_stamp_ns'] for audit.
        source_ns = round(source_time * 1e9) if source_time else 0
        if source_ns < 0:
            self._count('negative_source_time')
            return
        meta = self.sample_metadata(sample)
        with self._lock:
            # Samples behind the last accepted stamp, held until the clock shows whether it really went back.
            behind = self.__dict__.setdefault('_behind', [])
            previous = self._history[-1].source_stamp_ns if self._history else None
            # A zero stamp means "no stamp" (e.g. an unfilled IMU stamp field), as in
            # VLDrive; deduplicating it would drop every packet after the first.
            if source_ns and previous:
                if source_ns == previous:
                    # The same simulator instant again: not a new observation.
                    self._count('duplicate_source_time')
                    return
                if source_ns < previous:
                    behind.append((sample, received_ns, source_time, meta, source_ns))
                    if len(behind) < RESET_CONFIRM_SAMPLES:
                        return
                    # Simulator reset/restart: never join samples across it. The held samples start the new clock.
                    self._history.clear()
                    self.clock_epoch += 1
                    self._count('source_time_resets')
                    held, behind[:] = behind[:-1], []
                    for args in held:
                        if not self._history or args[4] > self._history[-1].source_stamp_ns:
                            self._append(*args)
                    if self._history and source_ns <= self._history[-1].source_stamp_ns:
                        return
                elif behind:
                    # The old clock goes on: the held samples carried a bad stamp. Drop them.
                    self._count('source_time_glitches', len(behind))
                    behind.clear()
            self._append(sample, received_ns, source_time, meta, source_ns)

    def _append(self, sample, received_ns, source_time, meta, source_ns):
        """Accept one sample (lock held)."""
        self._latest = sample
        self._stamp = time.time()
        self._monotonic_stamp = received_ns / 1e9
        self.samples += 1
        self._history.append(ReceivedSample(sample, self._monotonic_stamp,
                                            self.samples, source_time, meta,
                                            received_ns, source_ns, self.clock_epoch))

    def source_time(self, sample):
        return sample.get('sim_time') if isinstance(sample, dict) else None

    def sample_metadata(self, sample):
        """Per-sample facts that are not the sample itself (e.g. a LiDAR frame stamp)."""
        return None

    def _count(self, key, amount=1):
        # Receivers built with object.__new__ (parser tests) have no __init__ state.
        lock = self.__dict__.get('_counts_lock')
        if lock is None:
            lock = self.__dict__.setdefault('_counts_lock', threading.Lock())
        with lock:
            self.__dict__.setdefault('audit_counts', Counter())[key] += amount

    def _set_count(self, key, value):
        lock = self.__dict__.setdefault('_counts_lock', threading.Lock())
        with lock:
            self.__dict__.setdefault('audit_counts', Counter())[key] = value

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
            return self._latest, time.monotonic() - self._monotonic_stamp

    def snapshot(self):
        """Atomic sample, age, sequence and monotonic receipt time (not sim time)."""
        with self._lock:
            if self._latest is None:
                return None, float('inf'), 0, None
            stamp = self._monotonic_stamp
            return self._latest, time.monotonic() - stamp, self.samples, stamp

    def diagnostics(self):
        """Packet/sample counts and the reasons datagrams were rejected."""
        lock = self.__dict__.setdefault('_counts_lock', threading.Lock())
        with lock:
            counters = dict(self.__dict__.get('audit_counts', ()))
        return dict(port=getattr(self, 'port', None), packets=getattr(self, 'packets', 0),
                    samples=getattr(self, 'samples', 0),
                    socket_drop_counter_available=getattr(self, '_rx_overflow_option', None) is not None,
                    counters=counters)

    def close(self):
        self._running = False
        if self._shared_port is not None:
            self._shared_port.unsubscribe(self._channel)
        try:
            if self._socket is not None:
                self._socket.close()
        except OSError:
            pass

        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

Receiver = _Receiver
Packet = _Packet
