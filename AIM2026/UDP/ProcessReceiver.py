"""Receive the heavy sensors (cameras, LiDAR) in their own processes.

Camera JPEG reassembly and LiDAR frame decoding are CPU-bound Python. In the same interpreter as the encoder they
compete for the GIL: the LiDAR demux queue overflows (frames never complete), the sensors age past the freshness
limit and inference takes seconds instead of ~0.1 s. One receive-only child per group owns the sockets, the shared
port demultiplexers and the receivers; only the newest complete sample crosses into the parent through a bounded
shared-memory slot. The proxy has the receiver read API (``recent``, ``latest``, ``snapshot``, ``diagnostics``), so
callers do not change. No GPU, model or simulator control runs in a child.
"""
import json
import multiprocessing as mp
import os
import pickle
import threading
import time
from collections import deque

from .Common import ReceivedSample
from .SharedPort import SharedPort

POLL_S = .002
STATS_PERIOD_S = .25
STATS_BYTES = 16384
DEFAULT_SLOT_BYTES = 8 << 20


class _Slot:
    """Newest pickled ReceivedSample plus its sequence (bounded, single writer, lock-protected)."""

    def __init__(self, context, capacity):
        self.buffer = context.RawArray('B', capacity)
        self.meta = context.RawArray('q', 2)  # byte length, published sequence
        self.stats = context.RawArray('B', STATS_BYTES)
        self.stats_size = context.RawValue('i', 0)
        self.lock = context.Lock()

    def publish(self, sample, data):
        if len(data) > len(self.buffer):
            return False
        if not self.lock.acquire(timeout=.5):
            return False
        try:
            memoryview(self.buffer).cast('B')[:len(data)] = data
            self.meta[0], self.meta[1] = len(data), sample.sequence
        finally:
            self.lock.release()
        return True

    def read(self, after):
        """(sequence, bytes) of a sample newer than ``after``, else None."""
        if self.meta[1] <= after:
            return None
        if not self.lock.acquire(timeout=.5):
            return None
        try:
            length, sequence = self.meta[0], self.meta[1]
            data = bytes(memoryview(self.buffer).cast('B')[:length])
        finally:
            self.lock.release()
        return (sequence, data) if sequence > after else None

    def write_stats(self, stats):
        data = json.dumps(stats).encode('utf-8')
        if len(data) > STATS_BYTES or not self.lock.acquire(timeout=.5):
            return
        try:
            memoryview(self.stats).cast('B')[:len(data)] = data
            self.stats_size.value = len(data)
        finally:
            self.lock.release()

    def read_stats(self):
        if not self.lock.acquire(timeout=.5):
            return {}
        try:
            data = bytes(memoryview(self.stats).cast('B')[:self.stats_size.value])
        finally:
            self.lock.release()
        return json.loads(data) if data else {}


def _child(specs, slots, stop, parent_pid):
    shared, receivers, last = {}, {}, {}
    try:
        for spec in specs:
            key = (spec['bind'], spec['port'])
            if key not in shared:
                shared[key] = SharedPort(*key)
        for spec in specs:
            if spec.get('forward'):
                # A light sensor (GPS/IMU) on a shared port stays in the parent: its datagrams are passed on as-is.
                shared[(spec['bind'], spec['port'])].subscribe_forward(
                    spec['source_ip'], spec['source_port'], tuple(spec['forward']))
                continue
            receivers[spec['name']] = spec['factory'](
                spec['port'], bind=spec['bind'], name=spec['name'], drop_audit=True,
                shared_port=shared[(spec['bind'], spec['port'])], source_ip=spec['source_ip'],
                source_port=spec['source_port'], **spec['options'])
            last[spec['name']] = 0
        for port in shared.values():
            port.start()
        oversize = {name: 0 for name in receivers}
        next_stats = 0.
        while not stop.is_set():
            if os.getppid() != parent_pid:
                break
            time.sleep(POLL_S)
            for name, receiver in receivers.items():
                records = receiver.recent()
                if records and records[-1].sequence > last[name]:
                    sample = records[-1]
                    data = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                    if slots[name].publish(sample, data):
                        last[name] = sample.sequence
                    else:
                        oversize[name] += 1
                        last[name] = sample.sequence
            now = time.monotonic()
            if now >= next_stats:
                next_stats = now + STATS_PERIOD_S
                for name, receiver in receivers.items():
                    stats = receiver.diagnostics()
                    stats['counters'] = dict(stats.get('counters') or {})
                    if oversize[name]:
                        stats['counters']['oversize_or_locked_samples'] = oversize[name]
                    stats['clock_epoch'] = getattr(receiver, 'clock_epoch', 0)
                    slots[name].write_stats(stats)
    finally:
        for receiver in receivers.values():
            receiver.close()
        for port in shared.values():
            port.close()


class ProcessReceiverProxy:
    """Receiver read API over a child-owned slot."""

    def __init__(self, name, port, slot, history_size):
        self.name, self.port = name, int(port)
        self._slot, self._after = slot, 0
        self._history = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def _refresh(self):
        with self._lock:
            update = self._slot.read(self._after)
            if update is not None:
                sequence, data = update
                self._history.append(pickle.loads(data))
                self._after = sequence
            return self._history[-1] if self._history else None

    @property
    def samples(self):
        return int(self._slot.read_stats().get('samples', 0))

    @property
    def packets(self):
        return int(self._slot.read_stats().get('packets', 0))

    @property
    def clock_epoch(self):
        return int(self._slot.read_stats().get('clock_epoch', 0))

    def recent(self):
        self._refresh()
        with self._lock:
            return tuple(self._history)

    def latest(self):
        sample = self._refresh()
        if sample is None:
            return None, float('inf')
        return sample.value, time.monotonic() - sample.received_at

    def snapshot(self):
        sample = self._refresh()
        if sample is None:
            return None, float('inf'), 0, None
        return sample.value, time.monotonic() - sample.received_at, sample.sequence, sample.received_at

    def diagnostics(self):
        stats = dict(self._slot.read_stats())
        stats.setdefault('port', self.port)
        stats.setdefault('packets', 0)
        stats.setdefault('samples', 0)
        stats.setdefault('socket_drop_counter_available', False)
        stats.setdefault('counters', {})
        stats['process_isolated'] = True
        return stats

    def close(self):
        pass


class ProcessReceiverGroup:
    """One child process hosting several receivers, and the proxies that read them."""

    def __init__(self, specs, history_sizes, slot_bytes=None):
        self._context = mp.get_context('spawn')
        self._stop = self._context.Event()
        capacity = dict(slot_bytes or {})
        hosted = [spec for spec in specs if not spec.get('forward')]
        self._slots = {spec['name']: _Slot(self._context, capacity.get(spec['name'], DEFAULT_SLOT_BYTES))
                       for spec in hosted}
        self.proxies = {spec['name']: ProcessReceiverProxy(spec['name'], spec['port'], self._slots[spec['name']],
                                                           history_sizes.get(spec['name'], 16))
                        for spec in hosted}
        self._process = self._context.Process(
            target=_child, args=(specs, self._slots, self._stop, os.getpid()),
            name='udp_receivers_' + '_'.join(spec['name'] for spec in hosted), daemon=True)

    def start(self, ready_timeout=10.):
        self._process.start()
        return self

    def alive(self):
        return self._process.is_alive()

    def close(self):
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(timeout=3.)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.)
