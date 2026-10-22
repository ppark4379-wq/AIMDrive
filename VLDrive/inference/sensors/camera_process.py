"""Isolate camera packet processing from the encoder's Python interpreter.

One receive-only child owns all camera sockets. Bounded shared slots contain
only the newest complete JPEG, its original receive time and sample identity.
No GPU, image decoding, frame queue, or simulator control runs in this worker.
"""
import json
import multiprocessing as mp
import os
import time

from inference.sensors.morai_udp import CameraReceiver


class _Slot:
    def __init__(self, context, capacity):
        self.image = context.RawArray('B', capacity)
        self.meta = context.RawArray('q', 4)  # bytes, ID, receipt ns, source stamp ns
        self.stats = context.RawArray('B', 8192)
        self.stats_size = context.RawValue('i', 0)
        self.lock = context.Lock()

    def publish(self, image, sample_id, received_ns, source_stamp_ns=0):
        if len(image) > len(self.image):
            raise ValueError('Camera image exceeds shared slot capacity')
        if not self.lock.acquire(timeout=.5):
            raise RuntimeError('Camera shared slot write timed out')
        try:
            memoryview(self.image).cast('B')[:len(image)] = image
            self.meta[:] = (len(image), sample_id, received_ns, source_stamp_ns)
        finally:
            self.lock.release()

    def write_stats(self, stats):
        data = json.dumps(stats).encode('utf-8')
        if len(data) > len(self.stats):
            raise ValueError('Camera diagnostic record exceeds capacity')
        if not self.lock.acquire(timeout=.5):
            raise RuntimeError('Camera statistics write timed out')
        try:
            memoryview(self.stats).cast('B')[:len(data)] = data
            self.stats_size.value = len(data)
        finally:
            self.lock.release()


def _camera_worker(ports, bind, diagnostics, slots, stop, ready, owner_pid):
    receivers = {}
    try:
        for key, port in ports.items():
            receivers[key] = CameraReceiver(port, bind=bind, name=key, diagnostics=diagnostics)
        for key, receiver in receivers.items():
            slots[key].write_stats(receiver.diagnostics())
        ready.send(None)
        ready.close()
        last_ids = dict.fromkeys(ports, 0)
        last_stats = 0.
        while not stop.value and os.getppid() == owner_pid:
            now = time.monotonic()
            for key, receiver in receivers.items():
                if not receiver._thread.is_alive():
                    raise RuntimeError('Camera receive thread stopped: ' + key)
                sample, _, identity = receiver.latest_snapshot()
                if sample is not None and identity['sample_id'] != last_ids[key]:
                    slots[key].publish(sample, identity['sample_id'], identity['received_ns'],
                                       identity.get('source_stamp_ns', 0))
                    last_ids[key] = identity['sample_id']
                if now - last_stats >= .05:
                    slots[key].write_stats(receiver.diagnostics())
            if now - last_stats >= .05:
                last_stats = now
            time.sleep(.002)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        try:
            ready.send(str(exc))
        except (OSError, ValueError):
            pass
        raise
    finally:
        ready.close()
        for key, receiver in receivers.items():
            receiver.close()
            slots[key].write_stats(receiver.diagnostics())


class _CameraView:
    def __init__(self, group, key, port):
        self.group, self.key, self.port = group, key, port
        self._sample, self._id = None, 0

    def latest_snapshot(self):
        self.group.check_alive()
        slot = self.group.slots[self.key]
        if not slot.lock.acquire(timeout=.2):
            raise RuntimeError('Camera shared slot read timed out: ' + self.key)
        try:
            size, sample_id, received_ns, source_stamp_ns = slot.meta[:]
            if sample_id != self._id:
                self._sample = bytes(memoryview(slot.image).cast('B')[:size])
                self._id = sample_id
            sample = self._sample
        finally:
            slot.lock.release()
        age = (time.monotonic_ns() - received_ns) / 1e9 if received_ns else float('inf')
        return sample, age, dict(sample_id=sample_id, received_ns=received_ns,
                                source_stamp_ns=source_stamp_ns)

    def latest(self):
        sample, age, _ = self.latest_snapshot()
        return sample, age

    def diagnostics(self):
        slot = self.group.slots[self.key]
        if not slot.lock.acquire(timeout=.2):
            raise RuntimeError('Camera statistics read timed out: ' + self.key)
        try:
            data = bytes(memoryview(slot.stats).cast('B')[:slot.stats_size.value])
        finally:
            slot.lock.release()
        result = json.loads(data) if data else dict(packets=0, samples=0, counters={})
        return dict(result, receiver_mode='process', receiver_pid=self.group.process.pid)

    @property
    def packets(self):
        return self.diagnostics()['packets']

    @property
    def samples(self):
        return self.diagnostics()['samples']

    def close(self):
        pass  # The group owns the worker and all four sockets.


class CameraProcessGroup:
    def __init__(self, ports, bind='0.0.0.0', diagnostics=False, startup_timeout=15.):
        context = mp.get_context('spawn')  # Never fork a process that owns CUDA.
        self.slots = {key: _Slot(context, CameraReceiver.MAX_IMAGE_BYTES) for key in ports}
        # A killed worker must not leave an Event's internal semaphore locked.
        self.stop = context.RawValue('b', 0)
        self.closed = False
        reader, writer = context.Pipe(duplex=False)
        self.process = context.Process(target=_camera_worker,
            args=(dict(ports), bind, diagnostics, self.slots, self.stop, writer, os.getpid()),
            name='ve_camera_receive', daemon=True)
        self.process.start()
        writer.close()
        try:
            if not reader.poll(startup_timeout):
                raise RuntimeError('Camera receiver process startup timed out')
            error = reader.recv()
            if error is not None:
                raise RuntimeError('Camera receiver process startup failed: ' + error)
            self.check_alive()
        except BaseException:
            self.close()
            raise
        finally:
            reader.close()
        self.cameras = {key: _CameraView(self, key, port) for key, port in ports.items()}

    def check_alive(self):
        if self.closed or not self.process.is_alive():
            raise RuntimeError('Camera receiver process is unavailable (exit={})'.format(self.process.exitcode))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.value = 1
        self.process.join(timeout=2.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)
