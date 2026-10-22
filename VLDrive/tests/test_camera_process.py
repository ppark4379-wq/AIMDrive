"""Exercise real process/socket lifetime, image identity and stale-frame handling."""
import socket
import multiprocessing as mp
import time
import unittest

from inference.sensors.camera_process import CameraProcessGroup
from inference.sensors.morai_udp import CAMERA_PREFIX


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fragments(image, stamp):
    for index, offset in enumerate(range(0, len(image), 1100)):
        part = image[offset:offset + 1100]
        tail = b'EI' if offset + len(part) == len(image) else b'AI'
        yield CAMERA_PREFIX.pack(b'MOR', stamp, 0, index, len(part)) + part + tail


def worker_owner(pipe, port):
    group = CameraProcessGroup({'front': port}, bind='127.0.0.1')
    pipe.send(group.process.pid)
    pipe.close()
    while True:
        time.sleep(1.)


class CameraProcessTests(unittest.TestCase):
    def start(self, keys=('front',)):
        ports = {key: free_port() for key in keys}
        group = CameraProcessGroup(ports, bind='127.0.0.1', diagnostics=True)
        self.addCleanup(group.close)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        return group, ports, sender

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail('Timed out waiting for camera worker')

    def send(self, sender, port, packets):
        for packet in packets:
            sender.sendto(packet, ('127.0.0.1', port))

    def test_two_cameras_keep_exact_content_identity_and_original_age(self):
        group, ports, sender = self.start(('front', 'rear'))
        image = b'\xff\xd8' + b'front' * 1000 + b'\xff\xd9'
        rear = b'\xff\xd8rear\xff\xd9'
        begin = time.monotonic_ns()
        self.send(sender, ports['front'], fragments(image, 10))
        self.send(sender, ports['rear'], fragments(rear, 10))
        self.wait_for(lambda: all(v.latest()[0] is not None for v in group.cameras.values()))
        received, age, identity = group.cameras['front'].latest_snapshot()
        self.assertEqual(received, image)
        self.assertEqual(group.cameras['rear'].latest()[0], rear)
        self.assertEqual(identity['sample_id'], 1)
        self.assertEqual(identity['source_stamp_ns'], 10_000_000_000)
        self.assertGreaterEqual(identity['received_ns'], begin)
        time.sleep(.05)
        again, new_age, same = group.cameras['front'].latest_snapshot()
        self.assertEqual((again, same), (image, identity))
        self.assertGreater(new_age, age + .04)
        self.wait_for(lambda: group.cameras['front'].diagnostics()['samples'] == 1)
        self.assertEqual(group.cameras['front'].diagnostics()['counters'].get('discarded_torn_images', 0), 0)

    def test_missing_fragment_does_not_refresh_valid_image_then_recovers(self):
        group, ports, sender = self.start()
        view = group.cameras['front']
        first = b'\xff\xd8first\xff\xd9'
        self.send(sender, ports['front'], fragments(first, 10))
        self.wait_for(lambda: view.latest()[0] == first)
        original = view.latest_snapshot()[2]
        broken = list(fragments(b'\xff\xd8' + b'x' * 5000 + b'\xff\xd9', 11))
        self.send(sender, ports['front'], broken[:1] + broken[2:])
        self.wait_for(lambda: view.diagnostics()['counters'].get('discarded_torn_images', 0) == 1)
        self.assertEqual(view.latest_snapshot()[2], original)
        self.assertEqual(view.latest()[0], first)
        final = b'\xff\xd8final\xff\xd9'
        self.send(sender, ports['front'], fragments(final, 12))
        self.wait_for(lambda: view.latest()[0] == final)
        self.assertEqual(view.latest_snapshot()[2]['sample_id'], 2)
        self.assertEqual(view.latest_snapshot()[2]['source_stamp_ns'], 12_000_000_000)

    def test_worker_death_fails_closed_and_close_releases_ports(self):
        group, ports, _ = self.start()
        group.process.terminate()
        group.process.join(timeout=2.)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            group.cameras['front'].latest_snapshot()
        group.close()
        group.close()
        self.assertFalse(group.process.is_alive())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', ports['front']))

    def test_bind_conflict_does_not_leave_worker_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
            occupied.bind(('127.0.0.1', 0))
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                CameraProcessGroup({'front': occupied.getsockname()[1]}, bind='127.0.0.1')

    def test_worker_releases_port_after_owner_is_killed(self):
        context = mp.get_context('spawn')
        reader, writer = context.Pipe(duplex=False)
        port = free_port()
        owner = context.Process(target=worker_owner, args=(writer, port))
        owner.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(10.))
            self.assertGreater(reader.recv(), 0)
            owner.terminate()
            owner.join(timeout=2.)
            def released():
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    try:
                        probe.bind(('127.0.0.1', port))
                        return True
                    except OSError:
                        return False
            self.wait_for(released)
        finally:
            reader.close()
            if owner.is_alive():
                owner.terminate()
                owner.join(timeout=2.)


if __name__ == '__main__':
    unittest.main()
