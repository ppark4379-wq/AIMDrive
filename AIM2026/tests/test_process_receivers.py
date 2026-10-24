"""Heavy sensors received in child processes: same read API, same samples, clean shutdown."""
import socket
import struct
import time

from AIM2026.UDP.Camera import CameraReceiver
from AIM2026.UDP.ProcessReceiver import ProcessReceiverGroup


def free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def fragment(sec, index, body, last):
    return struct.pack('<3s4i', b'MOR', sec, 1, index, len(body)) + body + (b'EI' if last else b'XX')


def wait_for(predicate, timeout=15.):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(.02)
    return False


def test_camera_child_returns_complete_jpegs_through_proxy():
    port = free_port()
    spec = dict(name='rgb_front', factory=CameraReceiver, port=port, bind='127.0.0.1', source_ip=None,
                source_port=None, options=dict(history_size=4))
    group = ProcessReceiverGroup([spec], {'rgb_front': 4}).start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        proxy = group.proxies['rgb_front']
        image = b'\xff\xd8' + bytes(range(200)) * 3 + b'\xff\xd9'
        half = len(image) // 2

        def send_frame(sec):
            sender.sendto(fragment(sec, 0, image[:half], False), ('127.0.0.1', port))
            sender.sendto(fragment(sec, 1, image[half:], True), ('127.0.0.1', port))

        assert wait_for(lambda: (send_frame(1) or True) and proxy.recent(), timeout=20.)
        sample = proxy.recent()[-1]
        assert sample.value == image
        assert sample.sequence >= 1 and sample.received_at <= time.monotonic()
        value, age = proxy.latest()
        assert value == image and 0 <= age < 30
        for sec in range(2, 5):
            send_frame(sec)
            time.sleep(.05)
        assert wait_for(lambda: proxy.snapshot()[2] >= 2)
        assert wait_for(lambda: proxy.diagnostics()['counters'].get('complete_images', 0) >= 2)  # stats are periodic
        stats = proxy.diagnostics()
        assert stats['process_isolated'] is True and stats['port'] == port
    finally:
        sender.close()
        group.close()
    assert not group.alive()


def test_proxy_before_any_sample_matches_empty_receiver():
    port = free_port()
    spec = dict(name='rgb_rear', factory=CameraReceiver, port=port, bind='127.0.0.1', source_ip=None,
                source_port=None, options=dict(history_size=4))
    group = ProcessReceiverGroup([spec], {'rgb_rear': 4}).start()
    try:
        proxy = group.proxies['rgb_rear']
        assert proxy.recent() == ()
        assert proxy.latest() == (None, float('inf'))
        assert proxy.snapshot() == (None, float('inf'), 0, None)
        stats = proxy.diagnostics()
        assert stats['packets'] == 0 and stats['samples'] == 0 and stats['counters'] == {}
    finally:
        group.close()
