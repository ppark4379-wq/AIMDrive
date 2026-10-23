"""One UDP destination socket, demultiplexed by MORAI sender IP/port.

Each sensor retains its own parser thread and JPEG assembly state. Never bind
multiple SO_REUSEADDR sockets to the shared camera/GPS/IMU destination.
"""
import queue
import socket
import threading
import time


class SharedPort:
    def __init__(self, bind, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
            self.socket.bind((bind, port))
            self.socket.settimeout(.2)
        except BaseException:
            self.socket.close()
            raise
        self.routes = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def subscribe(self, source_ip, source_port, on_drop):
        channel = queue.Queue(maxsize=128)
        with self.lock:
            self.routes.append((source_ip, source_port, channel, on_drop))
        return channel

    def unsubscribe(self, channel):
        with self.lock:
            self.routes = [row for row in self.routes if row[2] is not channel]

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name='sensor_udp_demux')
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                payload, peer = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            received_ns = time.monotonic_ns()
            with self.lock:
                routes = tuple(self.routes)
            for ip, port, channel, on_drop in routes:
                if (ip is None or peer[0] == ip) and (port is None or peer[1] == port):
                    try:
                        channel.put_nowait((payload, received_ns))
                    except queue.Full:
                        on_drop('shared_queue_drops')
                    break

    def close(self):
        self.stop.set()
        self.socket.close()
        if self.thread is not None:
            self.thread.join(timeout=1.)
