"""
Generic non-blocking UDP receiver. One instance per MORAI topic/port.
Keeps only the most recent parsed sample -- for frame-sync purposes we
only need "what do we know right now", not packet history.
"""
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any


@dataclass
class Sample:
    data: Any
    recv_time: float


class UdpTopicReceiver:
    def __init__(self, name: str, ip: str, port: int, buf_size: int,
                 parse_fn: Callable[[bytes], Any]):
        self.name = name
        self.port = port
        self.buf_size = buf_size
        self.parse_fn = parse_fn

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((ip, port))
        self._sock.settimeout(0.5)

        self._lock = threading.Lock()
        self._latest: Optional[Sample] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "UdpTopicReceiver":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw, _addr = self._sock.recvfrom(self.buf_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                parsed = self.parse_fn(raw)
            except Exception as e:
                print(f"[{self.name}] parse error: {e}")
                continue
            with self._lock:
                self._latest = Sample(data=parsed, recv_time=time.time())

    @property
    def latest(self) -> Optional[Sample]:
        with self._lock:
            return self._latest
