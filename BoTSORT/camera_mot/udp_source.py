"""One UDP socket/assembler per camera, with bounded fair frame scheduling."""
import selectors
import socket
import threading
import time

from camera_mot.streams import CameraBuffers
from tools.morai_udp import JpegAssembler


class UdpCameraSource(CameraBuffers):
    def __init__(self, cameras, bind_ip='0.0.0.0', sender_ip=None, index_base=0):
        super().__init__(cameras, 'udp', 'morai_udp_header_raw')
        self.selector = selectors.DefaultSelector()
        self.sockets = {}
        self.assemblers = {key: JpegAssembler(index_base=index_base) for key in cameras}
        self.assembly_lock = threading.Lock()
        self.sender_ip = sender_ip
        self.stop_event = threading.Event()
        self.error = None
        self.packet_counts = {key: 0 for key in cameras}
        self.rejected_senders = {key: 0 for key in cameras}
        try:
            for key, config in cameras.items():
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets[key] = sock
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                sock.bind((bind_ip, config['udp_port']))
                sock.setblocking(False)
                self.selector.register(sock, selectors.EVENT_READ, key)
        except Exception:
            self.selector.close()
            for sock in self.sockets.values():
                sock.close()
            raise
        self.addresses = {key: sock.getsockname() for key, sock in self.sockets.items()}
        self.thread = threading.Thread(target=self._receive, name='camera-udp', daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stop_event.is_set():
                # One datagram per ready socket avoids starvation by a busy camera.
                for event, _ in self.selector.select(timeout=0.05):
                    try:
                        packet, sender = event.fileobj.recvfrom(65535)
                    except BlockingIOError:
                        continue
                    key = event.data
                    with self.assembly_lock:
                        if self.sender_ip and sender[0] != self.sender_ip:
                            self.rejected_senders[key] += 1
                            continue
                        self.packet_counts[key] += 1
                        received_ns = time.monotonic_ns()
                        frame = self.assemblers[key].feed(packet, received_ns / 1e9)
                        if frame is not None:
                            self.put(self.cameras[key]['frame_id'], *frame.stamp,
                                     frame.jpeg, received_ns)
                with self.assembly_lock:
                    for assembler in self.assemblers.values():
                        assembler.expire(time.monotonic())
        except Exception as exc:
            self.error = exc
            # Keep the source consumable until get() reports the worker failure.
            # Marking it closed here lets a consumer's `while not closed` loop
            # mistake an asynchronous receiver failure for successful completion.
            with self.condition:
                self.condition.notify_all()

    def get(self, timeout=0.1):
        frame = super().get(timeout)
        if self.error:
            raise RuntimeError('UDP receive worker failed') from self.error
        return frame

    def reset(self, camera):
        with self.assembly_lock:
            assembler = self.assemblers[camera]
            assembler.incomplete_frames += len(assembler.pending)
            assembler.pending.clear()
            assembler.completed.clear()
            super().reset(camera)

    def stats(self, now_ns=None):
        # Match the worker's lock order (assembler -> frame buffer).
        with self.assembly_lock:
            result = super().stats(now_ns)
            for key, assembler in self.assemblers.items():
                result[key].update(datagrams=self.packet_counts[key],
                    rejected_sender_packets=self.rejected_senders[key],
                    invalid_packets=assembler.invalid_packets,
                    incomplete_frames=assembler.incomplete_frames,
                    pending_assemblies=len(assembler.pending))
            return result

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.selector.close()
        for sock in self.sockets.values():
            sock.close()
        super().close()
