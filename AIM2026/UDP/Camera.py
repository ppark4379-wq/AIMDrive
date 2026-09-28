import struct
from .Common import _Receiver, _Packet

CAMERA_HEADER = b"MOR"
CAMERA_BBOX_HEADER = b"BOX"
CAMERA_FRAGMENT_BYTES = 65000
CAMERA_JPEG_CAPACITY = 64979
CAMERA_PREFIX = struct.Struct("<3s4i")  

class CameraReceiver(_Receiver):
    def __init__(self, port, **kwargs):
        self._buffer = bytearray()
        self._expect_index = 0
        self._intact = False
        self._frame_stamp = None
        self.torn_images = 0
        super().__init__(port, recv_bytes=CAMERA_FRAGMENT_BYTES + 64, **kwargs)

    def _consume(self, payload):
        if len(payload) < CAMERA_PREFIX.size + 2:
            return None
        header, sec, nsec, index, size = CAMERA_PREFIX.unpack_from(payload, 0)
        if header != CAMERA_HEADER:
            return None 
        frame_stamp = (sec, nsec)
        if frame_stamp != self._frame_stamp:
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            self._frame_stamp = frame_stamp
        tail = payload[-2:]
        body = payload[CAMERA_PREFIX.size:-2]
        take = min(max(int(size), 0), len(body)) if size else len(body)
        if index == 0:
            self._buffer = bytearray()
            self._intact = True
        elif index != self._expect_index:
            self._intact = False 
        self._buffer += body[:take]
        self._expect_index = index + 1
        if tail == b"EI":
            image = bytes(self._buffer) if self._intact else None
            if image is None:
                self.torn_images += 1
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            return image
        return None
