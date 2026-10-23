import struct
from .Common import _Receiver, _Packet

CAMERA_HEADER = b"MOR"
CAMERA_BBOX_HEADER = b"BOX"
CAMERA_FRAGMENT_BYTES = 65000
CAMERA_JPEG_CAPACITY = 64979
CAMERA_PREFIX = struct.Struct("<3s4i")  # header, sec, nsec, index, size
CAMERA_MAX_IMAGE_BYTES = 16 * 1024 * 1024
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

class CameraReceiver(_Receiver):
    """Reassemble one JPEG from 'MOR' fragments; never emit a torn image.

    Each fragment carries its own payload length in ``size``. A fragment whose
    size/index cannot be honoured is not repaired: the bytes it lacks are not
    recoverable, and a guessed JPEG either fails to decode or reaches the
    encoder corrupted. The whole frame is dropped and the next one is used.
    """

    def __init__(self, port, **kwargs):
        self._buffer = bytearray()
        self._expect_index = 0
        self._intact = False
        self._frame_stamp = None
        self.torn_images = 0
        super().__init__(port, recv_bytes=CAMERA_FRAGMENT_BYTES + 64, **kwargs)

    def source_time(self, sample):
        sec, nsec = self._frame_stamp
        return sec + nsec / 1e9

    def _discard(self, reason):
        self._count(reason)
        self._buffer = bytearray()
        self._expect_index = 0
        self._intact = False

    def _consume(self, payload):
        if len(payload) < CAMERA_PREFIX.size + 2:
            self._count('short_fragments')
            return None
        header, sec, nsec, index, size = CAMERA_PREFIX.unpack_from(payload, 0)
        if header != CAMERA_HEADER:
            return None  # 'BOX' bounding boxes are not used here
        tail = payload[-2:]
        body = payload[CAMERA_PREFIX.size:-2]
        if index < 0 or size <= 0 or size > len(body):
            self._discard('invalid_size_or_index')
            return None
        frame_stamp = (sec, nsec)
        if frame_stamp != self._frame_stamp:
            if self._buffer:
                self._count('unfinished_frames_replaced')
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            self._frame_stamp = frame_stamp
        if index == 0:
            self._buffer = bytearray()
            self._intact = True
        elif index != self._expect_index:
            self._count('fragment_index_gaps')
            self._intact = False
        if self._intact and len(self._buffer) + size > CAMERA_MAX_IMAGE_BYTES:
            self._count('oversize_images')
            self._intact = False
        if self._intact:
            self._buffer += body[:size]
        else:
            self._buffer = bytearray()
        self._expect_index = index + 1
        if tail == b"EI":
            image = bytes(self._buffer) if self._intact else None
            if image is not None and not (image.startswith(JPEG_SOI) and image.endswith(JPEG_EOI)):
                self._count('jpeg_marker_errors')
                image = None
            if image is None:
                self.torn_images += 1
                self._count('discarded_torn_images')
            else:
                self._count('complete_images')
            self._buffer = bytearray()
            self._expect_index = 0
            self._intact = False
            return image
        return None
