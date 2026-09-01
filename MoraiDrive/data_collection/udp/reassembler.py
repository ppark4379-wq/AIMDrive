"""
Reassembles a large payload (camera JPEG / lidar point cloud) that arrives
as multiple UDP chunks, since a full image/point cloud rarely fits in one
~1500-byte datagram.

TODO: HEADER_FORMAT/HEADER_SIZE below is a PLACEHOLDER layout
(frame_id: uint32, chunk_idx: uint16, chunk_total: uint16). Swap it for
MORAI's actual per-packet chunk header once you have the spec -- the
buffering/completion-detection logic below doesn't depend on the exact
header bytes, only on knowing frame_id/chunk_idx/chunk_total per packet.

Only needed if MORAI actually fragments camera/lidar data itself over UDP.
If it instead hands you one full payload per socket.recvfrom() call (many
sim bridges do, since OS-level IP fragmentation handles reassembly up to
~64KB transparently), you don't need this at all -- just feed raw bytes
straight into udp/parsers.py.
"""
import struct
from collections import defaultdict
from typing import Dict, Optional, Tuple

HEADER_FORMAT = "<IHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class ChunkReassembler:
    def __init__(self):
        self._buffers: Dict[int, Dict[int, bytes]] = defaultdict(dict)
        self._totals: Dict[int, int] = {}

    def feed(self, raw: bytes) -> Optional[Tuple[int, bytes]]:
        """Returns (frame_id, complete_payload) once a frame is fully assembled."""
        frame_id, chunk_idx, chunk_total = struct.unpack(
            HEADER_FORMAT, raw[:HEADER_SIZE]
        )
        payload = raw[HEADER_SIZE:]

        self._totals[frame_id] = chunk_total
        self._buffers[frame_id][chunk_idx] = payload

        if len(self._buffers[frame_id]) == chunk_total:
            chunks = self._buffers.pop(frame_id)
            self._totals.pop(frame_id)
            full = b"".join(chunks[i] for i in range(chunk_total))
            return frame_id, full
        return None
