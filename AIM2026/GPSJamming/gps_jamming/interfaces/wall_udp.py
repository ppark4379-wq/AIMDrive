"""ROS-independent transport of the exact RViz wall geometry (metres, FLU)."""
import json
import math
import socket
import time
import uuid

SCHEMA = 'gps_jamming_wall_display_v1'
PORT = 12051
MAX_BYTES = 60000


def line_edges(boundaries, side):
    """LINE_LIST pairs, preserving segment breaks and every original coordinate."""
    return [[a, b] for segment in boundaries[side]
            for a, b in zip(segment['points'], segment['points'][1:])]


def decode_packet(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Wall datagram too large')
    packet = json.loads(data)
    if packet['schema'] != SCHEMA or packet['units'] != 'm' or packet['axes'] != 'FLU':
        raise ValueError('Unsupported wall contract')
    if not isinstance(packet['session'], str) or not packet['session']:
        raise ValueError('Missing sender session')
    if type(packet['sequence']) is not int or packet['sequence'] < 0:
        raise ValueError('Invalid sequence')
    boundary = packet['boundaries']
    if boundary['frame_id'] != 'base_link':
        raise ValueError('Expected base_link geometry')
    for side in ('left', 'right'):
        for segment in boundary[side]:
            for point in segment['points']:
                if len(point) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in point):
                    raise ValueError('Expected finite XYZ')
    return packet


class WallUDPSender:
    def __init__(self, host='127.0.0.1', port=PORT):
        self.destination = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = str(uuid.uuid4())
        self.sequence = 0

    def send(self, display):
        packet = dict(schema=SCHEMA, session=self.session, sequence=self.sequence,
                      sent_unix_ns=time.time_ns(), units='m', axes='FLU',
                      geometry_source='rviz_wall_markers', boundaries=display)
        self.sequence += 1
        data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode('utf-8')
        decode_packet(data)
        self.socket.sendto(data, self.destination)

    def close(self):
        self.socket.close()


class WallUDPReceiver:
    """Call poll each planning cycle; None means absent/expired input, not free road.

    Freshness uses local arrival time; source scan timestamps remain in boundaries.
    This is not a timestamp synchronizer or a world-coordinate conversion.
    """
    def __init__(self, bind='127.0.0.1', port=PORT, max_age_s=0.5):
        if max_age_s <= 0:
            raise ValueError('max_age_s must be positive')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((bind, port))
        self.socket.setblocking(False)
        self.max_age_s = max_age_s
        self.packet = None
        self.received = 0.0
        self.retired_sessions = set()
        self.rejected = 0

    def poll(self):
        # Bound work per control cycle even if an input floods the socket.
        for _ in range(256):
            try:
                data, _ = self.socket.recvfrom(MAX_BYTES + 1)
            except BlockingIOError:
                break
            try:
                packet = decode_packet(data)
                if packet['session'] in self.retired_sessions:
                    raise ValueError('Retired sender')
                if self.packet is not None:
                    if packet['session'] == self.packet['session']:
                        if packet['sequence'] <= self.packet['sequence']:
                            raise ValueError('Duplicate or out-of-order frame')
                    else:
                        self.retired_sessions.add(self.packet['session'])
                self.packet, self.received = packet, time.monotonic()
            except (ValueError, KeyError, TypeError, OverflowError):
                self.rejected += 1
        if time.monotonic() - self.received > self.max_age_s:
            return None
        return self.packet

    def close(self):
        self.socket.close()
