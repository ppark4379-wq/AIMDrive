"""VE detections -> planning/control over loopback UDP (JSON).

The wire is the existing ``morai_ve_udp_objects_v2`` contract of VLDrive
``inference/ve_objects_udp.py``: packets from either side decode with the
same checks. Timestamps are host monotonic nanoseconds, so the age checks
hold only when sender and receiver share one clock; the target is loopback.
"""
import ipaddress
import json
import math
import socket
import time
import uuid
from dataclasses import dataclass

from .Common import _Receiver

SCHEMA = 'morai_ve_udp_objects_v2'
DEFAULT_PORT = 12050
MAX_PACKET_BYTES = 60000
MAX_AGE_S = 0.5
MAX_BOXES = 256
MAX_BOX_SIZE_M = 30.0
MAX_SPEED_MPS = 100.0
OUTPUT_FRAME = 'projected_grid'
SCENE_ASSUMPTION = 'unknown'
GPS_POSITION = 'GPS'
DR_POSITION = 'GPS_dead_reckoning'
UDP_STATUS = 'UDP_status_speed_IMU_heading'


class NotReady(ValueError):
    """The sender declared the packet unusable; ``reason`` says why."""

    def __init__(self, reason, generation, sequence):
        super().__init__('VE not ready: ' + str(reason))
        self.reason, self.generation, self.sequence = reason, generation, sequence


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ': expected a finite number')
    return float(value)


def _monotonic_ns(value, name):
    if type(value) is not int:
        raise ValueError(name + ': expected integer monotonic nanoseconds')
    return value


@dataclass(frozen=True)
class Box:
    """One class-agnostic BEV footprint in map (projected grid) coordinates."""
    x: float
    y: float
    heading_deg: float
    width_m: float
    length_m: float
    speed_mps: float = None  # VE scalar speed; None when the head abstains

    def to_wire(self):
        width, length = _finite(self.width_m, 'box.width_m'), _finite(self.length_m, 'box.length_m')
        if not 0 < width <= MAX_BOX_SIZE_M or not 0 < length <= MAX_BOX_SIZE_M:
            raise ValueError('box: footprint must be within 0..%g m' % MAX_BOX_SIZE_M)
        speed = None if self.speed_mps is None else _finite(self.speed_mps, 'box.speed_mps')
        if speed is not None and not 0 <= speed <= MAX_SPEED_MPS:
            raise ValueError('box.speed_mps: expected 0..%g' % MAX_SPEED_MPS)
        return dict(center_world=[_finite(self.x, 'box.x'), _finite(self.y, 'box.y')],
                    heading_deg=_finite(self.heading_deg, 'box.heading_deg'),
                    width_m=width, length_m=length, speed_mps=speed)

    @classmethod
    def from_wire(cls, row):
        center = row['center_world']
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError('box.center_world: expected [x, y]')
        box = cls(center[0], center[1], row['heading_deg'], row['width_m'], row['length_m'],
                  row.get('speed_mps'))
        box.to_wire()
        return box


@dataclass(frozen=True)
class EgoPose:
    """Pose used to place the boxes (perception time, not the latest control pose)."""
    x: float
    y: float
    yaw_raw_deg: float   # sensor heading, true ENU degrees
    yaw_grid_rad: float  # heading on the projected map grid
    speed_mps: float

    def to_wire(self):
        return {key: _finite(getattr(self, key), 'ego_pose.' + key)
                for key in ('x', 'y', 'yaw_raw_deg', 'yaw_grid_rad', 'speed_mps')}

    @classmethod
    def from_wire(cls, row):
        if not isinstance(row, dict):
            raise ValueError('ego_pose: expected an object')
        pose = cls(*(row[key] for key in ('x', 'y', 'yaw_raw_deg', 'yaw_grid_rad', 'speed_mps')))
        pose.to_wire()
        return pose


def _signal(value, name):
    """``{'code': int|None, 'confidence': float|None}``; None means no head output."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(name + ': expected an object')
    code, confidence = value.get('code'), value.get('confidence')
    if code is not None and type(code) is not int:
        raise ValueError(name + '.code: expected an integer MORAI status or null')
    if confidence is not None:
        confidence = _finite(confidence, name + '.confidence')
        if not 0 <= confidence <= 1:
            raise ValueError(name + '.confidence: expected 0..1')
    return dict(code=code, confidence=confidence)


@dataclass(frozen=True)
class Detection:
    generation: str
    sequence: int
    checkpoint_sha256: str
    sent_ns: int
    sensor_received_ns: int
    ego_pose: EgoPose
    boxes: tuple
    ego_position_source: str
    ego_status_source: str
    signal: dict = None
    native_signal: dict = None
    position_identity: dict = None

    def sensor_age_s(self, now_ns=None):
        return ((time.monotonic_ns() if now_ns is None else now_ns) - self.sensor_received_ns) / 1e9


def encode(packet):
    data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('Detection packet exceeds %d bytes' % MAX_PACKET_BYTES)
    return data


def decode(data, now_ns=None, *, generation=None, checkpoint_sha256=None, max_age_s=MAX_AGE_S):
    """Validated Detection; NotReady for a declared-unusable packet, ValueError otherwise.

    ``generation``/``checkpoint_sha256`` pin the expected sender when given.
    """
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('Oversize detection datagram')
    packet = json.loads(data)
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
        raise ValueError('Not a %s packet' % SCHEMA)
    sender, digest = packet.get('generation'), packet.get('checkpoint_sha256')
    if not isinstance(sender, str) or not sender or not isinstance(digest, str) or not digest:
        raise ValueError('Missing sender generation or checkpoint identity')
    if ((generation is not None and sender != generation)
            or (checkpoint_sha256 is not None and digest != checkpoint_sha256)):
        raise ValueError('Detection sender identity mismatch')
    if packet.get('output_frame') != OUTPUT_FRAME or packet.get('scene_assumption') != SCENE_ASSUMPTION:
        raise ValueError('Detection frame/scene contract mismatch')
    sequence = packet.get('sequence')
    if type(sequence) is not int or sequence < 1:
        raise ValueError('Invalid detection sequence')
    sent_ns = _monotonic_ns(packet.get('sent_ns'), 'sent_ns')
    if not 0 <= now_ns - sent_ns <= max_age_s * 1e9:
        raise ValueError('Stale or future detection datagram')
    if packet.get('ready') is not True:
        raise NotReady(packet.get('reason'), sender, sequence)
    received_ns = _monotonic_ns(packet.get('sensor_received_ns'), 'sensor_received_ns')
    if not 0 < received_ns <= sent_ns:
        raise ValueError('Invalid sensor receipt timestamp')
    if now_ns - received_ns > max_age_s * 1e9:
        raise ValueError('Detection sensor snapshot expired')
    boxes = packet.get('boxes')
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError('Invalid or too many detection boxes')
    sources = packet.get('ego_position_source'), packet.get('ego_status_source')
    if not all(isinstance(value, str) and value for value in sources):
        raise ValueError('Missing ego pose sources')
    position_identity = (packet.get('selected_samples') or {}).get('gps')
    if sources[0] == DR_POSITION and (not isinstance(position_identity, dict)
                                    or position_identity.get('position_mode') != 'dead_reckoning'):
        raise ValueError('DR detection requires explicit position provenance')
    return Detection(sender, sequence, digest, sent_ns, received_ns,
                     EgoPose.from_wire(packet.get('ego_pose')),
                     tuple(Box.from_wire(row) for row in boxes), *sources,
                     _signal(packet.get('signal'), 'signal'),
                     _signal(packet.get('native_signal'), 'native_signal'), position_identity)


class DetectionSender:
    """Send VE boxes, perception pose and signal heads to a planning process."""

    def __init__(self, checkpoint_sha256, host='127.0.0.1', port=DEFAULT_PORT, *,
                 generation=None, max_age_s=MAX_AGE_S):
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            raise ValueError('checkpoint_sha256 identifies the VE weights and is required')
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError('Detection target must be loopback: monotonic timestamps are per host')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid detection port')
        self.target = (host, port)
        self.checkpoint_sha256 = checkpoint_sha256
        self.generation = generation or str(uuid.uuid4())
        self.max_age_ns = int(max_age_s * 1e9)
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _header(self, ready, reason):
        self.sequence += 1
        return dict(schema=SCHEMA, generation=self.generation, sequence=self.sequence,
                    checkpoint_sha256=self.checkpoint_sha256, output_frame=OUTPUT_FRAME,
                    scene_assumption=SCENE_ASSUMPTION, ready=ready, reason=reason,
                    sensor_received_ns=0, boxes=[], ego_pose=None)

    def _send(self, packet):
        packet['datagram_sequence'] = self.sequence
        packet['sent_ns'] = time.monotonic_ns()
        return self.socket.sendto(encode(packet), self.target)

    def send(self, boxes, ego_pose, sensor_received_at, *, signal=None, native_signal=None,
             ego_position_source=GPS_POSITION, ego_status_source=UDP_STATUS, position_identity=None):
        """``sensor_received_at``: time.monotonic() receipt of the oldest sensor used.

        Inputs that expired during inference are sent as not ready, never as boxes.
        """
        received_ns = int(round(_finite(sensor_received_at, 'sensor_received_at') * 1e9))
        wire_boxes = [box.to_wire() for box in boxes]
        if len(wire_boxes) > MAX_BOXES:
            raise ValueError('At most %d detection boxes' % MAX_BOXES)
        wire_pose = ego_pose.to_wire()
        signals = _signal(signal, 'signal'), _signal(native_signal, 'native_signal')
        if not all(isinstance(value, str) and value for value in (ego_position_source, ego_status_source)):
            raise ValueError('Ego pose sources must be named')
        if ego_position_source == DR_POSITION and (not isinstance(position_identity, dict)
                or position_identity.get('position_mode') != 'dead_reckoning'):
            raise ValueError('DR detection requires explicit position provenance')
        now_ns = time.monotonic_ns()
        if not 0 < received_ns <= now_ns:
            raise ValueError('sensor_received_at must be a past time.monotonic() value')
        if now_ns - received_ns > self.max_age_ns:
            return self._send(self._header(False, 'sensors_expired_during_inference'))
        packet = self._header(True, None)
        packet.update(sensor_received_ns=received_ns, boxes=wire_boxes, ego_pose=wire_pose,
                      ego_position_source=ego_position_source, ego_status_source=ego_status_source,
                      signal=signals[0], native_signal=signals[1])
        if position_identity is not None:
            packet['selected_samples'] = dict(gps=dict(position_identity))
        return self._send(packet)

    def send_not_ready(self, reason):
        """Tell the planner this cycle has no usable perception (e.g. missing sensors)."""
        return self._send(self._header(False, str(reason)))

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DetectionReceiver(_Receiver):
    """Latest valid detection, and control pose, on one loopback port.

    Both streams share the port (as the Expert GPS runner expects), so one
    socket must read them: two sockets on a port would split the datagrams.
    Each stream keeps its own sequence order. Pin ``generation`` to one sender
    when another publisher could share the port; otherwise a new generation
    restarts the sequence check.
    """

    def __init__(self, port=DEFAULT_PORT, bind='127.0.0.1', *, generation=None,
                 checkpoint_sha256=None, max_age_s=MAX_AGE_S, **kwargs):
        self.expected_generation = generation
        self.expected_checkpoint = checkpoint_sha256
        self.max_age_s = max_age_s
        self._order = {}
        self.ready = False
        self.reason = 'waiting_for_detection'
        self._pose = None
        self.pose_ready = False
        self.pose_reason = 'waiting_for_control_pose'
        super().__init__(port, bind=bind, recv_bytes=MAX_PACKET_BYTES + 1, **kwargs)

    def _in_order(self, generation, sequence, stream='detection'):
        last_generation, last_sequence = self.__dict__.setdefault('_order', {}).get(stream, (None, 0))
        if generation != last_generation:
            if last_generation is not None:
                self._count(stream + '_generation_changes')
            last_sequence = 0
        if sequence <= last_sequence:
            self._count(stream + '_out_of_order')
            return False
        self._order[stream] = (generation, sequence)
        return True

    def _consume_pose(self, payload):
        from . import Pose
        try:
            pose = Pose.decode_pose(payload, time.monotonic_ns(), generation=self.expected_generation,
                                    checkpoint_sha256=self.expected_checkpoint)
        except NotReady as exc:
            if self._in_order(exc.generation, exc.sequence, 'pose'):
                self.pose_ready, self.pose_reason = False, exc.reason
                self._count('pose_not_ready')
            return
        except (ValueError, KeyError, TypeError) as exc:
            self._count('pose_rejected')
            self.pose_reason = 'rejected: ' + str(exc)
            return
        if self._in_order(pose.generation, pose.sequence, 'pose'):
            self._pose, self.pose_ready, self.pose_reason = pose, True, None

    def fresh_pose(self, now_ns=None):
        """Latest ready control pose whose sensors are within its 0.15 s limit, else None."""
        from .Pose import MAX_AGE_S as POSE_MAX_AGE_S
        pose = self._pose
        if pose is None or not self.pose_ready or pose.sensor_age_s(now_ns) > POSE_MAX_AGE_S:
            return None
        return pose

    def _consume(self, payload):
        try:
            schema = json.loads(payload).get('schema')
        except (ValueError, AttributeError):
            schema = None
        from .Pose import SCHEMA as POSE_SCHEMA
        if schema == POSE_SCHEMA:
            self._consume_pose(payload)
            return None
        try:
            detection = decode(payload, time.monotonic_ns(), generation=self.expected_generation,
                               checkpoint_sha256=self.expected_checkpoint, max_age_s=self.max_age_s)
        except NotReady as exc:
            if self._in_order(exc.generation, exc.sequence):
                self.ready, self.reason = False, exc.reason
                self._count('not_ready')
            return None
        except (ValueError, KeyError, TypeError) as exc:
            self._count('rejected')
            self.reason = 'rejected: ' + str(exc)
            return None
        if not self._in_order(detection.generation, detection.sequence):
            return None
        self.ready, self.reason = True, None
        return detection

    def fresh(self, now_ns=None):
        """Latest ready detection whose sensors are within max_age_s, else None."""
        detection, _ = self.latest()
        if detection is None or not self.ready or detection.sensor_age_s(now_ns) > self.max_age_s:
            return None
        return detection
