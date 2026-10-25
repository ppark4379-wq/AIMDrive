"""MORAI GT UDP wire formats, isolated from VE and the Expert planner.

Protocol references:
https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1
https://github.com/MORAI-Autonomous/MORAI-DriveExample_UDP/tree/main/network
"""
import math
import struct


OBJECT = struct.Struct('<hh16f')
CONTROL = struct.Struct('<14si3i3b6f2s')
CONTROL_SINGLE_STEER = struct.Struct('<14si3i3b5f2s')


def _body(data, header, timestamp_excluded_length=False):
    start = len(header) + 16
    if not data.startswith(header) or data[-2:] != b'\r\n' or len(data) < start + 2:
        raise ValueError('잘못된 UDP 헤더·길이·종료 바이트')
    size = struct.unpack_from('<i', data, len(header))[0]
    actual = len(data) - start - 2
    allowed = (actual, actual-8) if timestamp_excluded_length else (actual,)
    if size not in allowed:
        raise ValueError('UDP 선언 길이와 실제 데이터 길이가 다릅니다')
    return data[start:-2]


def _finite(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('UDP 데이터에 비정상 숫자가 있습니다')


def _stamp(body):
    sec, nsec = struct.unpack_from('<ii', body)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise ValueError('UDP 시각이 유효하지 않습니다')
    return sec * 1_000_000_000 + nsec


def parse_ego(data):
    # 229: older full GT; 245: front/rear steer and lane boundary fields.
    # Competition status (181 bytes) is deliberately not a GT pose source.
    if len(data) not in (229, 245):
        raise ValueError('Ego Vehicle Status 전체 GT(229/245 bytes)가 필요합니다; 수신={} bytes'.format(len(data)))
    body = _body(data, b'#MoraiInfo$')
    stamp_ns = _stamp(body)
    ctrl_mode, gear = struct.unpack_from('<bb', body, 8)
    speed_kmh = struct.unpack_from('<f', body, 10)[0]
    position = struct.unpack_from('<3f', body, 50)
    yaw_deg = struct.unpack_from('<f', body, 70)[0]
    accel, brake = struct.unpack_from('<2f', body, 18)
    steering = struct.unpack_from('<f', body, 110)[0]
    _finite((speed_kmh, *position, yaw_deg, accel, brake, steering))
    # The current competition build emits mode 0 while parked, before the
    # scenario enables Auto. Telemetry is valid in this state too.
    if ctrl_mode not in (0, 1, 2) or gear not in range(6):
        raise ValueError('자차 모드 또는 기어가 유효하지 않습니다')
    return dict(stamp_ns=stamp_ns, position=position, heading_deg=yaw_deg,
                speed_mps=speed_kmh / 3.6, ctrl_mode=ctrl_mode, gear=gear,
                accel=accel, brake=brake, steering=steering, layout_bytes=len(data))


def parse_objects(data):
    body = _body(data, b'#MoraiObjInfo$', timestamp_excluded_length=True)
    # The timestamp was added once per datagram in 23.R1. Records are
    # 68 bytes in older versions, 106 with the 38-byte MGeo link ID.
    if len(body) not in (8 + 20 * 68, 8 + 20 * 106):
        raise ValueError('지원하지 않는 Object Info 배열 크기: {}'.format(len(body)))
    stamp_ns = _stamp(body)
    stride = (len(body) - 8) // 20
    objects = []
    ids = set()
    for index in range(20):
        values = OBJECT.unpack_from(body, 8 + index * stride)
        actor_id, kind = values[:2]
        if actor_id == 0:  # Zero-filled, unused slots; not a missing packet.
            continue
        if kind == -1:  # Some versions include Ego in the list.
            continue
        if kind not in (0, 1, 2) or actor_id in ids:
            raise ValueError('장애물 타입 또는 중복 ID가 유효하지 않습니다')
        _finite(values[2:])
        size = values[6:9]
        if any(v < 0 for v in size):
            raise ValueError('장애물 크기는 음수일 수 없습니다')
        ids.add(actor_id)
        objects.append(dict(unique_id=actor_id, type=kind, position=values[2:5],
                            heading_deg=values[5], size=size,
                            velocity_kmh=values[12:15], acceleration=values[15:18]))
    return dict(stamp_ns=stamp_ns, objects=objects, record_bytes=stride)


def parse_signal(data):
    body = _body(data, b'#TrafficLight$')
    if len(body) != 16:
        raise ValueError('신호등 데이터는 16 bytes여야 합니다')
    signal_id = body[:12].split(b'\0', 1)[0].decode('ascii').strip()
    kind, status = struct.unpack_from('<hh', body, 12)
    # Preserve newer signal types and combined status bits for the existing
    # Expert policy; the transport must not invent a red/green interpretation.
    if not signal_id or kind < 0 or status < -1:
        raise ValueError('신호등 ID·타입·상태가 유효하지 않습니다')
    return dict(signal_id=signal_id, type=kind, status=status)


def pack_control(accel, brake, front_steer_rad=0., rear_steer_rad=0., layout='single_steer55'):
    """Encode supplied wire values without an implicit unit conversion.

    Steering argument names retain the earlier API; the UDP wire takes
    normalized values. Convert planner radians with normalized_steering first.
    The default single-steer layout was verified in both MORAI builds.
    """
    _finite((accel, brake, front_steer_rad, rear_steer_rad))
    if not 0 <= accel <= 1 or not 0 <= brake <= 1:
        raise ValueError('페달 입력은 0~1이어야 합니다')
    if layout == 'single_steer55':
        if rear_steer_rad != 0:
            raise ValueError('55-byte 형식에서는 후륜 조향을 보낼 수 없습니다')
        return CONTROL_SINGLE_STEER.pack(b'#MoraiCtrlCmd$', 23, 0, 0, 0, 2, 4, 1,
                                         0., 0., accel, brake, front_steer_rad, b'\r\n')
    if layout != 'front_rear59':
        raise ValueError('지원하지 않는 UDP 제어 형식입니다')
    return CONTROL.pack(b'#MoraiCtrlCmd$', 27, 0, 0, 0, 2, 4, 1,
                        0., 0., accel, brake, front_steer_rad, rear_steer_rad, b'\r\n')


def normalized_steering(front_steer_rad, max_steer_rad):
    """MORAI UDP steering wire = desired wheel angle / physical angle limit.

    The legacy pack_control entry point encodes supplied wire values unchanged.
    A planner producing radians must convert them before using that encoder.
    """
    _finite((front_steer_rad, max_steer_rad))
    if max_steer_rad <= 0:
        raise ValueError('최대 조향각은 양수여야 합니다')
    return max(-1., min(1., front_steer_rad / max_steer_rad))
