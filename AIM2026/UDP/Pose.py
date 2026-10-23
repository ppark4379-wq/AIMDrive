"""Control pose (GPS position, IMU heading, status speed) -> planning/control.

Competition rules 2-6-2 / 2-8 fix the sources:
  position  GPS sensor (UDP); Competition Vehicle Status withholds pos_x/y/z
  heading   IMU sensor (UDP) quaternion, true ENU, converted to the map grid
  speed     Competition Vehicle Status (UDP) signed velocity, km/h / 3.6
Status attitude and any simulator GT are never used.

The wire is VLDrive's ``morai_ve_control_pose_v1`` (utils/common/ve_control_pose.py),
which the Expert GPS runner reads beside detection packets on the same port.
Both streams must carry the same sender generation and checkpoint identity.
Timestamps are host monotonic nanoseconds, so the target is loopback.
"""
import ipaddress
import json
import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from .Detection import NotReady, _finite, _monotonic_ns, OUTPUT_FRAME, DEFAULT_PORT, GPS_POSITION, UDP_STATUS

SCHEMA = 'morai_ve_control_pose_v1'
MAX_AGE_S = 0.15
MAX_PACKET_BYTES = 8000
DEFAULT_HZ = 50.0  # IMU maximum under rule 2-6-2
MAX_SPEED_MPS = 100.0
# Projected map basis of the competition route (UTM 52N minus the map origin).
MAP_CRS = '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs'
MAP_ORIGIN_XYZ = (302595.0, 4124145.0, 0.0)


def _quaternion_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not 0.9 < norm < 1.1:
        raise ValueError('IMU quaternion is not a rotation')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _euler_xyz_matrix(roll_deg, pitch_deg, yaw_deg):
    """Extrinsic x-y-z (sensor mounting rot_deg), i.e. Rz @ Ry @ Rx."""
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


class GridProjector:
    """GPS fix + IMU orientation -> vehicle pose on the projected map grid.

    ``gps_mount``/``imu_mount_deg`` are the sensor ``pos``/``rot_deg`` from the
    sensor setting; with zero mounts this equals VLDrive's control pose.
    """

    def __init__(self, gps_mount=(0., 0., 0.), imu_mount_deg=(0., 0., 0.),
                 crs=MAP_CRS, origin_xyz=MAP_ORIGIN_XYZ):
        from pyproj import CRS, Proj, Transformer
        self.crs = CRS.from_user_input(crs)
        if not self.crs.is_projected:
            raise ValueError('A projected map CRS is required')
        self.to_map = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        self.proj = Proj(self.crs)
        self.origin = np.asarray(origin_xyz, dtype=float)
        self.gps_mount = np.asarray(gps_mount, dtype=float)
        self.imu_mount = _euler_xyz_matrix(*imu_mount_deg)
        if self.origin.shape != (3,) or self.gps_mount.shape != (3,) or not (
                np.isfinite(self.origin).all() and np.isfinite(self.gps_mount).all()):
            raise ValueError('Finite map origin and GPS mount required')

    def project(self, gps, imu):
        """dict(x, y, yaw_raw_rad, yaw_grid_rad) of the vehicle reference point."""
        lat, lon = gps['latitude'], gps['longitude']
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Invalid GPS fix')
        east, north = self.to_map.transform(lon, lat)
        position = np.array([east, north, gps.get('altitude') or 0.]) - self.origin
        body = _quaternion_matrix(*imu['quaternion']) @ self.imu_mount.T
        yaw_raw = math.atan2(body[1, 0], body[0, 0])
        convergence = self.proj.get_factors(lon, lat).meridian_convergence
        if not math.isfinite(convergence):
            raise ValueError('Projection convergence unavailable')
        c = math.radians(convergence)
        grid = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]]) @ body
        position = position - grid @ self.gps_mount
        return dict(x=float(position[0]), y=float(position[1]), yaw_raw_rad=yaw_raw,
                    yaw_grid_rad=math.atan2(grid[1, 0], grid[0, 0]))


@dataclass(frozen=True)
class ControlPose:
    generation: str
    sequence: int
    sent_ns: int
    sensor_received_ns: int
    x: float
    y: float
    yaw_raw_deg: float
    yaw_grid_rad: float
    speed_mps: float          # signed; negative while reversing
    ego_control: dict = None  # accel/brake/steer/gear the vehicle reports

    def sensor_age_s(self, now_ns=None):
        return ((time.monotonic_ns() if now_ns is None else now_ns) - self.sensor_received_ns) / 1e9


def decode_pose(data, now_ns=None, *, generation=None, checkpoint_sha256=None, max_age_s=MAX_AGE_S):
    """Validated ControlPose; NotReady for a declared-unusable packet, ValueError otherwise.

    Mirrors VLDrive ``decode_control_pose``: every sensor receipt must be fresh,
    and the oldest of them must be the reported ``sensor_received_ns``.
    """
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('Oversize control pose datagram')
    packet = json.loads(data)
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
        raise ValueError('Not a %s packet' % SCHEMA)
    sender, digest = packet.get('generation'), packet.get('checkpoint_sha256')
    if not isinstance(sender, str) or not sender or not isinstance(digest, str) or not digest:
        raise ValueError('Missing sender generation or checkpoint identity')
    if ((generation is not None and sender != generation)
            or (checkpoint_sha256 is not None and digest != checkpoint_sha256)):
        raise ValueError('Control pose sender identity mismatch')
    if (packet.get('output_frame') != OUTPUT_FRAME or packet.get('ego_position_source') != GPS_POSITION
            or packet.get('ego_status_source') != UDP_STATUS):
        raise ValueError('Control pose frame/source mismatch')
    sequence = packet.get('sequence')
    if type(sequence) is not int or sequence < 1:
        raise ValueError('Invalid control pose sequence')
    sent_ns = _monotonic_ns(packet.get('sent_ns'), 'sent_ns')
    if not 0 <= now_ns - sent_ns <= max_age_s * 1e9:
        raise ValueError('Control pose delivery expired')
    if packet.get('ready') is not True:
        raise NotReady(packet.get('reason'), sender, sequence)
    received_ns = _monotonic_ns(packet.get('sensor_received_ns'), 'sensor_received_ns')
    if not 0 < received_ns <= sent_ns or now_ns - received_ns > max_age_s * 1e9:
        raise ValueError('Control pose sensors expired')
    pose = packet.get('ego_pose')
    if not isinstance(pose, dict):
        raise ValueError('Missing ego_pose')
    values = {key: _finite(pose.get(key), 'ego_pose.' + key)
              for key in ('x', 'y', 'yaw_raw_deg', 'yaw_grid_rad', 'speed_mps')}
    if abs(values['speed_mps']) > MAX_SPEED_MPS:
        raise ValueError('Invalid control speed')
    samples = packet.get('selected_samples') or {}
    receipts = [samples.get('gps', {}).get('received_ns'),
                samples.get('vehicle_status', {}).get('received_ns'),
                samples.get('vehicle_status', {}).get('imu_received_ns')]
    if any(type(stamp) is not int or not received_ns <= stamp <= sent_ns for stamp in receipts):
        raise ValueError('Invalid control pose sample receipts')
    if min(receipts) != received_ns:
        raise ValueError('Control pose oldest sample mismatch')
    return ControlPose(sender, sequence, sent_ns, received_ns, **values,
                       ego_control=packet.get('ego_control'))


class PoseSender:
    """Sample the latest GPS/IMU/status at a fixed rate and send the control pose.

    The stream is independent of the VE loop: control uses the newest pose,
    while detections keep the pose they were transformed with.
    """

    def __init__(self, gps, imu, status, projector, checkpoint_sha256, generation,
                 host='127.0.0.1', port=DEFAULT_PORT, *, hz=DEFAULT_HZ, max_age_s=MAX_AGE_S):
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            raise ValueError('checkpoint_sha256 is required and must match the detection sender')
        if not isinstance(generation, str) or not generation:
            raise ValueError('generation is required and must match the detection sender')
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError('Control pose target must be loopback: monotonic timestamps are per host')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid control pose port')
        if not math.isfinite(hz) or not 1 <= hz <= 100:
            raise ValueError('Control pose rate must be 1..100 Hz')
        self.gps, self.imu, self.status, self.projector = gps, imu, status, projector
        self.checkpoint_sha256, self.generation = checkpoint_sha256, generation
        self.target, self.period, self.max_age_s = (host, port), 1.0 / hz, max_age_s
        self.sequence = 0
        self.reason = 'not_started'
        self.error = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='control_pose', daemon=True)

    def _not_ready(self, reason, samples):
        self.reason = reason
        return dict(ready=False, reason=reason, sensor_received_ns=0, ego_pose=None,
                    selected_samples=samples)

    def _body(self):
        gps, gps_age, gps_seq, gps_at = self.gps.snapshot()
        imu, imu_age, imu_seq, imu_at = self.imu.snapshot()
        status, status_age, status_seq, status_at = self.status.snapshot()
        ns = lambda at: int(round(at * 1e9)) if at is not None else 0
        samples = dict(
            gps=dict(sample_id=gps_seq, received_ns=ns(gps_at), source_stamp_ns=0),
            vehicle_status=dict(sample_id=status_seq, received_ns=ns(status_at),
                                source_stamp_ns=int(round(status['sim_time'] * 1e9)) if status else 0,
                                imu_received_ns=ns(imu_at),
                                imu_source_stamp_ns=int(round(imu['sim_time'] * 1e9))
                                if imu and imu.get('sim_time') else 0))
        for name, value, age in (('gps', gps, gps_age), ('imu', imu, imu_age),
                                 ('vehicle_status', status, status_age)):
            if value is None:
                return self._not_ready('missing_' + name, samples)
            if not 0 <= age <= self.max_age_s:
                return self._not_ready('stale_' + name, samples)
        if status.get('competition_variant') is not True:
            return self._not_ready('competition_status_required', samples)
        speed = status['speed']  # Competition Vehicle Status signed_vel / 3.6
        if not math.isfinite(speed) or abs(speed) > MAX_SPEED_MPS:
            return self._not_ready('invalid_status_speed', samples)
        try:
            pose = self.projector.project(gps, imu)
        except (ValueError, KeyError, TypeError) as exc:
            return self._not_ready('projection_failed: ' + str(exc), samples)
        self.reason = None
        return dict(ready=True, reason=None,
                    sensor_received_ns=min(ns(gps_at), ns(imu_at), ns(status_at)),
                    ego_pose=dict(x=pose['x'], y=pose['y'], yaw_raw_deg=math.degrees(pose['yaw_raw_rad']),
                                  yaw_grid_rad=pose['yaw_grid_rad'], speed_mps=float(speed)),
                    ego_control={key: status[key] for key in ('accel', 'brake', 'steer', 'gear')},
                    selected_samples=samples)

    def sample(self):
        """One complete packet; not ready when any source is missing or older than max_age_s."""
        self.sequence += 1
        packet = dict(schema=SCHEMA, generation=self.generation, sequence=self.sequence,
                      checkpoint_sha256=self.checkpoint_sha256, output_frame=OUTPUT_FRAME,
                      ego_position_source=GPS_POSITION, ego_status_source=UDP_STATUS)
        packet.update(self._body())
        packet['sent_ns'] = time.monotonic_ns()
        return packet

    def send_once(self):
        data = json.dumps(self.sample(), allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(data) > MAX_PACKET_BYTES:
            raise ValueError('Control pose datagram oversized')
        return self.socket.sendto(data, self.target)

    def _run(self):
        try:
            while not self._stop.is_set():
                begin = time.monotonic()
                self.send_once()
                self._stop.wait(max(.001, self.period - (time.monotonic() - begin)))
        except Exception as exc:
            # The consumer brakes when this stream expires; keep the cause visible.
            self.error = str(exc)
            print('[ControlPose] ' + self.error, flush=True)

    def start(self):
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread.ident is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.)
        self.socket.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
