"""Latest GPS/IMU/status for control, independent of the VE forward loop.

This optional stream shares the existing loopback VE socket. It never replaces
the perception-time pose used to transform detections, nor supplies GT pose.
"""
import json
import math
import socket
import threading
import time

SCHEMA = 'morai_ve_control_pose_v1'
MAX_AGE_S = .15
MAX_BYTES = 8000
POSITION_SOURCE = 'GPS'
# Position held through a GPS blackout (inference.sensors.gps_blackout_hold); accepted only when enabled.
DEAD_RECKONING_SOURCE = 'GPS_dead_reckoning'
STATUS_SOURCE = 'UDP_status_speed_IMU_heading'


def decode_control_pose(packet, generation, checkpoint_sha256, now_ns, position_sources=(POSITION_SOURCE,)):
    if (packet.get('schema') != SCHEMA or packet.get('generation') != generation
            or packet.get('checkpoint_sha256') != checkpoint_sha256
            or packet.get('output_frame') != 'projected_grid'
            or packet.get('ego_position_source') not in position_sources
            or packet.get('ego_status_source') != STATUS_SOURCE):
        raise ValueError('Control pose identity/source mismatch')
    if type(packet.get('sequence')) is not int or packet['sequence'] < 1:
        raise ValueError('Invalid control pose sequence')
    sent, received = packet.get('sent_ns'), packet.get('sensor_received_ns')
    if type(sent) is not int or not 0 <= now_ns-sent <= MAX_AGE_S*1e9:
        raise ValueError('Control pose delivery expired')
    if packet.get('ready') is not True:
        raise ValueError('Control pose not ready: '+str(packet.get('reason')))
    if type(received) is not int or not 0 < received <= sent or now_ns-received > MAX_AGE_S*1e9:
        raise ValueError('Control pose sensors expired')
    pose = packet['ego_pose']
    for key in ('x', 'y', 'yaw_raw_deg', 'yaw_grid_rad', 'speed_mps'):
        value = pose[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid control pose value')
    if abs(pose['speed_mps']) > 100:
        raise ValueError('Invalid control speed')
    samples = packet['selected_samples']
    receipts = [samples['gps']['received_ns'], samples['vehicle_status']['received_ns'],
                samples['vehicle_status']['imu_received_ns']]
    if any(type(stamp) is not int or not received <= stamp <= sent for stamp in receipts):
        raise ValueError('Invalid control pose sample receipts')
    # A recent GPS fix must not mask an old heading/status sample (or vice versa).
    if min(receipts) != received:
        raise ValueError('Control pose oldest sample mismatch')
    return packet


class ControlPoseStream:
    def __init__(self, status, gps, basis, generation, checkpoint_sha256,
                 target=('127.0.0.1', 12050), hz=50.):
        from pyproj import Transformer
        from data.src.loaders.morai_map_geometry import validate_map_basis
        if not math.isfinite(hz) or not 1 <= hz <= 100:
            raise ValueError('Control pose frequency must be 1..100Hz')
        if target[0] != '127.0.0.1':
            raise ValueError('Control pose stream must stay on loopback')
        self.status, self.gps = status, gps
        self.geometry = validate_map_basis(basis)
        self.utm = Transformer.from_crs('EPSG:4326', basis['crs'], always_xy=True)
        self.origin = basis['origin_xyz']
        self.generation, self.digest = generation, checkpoint_sha256
        self.target, self.period = target, 1./hz
        self.sequence = 0
        self.stop_event = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.error = None
        self.thread = threading.Thread(target=self._run, name='ve_control_pose', daemon=True)

    def sample(self):
        status, status_age, status_id = self.status.latest_snapshot()
        fix, gps_age, gps_id = self.gps.latest_snapshot()
        self.sequence += 1
        packet = dict(schema=SCHEMA, generation=self.generation, sequence=self.sequence,
            checkpoint_sha256=self.digest, output_frame='projected_grid',
            ego_position_source=POSITION_SOURCE, ego_status_source=STATUS_SOURCE,
            ready=False, reason='GPS_IMU_status_stale_or_missing', sensor_received_ns=0,
            ego_pose=None, selected_samples=dict(gps=gps_id, vehicle_status=status_id))
        if (status is not None and fix is not None
                and 0 <= status_age <= MAX_AGE_S and 0 <= gps_age <= MAX_AGE_S):
            east, north = self.utm.transform(fix['longitude'], fix['latitude'])
            x, y = east-self.origin[0], north-self.origin[1]
            yaw = status['yaw']
            grid = yaw+math.radians(self.geometry.convergence_degrees((x, y, 0.)))
            if gps_id.get('position_mode') == 'dead_reckoning':
                packet.update(ego_position_source=DEAD_RECKONING_SOURCE,
                              dead_reckoning=dict(hold_s=gps_id.get('hold_s'),
                                                  distance_m=gps_id['dead_reckoning']['distance_m']))
            packet.update(ready=True, reason=None,
                sensor_received_ns=min(status_id['received_ns'], gps_id['received_ns']),
                ego_pose=dict(x=x, y=y, yaw_raw_deg=math.degrees(yaw), yaw_grid_rad=grid,
                              speed_mps=status['signed_speed_mps']),
                ego_control={k:status[k] for k in ('accel', 'brake', 'steer', 'gear')})
        packet['sent_ns'] = time.monotonic_ns()
        return packet

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop_event.is_set():
                begin = time.monotonic()
                packet = self.sample()
                data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode()
                if len(data) > MAX_BYTES:
                    raise ValueError('Control pose datagram oversized')
                self.socket.sendto(data, self.target)
                self.stop_event.wait(max(.001, self.period-(time.monotonic()-begin)))
        except Exception as exc:
            # The consumer will brake when this stream expires. Also surface
            # the failure in the VE process log instead of hiding thread death.
            self.error = str(exc)
            print('[ControlPose] '+self.error, flush=True)

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.)
        self.socket.close()
