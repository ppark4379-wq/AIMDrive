"""MORAI UDP sensors -> VE perception -> rule-based Expert (loopback UDP).

python AIM2026/Inference/main.py --check        # config and weight files only; no sockets, no model
python AIM2026/Inference/main.py                # run until Ctrl+C
python AIM2026/Inference/main.py --seconds 60

For every new LiDAR frame: the newest four cameras, the vehicle-status speed
aligned to the LiDAR source time (causal <= 50 ms, or interpolation between
received samples bracketing that time within 100 ms), the IMU
heading and the GPS fix -> VisionRuntime -> one Detection packet. The control
pose (GPS / IMU / status, 50 Hz) is sent independently by the UDP manager.
Missing or stale inputs send a not-ready packet, never old boxes. This process
never sends vehicle control; the Expert does.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AIM2026.Inference.settings import read_config
from AIM2026.Inference.sensors.position import position_at_frame, position_identity

CAMERAS = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')
SPEED_ALIGNMENT_S = 0.05  # status/IMU sample at or before the LiDAR source time
LIDAR_STAMP_RESOLUTION_S = 1e-6  # LiDAR stamps are whole microseconds; status stamps are ns
# MORAI IMU arrives at ~40 Hz with source gaps up to ~80 ms (p99 62 ms measured), so a
# 50 ms window misses ~5% of LiDAR frames. An older sample (<= 100 ms) is rotated to
# the LiDAR time with its own body gyro rate instead of declaring the frame unusable.
IMU_PROPAGATION_S = 0.10
STATUS_INTERPOLATION_S = 0.10  # bounded bracketing only; never extrapolate speed


def _causal(records, source_time, tolerance=SPEED_ALIGNMENT_S):
    """Newest record stamped at or before source_time, within tolerance (simulator clock).

    A sample from the same instant may read up to 1 us later than the truncated
    LiDAR stamp; it is still causal and must not be skipped.
    """
    candidates = [r for r in records if r.source_time
                  and -LIDAR_STAMP_RESOLUTION_S <= source_time - r.source_time <= tolerance + 1e-9]
    return max(candidates, key=lambda r: r.source_time) if candidates else None


def aligned_status(records, source_time):
    """Use a causal sample, else interpolate two already received speed samples.

    A render-limited simulator may publish status every 60--70 ms. Dropping
    each LiDAR time in the last 10--20 ms of that interval causes false input
    outages. A bounded bracket estimates the speed at the actual LiDAR time;
    it neither extends freshness limits nor assumes an old speed is current.
    """
    causal = _causal(records, source_time)
    if causal is not None:
        return causal, 'causal_lidar_source_50ms'
    before = [r for r in records if r.source_time and r.source_time < source_time]
    after = [r for r in records if r.source_time and r.source_time >= source_time]
    if not before or not after:
        return None, None
    left, right = max(before, key=lambda r: r.source_time), min(after, key=lambda r: r.source_time)
    span = right.source_time-left.source_time
    if not 0 < span <= STATUS_INTERPOLATION_S or left.clock_epoch != right.clock_epoch:
        return None, None
    a, b = float(left.value['speed']), float(right.value['speed'])
    if not (math.isfinite(a) and math.isfinite(b)):
        return None, None
    ratio = (source_time-left.source_time)/span
    value = dict(left.value, speed=a+(b-a)*ratio)
    return dataclasses.replace(left, value=value, source_time=source_time,
        source_stamp_ns=int(round(source_time*1e9)),
        received_at=min(left.received_at, right.received_at),
        meta=dict(left.meta or {}, speed_interpolation=dict(
            before_source_time=left.source_time, after_source_time=right.source_time,
            before_sequence=left.sequence, after_sequence=right.sequence))), 'interpolated_lidar_source_100ms'


def propagate_imu(value, dt):
    """IMU sample rotated forward by dt seconds with its body angular velocity.

    q(t+dt) = q(t) * exp(0.5 * omega * dt), omega in the body frame (rad/s),
    quaternion ordered (x, y, z, w). Only orientation fields change.
    """
    omega = value.get('angular_velocity')
    if dt <= 0 or not omega:
        return value
    wx, wy, wz = (float(v) for v in omega)
    rate = math.sqrt(wx * wx + wy * wy + wz * wz)
    if not math.isfinite(rate):
        raise ValueError('Non-finite IMU angular velocity')
    if rate * dt < 1e-12:
        return value
    half = .5 * rate * dt
    s = math.sin(half) / rate
    a, b, c, d = wx * s, wy * s, wz * s, math.cos(half)
    x, y, z, w = value['quaternion']
    q = (w * a + x * d + y * c - z * b, w * b - x * c + y * d + z * a,
         w * c + x * b - y * a + z * d, w * d - x * a - y * b - z * c)
    nx, ny, nz, nw = q
    return dict(value, quaternion=q, propagated_s=dt,
                yaw=math.atan2(2. * (nw * nz + nx * ny), 1. - 2. * (ny * ny + nz * nz)))


def causal_imu(records, source_time):
    """(IMU value at source_time, method) from the causal window, else (None, None)."""
    record = _causal(records, source_time)
    if record is not None:
        return record, 'causal_50ms'
    record = _causal(records, source_time, IMU_PROPAGATION_S)
    if record is None:
        return None, None
    dt = max(0., source_time - record.source_time)
    return dataclasses.replace(record, value=propagate_imu(record.value, dt)), 'gyro_propagated'


class VePublisher:
    """One VE inference per new LiDAR frame, published as a Detection packet."""

    def __init__(self, udp, runtime, projector, *, max_age_s=.5):
        self.udp, self.runtime, self.projector = udp, runtime, projector
        self.max_age_s = max_age_s
        self.last_lidar = 0
        self.frames = self.ready_frames = self.expired_frames = 0
        self.reason = 'waiting_for_lidar'
        self.last = None
        self.imu_method = None
        self.imu_methods = {}

    def _fresh(self, record, now):
        return record is not None and record.value is not None and 0 <= now - record.received_at <= self.max_age_s

    def _inputs(self, recent, lidar, now):
        """(selected records, alignment) or (None, missing reasons)."""
        missing, chosen = [], dict(lidar=lidar)
        if not self._fresh(lidar, now):
            missing.append('lidar')
        for name in CAMERAS:
            record = recent[name][-1] if recent.get(name) else None
            chosen[name] = record
            if not self._fresh(record, now):
                missing.append(name)
        statuses, imus = recent.get('vehicle_status', ()), recent.get('imu', ())
        reference = statuses[-1] if statuses else None
        stamp_us = (lidar.meta or {}).get('source_stamp_us_within_hour')
        lidar_time = None
        if stamp_us is None:
            # azimuth_wrap frames carry no source time: newest status by receipt.
            chosen['vehicle_status'], alignment = reference, 'receipt'
        elif not self._fresh(reference, now):
            chosen['vehicle_status'], alignment = None, None
        else:
            from AIM2026.UDP.LiDAR import resolve_hour_stamp
            try:
                lidar_time = resolve_hour_stamp(stamp_us, reference.source_time)
            except (ValueError, TypeError):
                missing.append('lidar_source_clock')
            chosen['vehicle_status'], alignment = (aligned_status(statuses, lidar_time)
                                                   if lidar_time else (None, None))
            if lidar_time and chosen['vehicle_status'] is None:
                missing.append('vehicle_status_source_alignment')
        if not self._fresh(chosen['vehicle_status'], now) and 'vehicle_status_source_alignment' not in missing:
            missing.append('vehicle_status')
        # A stamped IMU uses the causal window (gyro-propagated up to 100 ms); the
        # 107-byte IMU has no stamp.
        newest_imu = imus[-1] if imus else None
        imu, self.imu_method = causal_imu(imus, lidar_time) if lidar_time else (None, None)
        if imu is None and newest_imu is not None and not newest_imu.source_time:
            imu, self.imu_method = newest_imu, 'receipt'
        if self.imu_method:
            self.imu_methods[self.imu_method] = self.imu_methods.get(self.imu_method, 0) + 1
        chosen['imu'] = imu
        if not self._fresh(imu, now):
            missing.append('imu')
        chosen['gps'] = position_at_frame(recent.get('gps', ()), lidar, lidar_time,
                                          max_skew_s=min(.15, self.max_age_s))
        if not self._fresh(chosen['gps'], now):
            missing.append('gps')
        return (None, missing) if missing else (chosen, alignment)

    def step(self):
        """Process the newest LiDAR frame once; None when no new frame has arrived."""
        from AIM2026.UDP import EgoPose
        recent = self.udp.recent()
        lidar = recent['lidar'][-1] if recent.get('lidar') else None
        if lidar is None or lidar.sequence <= self.last_lidar:
            return None
        self.last_lidar = lidar.sequence
        self.frames += 1
        now = time.monotonic()
        chosen, detail = self._inputs(recent, lidar, now)
        if chosen is None:
            return self._not_ready('missing:' + ','.join(detail))
        try:
            pose = self.projector.project(chosen['gps'].value, chosen['imu'].value)
            speed = float(chosen['vehicle_status'].value['speed'])
            result = self.runtime.infer({name: chosen[name].value for name in CAMERAS},
                                        lidar.value, speed, pose)
        except ValueError as exc:
            return self._not_ready('input_rejected: ' + str(exc))
        received = {name: chosen[name].received_at for name in (*CAMERAS, 'lidar', 'vehicle_status', 'gps', 'imu')}
        oldest_name = min(received, key=received.get)
        oldest = received[oldest_name]
        identity = position_identity(chosen['gps'])
        if identity['position_mode'] not in ('gps', 'dead_reckoning'):
            return self._not_ready('position_unavailable')
        ego_pose = EgoPose(pose['x'], pose['y'], math.degrees(pose['yaw_raw_rad']), pose['yaw_grid_rad'], speed)
        sent_at = time.monotonic()
        self.udp.send_detection(result.boxes, ego_pose, oldest, signal=result.signal,
                                native_signal=result.native_signal, position_identity=identity,
                                object_received_at=lidar.received_at,
                                ego_position_source='GPS_dead_reckoning'
                                if identity['position_mode']=='dead_reckoning' else 'GPS')
        # The sender downgrades a frame whose sensors aged past 0.5 s during inference.
        sender = getattr(self.udp, 'detection', None)
        delivered = getattr(sender, 'last_ready', True) is not False
        self.reason = None if delivered else getattr(sender, 'last_reason', 'not_ready')
        if delivered:
            self.ready_frames += 1
        else:
            self.expired_frames += 1
        self.last = dict(objects=len(result.boxes), inference_ms=round(result.inference_ms, 1),
                         signal=result.signal, native_signal=result.native_signal, delivered=delivered,
                         speed_alignment=detail, sensor_age_ms=round((sent_at - oldest) * 1e3, 1),
                         # Age budget: which input was oldest and how old each was when inference began.
                         oldest_input=oldest_name,
                         input_age_ms={name: round((now - at) * 1e3, 1) for name, at in received.items()},
                         position_identity=identity,
                         pose=dict(x=round(pose['x'], 2), y=round(pose['y'], 2)))
        return 'ready' if delivered else self.reason

    def _not_ready(self, reason):
        self.reason = reason
        self.udp.send_detection_not_ready(reason)
        return reason

    def status(self):
        streams = {name: dict(packets=d['packets'], samples=d['samples'],
                              **({'rejected': d['counters']} if d['counters'] else {}))
                   for name, d in self.udp.diagnostics().items()}
        return dict(frames=self.frames, ready_frames=self.ready_frames, expired_frames=self.expired_frames,
                    reason=self.reason,
                    last=self.last, imu_alignment=dict(self.imu_methods), pose_reason=getattr(self.udp.pose, 'reason', None), streams=streams,
                    localization=self.udp.localization.diagnostics()
                    if getattr(self.udp, 'localization', None) is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config.yaml')
    parser.add_argument('--check', action='store_true', help='Validate config and weight files; no sockets, no model')
    parser.add_argument('--seconds', type=float, default=0., help='Stop after this many seconds (0 = until Ctrl+C)')
    parser.add_argument('--generation', help='Shared sender identity when launched by the common Expert')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.generation is not None:
        if not args.generation.strip():
            parser.error('--generation must be nonempty')
        config['Detection']['generation'] = args.generation
    options, vision = config['Inference'], config['Inference']['vision']

    from AIM2026.UDP import UDPManager
    from AIM2026.Inference.vision.checkpoint import resolve
    udp = UDPManager(config)
    udp._validate_ports()
    weights, _sha, head = resolve(vision['checkpoint'])
    files = [weights] + ([head] if vision['signal_head'] and head else [])
    if vision['signal_head'] and head is None:
        raise ValueError('No signal head is registered for %s; set Inference.vision.signal_head: false'
                         % vision['checkpoint'])
    absent = [str(p) for p in files if not p.is_file()]
    if args.check:
        print('Configuration OK; no sockets opened, no model loaded.')
        print('VE %s weights: %s' % (vision['checkpoint'], 'present' if not absent else 'MISSING ' + ', '.join(absent)))
        print('Detection/pose -> %s:%s' % (config['Detection'].get('ip') or '127.0.0.1', config['Detection']['port']))
        return 1 if absent else 0
    if absent:
        raise FileNotFoundError('Put the VE weights in AIM2026/Weights/VE: ' + ', '.join(absent))

    from AIM2026.Inference.vision import VisionRuntime, lidar_to_base
    lidar = config['LiDAR']
    runtime = VisionRuntime(vision['checkpoint'], device=options['device'], signal_head=vision['signal_head'],
                            lidar_to_base=lidar_to_base(lidar['pos'], lidar['rot_deg']),
                            cpu_threads=options['cpu_threads'])
    print('[VE] %s loaded (%s), signal head=%s' % (vision['checkpoint'], options['device'],
                                                  runtime.signal_head is not None), flush=True)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    from AIM2026.Inference.timer_resolution import high_resolution_timer
    timer = high_resolution_timer()
    timer.__enter__()
    udp.start()
    try:
        publisher = VePublisher(udp, runtime, udp.pose.projector, max_age_s=vision['max_sensor_age_s'])
        print('[VE] receiving: %s -> Detection/pose %s:%s' % (', '.join(udp.receivers),
              config['Detection'].get('ip') or '127.0.0.1', config['Detection']['port']), flush=True)
        started = next_status = time.monotonic()
        while not stop and not (args.seconds and time.monotonic() - started >= args.seconds):
            if udp.pose.error:
                raise RuntimeError('Control pose stream stopped: ' + udp.pose.error)
            if publisher.step() is None:
                time.sleep(.002)
            if time.monotonic() >= next_status:
                row = publisher.status()
                print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
                if options.get('status_file'):
                    target = Path(options['status_file'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(row, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
                next_status = time.monotonic() + options['status_interval_s']
    finally:
        try:
            if udp.running:
                udp.send_detection_not_ready('publisher_stopped')
        finally:
            udp.close()
            timer.__exit__(None, None, None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
