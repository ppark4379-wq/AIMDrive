"""GPS/DR position service between shared UDP inputs and the common Expert.

No perception model, driving policy or control command lives here. All sensor
sockets remain owned by UDPManager. Development GPS masking is opt-in.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time

from .adapters import GpsView, StatusView, receipt_ns
from .gps_aligned_hold import AlignedGpsBlackoutHold
from .gps_blackout_hold import SectionZone


class _Recording:
    """Asynchronous allowed-input log; never feeds recorded references to DR."""

    def __init__(self, path, plan):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        self.queue = queue.Queue(maxsize=20000)
        self.dropped = 0
        self.error = None
        self._last = {}
        self.thread = threading.Thread(target=self._write, name='localization_recording', daemon=True)
        self.thread.start()

    def _put(self, kind, row):
        if self.error is not None:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait((kind, row))
        except queue.Full:
            self.dropped += 1

    def inputs(self, sensor, records, poll_id):
        last = self._last.get(sensor, 0)
        for record in records:
            if record.sequence <= last:
                continue
            value = dict(record.value)
            if sensor == 'status':
                value = {key: value[key] for key in ('signed_speed', 'sim_time') if key in value}
            row = dict(sensor=sensor, received_ns=receipt_ns(record),
                       sample_id=record.sequence, value=value,
                       clock_epoch=getattr(record, 'clock_epoch', 0),
                       observed_poll_id=poll_id,
                       source_stamp_ns=(round(value['sim_time'] * 1e9)
                                        if value.get('sim_time') is not None else 0))
            if sensor == 'gps' and (record.meta or {}).get('nmea'):
                row['nmea'] = record.meta['nmea']
            self._put('sensor_audit', row)
            last = record.sequence
        self._last[sensor] = last

    def pose(self, now_ns, fix, meta, hold, completed_ns, poll_id, age):
        position = None
        if fix is not None:
            x, y = hold._grid(fix)
            state = hold.track.at(meta.get('received_ns', now_ns))
            yaw = speed = None
            if state is not None:
                yaw = state['yaw'] + math.radians(hold.core.convergence_deg((x, y)))
                speed = state['speed_mps']
            position = dict(x=x, y=y, yaw_grid_rad=yaw, speed_mps=speed)
        self._put('poses', dict(monotonic_ns=now_ns, position=position,
                                poll_id=poll_id,
                                estimator_started_ns=now_ns,
                                estimator_completed_ns=completed_ns,
                                returned_age_s=age if math.isfinite(age) else None,
                                selected_samples=dict(gps=meta)))

    def _write(self):
        try:
            with (self.path / 'sensor_audit.jsonl').open('w') as sensors, (self.path / 'poses.jsonl').open('w') as poses:
                streams = dict(sensor_audit=sensors, poses=poses)
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    kind, row = item
                    streams[kind].write(json.dumps(row, allow_nan=False) + '\n')
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=2.)
            except queue.Full:
                self.error = self.error or 'recording_queue_full_on_close'
            self.thread.join(timeout=5.)
        (self.path / 'recording_status.json').write_text(json.dumps(
            dict(dropped_rows=self.dropped, error=self.error,
                 writer_finished=not self.thread.is_alive()), indent=2) + '\n')


class LocalizationService:
    """Serializes estimator calls and exposes both legacy and AIM2026 APIs."""

    def __init__(self, receivers, settings, *, record_dir=None):
        self.basis = settings['map_basis']
        self.gps = GpsView(receivers['gps'])
        self.status = StatusView(receivers['vehicle_status'], receivers['imu'])
        test = settings.get('test_blackout') or {}
        zone = None
        if test.get('enabled') is True:
            zone = SectionZone(**{key: test[key] for key in
                                 ('start_xy', 'end_xy', 'half_width_m', 'margin_m') if key in test})
        self.hold = AlignedGpsBlackoutHold(
            self.gps, self.status, self.basis,
            max_hold_s=settings.get('max_hold_s', 120.),
            max_distance_m=settings.get('max_distance_m', 250.),
            anchor_window_s=settings.get('anchor_window_s', 1.), simulate_zone=zone)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._history = deque(maxlen=4096)
        self._sequence = 0
        self._poll_id = 0
        self._key = None
        self._latest = (None, float('inf'), {})
        self._input_epochs = None
        self._position_epoch = 0
        self.error = None
        self._receivers = receivers
        self.recorder = None
        if record_dir is not None:
            paths = [Path(__file__), Path(__file__).with_name('gps_aligned_hold.py'),
                     Path(__file__).with_name('gps_blackout_hold.py'),
                     Path(__file__).with_name('adapters.py'), Path(__file__).with_name('map_geometry.py')]
            plan = dict(schema='aim2026_localization_recording_v1', map_basis=self.basis,
                        gps_hold_method='aligned', gps_anchor_window_s=settings.get('anchor_window_s', 1.),
                        max_hold_s=settings.get('max_hold_s', 120.),
                        max_distance_m=settings.get('max_distance_m', 250.),
                        simulate_zone=({key: test[key] for key in ('start_xy', 'end_xy', 'half_width_m', 'margin_m')
                                        if key in test} if zone is not None else None),
                        driving_inputs=['GPS_outside_blackout', 'IMU_orientation', 'signed_status_speed'],
                        GT_driving_inputs=False,
                        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
            self.recorder = _Recording(record_dir, plan)

    @property
    def port(self):
        return self.gps.port

    @property
    def packets(self):
        return self.gps.packets

    @property
    def samples(self):
        return self.gps.samples

    def poll(self):
        from AIM2026.UDP.Common import ReceivedSample
        with self._lock:
            if self._closed:
                return None, float('inf'), dict(position_mode='unavailable', reason='localization_closed')
            self._poll_id += 1
            now_ns = time.monotonic_ns()
            speeds = imus = gps_records = ()
            try:
                # Freeze the same inputs used by this call. A datagram arriving
                # during computation belongs to the next call and its audit tick.
                gps_records = self.gps.refresh()
                speeds, imus = self.status.refresh()
                now_ns = time.monotonic_ns()
                epochs = tuple(getattr(records[-1], 'clock_epoch', 0) if records else 0
                               for records in (speeds, imus))
                if self._input_epochs is not None and epochs != self._input_epochs:
                    self._history.clear()
                    self._key = None
                    self._position_epoch += 1
                self._input_epochs = epochs
                fix, age, meta = self.hold.latest_snapshot()
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                fix, age, meta = None, float('inf'), dict(position_mode='unavailable', reason='localization_exception')
            completed_ns = time.monotonic_ns()
            meta = dict(meta)
            meta['clock_epoch'] = self._position_epoch
            if fix is None:
                # Availability changes now; the last GPS sensor receipt remains
                # old. Keep the two times separate so frame selectors cannot
                # skip this refusal and reuse an earlier valid DR position.
                previous = self._latest[2]
                same_refusal = self._latest[0] is None and previous.get('reason') == meta.get('reason')
                meta['availability_changed_ns'] = (previous.get('availability_changed_ns', now_ns)
                                                   if same_refusal else now_ns)
            self._latest = fix, age, dict(meta)
            stamp = meta.get('received_ns', now_ns) if fix is not None else meta['availability_changed_ns']
            key = (meta.get('position_mode'), stamp if fix else meta.get('reason'), meta.get('sample_id'),
                   meta.get('source_stamp_ns'), tuple(sorted(fix.items())) if fix else None)
            source = meta.get('source_stamp_ns', 0) if fix is not None else 0
            if key != self._key:
                self._sequence += 1
                record = ReceivedSample(fix, stamp / 1e9, self._sequence,
                                        source / 1e9 if source else None, dict(meta),
                                        received_ns=stamp, source_stamp_ns=source,
                                        clock_epoch=self._position_epoch)
                self._history.append(record)
                self._key = key
            if self.recorder is not None:
                try:
                    self.recorder.inputs('status', speeds, self._poll_id)
                    self.recorder.inputs('imu', imus, self._poll_id)
                    self.recorder.inputs('gps', gps_records, self._poll_id)
                    self.recorder.pose(now_ns, fix, meta, self.hold, completed_ns, self._poll_id, age)
                except Exception as exc:
                    self.recorder.error = str(exc)
            return fix, age, dict(meta)

    def latest_snapshot(self):
        return self.poll()

    def latest(self):
        fix, age, _ = self.latest_snapshot()
        return fix, age

    def snapshot(self):
        with self._lock:
            fix, age, meta = self.poll()
            stamp = meta.get('received_ns')
            return fix, age, self._sequence, stamp / 1e9 if stamp else None

    def recent(self):
        with self._lock:
            self.poll()
            return tuple(self._history)

    def diagnostics(self):
        with self._lock:
            result = self.hold.diagnostics()
            result['localization'] = dict(enabled=True, method='aligned',
                position_mode=self._latest[2].get('position_mode', 'unavailable'),
                reason=self._latest[2].get('reason'), error=self.error,
                history_size=len(self._history), shared_sensor_sockets=True,
                recording=(dict(path=str(self.recorder.path), dropped_rows=self.recorder.dropped,
                                error=self.recorder.error) if self.recorder else None))
            return result

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='localization', daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            begin = time.monotonic()
            self.poll()
            self._stop.wait(max(.001, .02 - (time.monotonic() - begin)))

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._history.clear()
            self._latest = (None, float('inf'), dict(position_mode='unavailable', reason='localization_closed'))
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:
                    self.recorder.error = str(exc)
            # GpsView.close is deliberately non-owning.
            self.hold.close()


def create_localization(manager, config):
    """Build from the common config; absent/disabled sections preserve old callers."""
    options = config.get('Localization') or {}
    enabled = options.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('Localization.enabled must be boolean')
    if not enabled:
        return None
    import yaml
    aim_root = Path(__file__).resolve().parents[3]
    path = Path(options.get('config', 'GPSJamming/configs/localization.yaml'))
    if not path.is_absolute():
        path = aim_root / path
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict):
        raise ValueError('Localization configuration must be a mapping')
    for label, values in (('GPS.pos', config['GPS']['pos']), ('IMU.rot_deg', config['IMU']['rot_deg'])):
        if any(float(value) != 0. for value in values):
            raise ValueError(label + ': improved DR requires the original zero-mount calibration')
    directory = options.get('record_dir')
    if directory is not None:
        directory = Path(directory)
        if not directory.is_absolute():
            directory = aim_root / directory
    return LocalizationService(manager.receivers, settings, record_dir=directory)
