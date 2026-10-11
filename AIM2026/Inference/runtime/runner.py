"""UDP capture, VE and Joint workers; fixed-rate latest-prediction control."""
import json
from pathlib import Path
import threading
import time
import traceback

from AIM2026.Control.waypoint import WaypointController, remaining_waypoints, to_world
from AIM2026.UDP import UDPManager
from ..sensors.synchronizer import SensorSynchronizer
from .history import InstructionHistory


class LatestSlot:
    """Single replaceable item: slow consumers never accumulate queued jobs."""
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None

    def put(self, value):
        with self.lock:
            self.value = value

    def take(self):
        with self.lock:
            value, self.value = self.value, None
            return value


class LiveRunner:
    def __init__(self, config, models, *, control=False):
        self.config, self.models, self.control_enabled = config, models, control
        self.options = config['Inference']
        options = self.options
        self.udp = UDPManager(config)
        self.sensors = SensorSynchronizer(self.udp, models.projector,
            max_age_s=options['sensors']['max_age_s'], max_skew_s=options['sensors']['max_skew_s'])
        self.controller = (WaypointController(positive_steer=config['Control']['positive_steer'],
            max_speed_mps=config['Control'].get('max_speed_mps')) if control else None)
        self.history = InstructionHistory(capacity=options['history']['capacity'],
            stride=options['history']['stride'], max_gap_s=options['history']['max_gap_s'])
        self.stop_event = threading.Event()
        self.raw = LatestSlot()
        self.prediction_lock = threading.Lock()
        self.prediction = None
        self.error = None
        self.threads = []
        self.counts = dict(captured=0, encoded=0, predicted=0, discarded=0)
        self.status = dict(state='starting', control_enabled=control)

    def _fail(self, name):
        self.error = name + ': ' + traceback.format_exc()
        self.stop_event.set()

    def _vision_worker(self):
        try:
            stream = self.models.stream()
            period = 1. / self.options['encoder_hz']
            last_time = None
            with self.models.stream_context(stream):
                while not self.stop_event.wait(.002):
                    frame = self.raw.take()
                    if frame is None:
                        continue
                    if time.monotonic() - frame.received_at > self.options['sensors']['max_age_s']:
                        continue
                    if last_time is not None and 0 < frame.source_time - last_time < period - .005:
                        continue
                    if last_time is not None and (frame.source_time <= last_time
                            or frame.source_time - last_time > self.options['history']['max_gap_s']):
                        self.models.navigation.reset()
                    last_time = frame.source_time
                    try:
                        feature = self.models.encode(frame)
                    except ValueError as exc:
                        # A missing pose/path is recoverable; a model/recipe
                        # failure must surface instead of silently driving.
                        if str(exc) != 'GPS/IMU pose cannot be projected onto the planned route':
                            raise
                        self.history.reset('invalid_route_projection')
                        self.models.situation_frames.clear()
                        self.status['input_error'] = str(exc)
                        continue
                    self.status.pop('input_error', None)
                    self.history.append(feature)
                    self.counts['encoded'] += 1
        except BaseException:
            self._fail('vision_worker')

    def _joint_worker(self):
        try:
            stream = self.models.stream()
            last_sequence = None
            period = 1. / self.options['prediction_hz']
            next_start = 0.
            with self.models.stream_context(stream):
                while not self.stop_event.wait(.005):
                    if time.monotonic() < next_start:
                        continue
                    generation, frames = self.history.snapshot()
                    if not frames or frames[-1].sequence == last_sequence:
                        continue
                    if time.monotonic() - frames[-1].received_at > self.options['max_prediction_age_s']:
                        continue
                    last_sequence = frames[-1].sequence
                    start = time.monotonic()
                    result = self.models.predict(frames)
                    result.update(generation=generation, duration_s=time.monotonic()-start,
                                  world_points=to_world(result['waypoints'], result['anchor'].pose))
                    current_generation, _ = self.history.snapshot()
                    if generation == current_generation:
                        with self.prediction_lock:
                            self.prediction = result
                        self.counts['predicted'] += 1
                    else:
                        self.counts['discarded'] += 1
                    next_start = max(start + period, time.monotonic())
        except BaseException:
            self._fail('joint_worker')

    def _command(self, prediction, now):
        state, reason = self.sensors.current_state()
        if self.error or self.stop_event.is_set():
            return self.controller.stop('runtime_stopping')
        if reason:
            return self.controller.stop(reason)
        if prediction is None:
            return self.controller.stop('waiting_for_prediction')
        generation, _ = self.history.snapshot()
        if prediction['generation'] != generation:
            return self.controller.stop('instruction_history_changed')
        age = now - prediction['anchor'].received_at
        if not 0 <= age <= self.options['max_prediction_age_s']:
            return self.controller.stop('stale_prediction')
        # Receipt age is a freshness bound; trajectory timestamps are simulator
        # seconds. Do not confuse slow simulation time with GPU/wall latency.
        sim_age = state['source_time'] - prediction['anchor'].source_time
        if not 0 <= sim_age <= 1.5:
            return self.controller.stop('prediction_sim_time_expired')
        wp = remaining_waypoints(prediction['world_points'], state['pose'], sim_age)
        return self.controller.step(wp, state['speed_mps'])

    def _report(self, prediction, command, now):
        status = dict(self.status, counts=dict(self.counts), sensor_status=self.sensors.reason,
                      situation_status=self.models.situation_reason, error=self.error)
        if prediction is not None:
            f = prediction['anchor']
            status.update(prediction_age_s=now-f.received_at, inference_s=prediction['duration_s'],
                source_time=f.source_time, history_frames=prediction['frames'],
                padded_tokens_bound=prediction['padded_tokens_bound'],
                instruction=f.navigation['instruction'], zone_id=f.navigation['zone_id'],
                situation=f.situation, waypoints=prediction['waypoints'].tolist())
        if command is not None:
            status['control'] = vars(command)
        output = Path(self.options['status_file'])
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix('.pending')
        pending.write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        pending.replace(output)
        print(json.dumps({k: status[k] for k in ('state', 'counts', 'sensor_status', 'history_frames',
            'inference_s', 'prediction_age_s', 'zone_id', 'control') if k in status}), flush=True)

    def run(self):
        next_control = next_report = 0.
        command = prediction = None
        try:
            self.udp.start()
            if self.control_enabled and self.udp.cmd is None:
                raise ValueError('Ego.ip and Ego.cmd_port are required for --control')
            for name, fn in (('vision', self._vision_worker), ('joint', self._joint_worker)):
                thread = threading.Thread(target=fn, name=name, daemon=True)
                self.threads.append(thread)
                thread.start()
            self.status['state'] = 'running'
            while not self.stop_event.wait(.005):
                now = time.monotonic()
                frame = self.sensors.next_frame()
                if frame is not None:
                    self.raw.put(frame)
                    self.counts['captured'] += 1
                with self.prediction_lock:
                    prediction = self.prediction
                if self.control_enabled and now >= next_control:
                    command = self._command(prediction, now)
                    self.udp.send_control(command.accel, command.brake, command.steer)
                    next_control = now + .1  # LMDrive PID's 10 Hz nominal update cadence.
                if now >= next_report:
                    self._report(prediction, command, now)
                    next_report = now + 1.
            if self.error:
                raise RuntimeError(self.error)
        finally:
            self.stop_event.set()
            self.status['state'] = 'stopped'
            try:
                if self.control_enabled and self.udp.running and self.udp.cmd:
                    command = self.controller.stop('runtime_stopped')
                    self.udp.send_control(command.accel, command.brake, command.steer)
            finally:
                self.udp.close()
            for thread in self.threads:
                thread.join(timeout=2.)
            self._report(prediction, command, time.monotonic())
