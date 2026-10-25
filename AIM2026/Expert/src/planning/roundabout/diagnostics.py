"""Optional roundabout evidence capture; never used to select a control action."""

import atexit
import json
import math
import queue
import threading
import time

from ...config.parameters import lattice as p
from ..highway import parameters as vehicle_parameters


_writer = None
_input_context = {}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _JsonWriter:
    """Bounded queue: disk writes and JSON encoding stay off the control thread."""

    def __init__(self, path):
        self.stream = open(path, 'a', encoding='utf-8')
        self.queue = queue.Queue(maxsize=256)
        self.dropped = 0
        self.failed = False
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='roundabout-debug', daemon=True)
        self.thread.start()

    def submit(self, record):
        if self.failed or self.closed:
            return False
        try:
            self.queue.put_nowait(dict(record, dropped_records=self.dropped))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        try:
            while True:
                record = self.queue.get()
                if record is None:
                    break
                self.stream.write(json.dumps(_json_safe(record), ensure_ascii=False,
                                             allow_nan=False, separators=(',', ':')) + '\n')
                self.stream.flush()
        except Exception as error:
            self.failed = True
            print('[ROUNDABOUT_DEBUG] disabled after write error: {}'.format(error))
        finally:
            self.stream.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if not self.failed:
            try:
                self.queue.put(None, timeout=1.)
            except queue.Full:
                return
        self.thread.join(timeout=2.)


def configure(path):
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None
    if not p.ROUNDABOUT_DEBUG_ENABLED:
        return
    try:
        _writer = _JsonWriter(path)
        atexit.register(_writer.close)
        print('[ROUNDABOUT_DEBUG] file: {}'.format(path))
    except OSError as error:
        print('[ROUNDABOUT_DEBUG] disabled: {}'.format(error))


def input_context(**values):
    global _input_context
    if _writer is not None:
        _input_context = values


def record_control(result, target, command, dt, steering):
    frame = getattr(result, 'roundabout_debug_frame', None)
    if frame is not None and _writer is not None:
        details = result.debug_details.get('roundabout_entry', {})
        _writer.submit(dict(kind='control', frame=frame, wall_time=time.time(),
                            target_mps=target, pedal_command=command, control_dt_s=dt,
                            front_steer_rad=steering, speed_limits_mps=details.get('speed_limits'),
                            speed_source=details.get('speed_source')))


class RoundaboutDiagnostics:
    def __init__(self):
        self.frame = 0
        self.last_signature = None
        self.last_emit = -math.inf
        self.last_input_ids = set()
        self.last_tracked_ids = set()
        self.was_active = False
        self.disabled = False
        self.checks = None
        self.tracks = []
        self.predictions = None

    def begin(self):
        self.frame += 1
        self.started = time.perf_counter()
        self.checks = [] if not self.disabled and p.ROUNDABOUT_DEBUG_ENABLED and _writer is not None and not _writer.failed else None
        self.tracks = []
        self.predictions = None

    def finish(self, planner, ego, path, entries, state, desired, before, result):
        if self.checks is None:
            return
        now = time.monotonic()
        if result is None:
            if self.was_active:
                _writer.submit(dict(kind='exit', frame=self.frame, wall_time=time.time(),
                                    ego_xy=[ego.position.x, ego.position.y], input=dict(_input_context)))
            self.was_active = False
            self.last_signature = None
            return
        d = result.debug_details['roundabout_entry']
        input_ids = {str(entry['object_info'].unique_id) for entry in entries}
        tracked_ids = {str(obj.unique_id) for obj, reason, _ in self.tracks
                       if reason in ('base_range', 'rear_extension')}
        signature = (d.get('phase'), d.get('entering'), d.get('committed'),
                     str(d.get('front_id')), str(d.get('blocker_unique_id')),
                     result.acc_target_velocity <= .05, tuple(sorted(input_ids)),
                     tuple(sorted(tracked_ids)), d.get('following_speed_adjusted'))
        event = signature != self.last_signature
        self.was_active = True
        if not event and now - self.last_emit < p.ROUNDABOUT_DEBUG_INTERVAL:
            return
        self.last_emit = now
        self.last_signature = signature
        tracked_by_object = {id(obj): (reason, match) for obj, reason, match in self.tracks}
        objects = []
        for entry in entries:
            obj = entry['object_info']
            reason, match = tracked_by_object.get(id(obj), ('not_evaluated', None))
            objects.append(dict(id=obj.unique_id, name=obj.name, type=obj.type, static=obj.is_static,
                                xy=[obj.position.x, obj.position.y], speed_mps=obj.velocity,
                                heading_deg=obj.heading_deg, size_m=[obj.size_x, obj.size_y],
                                distance_m=obj.position.distance(ego.position), filter=reason,
                                map_link=match.lane_link.id if match else None,
                                map_s_m=match.s if match else None,
                                map_offset_m=match.distance if match else None,
                                prediction_model=('map' if match and obj.velocity >= .05 else 'heading')))
        prediction_samples = []
        if self.predictions is not None:
            tracks, predictions = self.predictions
            stride = max(1, round(2. / p.ROUNDABOUT_PREDICTION_STEP))
            indices = sorted(set(range(0, len(predictions), stride)) | {len(predictions) - 1})
            for j, (obj, _) in enumerate(tracks):
                prediction_samples.append(dict(id=obj.unique_id, samples=[
                    dict(t_s=i * p.ROUNDABOUT_PREDICTION_STEP,
                         alternatives=[[pt.x, pt.y, math.degrees(yaw)] for pt, yaw in predictions[i][j]])
                    for i in indices]))
        record = dict(kind='plan', frame=self.frame, wall_time=time.time(), event=event,
                      elapsed_since_previous_emit_s=None if not hasattr(self, 'previous_emit') else now-self.previous_emit,
                      planner_ms=(time.perf_counter()-self.started)*1000., input=dict(_input_context),
                      ego=dict(xy=[ego.position.x, ego.position.y], yaw_deg=math.degrees(ego.yaw), speed_mps=ego.velocity),
                      before=before, decision=dict(d), approach_state=dict(state), desired_mps=desired,
                      objects=objects, object_predictions=prediction_samples, checks=self.checks,
                      input_ids_added=sorted(input_ids-self.last_input_ids), input_ids_removed=sorted(self.last_input_ids-input_ids),
                      tracked_ids_added=sorted(tracked_ids-self.last_tracked_ids), tracked_ids_removed=sorted(self.last_tracked_ids-tracked_ids),
                      route_xy=[[pt.x, pt.y] for pt in path],
                      collision_geometry=dict(ego_length_m=vehicle_parameters.VEHICLE_LENGTH,
                                              ego_width_m=vehicle_parameters.VEHICLE_WIDTH,
                                              margin_m=vehicle_parameters.COLLISION_MARGIN),
                      parameters={name: getattr(p, name) for name in dir(p) if name.startswith('ROUNDABOUT_')
                                  and 'MONITOR' not in name})
        self.previous_emit = now
        self.last_input_ids, self.last_tracked_ids = input_ids, tracked_ids
        if _writer.submit(record):
            result.roundabout_debug_frame = self.frame
        if event:
            print('[ROUNDABOUT_DEBUG] frame={} phase={} entering={} merged={} tracked={}/{} conflict={} hold_gap={} target={:.1f}km/h'.format(
                self.frame, d.get('phase'), d.get('entering'), d.get('committed'), len(tracked_ids), len(entries),
                d.get('blocker_unique_id'), d.get('hold_gap'), result.acc_target_velocity*3.6))
