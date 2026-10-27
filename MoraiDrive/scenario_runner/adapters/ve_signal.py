"""Mapped VE signal trials. No GT signal subscription or phase input.

Map geometry binds the target ID; it does not determine its lamp state. Code 0
is unavailable evidence, never synthetic red/green. The full-route trial adds
0 to stop statuses in its private runtime config; LCS stays unsupported.
"""
from collections import deque
import json
import math
import threading
import time

TARGET_ID = 'C1256W000052'
# A permission (green / left arrow) that was valid is kept this long when the head loses confidence: the vehicle that
# has started through the intersection sees the lamp from a worse angle and must not stop in the middle of it.
PERMISSION_LATCH_S = 4.
TOPIC = '/ve/GetTrafficLightStatus'


class SignalPolicy:
    permission_codes = (33,)
    target_id = TARGET_ID
    head_source = 'three_class'
    accepted_codes = (1, 16, 33)
    # Gate constants. Named so the read-only audit and the offline diagnosis report the real values.
    FRESH_MAX_S = .5
    REQUIRED_SAMPLES = 3
    REQUIRED_HOLD_S = .3
    MIN_CONFIDENCE = .9
    PERMISSION_LATCH_S = PERMISSION_LATCH_S

    def __init__(self):
        self.audit_counters = dict(permission_resets=0, stale_invalidations=0, last_reset_reason=None)
        self._audit_raw = None
        self._audit_gate = None
        self.invalidate()

    def _reset_allow(self, reason):
        """Same reset as before; the audit only notes that accumulated permission was lost."""
        if getattr(self, 'allow_samples', 0):
            self.audit_counters['permission_resets'] += 1
            self.audit_counters['last_reset_reason'] = reason
        self.allow_since, self.allow_samples = None, 0

    def invalidate(self, reason='invalidated'):
        self.code = None
        self.confidence = None
        self.received = None
        self._reset_allow(reason)
        self.bound = False
        self.latched_until = None

    def accept(self, packet, points, now):
        previous_code = self.code
        self.bound = self.bind(packet['ego_pose'], points)
        row = self.select_signal(packet)
        code, confidence = row.get('code'), row.get('confidence')
        self.confidence = confidence
        # Freshness counts from the VE output (sent_ns), not from the oldest sensor of the packet: that sensor (IMU or
        # vehicle status, already limited to 0.5 s by the Inference) put the phase age at 350-490 ms in normal running,
        # so one late frame crossed FRESH_MAX_S, reset a granted green and braked the car short of the line (7 of the
        # GO->STOP drops in 223 recorded runs; 3 left, all from VE gaps > 0.5 s or VE reading red).
        self.received = packet.get('sent_ns', packet['sensor_received_ns']) / 1e9
        valid = (type(code) is int and code in self.accepted_codes
                 and type(confidence) in (float, int) and math.isfinite(confidence)
                 and self.MIN_CONFIDENCE <= confidence <= 1 and self.bound
                 and 0 <= now-self.received <= self.FRESH_MAX_S)
        fresh = 0 <= now-self.received <= self.FRESH_MAX_S
        latched = False
        if valid:
            self.code = code
            self.latched_until = now+self.PERMISSION_LATCH_S if code in self.permission_codes else None
        elif (previous_code in self.permission_codes and self.latched_until is not None
                and now <= self.latched_until and self.bound and fresh
                and not (code == 1 and type(confidence) in (float, int) and confidence >= self.MIN_CONFIDENCE)):
            self.code = previous_code      # permission kept briefly; a confident red always cancels it
            latched = True
        else:
            self.code = None
            self.latched_until = None
        self._audit_raw = dict(latched=latched, code=code if type(code) is int else None,
                               confidence=float(confidence) if type(confidence) in (float, int) and math.isfinite(confidence) else None,
                               valid=bool(valid), reject=None if valid else self._reject_reason(code, confidence, now))
        if self.code != previous_code:
            self._reset_allow('code_changed')
        if self.code not in self.permission_codes:
            self._reset_allow('code_not_permission')
        else:
            if self.allow_since is None:
                self.allow_since = now
            self.allow_samples += 1

    def _reject_reason(self, code, confidence, now):
        if not (type(code) is int and code in self.accepted_codes):
            return 'no_signal_in_packet' if code is None else 'code_not_accepted'
        if not (type(confidence) in (float, int) and math.isfinite(confidence) and self.MIN_CONFIDENCE <= confidence <= 1):
            return 'low_confidence'
        if not self.bound:
            return 'unbound_or_unsupported'
        return 'stale_at_accept'

    def select_signal(self, packet):
        return packet.get('signal') or {}

    def bind(self, pose, points):
        if not points:
            return False
        distances = [(float(p[0])-pose['x'])**2 + (float(p[1])-pose['y'])**2 for p in points]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        return 1185-1031 <= nearest <= 1297-1031 and distances[nearest] <= 2.0**2

    def output(self, now):
        if self.received is None or not 0 <= now-self.received <= self.FRESH_MAX_S:
            had_packet = self.received is not None
            if had_packet:
                self.audit_counters['stale_invalidations'] += 1
            self._audit_gate = dict(output=0, reason='stale_invalidated' if had_packet else 'no_packet')
            self.invalidate('stale' if had_packet else 'no_packet')
            return 0
        if self.code in self.permission_codes and (self.allow_samples < self.REQUIRED_SAMPLES
                                                   or now-self.allow_since < self.REQUIRED_HOLD_S):
            self._audit_gate = dict(output=0, reason='debounce_samples' if self.allow_samples < self.REQUIRED_SAMPLES
                                    else 'debounce_duration')
            return 0
        result = self.code or 0
        self._audit_gate = dict(output=result, reason=None if self.code else (self._audit_raw or {}).get('reject') or 'no_valid_code')
        return result

    def audit(self, now):
        """Read-only snapshot of the gate after output(now). Only states that exist are reported."""
        raw, gate = self._audit_raw or {}, self._audit_gate or {}
        received = self.received
        row = dict(
            raw_signal_code=raw.get('code'), raw_signal_confidence=raw.get('confidence'),
            raw_accepted_by_gate=raw.get('valid'), raw_reject_reason=raw.get('reject'),
            permission_latched=raw.get('latched'), latch_s=self.PERMISSION_LATCH_S,
            received_timestamp=received,
            signal_age_ms=None if received is None else (now-received)*1000,
            accepted_code=self.code, head_source=self.head_source, target_id=self.target_id, bound=self.bound,
            allow_samples=self.allow_samples,
            allow_duration_ms=None if self.allow_since is None else (now-self.allow_since)*1000,
            required_samples=self.REQUIRED_SAMPLES, required_duration_ms=self.REQUIRED_HOLD_S*1000,
            fresh_max_ms=self.FRESH_MAX_S*1000,
            gate_output_code=gate.get('output'), gate_rejection_reason=gate.get('reason'),
            permission_resets=self.audit_counters['permission_resets'],
            stale_invalidations=self.audit_counters['stale_invalidations'],
            last_reset_reason=self.audit_counters['last_reset_reason'])
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}


class FullRouteSignalPolicy(SignalPolicy):
    """Bind phase to the next mapped stop. No simulator light state is read.

    Yellow on straight signals: the red/green head reads a yellow lamp as green (it never saw yellow in training), so a
    car reaching the line during the 4 s yellow crossed it. When the packet carries the five-phase head
    (``phase_signal``), a green from the red/green head is turned into a stop while at least PHASE_STOP_VOTES of the
    last PHASE_WINDOW phase frames say yellow or red, the stop line is within PHASE_RANGE_M (where that head is
    reliable) and the car can still stop comfortably before it; closer or faster it keeps going (dilemma zone). Once
    taken, the stop holds for PHASE_HOLD_S on that signal, so neither a phase frame flickering back to green nor the
    red/green head reading green again after a red frame (it does so for a lamp seen from closer than ~11 m) sends
    the car off during the yellow; it is taken once per signal and only by a car moving at PHASE_MIN_SPEED_MPS or
    more, so a false yellow costs at most that long (a real one is red by then and the red/green head reads red from
    12 m and beyond). A head that kept reading a green lamp as yellow held a waiting car for 90 s before. Right after a red -> green start the check
    waits PHASE_ARM_S (only after a red of RED_CONFIRM_S or longer): a fresh green is never followed by yellow
    that soon.
    A green from the red/green head less than RED_CONFIRM_S after it read red (RED_MIN_SAMPLES in a row) needs the
    five-phase head to agree (its last PHASE_GO_FRAMES frames green): a real red lasts 60-90 s, so such a green is
    either the end of a false red (the five-phase head agrees at once) or that head reading the yellow/red lamp as
    green from closer than ~11 m, which sent a car stopped 6 m before the line through the red.
    Without ``phase_signal`` the decisions are unchanged.
    """
    permission_codes = (16, 33)

    OFF_ROUTE_M = 6.
    PHASE_STOP_CODES = (1, 4, 33)
    PHASE_MIN_CONFIDENCE = .6
    PHASE_WINDOW, PHASE_STOP_VOTES = 3, 3   # 2 of 3: 3 false stops in 90 replayed approaches, 3 of 3: 1, same yellows
    PHASE_RANGE_M = (12., 50.)
    PHASE_ARM_S = 3.
    PHASE_HOLD_S = 4.
    PHASE_GO_CODES = (16, 48)
    RED_CONFIRM_S, RED_MIN_SAMPLES = 8., 3
    PHASE_MIN_SPEED_MPS = 2.
    PHASE_GO_FRAMES = 5                    # 3 green blips released two cars stopped at 10.5-11 m into the red
    COMFORT_DECEL_MPS2 = 3.
    STOP_MARGIN_M = 2.

    def __init__(self, points, targets, wide_zone=None, wide_zones=()):
        """``wide_zone=(first_index, last_index, limit_m)`` widens the off-route limit where the Expert is
        designed to leave the global path (highway lane changes stay within its +-6 m zone around the centreline).
        ``wide_zones`` adds more such ranges (e.g. the GPS-blackout tunnel, where avoidance may use the whole road)."""
        super().__init__()
        self.wide_zone = wide_zone
        self.wide_zones = ([wide_zone] if wide_zone else []) + list(wide_zones)
        self.points, self.targets = list(points), sorted(targets, key=lambda r:r['path_index'])
        self.route_s = [0.]
        for a,b in zip(points,points[1:]):
            self.route_s.append(self.route_s[-1]+math.hypot(a[0]-b[0],a[1]-b[1]))
        self.last_index = 0
        self.target_id = ''
        self.reason = 'waiting'
        self.head_source = 'unsupported'
        self.distance = None
        self.phase_stop = False
        self._phase_calls = deque(maxlen=self.PHASE_WINDOW)
        self._green_start = None
        self._phase_hold = None            # (signal id, hold end)
        self._phase_used = None            # signal id whose one phase stop is spent
        self._phase_go = deque(maxlen=self.PHASE_GO_FRAMES)
        self._red = None                   # [signal id, first red time, reds in a row, confirmed red start]

    def accept(self, packet, points, now):
        super().accept(packet, points, now)
        self.phase_stop = False
        native = self.code
        if self._red is None or self._red[0] != self.target_id:
            self._red = [self.target_id, None, 0, None]
        if native == 16 and self._red[2] >= self.RED_MIN_SAMPLES and now - self._red[1] >= self.RED_CONFIRM_S:
            self._green_start = now        # only a real red ends in a fresh green; red frames on a yellow lamp do not
        if native == 1:
            if not self._red[2]:
                self._red[1] = now
            self._red[2] += 1
            if self._red[2] >= self.RED_MIN_SAMPLES and self._red[3] is None:
                self._red[3] = self._red[1]
        elif native == 16:
            self._red[2] = 0
        phase = packet.get('phase_signal')
        if self.head_source != 'native_binary' or not isinstance(phase, dict):
            self._phase_calls.clear()
            self._phase_go.clear()
            return
        confidence = phase.get('confidence')
        sure = type(confidence) in (float, int) and confidence >= self.PHASE_MIN_CONFIDENCE
        self._phase_calls.append(phase.get('code') in self.PHASE_STOP_CODES and sure)
        self._phase_go.append(phase.get('code') in self.PHASE_GO_CODES and sure)
        if native == 16 and self._red[3] is not None:
            if now - self._red[3] >= self.RED_CONFIRM_S or (len(self._phase_go) == self.PHASE_GO_FRAMES and all(self._phase_go)):
                self._red[3] = None
            else:
                self.code = 1
                self.latched_until = None
                self._reset_allow('green_unconfirmed')
                self.phase_stop = True
                return
        hold = self._phase_hold
        if hold is not None and (hold[0] != self.target_id or now >= hold[1]):
            hold = self._phase_hold = None
        if native != 16:
            return
        if hold is None:
            if sum(self._phase_calls) < self.PHASE_STOP_VOTES or self._phase_used == self.target_id:
                return
            if self._green_start is not None and now - self._green_start < self.PHASE_ARM_S:
                return
            if self.distance is None or not self.PHASE_RANGE_M[0] <= self.distance <= self.PHASE_RANGE_M[1]:
                return
            speed = abs(float(packet['ego_pose'].get('speed_mps') or 0.))
            if speed < self.PHASE_MIN_SPEED_MPS:
                return                     # waiting at the line: a green there cannot turn yellow before the car goes
            if self.distance - self.STOP_MARGIN_M <= speed * speed / (2. * self.COMFORT_DECEL_MPS2):
                return                     # too close to stop comfortably: keep going through the yellow
            self._phase_hold = (self.target_id, now + self.PHASE_HOLD_S)
            self._phase_used = self.target_id
        self.code = 1
        self.latched_until = None
        self._reset_allow('phase_yellow')
        self.phase_stop = True

    def select_signal(self, packet):
        key = {'native_binary': 'native_signal', 'three_class': 'signal'}.get(self.head_source)
        return (packet.get(key) or {}) if key else {}

    def bind(self, pose, points):
        self.head_source = 'unsupported'
        self.accepted_codes = ()
        self.permission_codes = ()
        if not self.points:
            self.reason = 'no_route'
            return False
        indices = range(max(0,self.last_index-30), min(len(self.points),self.last_index+201))
        nearest = min(indices,key=lambda i:(self.points[i][0]-pose['x'])**2+(self.points[i][1]-pose['y'])**2)
        lateral = math.hypot(self.points[nearest][0]-pose['x'],self.points[nearest][1]-pose['y'])
        limit = self.OFF_ROUTE_M
        for first, last, wide in self.wide_zones:
            if first <= nearest <= last:
                limit = max(limit, wide)
        if lateral > limit:
            self.reason = 'off_route'
            return False
        self.last_index = nearest
        target = next((r for r in self.targets if r['path_index'] >= nearest),None)
        identity = target['signal_id'] if target else ''
        if identity != self.target_id:
            self._reset_allow("target_changed")
        self.target_id = identity
        if target is None:
            self.reason = 'no_upcoming_signal'
            self.distance = None
            return False
        distance = self.route_s[target['path_index']]-self.route_s[nearest]
        self.distance = distance
        if target['signal_movement'] == 'straight' and identity.startswith('C'):
            self.head_source = 'native_binary'
            self.accepted_codes, self.permission_codes = (1, 16), (16,)
        elif target['signal_movement'] == 'left' and identity == TARGET_ID:
            self.head_source = 'three_class'
            self.accepted_codes, self.permission_codes = (1, 16, 33), (33,)
        supported = self.head_source != 'unsupported'
        self.reason = ('unsupported_signal' if not supported else
                       'outside_60m' if distance > 60 else 'bound')
        return supported and distance <= 60


def verify_signal_topology(state, own_node, expert_pid, allow_starting=False):
    pubs, subs = dict(state[0]), dict(state[1])
    if pubs.get(TOPIC) != [own_node]:
        raise RuntimeError('VE 신호 발행자가 중복되었거나 종료되었습니다')
    prefix = '/morai_standard_{}_'.format(expert_pid)
    experts = {n for nodes in subs.values() for n in nodes if n.startswith(prefix)}
    if experts.intersection(subs.get('/GetTrafficLightStatus', [])):
        raise RuntimeError('Expert에 GT 신호 구독이 남아 있습니다')
    owners = pubs.get('/ctrl_cmd', [])
    ready = bool(owners) and set(owners).issubset(subs.get(TOPIC, []))
    if not ready and not allow_starting:
        raise RuntimeError('Expert의 VE 신호 입력 연결이 끊겼습니다')
    return ready


def build_signal_targets(context, config):
    """Use the same open/closed route contract as this invocation's Expert."""
    import importlib.util
    from pathlib import Path
    expert_root = Path(context.global_cfg['paths']['expert_root'])
    source = expert_root/'src/planning/traffic/route_targets.py'
    if not source.is_file():
        source = expert_root/'src/expert/src/planning/traffic/route_targets.py'
    spec = importlib.util.spec_from_file_location('ve_trial_route_signals', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_route_signal_targets(
        context.route_points, config['map']['data_dir'],
        config['map']['route_signal_route_links'],
        is_closed_path=bool(config['map'].get('is_closed_path', False)))


class SignalPublisher:
    def __init__(self, context, session, output):
        from morai_msgs.msg import GetTrafficLightStatus
        self.context, self.session = context, session
        self.message_type = GetTrafficLightStatus
        self.policy, self.lock = SignalPolicy(), threading.Lock()
        self.stream = (output/'signal_delivery.jsonl').open('w', buffering=1)
        self.publisher = session.ros.Publisher(TOPIC, GetTrafficLightStatus, queue_size=1)
        self.last_log = 0.
        # Asynchronous sensor inference uses monotonic receipt timestamps.
        # ROS simulated time may stay at zero or reset between scenarios;
        # neither must suspend delivery or stale-evidence invalidation.
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._publish_loop, name='ve_signal_publisher', daemon=True)
        self.thread.start()

    def _publish_loop(self):
        while not self.stop_event.is_set():
            self.tick(None)
            if self.stop_event.wait(.1):
                return

    def validate_route(self):
        if self.context.cfg.get('ve_full_route_signals'):
            from pathlib import Path
            controller = self.context.expert_controller
            path = Path(controller.runtime_config_path)
            config = json.loads(path.read_text())
            points = self.context.route_points
            document = build_signal_targets(self.context, config)
            ignored = set(config['planning']['traffic_light_stop'].get('ignored_signal_ids',[]))
            targets = [r for r in document['targets'] if r['signal_id'] not in ignored]
            with self.lock:
                self.policy = FullRouteSignalPolicy(points, targets)
            # Runtime-only fail-closed handling. Original Expert files stay untouched.
            settings = config['planning']['traffic_light_stop']
            settings['stop_statuses'] = sorted(set(settings.get('stop_statuses',[1])) | {0})
            settings['hold_on_stale'] = True
            config['map']['traffic_light_control'] = False
            path.write_text(json.dumps(config,indent=2)+'\n')
            (path.parent/'ve_signal_targets.json').write_text(json.dumps(document,indent=2)+'\n')
            print('[VE 풀루트 신호] '+repr([(r['signal_id'],r['signal_movement']) for r in targets]),flush=True)
            return
        window = getattr(self.context, 'mission_route', None)
        if (window is None or window.first_index != 1031 or window.last_index != 1348
                or len(self.context.route_points) != 318):
            raise RuntimeError('VE 신호 ID 연결은 검증된 1031..1348 시험 경로에서만 허용됩니다')

    def accept(self, packet):
        with self.lock:
            # Flush any previous permission on a gap before accepting a new sample.
            self.policy.output(time.monotonic())
            self.policy.accept(packet, getattr(self.context, 'route_points', []), time.monotonic())

    def invalidate(self):
        with self.lock:
            self.policy.invalidate()

    def tick(self, _event):
        try:
            now = time.monotonic()
            with self.lock:
                code = self.policy.output(now)
                row = dict(monotonic=now, target_id=self.policy.target_id, binding_reason=getattr(self.policy,'reason','single_intersection'), prediction=self.policy.code,
                           head_source=self.policy.head_source,
                           confidence=self.policy.confidence, route_bound=self.policy.bound,
                           sensor_received=self.policy.received, delivered_code=code)
            message = self.message_type()
            message.trafficLightIndex = row['target_id']
            message.trafficLightStatus = code
            message.trafficLightType = 0
            if hasattr(message, 'header'):
                message.header.stamp = self.session.ros.Time.now()
            self.publisher.publish(message)
            self.stream.write(json.dumps(row, allow_nan=False)+'\n')
            if now-self.last_log >= 2:
                print('[VE 신호] ID={} head={} 예측={} 신뢰도={} 구간={} Expert전달={} 사유={} (0=미확정)'.format(
                    row['target_id'], row['head_source'], row['prediction'], row['confidence'], row['route_bound'], code, row['binding_reason']), flush=True)
                self.last_log = now
        except Exception as exc:
            self.invalidate()
            self.session.error = 'VE signal bridge: '+repr(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.publisher.unregister()
        self.stream.close()
