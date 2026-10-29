"""Source-clock wheel/IMU odometry and causal, motion-compensated GPS anchoring.

No ground-truth inputs or tunnel GPS corrections. GPS receipt alignment remains
an explicit approximation when the receiver provides no precise acquisition time.

While GPS is available the published position is the odometry plus the GPS offset (fused), not the raw fix: NMEA
positions arrive at 10 Hz in 0.15-0.19 m steps, so the raw fix held still for several control ticks and then jumped,
and steering followed those steps. The offset (mean of GPS minus odometry over the anchor window) is followed with a
time constant instead of being applied at once; a large offset change (GPS back after a blackout) is followed faster.
Dead reckoning (GPS lost) keeps the last window offset unchanged.
Each fused position carries its status sample's clocks, like the dead-reckoning output.
"""
from collections import deque
import math
import time

from .gps_blackout_hold import GpsBlackoutHold, MAX_STATUS_GAP_S

OFFSET_TAU_S = .7          # time constant for following a new GPS offset [s]
OFFSET_FAST_GAP_M = .5     # an offset change larger than this ...
OFFSET_FAST_TAU_S = .15    # ... is followed with this time constant [s]
# Status samples without a matching IMU sample wait (no position meanwhile) for the next IMU sample; once it arrives
# within this time of the last integrated sample, the heading over the gap is interpolated between the two.
IMU_GAP_MAX_S = 1.
# Longest source-clock interval between two status samples that is still integrated (trapezoid of their speeds and
# headings). A lagging simulator sent status and IMU 0.3-0.54 s apart (10-27): with the old 0.25 s limit the tunnel
# dead reckoning ended there for good. No position is published while the next sample is late.
TRACK_GAP_MAX_S = 1.


def forward_vector(value, dt=0.):
    """Body-forward vector in true ENU; optionally propagate a recent IMU gyro."""
    q = value.get('quaternion')
    if q is None:
        return math.cos(value['yaw']), math.sin(value['yaw'])
    x, y, z, w = q
    omega = value.get('angular_velocity', (0., 0., 0.))
    angle = math.sqrt(sum(v*v for v in omega))*dt
    if abs(angle) > 1e-12:
        scale = math.sin(angle/2)/math.sqrt(sum(v*v for v in omega))
        a, b, c = (v*scale for v in omega)
        d = math.cos(angle/2)
        x, y, z, w = (w*a+x*d+y*c-z*b, w*b-x*c+y*d+z*a,
                      w*c+x*b-y*a+z*d, w*d-x*a-y*b-z*c)
    return 1-2*(y*y+z*z), 2*(x*y+w*z)


def aligned_forward(history, identity):
    """Match IMU and wheel acquisition clocks, using only already received data."""
    target = identity.get('source_stamp_ns', 0)
    stamped = [(v, i) for v, i in history if target and i.get('source_stamp_ns', 0)]
    before = [(v, i) for v, i in stamped if i['source_stamp_ns'] <= target]
    after = [(v, i) for v, i in stamped if i['source_stamp_ns'] >= target]
    if before:
        v0, i0 = before[-1]
        dt = (target-i0['source_stamp_ns'])/1e9
        if dt <= .05:
            if after:
                v1, i1 = after[0]
                span = (i1['source_stamp_ns']-i0['source_stamp_ns'])/1e9
                if 0 < span <= .05:
                    fraction = dt/span
                    a, b = forward_vector(v0), forward_vector(v1)
                    return tuple(x+fraction*(y-x) for x, y in zip(a, b)), dict(
                        method='source_interpolation', source_delta_s=0.)
            return forward_vector(v0, dt), dict(method='source_gyro_propagation', source_delta_s=dt)
    # An unstamped protocol cannot establish acquisition alignment. Keep the
    # historical receipt-causal fallback and disclose it rather than invent time.
    for value, iid in reversed(history):
        age = (identity['received_ns']-iid['received_ns'])/1e9
        if 0 <= age <= MAX_STATUS_GAP_S:
            return forward_vector(value), dict(method='receipt_causal', source_delta_s=None)
    return None, dict(method='missing')


def bridged_forward(last, history, identity, now_ns):
    """Heading for a status sample that no IMU sample matches: from the last integrated state to the next IMU sample.

    A short IMU gap (a lost packet, an IMU history emptied by a source-clock reset) used to end dead reckoning for
    good: the error stayed until GPS returned, so a car in the tunnel blackout stopped there. Returns
    (forward, alignment); forward None with method 'waiting' while the next IMU sample may still come in time.
    """
    if last is None:
        return None, dict(method='missing')
    receipt, start = identity['received_ns'], last['received_ns']
    after = next(((v, i) for v, i in history if i['received_ns'] >= receipt), None)
    if after is not None:
        span = after[1]['received_ns']-start
        if 0 < span <= IMU_GAP_MAX_S*1e9:
            fraction = (receipt-start)/span
            a, b = last['forward'], forward_vector(after[0])
            return tuple(x+fraction*(y-x) for x, y in zip(a, b)), dict(
                method='gap_interpolation', source_delta_s=None, gap_s=round(span/1e9, 3))
        return None, dict(method='missing')
    if now_ns-start <= IMU_GAP_MAX_S*1e9:
        return None, dict(method='waiting')
    return None, dict(method='missing')


class MotionTrack:
    """Continuous relative odometry, independent of GPS anchors and polling batches."""
    def __init__(self, convergence):
        self.convergence = convergence
        self.states = deque(maxlen=4096)
        self.failure = None
        self.waiting = None         # why the newest status samples are not integrated yet (not a failure)
        # Receipt of the last state integrated across a gap. The motion over a gap is assumed, not measured: at a
        # run start MORAI held the car still for 0.38 s while the status kept its 4 m/s, and the track gained 1.4 m.
        self.discontinuity_ns = None

    def update(self, speed, imu, now_ns):
        last = self.states[-1] if self.states else None
        self.waiting = None
        seen = [(v, i) for v, i in imu if i['received_ns'] <= now_ns]
        for value, identity in speed:
            receipt = identity['received_ns']
            if receipt > now_ns or (last and receipt <= last['received_ns']):
                continue
            forward, alignment = aligned_forward(seen, identity)
            if forward is None:
                forward, alignment = bridged_forward(last, seen, identity, now_ns)
            if forward is None:
                if alignment['method'] == 'waiting':
                    self.waiting = 'imu_missing_or_stale'
                else:
                    self.failure = 'imu_missing_or_stale'
                break
            speed_mps = value['signed_speed']/3.6
            if not all(math.isfinite(v) for v in (*forward, speed_mps)) or abs(speed_mps) > 100:
                self.failure = 'invalid_status_sample'
                break
            angle = self.convergence()
            c, s = math.cos(angle), math.sin(angle)
            vx, vy = speed_mps*(c*forward[0]-s*forward[1]), speed_mps*(s*forward[0]+c*forward[1])
            source = identity.get('source_stamp_ns', 0)
            point, distance = (0., 0.), 0.
            if last:
                source_pair = source and last['source_stamp_ns']
                dt = ((source-last['source_stamp_ns']) if source_pair else (receipt-last['received_ns']))/1e9
                if not 0 < dt <= TRACK_GAP_MAX_S:
                    self.failure = 'invalid_motion_interval'
                    break
                point = (last['xy'][0]+.5*(last['velocity'][0]+vx)*dt,
                         last['xy'][1]+.5*(last['velocity'][1]+vy)*dt)
                distance = last['distance_m']+math.hypot(point[0]-last['xy'][0], point[1]-last['xy'][1])
                if dt > MAX_STATUS_GAP_S or alignment['method'] == 'gap_interpolation':
                    self.discontinuity_ns = receipt
            last = dict(received_ns=receipt, source_stamp_ns=source, xy=point, velocity=(vx, vy),
                        yaw=math.atan2(forward[1], forward[0]), forward=tuple(forward), speed_mps=speed_mps,
                        distance_m=distance, alignment=alignment)
            self.states.append(last)

    def at(self, received_ns):
        if not self.states or received_ns < self.states[0]['received_ns']:
            return None
        previous = self.states[0]
        for current in self.states:
            if current['received_ns'] < received_ns:
                previous = current
                continue
            span = current['received_ns']-previous['received_ns']
            fraction = (received_ns-previous['received_ns'])/span if span else 0.
            return dict(previous, xy=tuple(a+fraction*(b-a) for a, b in zip(previous['xy'], current['xy'])),
                        distance_m=previous['distance_m']+fraction*(current['distance_m']-previous['distance_m']))
        dt = (received_ns-previous['received_ns'])/1e9
        if dt > MAX_STATUS_GAP_S:
            return None
        return dict(previous, xy=tuple(a+dt*v for a, v in zip(previous['xy'], previous['velocity'])),
                    distance_m=previous['distance_m']+dt*math.hypot(*previous['velocity']))


class AlignedGpsBlackoutHold(GpsBlackoutHold):
    def __init__(self, *args, anchor_window_s=1., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(anchor_window_s) or not 0 <= anchor_window_s <= 5:
            raise ValueError('anchor window must be within 0..5 s')
        if self.core.lever != (0., 0.) or self.core.latency_s != 0.:
            raise ValueError('aligned hold currently requires zero, uncalibrated lever and latency settings')
        self.anchor_window_s = anchor_window_s
        self.offsets = deque()
        self.track = MotionTrack(lambda: math.radians(self.core.convergence_deg(self.core.xy or (0., 0.))))
        self.offset = None
        self.anchor_distance = 0.
        self.anchor_metadata = {}
        self.fuse_gps = True        # publish odometry + smoothed offset while GPS is available
        self.smooth_offset = self.smooth_ns = None

    def _anchor_on_fix(self, fix, identity, speed, imu):
        now = time.monotonic_ns()
        self.track.update(speed, imu, now)
        if self.track.failure:
            self.track = MotionTrack(self.track.convergence)
            self.offsets.clear()
            self.track.update(speed[-2:], imu, now)
        point = self.track.at(identity['received_ns'])
        raw = self._grid(fix)
        if point is None or self.track.failure:
            self.offsets.clear()
            self.offset = None
            self.smooth_offset = self.smooth_ns = None
            super()._anchor_on_fix(fix, identity, speed, imu)
            return
        while self.offsets and identity['received_ns']-self.offsets[0][0] > self.anchor_window_s*1e9:
            self.offsets.popleft()
        gap = self.track.discontinuity_ns
        if gap is not None and self.offsets and self.offsets[0][0] < gap:
            # GPS before a bridged gap does not describe the track after it: start the offset from this fix, as a
            # new track would (no 1 s mean across the gap, no smoothing from the old offset).
            while self.offsets and self.offsets[0][0] < gap:
                self.offsets.popleft()
            self.smooth_offset = self.smooth_ns = None
        self.offsets.append((identity['received_ns'], tuple(a-b for a, b in zip(raw, point['xy']))))
        self.offset = tuple(sum(item[1][k] for item in self.offsets)/len(self.offsets) for k in (0, 1))
        anchor = tuple(a+b for a, b in zip(self.offset, point['xy']))
        self.core.anchor(anchor, identity['received_ns'], point['yaw'], point['speed_mps'])
        self.anchor_distance = point['distance_m']
        self.anchor_metadata = dict(method='causal_motion_compensated_mean', window_s=self.anchor_window_s,
                                    fixes=len(self.offsets), correction_xy=[a-b for a, b in zip(anchor, raw)],
                                    GPS_inside_blackout_used=False)
        self._last_source_stamp_ns = 0
        self._last_status_identity = {}

    def _advance(self, now_ns, speed, imu):
        if self.offset is None:
            return super()._advance(now_ns, speed, imu)
        self.track.update(speed, imu, now_ns)
        rows = []
        for state in self.track.states:
            if state['received_ns'] <= self.core.last_ns or state['received_ns'] > now_ns:
                continue
            self.core.xy = self.core.ref_xy = tuple(a+b for a, b in zip(self.offset, state['xy']))
            self.core.last_ns = state['received_ns']
            # The followed offset is the one dead reckoning uses, up to now: when GPS returns, the new offset is
            # followed from here. Otherwise the whole blackout counts as elapsed time and the correction lands in one
            # tick (a 6-8 cm lateral and 0.3 m longitudinal step at the tunnel exit).
            self.smooth_offset, self.smooth_ns = self.offset, state['received_ns']
            horizontal = math.hypot(*state['velocity']) * (-1 if state['speed_mps'] < 0 else 1)
            self.core.last_sample = (horizontal, state['yaw'])
            self.core.distance_m = max(0., state['distance_m']-self.anchor_distance)
            self.core.steps += 1
            self._last_status_identity = dict(source_stamp_ns=state['source_stamp_ns'])
            self._record_state()
            rows.append(state)
        if self.track.failure:
            self.core.failure = self.track.failure
        elif self.core.distance_m > self.core.max_distance_m:
            self.core.failure = 'hold_distance_limit'
        return rows

    def _stale_reason(self):
        # Waiting for an IMU sample (or status samples a little late) is refused without ending the hold: a gap
        # the track cannot bridge sets its own failure (imu_missing_or_stale, invalid_motion_interval).
        if self.offset is None:
            return super()._stale_reason()
        return self.track.waiting or 'status_stale'

    def _comparison_metadata(self):
        return dict(super()._comparison_metadata(), anchor_filter=self.anchor_metadata,
                    motion_method='continuous_source_clock_quaternion_forward')

    def _smoothed(self, at_ns):
        """The GPS offset followed with OFFSET_TAU_S (OFFSET_FAST_TAU_S for a large change) up to ``at_ns``."""
        if self.smooth_offset is None:
            self.smooth_offset, self.smooth_ns = self.offset, at_ns
            return self.smooth_offset
        dt = max(0., (at_ns-self.smooth_ns)/1e9)
        gap = math.hypot(self.offset[0]-self.smooth_offset[0], self.offset[1]-self.smooth_offset[1])
        weight = 1.-math.exp(-dt/(OFFSET_FAST_TAU_S if gap > OFFSET_FAST_GAP_M else OFFSET_TAU_S))
        self.smooth_offset = tuple(a+weight*(b-a) for a, b in zip(self.smooth_offset, self.offset))
        self.smooth_ns = max(self.smooth_ns, at_ns)
        return self.smooth_offset

    def latest_snapshot(self):
        fix, age, identity = super().latest_snapshot()
        if fix is None or identity.get('position_mode') != 'gps' or not self.fuse_gps:
            return fix, age, identity
        with self.lock:
            if self.offset is None:
                return fix, age, identity
            now = time.monotonic_ns()
            self.track.update(*self._histories(), now)
            state = self.track.states[-1] if self.track.states and not self.track.failure else None
            if state is None or not 0 <= now-state['received_ns'] <= MAX_STATUS_GAP_S*1e9:
                return fix, age, identity           # no fresh odometry: the raw fix
            offset = self._smoothed(state['received_ns'])
            xy = tuple(a+b for a, b in zip(offset, state['xy']))
            fused = self._fix_from_grid(xy)
            raw = self._grid(fix)
            meta = dict(identity, received_ns=state['received_ns'], source_stamp_ns=state['source_stamp_ns'],
                        fused_gps=dict(raw_gps_received_ns=identity.get('received_ns'),
                                       raw_minus_fused_xy=[round(a-b, 3) for a, b in zip(raw, xy)],
                                       pending_offset_m=round(math.hypot(self.offset[0]-offset[0],
                                                                         self.offset[1]-offset[1]), 3)))
            return fused, (now-state['received_ns'])/1e9, meta
