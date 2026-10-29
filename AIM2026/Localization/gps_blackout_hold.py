"""Hold the GPS position through a GPS blackout with IMU heading and status speed.

While the last valid fix is fresh it passes through unchanged.
After it goes stale, the position is integrated from that fix with the allowed
UDP inputs only: IMU orientation (true ENU, converted to the map grid with the
meridian convergence) and the signed Competition Vehicle Status speed. Every
held sample is labelled ``position_mode='dead_reckoning'`` so consumers can
refuse it unless explicitly enabled. Limits on hold time, distance and missing
status samples return no position, which keeps the existing brake-on-stale
behaviour.

``simulate_zone`` is for development only: it discards real fixes whose own
position lies inside the zone, imitating a geographic blackout in a simulator
that does not black out GPS.
"""
from collections import deque
import math
import threading
import time

FRESH_GPS_S = .15
MAX_STATUS_GAP_S = .25


class DeadReckoner:
    """Socket-free core. Positions are map-grid metres; yaw inputs are true-ENU radians.

    ``xy`` is the GPS antenna position (what consumers treat as the GPS fix). The status speed belongs to the
    vehicle reference point, so the reference point is integrated and the physical antenna offset ``lever_xy``
    (vehicle frame: x forward, y left, metres) is re-applied with the current heading.

    ``latency_s`` is the GPS delay before receipt. A fix describes where the vehicle was that long before it
    arrived, so the anchor is moved forward by speed * latency along the heading. Without this the anchor
    error is frozen in the entry heading and turns sideways when the vehicle swerves. (Receipt-to-use age is
    already removed because integration starts at the receipt time.)
    """

    def __init__(self, convergence_deg, max_hold_s=120., max_distance_m=250., lever_xy=(0., 0.), latency_s=0.):
        if not (math.isfinite(max_hold_s) and max_hold_s > 0 and math.isfinite(max_distance_m) and max_distance_m > 0):
            raise ValueError('Dead-reckoning limits must be finite and positive')
        if len(lever_xy) != 2 or not all(math.isfinite(float(v)) and abs(float(v)) < 5 for v in lever_xy):
            raise ValueError('GPS lever arm must be two finite values under 5 m')
        if not (math.isfinite(latency_s) and 0 <= latency_s <= .5):
            raise ValueError('GPS latency must be within 0..0.5 s')
        self.latency_s = float(latency_s)
        self.convergence_deg = convergence_deg
        self.max_hold_s, self.max_distance_m = float(max_hold_s), float(max_distance_m)
        self.lever = (float(lever_xy[0]), float(lever_xy[1]))
        self.anchor_xy = self.anchor_ns = None
        self.xy = None
        self.ref_xy = None
        self.lever_active = False
        self.last_ns = None
        self.distance_m = 0.
        self.steps = 0
        self.failure = None
        self.last_sample = None      # (speed_mps, true_yaw_rad) of the previous status sample

    def _offset(self, grid_yaw):
        c, s = math.cos(grid_yaw), math.sin(grid_yaw)
        return self.lever[0]*c - self.lever[1]*s, self.lever[0]*s + self.lever[1]*c

    def anchor(self, xy, received_ns, true_yaw_rad=None, speed_mps=None):
        """Re-anchor on a GPS fix. Without a heading the lever arm and latency cannot be removed and are ignored."""
        self.anchor_xy = self.xy = (float(xy[0]), float(xy[1]))
        self.anchor_ns = self.last_ns = int(received_ns)
        self.distance_m, self.steps, self.failure, self.last_sample = 0., 0, None, None
        if not all(math.isfinite(v) for v in self.xy):
            self.failure = 'invalid_gps_anchor'
            return
        heading_known = true_yaw_rad is not None and math.isfinite(true_yaw_rad)
        self.lever_active = heading_known and self.lever != (0., 0.)
        self.ref_xy = self.xy
        if heading_known:
            grid = true_yaw_rad + math.radians(self.convergence_deg(self.xy))
            ox, oy = self._offset(grid) if self.lever_active else (0., 0.)
            lead = speed_mps*self.latency_s if speed_mps is not None and math.isfinite(speed_mps) and abs(speed_mps) <= 100 else 0.
            self.ref_xy = (self.xy[0]-ox+lead*math.cos(grid), self.xy[1]-oy+lead*math.sin(grid))
            if lead:
                self.xy = (self.ref_xy[0]+ox, self.ref_xy[1]+oy)
            if speed_mps is not None and math.isfinite(speed_mps) and abs(speed_mps) <= 100:
                self.last_sample = (speed_mps, true_yaw_rad)

    def step(self, received_ns, speed_mps, true_yaw_rad, dt_s=None):
        """Advance to a status sample received at ``received_ns``. Returns False once a limit is hit."""
        if self.failure is not None:
            return False
        if self.xy is None:
            self.failure = 'no_gps_anchor'
            return False
        if received_ns <= self.last_ns:
            return True
        gap = (received_ns - self.last_ns) / 1e9
        if gap > MAX_STATUS_GAP_S:
            self.failure = 'status_gap_%.2fs' % gap
            return False
        if not all(math.isfinite(v) for v in (speed_mps, true_yaw_rad)) or abs(speed_mps) > 100:
            self.failure = 'invalid_status_sample'
            return False
        dt = gap if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or not 0 < dt <= MAX_STATUS_GAP_S:
            self.failure = 'invalid_status_interval'
            return False
        # Trapezoid: mean speed and mean heading of the previous and current sample over the interval.
        speed, heading = speed_mps, true_yaw_rad
        if self.last_sample is not None:
            speed = .5*(self.last_sample[0] + speed_mps)
            heading = self.last_sample[1] + .5*((true_yaw_rad - self.last_sample[1] + math.pi) % (2*math.pi) - math.pi)
        yaw = heading + math.radians(self.convergence_deg(self.xy))
        x, y = self.ref_xy
        self.ref_xy = (x + speed*dt*math.cos(yaw), y + speed*dt*math.sin(yaw))
        if self.lever_active:
            ox, oy = self._offset(true_yaw_rad + math.radians(self.convergence_deg(self.ref_xy)))
            self.xy = (self.ref_xy[0]+ox, self.ref_xy[1]+oy)
        else:
            self.xy = self.ref_xy
        self.distance_m += abs(speed)*dt
        self.last_sample = (speed_mps, true_yaw_rad)
        self.last_ns = int(received_ns)
        self.steps += 1
        if (received_ns - self.anchor_ns)/1e9 > self.max_hold_s:
            self.failure = 'hold_time_limit'
        elif self.distance_m > self.max_distance_m:
            self.failure = 'hold_distance_limit'
        return self.failure is None

    def summary(self):
        return dict(anchor_xy=self.anchor_xy, anchor_received_ns=self.anchor_ns, distance_m=round(self.distance_m, 3),
                    steps=self.steps, failure=self.failure, lever_arm_applied=bool(getattr(self, 'lever_active', False)))


class SectionZone:
    """Rectangle along a start->end axis (development blackout injection)."""

    def __init__(self, start_xy, end_xy, half_width_m=15., margin_m=0.):
        dx, dy = end_xy[0]-start_xy[0], end_xy[1]-start_xy[1]
        self.length = math.hypot(dx, dy)
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError('Zone axis must have finite length')
        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in (half_width_m, margin_m)):
            raise ValueError('Zone width and margin must be finite and nonnegative')
        self.start, self.fwd = start_xy, (dx/self.length, dy/self.length)
        self.half_width, self.margin = float(half_width_m), float(margin_m)

    def contains(self, xy):
        rx, ry = xy[0]-self.start[0], xy[1]-self.start[1]
        s = rx*self.fwd[0] + ry*self.fwd[1]
        lateral = -rx*self.fwd[1] + ry*self.fwd[0]
        # Projection roundoff must not place an exact portal coordinate outside its own zone.
        epsilon = 1e-7
        return -self.margin-epsilon <= s <= self.length+self.margin+epsilon and abs(lateral) <= self.half_width+epsilon


class GpsBlackoutHold:
    """Drop-in for ValidGpsReceiver (``latest``, ``latest_snapshot``, ``diagnostics``, ``close``)."""

    def __init__(self, gps, status, basis, *, max_hold_s=120., max_distance_m=250., simulate_zone=None,
                 lever_xy=(0., 0.), latency_s=0.):
        from .map_geometry import validate_map_basis
        from pyproj import Transformer
        self.gps, self.status = gps, status
        self.geometry = validate_map_basis(basis)
        self.origin = basis['origin_xyz']
        self.forward = Transformer.from_crs('EPSG:4326', basis['crs'], always_xy=True)
        self.core = DeadReckoner(lambda xy: self.geometry.convergence_degrees((xy[0], xy[1], 0.)),
                                 max_hold_s, max_distance_m, lever_xy, latency_s)
        self.zone = simulate_zone
        self.lock = threading.Lock()
        self.counts = dict(gps=0, dead_reckoning=0, refused=0, simulated_drops=0)
        self.max_hold_seen_s = 0.
        self.max_distance_seen_m = 0.
        self.last_reason = None
        self._dropped = None     # (grid xy, receipt ns) of the last real fix discarded by the development zone
        self._last_source_stamp_ns = 0
        self._last_status_identity = {}
        self._trajectory = deque(maxlen=512)
        self._pending_drops = deque(maxlen=512)
        self._comparison = None
        self._comparisons = deque(maxlen=8)

    # ValidGpsReceiver compatibility (LiveFrameSource.stream_report reads port/packets/samples/latest)
    @property
    def port(self):
        return self.gps.port

    @property
    def packets(self):
        return self.gps.packets

    @property
    def samples(self):
        return self.gps.samples

    def close(self):
        self.gps.close()

    def latest(self):
        value, age, _ = self.latest_snapshot()
        return value, age

    def _grid(self, fix):
        east, north = self.forward.transform(fix['longitude'], fix['latitude'])
        return east-self.origin[0], north-self.origin[1]

    def _fix_from_grid(self, xy):
        lon, lat = self.geometry.inverse.transform(xy[0]+self.origin[0], xy[1]+self.origin[1])
        return dict(latitude=lat, longitude=lon, altitude=None)

    def _histories(self):
        with self.status.speed.history_lock:
            speed = list(self.status.speed.history)
        with self.status.imu.history_lock:
            imu = list(self.status.imu.history)
        return speed, imu

    @staticmethod
    def _causal_sample(history, received_ns):
        """Latest sample already received at the target, with a bounded receipt age."""
        for value, identity in reversed(history):
            age_ns = received_ns - identity['received_ns']
            if age_ns >= 0:
                return (value, identity) if age_ns <= MAX_STATUS_GAP_S*1e9 else (None, {})
        return None, {}

    def _status_samples(self, after_ns, now_ns, speed, imu):
        """Chronological status samples paired with a fresh, causally received IMU."""
        rows = []
        for value, identity in speed:
            received_ns = identity['received_ns']
            if not after_ns < received_ns <= now_ns:
                continue
            heading, heading_identity = self._causal_sample(imu, received_ns)
            if heading is None:
                return rows, 'imu_missing_or_stale'
            rows.append((value, identity, heading, heading_identity))
        return rows, None

    def _record_state(self):
        sample = self.core.last_sample
        yaw = (sample[1] + math.radians(self.core.convergence_deg(self.core.xy))) if sample else None
        self._trajectory.append(dict(received_ns=self.core.last_ns, xy=self.core.xy,
                                     yaw=yaw, speed=sample[0] if sample else None,
                                     distance_m=self.core.distance_m))

    def _advance(self, now_ns, speed, imu):
        rows, failure = self._status_samples(self.core.last_ns, now_ns, speed, imu)
        for value, sid, heading, heading_identity in rows:
            source = sid.get('source_stamp_ns', 0)
            dt = None
            if source and self._last_source_stamp_ns:
                dt = (source-self._last_source_stamp_ns)/1e9
            if not self.core.step(sid['received_ns'], value['signed_speed']/3.6, heading['yaw'], dt):
                break
            self._last_source_stamp_ns = source
            self._last_status_identity = dict(sid, imu_received_ns=heading_identity['received_ns'])
            self._record_state()
        if failure:
            self.core.failure = self.core.failure or failure
        return rows

    def _compare_pending(self):
        """Evaluate DR at the GPS *receipt* time, independently of the dropped GPS coordinates.

        Interpolating two adjacent DR states aligns monotonic receipt times. NMEA
        acquisition latency is unknown, so this is not a source-time ground-truth comparison.
        A GPS fix newer than the last DR state waits for another status sample.
        """
        ready = []
        while self._pending_drops:
            xy, received_ns = self._pending_drops[0]
            if not self._trajectory or received_ns > self._trajectory[-1]['received_ns']:
                break
            self._pending_drops.popleft()
            if received_ns < self._trajectory[0]['received_ns']:
                continue
            previous = self._trajectory[0]
            for current in self._trajectory:
                if current['received_ns'] < received_ns:
                    previous = current
                    continue
                span_ns = current['received_ns'] - previous['received_ns']
                weight = (received_ns-previous['received_ns'])/span_ns if span_ns else 0.
                if (span_ns > MAX_STATUS_GAP_S*1e9 or previous['yaw'] is None
                        or current['yaw'] is None):
                    break
                estimated = [a+weight*(b-a) for a, b in zip(previous['xy'], current['xy'])]
                yaw_delta = (current['yaw']-previous['yaw']+math.pi) % (2*math.pi)-math.pi
                comparison = dict(gps_received_ns=received_ns, dr_received_ns=received_ns,
                                  real_gps_xy=list(xy), estimated_xy=estimated,
                                  yaw_grid_rad=previous['yaw']+weight*yaw_delta,
                                  speed_mps=previous['speed']+weight*(current['speed']-previous['speed']),
                                  hold_s=(received_ns-self.core.anchor_ns)/1e9,
                                  distance_m=previous['distance_m']+weight*(current['distance_m']-previous['distance_m']),
                                  time_basis='receipt_monotonic', alignment_method='interpolated_dr_history',
                                  interpolation_span_ns=span_ns)
                ready.append(comparison)
                self._comparison = comparison
                self._comparisons.append(comparison)
                break
        return ready

    def _comparison_metadata(self):
        # Control and perception poll independently. Repeating a small window lets
        # either logger collect it without the other consumer draining evidence.
        return (dict(simulated_comparison=self._comparison, simulated_comparisons=list(self._comparisons))
                if self._comparisons else {})

    def _anchor_on_fix(self, fix, identity, speed, imu):
        heading, _ = self._causal_sample(imu, identity['received_ns'])
        velocity, _ = self._causal_sample(speed, identity['received_ns'])
        self.core.anchor(self._grid(fix), identity['received_ns'],
                         heading['yaw'] if heading else None,
                         velocity['signed_speed']/3.6 if velocity else None)
        self._last_source_stamp_ns = 0
        self._last_status_identity = {}

    def _stale_reason(self):
        """Reason for a recoverable refusal while the status samples are late, or None: the hold ends for good."""
        return None

    def _refuse(self, identity, reason):
        self.counts['refused'] += 1
        self.last_reason = reason
        return None, float('inf'), dict(identity, position_mode='unavailable', reason=reason,
                                       dead_reckoning=self.core.summary(), **self._comparison_metadata())

    def latest_snapshot(self):
        with self.lock:
            # Multiple consumers must not process an older GPS snapshot after a
            # newer caller has already anchored it.
            fix, age, identity = self.gps.latest_snapshot()
            now_ns = time.monotonic_ns()
            speed, imu = self._histories()
            usable = fix is not None and 0 <= age <= FRESH_GPS_S
            if usable and self.zone is not None and self.zone.contains(self._grid(fix)):
                usable = False
                dropped = (self._grid(fix), identity['received_ns'])
                if self._dropped is None or dropped[1] != self._dropped[1]:
                    self.counts['simulated_drops'] += 1
                    self._pending_drops.append(dropped)
                self._dropped = dropped
            if usable:
                # A fix must not be re-anchored on later IMU/speed samples every time
                # the consumer polls it. Match inputs to that fix's own receipt time.
                if identity['received_ns'] != self.core.anchor_ns:
                    # Complete evidence for the final masked fix before a recovery
                    # fix resets the DR state. The recovery GPS never enters _advance.
                    if self.core.anchor_xy is not None and self._pending_drops:
                        self._advance(identity['received_ns'], speed, imu)
                        self._compare_pending()
                    self._anchor_on_fix(fix, identity, speed, imu)
                    self._trajectory.clear()
                    self._pending_drops.clear()
                    self._dropped = None
                    self._record_state()
                self.counts['gps'] += 1
                self.last_reason = None
                return fix, age, dict(identity, position_mode='gps', **self._comparison_metadata())
            if self.core.anchor_xy is None:
                return self._refuse(identity, 'no_gps_anchor')
            rows = self._advance(now_ns, speed, imu)
            hold_s = (now_ns-self.core.anchor_ns)/1e9
            status_age = (now_ns-self.core.last_ns)/1e9
            if hold_s > self.core.max_hold_s:
                self.core.failure = self.core.failure or 'hold_time_limit'
            elif not 0 <= status_age <= MAX_STATUS_GAP_S:
                stale = self._stale_reason()
                if stale is not None and self.core.failure is None:
                    self.max_hold_seen_s = max(self.max_hold_seen_s, hold_s)
                    return self._refuse(identity, stale)        # no position now, the hold goes on
                self.core.failure = self.core.failure or 'status_stale'
            elif not rows and self.core.last_sample is None:
                self.core.failure = self.core.failure or 'no_status_sample'
            self.max_hold_seen_s = max(self.max_hold_seen_s, hold_s)
            self.max_distance_seen_m = max(self.max_distance_seen_m, self.core.distance_m)
            if self.core.failure is not None:
                return self._refuse(identity, self.core.failure)
            self._compare_pending()
            self.counts['dead_reckoning'] += 1
            held = self._fix_from_grid(self.core.xy)
            held['dead_reckoning'] = True
            out_id = dict(sample_id=identity.get('sample_id', 0), received_ns=self.core.last_ns,
                          source_stamp_ns=self._last_status_identity.get('source_stamp_ns', 0),
                          position_mode='dead_reckoning', last_gps_received_ns=self.core.anchor_ns,
                          hold_s=round(hold_s, 3), dead_reckoning=self.core.summary())
            if self.zone is not None and self._dropped is not None:
                out_id['simulated_real_gps_xy'] = [round(v, 3) for v in self._dropped[0]]
                out_id['simulated_real_gps_age_s'] = round((now_ns-self._dropped[1])/1e9, 3)
            out_id.update(self._comparison_metadata())
            return held, status_age, out_id

    def diagnostics(self):
        with self.lock:
            return dict(self.gps.diagnostics(), blackout_hold=dict(self.counts,
                max_hold_seen_s=round(self.max_hold_seen_s, 2), max_distance_seen_m=round(self.max_distance_seen_m, 2),
                last_reason=self.last_reason, simulated_zone=self.zone is not None,
                limits=dict(max_hold_s=self.core.max_hold_s, max_distance_m=self.core.max_distance_m)))
