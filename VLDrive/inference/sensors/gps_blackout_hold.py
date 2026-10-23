"""Hold the GPS position through a GPS blackout with IMU heading and status speed.

Competition rule v1.2: GPS is blacked out inside the GPS shadow zone and GPS/IMU
carry no noise. While the last valid fix is fresh it passes through unchanged.
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
        heading_known = true_yaw_rad is not None and math.isfinite(true_yaw_rad)
        self.lever_active = heading_known and self.lever != (0., 0.)
        self.ref_xy = self.xy
        if heading_known:
            grid = true_yaw_rad + math.radians(self.convergence_deg(self.xy))
            ox, oy = self._offset(grid) if self.lever_active else (0., 0.)
            lead = speed_mps*self.latency_s if speed_mps is not None and math.isfinite(speed_mps) else 0.
            self.ref_xy = (self.xy[0]-ox+lead*math.cos(grid), self.xy[1]-oy+lead*math.sin(grid))
            if lead:
                self.xy = (self.ref_xy[0]+ox, self.ref_xy[1]+oy)

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
        if not 0 < dt <= MAX_STATUS_GAP_S:
            dt = gap
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
        if self.length <= 0:
            raise ValueError('Zone axis must have length')
        self.start, self.fwd = start_xy, (dx/self.length, dy/self.length)
        self.half_width, self.margin = float(half_width_m), float(margin_m)

    def contains(self, xy):
        rx, ry = xy[0]-self.start[0], xy[1]-self.start[1]
        s = rx*self.fwd[0] + ry*self.fwd[1]
        lateral = -rx*self.fwd[1] + ry*self.fwd[0]
        return -self.margin <= s <= self.length+self.margin and abs(lateral) <= self.half_width


class GpsBlackoutHold:
    """Drop-in for ValidGpsReceiver (``latest``, ``latest_snapshot``, ``diagnostics``, ``close``)."""

    def __init__(self, gps, status, basis, *, max_hold_s=120., max_distance_m=250., simulate_zone=None,
                 lever_xy=(0., 0.), latency_s=0.):
        from data.src.loaders.morai_map_geometry import validate_map_basis
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

    def _status_samples(self, after_ns):
        """Chronological (status_value, status_identity, imu_value) received after ``after_ns``."""
        with self.status.speed.history_lock:
            speed = list(self.status.speed.history)
        with self.status.imu.history_lock:
            imu = list(self.status.imu.history)
        if speed and speed[0][1]['received_ns'] > after_ns + MAX_STATUS_GAP_S*1e9 and self.core.last_ns == after_ns:
            # The bounded history no longer reaches back to the anchor: samples were lost.
            return None
        rows, j = [], 0
        for value, identity in speed:
            if identity['received_ns'] <= after_ns:
                continue
            while j+1 < len(imu) and imu[j+1][1]['received_ns'] <= identity['received_ns']:
                j += 1
            if not imu or imu[j][1]['received_ns'] > identity['received_ns']:
                continue
            rows.append((value, identity, imu[j][0]))
        return rows

    def latest_snapshot(self):
        fix, age, identity = self.gps.latest_snapshot()
        with self.lock:
            usable = fix is not None and 0 <= age <= FRESH_GPS_S
            if usable and self.zone is not None and self.zone.contains(self._grid(fix)):
                usable = False
                self.counts['simulated_drops'] += 1
                self._dropped = (self._grid(fix), identity['received_ns'])
            if usable:
                with self.status.imu.history_lock:
                    latest_imu = self.status.imu.history[-1][0]['yaw'] if self.status.imu.history else None
                with self.status.speed.history_lock:
                    latest_speed = (self.status.speed.history[-1][0]['signed_speed']/3.6
                                    if self.status.speed.history else None)
                self.core.anchor(self._grid(fix), identity['received_ns'], latest_imu, latest_speed)
                self.counts['gps'] += 1
                return fix, age, dict(identity, position_mode='gps')
            if self.core.anchor_xy is None:
                self.counts['refused'] += 1
                self.last_reason = 'no_gps_anchor'
                return None, float('inf'), dict(identity, position_mode='unavailable', reason='no_gps_anchor')
            rows = self._status_samples(self.core.last_ns)
            if rows is None:
                self.core.failure = self.core.failure or 'status_history_overrun'
                rows = []
            previous_source = None
            last_identity = None
            for value, sid, imu in rows:
                dt = None
                if previous_source and sid.get('source_stamp_ns'):
                    dt = (sid['source_stamp_ns'] - previous_source)/1e9
                if not self.core.step(sid['received_ns'], value['signed_speed']/3.6, imu['yaw'], dt):
                    break
                previous_source = sid.get('source_stamp_ns') or None
                last_identity = sid
            hold_s = (time.monotonic_ns() - self.core.anchor_ns)/1e9
            self.max_hold_seen_s = max(self.max_hold_seen_s, hold_s)
            self.max_distance_seen_m = max(self.max_distance_seen_m, self.core.distance_m)
            if self.core.failure is not None:
                self.counts['refused'] += 1
                self.last_reason = self.core.failure
                return None, float('inf'), dict(identity, position_mode='unavailable', reason=self.core.failure,
                                                 dead_reckoning=self.core.summary())
            status_ns = self.core.last_ns
            status_age = (time.monotonic_ns() - status_ns)/1e9
            self.counts['dead_reckoning'] += 1
            held = self._fix_from_grid(self.core.xy)
            held['dead_reckoning'] = True
            out_id = dict(sample_id=identity.get('sample_id', 0), received_ns=status_ns,
                          source_stamp_ns=(last_identity or {}).get('source_stamp_ns', 0),
                          position_mode='dead_reckoning', last_gps_received_ns=identity.get('received_ns', 0),
                          hold_s=round(hold_s, 3), dead_reckoning=self.core.summary())
            if self.zone is not None and self._dropped is not None:
                # Development evidence only: where the discarded real fix was, for an offline error comparison.
                out_id['simulated_real_gps_xy'] = [round(v, 3) for v in self._dropped[0]]
                out_id['simulated_real_gps_age_s'] = round((time.monotonic_ns() - self._dropped[1])/1e9, 3)
            return held, status_age, out_id

    def diagnostics(self):
        with self.lock:
            return dict(self.gps.diagnostics(), blackout_hold=dict(self.counts,
                max_hold_seen_s=round(self.max_hold_seen_s, 2), max_distance_seen_m=round(self.max_distance_seen_m, 2),
                last_reason=self.last_reason, simulated_zone=self.zone is not None,
                limits=dict(max_hold_s=self.core.max_hold_s, max_distance_m=self.core.max_distance_m)))
