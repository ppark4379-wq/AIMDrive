"""Moving traffic in the Jamming (tunnel) section, on top of the map planner (corridor.py).

The map planner owns the path; traffic changes it only through boxes added to the map and caps the speed. One tick:

``before``: virtual boxes for the map planner
  * a slow car being overtaken: its footprint swept forward by its speed for OVERTAKE_SWEEP_S, so the path goes round
    the stretch it drives through, as round a standing obstacle;
  * an oncoming car, or a car-sized object in the opposite lane not judged yet (it may be one): the stretch of the
    opposite lane it may cover in ONCOMING_SWEEP_S. A detour that would meet it is then not planned; the map planner
    stops before the obstacle instead, and a car already beside it returns to its lane.
``after``: speed caps on the planned path
  * lead: a car-sized object in the path that is moving, stopped (not yet STALL_S) or never seen moving and not yet
    seen long enough to call it parked: followed at a time gap, or stopped behind;
  * yield: a pedestrian (or another small mover) in the path or walking into it within YIELD_HORIZON_S: stop before it
    and wait until it has left; a pedestrian is never driven round;
  * no stop inside the opposite lane: when the planned path is blocked further on, the car waits before it would
    cross the lane line.

Object classes and motion come from the tunnel map's tracks (Expert/tunnel_map.py): VE gives no class, so a size class
is taken from recent boxes, and motion from displacement. Overtaking is only for objects classed as cars: a pedestrian
can walk slower than the slowest car, so speed does not tell them apart.
"""
import math

import numpy as np

from ...config.parameters.jamming import (
    FOLLOW_MIN_GAP_M, FOLLOW_TIME_GAP_S, FOLLOW_STOP_GAP_M, FOLLOW_DECEL, LEAD_LATERAL_MARGIN_M, LEAD_LANE_OVERLAP_M,
    YIELD_HORIZON_S, YIELD_STEP_S, YIELD_CLEARANCE_M, YIELD_STOP_GAP_M, YIELD_DECEL, YIELD_RANGE_M,
    OVERTAKE_MAX_SPEED, OVERTAKE_CONFIRM_S, OVERTAKE_TRIGGER_GAP_M, OVERTAKE_SWEEP_S, OVERTAKE_SPEED_TOLERANCE,
    OVERTAKE_PASS_SPEED, OVERTAKE_RETURN_M, OPPOSITE_MIN_WIDTH_M, OVERTAKE_GROUP_GAP_M,
    OVERTAKE_ABORT_SPEED, OVERTAKE_CLEAR_M,
    BLOCKER_MIN_LONG_M, BLOCKER_WALL_MARGIN_M, BLOCKER_MEMORY_S, BLOCKER_RANGE_M, PENDING_ASSUMED_SPEED,
    ONCOMING_MIN_SPEED, TUNNEL_MAX_SPEED_KPH, TUNNEL_OVERTAKE_SPEED_KPH, TUNNEL_CLEARANCE_SPEED,
    TUNNEL_OVERTAKE_CLEARANCE_SPEED,
    ONCOMING_SWEEP_S, ENCROACHED_SWEEP_S, ENCROACH_TOLERANCE_M, GATE_MARGIN_M, GATE_DECEL, TRACK_STALE_S,
)
from ...config.parameters.static_obstacle import CORRIDOR_LANE_HALF_WIDTH
from ...config.parameters.jamming import VEHICLE_MIN_LONG_M

FORWARD = .5*math.pi      # section frame: +y is the direction of travel
LOST_TARGET_S = 1.        # an overtaking target not seen for this long is dropped [s]


def _stop_speed(distance, deceleration):
    return math.sqrt(2.*deceleration*max(0., distance))


class TunnelTraffic:
    def __init__(self, occupancy, corridor, section):
        self.map = occupancy          # TunnelObstacleMap: tracks (motion), virtual boxes
        self.corridor = corridor      # TunnelCorridorPlanner: route curve, vehicle size
        self.section = section
        self.overtaking = {}          # track id -> committed overtake
        self.slow_since = {}          # track id -> time a car-sized lead has been slow since
        self.blockers = []

    def clear(self):
        self.map.set_virtual([])
        self.corridor.MAX_SPEED = TUNNEL_MAX_SPEED_KPH/3.6
        self.corridor.CLEARANCE_SPEED = TUNNEL_CLEARANCE_SPEED
        self.overtaking, self.slow_since, self.blockers = {}, {}, []

    # --- geometry -----------------------------------------------------------------------------------------------
    def _lane_line(self, y):
        """Lateral position of the line between the own lane and the opposite lane (opposite lane: smaller x)."""
        return self._centre(y)-CORRIDOR_LANE_HALF_WIDTH

    def _centre(self, y):
        """Own lane centre (the route) at forward position(s) y."""
        base = self.corridor.base
        low, high = base.geometry.route_y
        return base.global_curve(np.clip(np.asarray(y, dtype=float), low, high))

    def _own_lane(self, x, y):
        return abs(x-float(self._centre(y))) <= CORRIDOR_LANE_HALF_WIDTH

    def _vehicle(self):
        planner = self.corridor.base.planner
        return planner.vehicle_length, planner.vehicle_half_width

    def _objects(self, now):
        """Recently seen tracks as dicts: id, box (map frame), size class, velocity, age."""
        motion = self.map.motion
        out = []
        for identity, track in motion.tracks.items():
            box = track.get('box')
            if box is None:
                continue
            history = track.get('history') or ()
            longs = sorted(h[3] for h in history) or [max(box['length_m'], box['width_m'])]
            wides = sorted(h[4] for h in history) or [min(box['length_m'], box['width_m'])]
            out.append(dict(id=identity, x=box['center_world'][0], y=box['center_world'][1],
                            heading=math.radians(box['heading_deg']), length=longs[len(longs)//2],
                            width=wides[len(wides)//2], size=track.get('size'), car=motion.car_like(identity),
                            mover=bool(track.get('mover')),
                            person=bool(track.get('person')), standing=motion.standing(identity),
                            velocity=motion.velocity(identity), age=now-track['seen']))
        return out

    # --- before the map planner ---------------------------------------------------------------------------------
    def before(self, ego):
        """Decide overtakes and opposite-lane blockers; put their boxes into the map for this tick."""
        now = self.map.motion.now
        if now is None or not self.corridor.ready:      # the route curve exists once the map planner is set up
            self.map.set_virtual([])
            return
        length, _ = self._vehicle()
        ex, ey = float(ego.position.x), float(ego.position.y)
        objects = self._objects(now)
        encroaching = self._encroaching_pose(ego)
        rectangles, footprints = [], []
        seen = set()
        for o in objects:
            seen.add(o['id'])
            # Only a car-sized box can be a car to go round: a row of barriers renumbered by the tracker leaves small
            # fragments that may have been called a car before.
            if o['size'] != 'vehicle' or o['age'] > LOST_TARGET_S or o['length'] < VEHICLE_MIN_LONG_M:
                self.overtaking.pop(o['id'], None)
                continue
            rear, front = o['y']-.5*o['length'], o['y']+.5*o['length']
            speed = o['velocity'][1]
            in_lane = self._own_lane(o['x'], o['y'])
            target = self.overtaking.get(o['id'])
            if target is not None and target.get('resumed') and not o['mover']:
                del self.overtaking[o['id']]                # standing again: on the map as a standing obstacle
                continue
            if target is not None:
                passed = ey > front+OVERTAKE_CLEAR_M
                alongside = encroaching or ey+length > rear
                if passed or (not alongside and speed > OVERTAKE_ABORT_SPEED):
                    del self.overtaking[o['id']]
                    continue
            elif o['mover'] and in_lane:
                # The speed estimate jitters by about 1 m/s: once counted slow, a car stays slow below the tolerance.
                limit = OVERTAKE_MAX_SPEED+(OVERTAKE_SPEED_TOLERANCE if o['id'] in self.slow_since else 0.)
                slow = -.5 <= speed <= limit and rear > ey
                since = self.slow_since.setdefault(o['id'], now) if slow else self.slow_since.pop(o['id'], None)
                gap = rear-(ey+length)
                if (slow and since is not None and now-since >= OVERTAKE_CONFIRM_S and gap <= OVERTAKE_TRIGGER_GAP_M
                        and self._room_to_pass(o, objects, ego)):
                    self.overtaking[o['id']] = dict(started=now)
                elif encroaching and ey+length > rear-5. and ey < front+OVERTAKE_CLEAR_M:
                    # A car that starts moving while this one is already going round it: keep its stretch blocked
                    # until it has been passed, or the path would swing back into it.
                    self.overtaking[o['id']] = dict(started=now, resumed=True)
                if o['id'] not in self.overtaking:
                    continue
            else:
                continue
            ahead = max(0., speed)*OVERTAKE_SWEEP_S
            rectangles.append((o['x'], .5*(rear+front+ahead), FORWARD, front+ahead-rear, o['width']))
            footprints.append((o['x'], o['y'], FORWARD, o['length'], o['width']))
        for identity in list(self.overtaking):
            if identity not in seen:
                del self.overtaking[identity]
        self.slow_since = {k: v for k, v in self.slow_since.items() if k in seen}
        self.blockers = []
        for o in objects:
            if o['person'] or o['size'] == 'person' or o['age'] > BLOCKER_MEMORY_S or o['id'] in self.overtaking:
                continue
            # Judged or seen moving: anything car-long enough; not judged yet: only what is car-sized (fragments of
            # standing objects are 1-3 m).
            if not o['car'] and (o['length'] < BLOCKER_MIN_LONG_M or not o['mover']):
                continue
            if o['standing'] and o['size'] is not None:
                continue                                    # judged standing; a far, unjudged one may be oncoming
            line = float(self._lane_line(o['y']))
            # Centred where a car drives in the opposite lane: boxes merged with the tunnel wall sit beyond it.
            if not line-2.*CORRIDOR_LANE_HALF_WIDTH+BLOCKER_WALL_MARGIN_M <= o['x'] < line:
                continue
            if not -5. < o['y']-ey <= BLOCKER_RANGE_M:
                continue
            vy = o['velocity'][1]
            if o['mover'] and vy > .5:
                continue                                    # driving away in the opposite lane
            speed = max(-vy, ONCOMING_MIN_SPEED) if o['mover'] else PENDING_ASSUMED_SPEED
            # Before going into the opposite lane anything that may be oncoming keeps it closed for ONCOMING_SWEEP_S.
            # Once in it, stopping there is the worst choice: only a car seen coming closes the short stretch it covers
            # while this car gets back; an unjudged one only its own footprint.
            sweep = ONCOMING_SWEEP_S if not encroaching else ENCROACHED_SWEEP_S if o['mover'] else 0.
            y = o['y']-speed*o['age']
            far, near = y+.5*o['length'], y-.5*o['length']-speed*sweep
            rectangles.append((o['x'], .5*(far+near), FORWARD, far-near, max(o['width'], 1.8)))
            footprints.append((o['x'], y, FORWARD, o['length'], o['width']))
            self.blockers.append(dict(id=o['id'], x=round(o['x'], 2), y=round(o['y'], 1), mover=o['mover'],
                                      assumed_speed=round(speed, 1), sweep_s=sweep))
        self.map.set_virtual(rectangles, footprints)
        # Overtaking a car: a higher cap, so the car spends less time in the opposite lane.
        self.corridor.MAX_SPEED = (TUNNEL_OVERTAKE_SPEED_KPH if self.overtaking else TUNNEL_MAX_SPEED_KPH)/3.6
        self.corridor.CLEARANCE_SPEED = TUNNEL_OVERTAKE_CLEARANCE_SPEED if self.overtaking else TUNNEL_CLEARANCE_SPEED

    def _opposite_end(self, start):
        """Forward position where the opposite lane stops being usable beyond ``start`` (the tunnel exit splits the
        two directions: the road there is the own lane only)."""
        c = self.corridor
        if c.road_y is None:
            return math.inf
        ys = c.road_y[c.road_y >= start]
        if not len(ys):
            return math.inf
        width = self._lane_line(ys)-np.interp(ys, c.road_y, np.nan_to_num(c.road_low, nan=np.inf))
        narrow = np.flatnonzero(~(width >= OPPOSITE_MIN_WIDTH_M))
        return float(ys[narrow[0]]) if len(narrow) else math.inf

    def _room_to_pass(self, target, objects, ego):
        """Whether going round ``target`` can end back in the own lane before the opposite lane ends. Cars further on
        in the own lane that would be where this car returns are passed in the same go (and so lengthen it);
        otherwise the car would be caught beside them, or at the exit."""
        length, _ = self._vehicle()
        ey = float(ego.position.y)
        pass_speed = max(float(ego.velocity), OVERTAKE_PASS_SPEED)
        group = [target]
        others = sorted((o for o in objects if o['id'] != target['id'] and o['car'] and o['age'] <= TRACK_STALE_S
                         and self._own_lane(o['x'], o['y']) and o['y'] > target['y']), key=lambda o: o['y'])
        while True:
            front = max(o['y']+.5*o['length'] for o in group)
            speed = max(max(0., o['velocity'][1]) for o in group)
            if pass_speed <= speed+.5:
                return False
            time = (front+OVERTAKE_CLEAR_M-ey)/(pass_speed-speed)
            back = ey+pass_speed*time+OVERTAKE_RETURN_M          # where this car is back in its lane
            caught = [o for o in others if o not in group
                      and o['y']-.5*o['length']+max(0., o['velocity'][1])*time < back+OVERTAKE_GROUP_GAP_M]
            if not caught:
                return back <= self._opposite_end(ey)
            group.append(caught[0])

    # --- after the map planner ----------------------------------------------------------------------------------
    def _encroaching_pose(self, ego):
        x, y, yaw = float(ego.position.x), float(ego.position.y), float(ego.yaw)
        return bool(self._encroach(np.array([x]), np.array([y]), np.array([yaw]))[0])

    def _encroach(self, x, y, yaw):
        """Whether the body at reference poses reaches into the opposite lane."""
        length, half = self._vehicle()
        c, s = np.cos(yaw), np.sin(yaw)
        out = np.zeros(len(x), dtype=bool)
        for along in (0., length):
            cx, cy = x+along*c-half*s, y+along*s+half*c        # left side corners (towards smaller x)
            out |= cx < self._lane_line(cy)-ENCROACH_TOLERANCE_M
        return out

    @staticmethod
    def _project(px, py, path_x, path_y, stations):
        """Station and signed lateral offset of points on the path polyline (nearest segment)."""
        ax, ay = path_x[:-1], path_y[:-1]
        dx, dy = np.diff(path_x), np.diff(path_y)
        lengths = np.maximum(np.hypot(dx, dy), 1e-9)
        tx, ty = dx/lengths, dy/lengths
        rx, ry = px[:, None]-ax[None, :], py[:, None]-ay[None, :]
        along = np.clip(rx*tx+ry*ty, 0., lengths)
        ox, oy = rx-along*tx, ry-along*ty
        distance = np.hypot(ox, oy)
        k = np.argmin(distance, axis=1)
        rows = np.arange(len(px))
        side = np.sign(tx[k]*oy[rows, k]-ty[k]*ox[rows, k])
        return stations[k]+along[rows, k], side*distance[rows, k], np.arctan2(ty[k], tx[k])

    def after(self, result, ego):
        """Cap the planned speed for traffic; returns the same result with debug details."""
        now = self.map.motion.now
        details = dict(overtaking=sorted(self.overtaking), blockers=self.blockers, leads=[], yields=[], kind=None)
        result.debug_details['jamming_traffic'] = details
        if now is None or not self.corridor.ready or len(result.path) < 2:
            return result
        length, half = self._vehicle()
        path = np.array([(p.x, p.y) for p in result.path], dtype=float)
        keep = np.r_[True, np.hypot(*np.diff(path, axis=0).T) > 1e-6]
        path = path[keep]
        if len(path) < 2:
            return result
        stations = np.r_[0., np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        yaw = np.arctan2(*np.diff(path, axis=0).T[::-1])
        encroach = self._encroach(path[:-1, 0], path[:-1, 1], yaw)
        entering = (not self._encroaching_pose(ego)) and bool(encroach.any())
        s_in = float(stations[int(np.argmax(encroach))]) if entering else math.inf
        caps = []
        for o in self._objects(now):
            if o['age'] > TRACK_STALE_S:
                continue
            vx, vy = o['velocity']
            if o['car'] and not o['standing'] and o['id'] not in self.overtaking:
                s, d, tangent = self._project(np.array([o['x']]), np.array([o['y']]), path[:, 0], path[:, 1], stations)
                rel = o['heading']-tangent[0]
                across = .5*(o['length']*abs(math.sin(rel))+o['width']*abs(math.cos(rel)))
                along = .5*(o['length']*abs(math.cos(rel))+o['width']*abs(math.sin(rel)))
                in_path = abs(d[0])-across <= half+LEAD_LATERAL_MARGIN_M
                in_lane = abs(o['x']-float(self._centre(o['y'])))-across <= CORRIDOR_LANE_HALF_WIDTH-LEAD_LANE_OVERLAP_M
                if not (in_path or in_lane) or s[0]+along <= length:
                    continue
                gap = s[0]-along-length
                if s[0] >= stations[-1]-1e-6:
                    # Beyond the end of a path cut short (blocked): the projection would put it at the path's end.
                    gap = o['y']-along-(float(ego.position.y)+length)
                lead_speed = max(0., vx*math.cos(tangent[0])+vy*math.sin(tangent[0])) if o['mover'] else 0.
                wanted = FOLLOW_MIN_GAP_M+FOLLOW_TIME_GAP_S*lead_speed
                if gap >= wanted:
                    cap = math.sqrt(lead_speed**2+2.*FOLLOW_DECEL*(gap-wanted))
                else:
                    cap = lead_speed*max(0., gap-FOLLOW_STOP_GAP_M)/max(wanted-FOLLOW_STOP_GAP_M, 1e-6)
                if lead_speed < 1. and s_in < s[0]-along:
                    # Going round it would start before reaching it: wait in the own lane.
                    cap = min(cap, _stop_speed(s_in-GATE_MARGIN_M, GATE_DECEL))
                caps.append((cap, 'lead'))
                details['leads'].append(dict(id=o['id'], gap=round(gap, 1), speed=round(lead_speed, 2),
                                             state='moving' if o['mover'] else 'waiting', cap=round(cap, 2)))
            elif o['person'] or (o['mover'] and o['size'] != 'vehicle'):
                radius = max(.3, .5*max(o['length'], o['width']))
                times = np.arange(0., YIELD_HORIZON_S+1e-9, YIELD_STEP_S)+o['age']
                if not o['mover']:
                    times = times[:1]
                px, py = o['x']+vx*times, o['y']+vy*times
                s, d, _ = self._project(px, py, path[:, 0], path[:, 1], stations)
                # Ahead of the front; a confirmed pedestrian also beside the car (it may step under the wheels). Other
                # small movers beside or behind are not driven into by going on (and are often box fragments).
                start = 0. if o['person'] else length
                inside = ((np.abs(d)-radius <= half+YIELD_CLEARANCE_M) & (s+radius > start)
                          & (s-radius <= length+YIELD_RANGE_M))
                if not inside.any():
                    continue
                conflict = float(np.min(s[inside]-radius))
                cap = _stop_speed(conflict-length-YIELD_STOP_GAP_M, YIELD_DECEL)
                caps.append((cap, 'yield'))
                details['yields'].append(dict(id=o['id'], person=o['person'], station=round(conflict, 1),
                                              cap=round(cap, 2)))
        phase = result.debug_details.get('static_avoidance', {}).get('phase')
        stop = result.debug_details.get('static_avoidance', {}).get('stop_distance_m')
        if entering and phase == 'blocked' and (stop is None or s_in < stop):
            caps.append((_stop_speed(s_in-GATE_MARGIN_M, GATE_DECEL), 'wait'))
        details['entry_station'] = None if not math.isfinite(s_in) else round(s_in, 1)
        if caps:
            cap, kind = min(caps)
            details['cap'] = round(cap, 2)
            if cap < result.acc_target_velocity-1e-3:
                result.acc_target_velocity = cap
                result.reason = 'jamming_traffic_'+kind
                details['kind'] = kind
        return result
