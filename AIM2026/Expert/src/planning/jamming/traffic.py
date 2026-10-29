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
from types import SimpleNamespace

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
    NEW_OBSTACLE_WINDOW_S, NEW_OBSTACLE_REACTION_S,
    NEW_OBSTACLE_SIDE_MARGIN_M, NEW_OBSTACLE_OBSERVE_S, NEW_OBSTACLE_CLEAR_S,
    NEW_OBSTACLE_STOP_OBSERVE_S, NEW_OBSTACLE_CAUTION_SPEED_KPH,
    NEW_OBSTACLE_CAUTION_ACCEL_MPS2, NEW_OBSTACLE_RECHECK_MOVE_M,
    YIELD_RESTART_SIDE_CLEARANCE_M,
    NEW_OBSTACLE_WATCH_EXTRA_M, NEW_OBSTACLE_WATCH_HOLD_S, NEW_OBSTACLE_WATCH_SPEED_KPH,
    STANDING_HOLD_RELEASE,
)
from ...config.parameters.static_obstacle import CORRIDOR_LANE_HALF_WIDTH
from ...config.parameters.jamming import VEHICLE_MIN_LONG_M
from ..common.dynamic_crossing import DynamicCrossingGuard

FORWARD = .5*math.pi      # section frame: +y is the direction of travel
LOST_TARGET_S = 1.        # an overtaking target not seen for this long is dropped [s]


def _stop_speed(distance, deceleration):
    return math.sqrt(2.*deceleration*max(0., distance))


def _new_obstacle_braking_distance(speed):
    """새 장애물 제동거리 [m]. 기존 2m 여유를 고정 포함하며 반응거리는 별도로 더한다."""
    return speed**2/(2.*YIELD_DECEL) + 2.0


class TunnelTraffic:
    def __init__(self, occupancy, corridor, section):
        self.map = occupancy          # TunnelObstacleMap: tracks (motion), virtual boxes
        self.corridor = corridor      # TunnelCorridorPlanner: route curve, vehicle size
        self.section = section
        self.overtaking = {}          # track id -> committed overtake
        self.slow_since = {}          # track id -> time a car-sized lead has been slow since
        self.blockers = []
        self._caution_time = None
        self._caution_target = 0.
        self._crossing_guard = None
        self._edge_watches = {}
        self._standing_released = False   # standing-body holds released by the planner's stall clock last tick

    def clear(self):
        self.map.set_virtual([])
        self.corridor.MAX_SPEED = TUNNEL_MAX_SPEED_KPH/3.6
        self.corridor.CLEARANCE_SPEED = TUNNEL_CLEARANCE_SPEED
        self.overtaking, self.slow_since, self.blockers = {}, {}, []
        self._caution_time, self._caution_target = None, 0.
        self._standing_released = False
        self._edge_watches.clear()
        if self._crossing_guard is not None:
            self._crossing_guard.reset()

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
    def _extension_objects(self, objects):
        """New stop/restart behaviour is for person/unknown size classes only.

        A large box is still unknown until TrackMotion confirms its size.
        On confirmation, discard its extension state immediately; ordinary
        vehicle following/overtaking and the static map keep ownership.
        """
        vehicles = {o['id'] for o in objects if o['size'] == 'vehicle'}
        for identity in vehicles:
            track = self.map.motion.tracks[identity]
            track.pop('new_obstacle_guard', None)
            track.pop('tunnel_side_release', None)
            self._edge_watches.pop(identity, None)
            if self._crossing_guard is not None:
                self._crossing_guard._hazards.pop(identity, None)
                self._crossing_guard._released.pop(identity, None)
                self._crossing_guard._acc_excluded.discard(identity)
        return [o for o in objects if o['id'] not in vehicles], vehicles

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

    @staticmethod
    def _recent_approach_velocity(history, now):
        """Short observed motion for crossing checks; no global motion relabelling."""
        recent = np.array([h[:3] for h in history if now-h[0] <= .4])
        if (len(recent) < 2 or recent[-1, 0]-recent[0, 0] < .09
                or np.linalg.norm(recent[-1, 1:]-recent[0, 1:]) < NEW_OBSTACLE_RECHECK_MOVE_M):
            return None
        t = recent[:, 0]-recent[:, 0].mean()
        return tuple((t @ (recent[:, 1:]-recent[:, 1:].mean(axis=0))) / (t @ t))

    def _edge_watch(self, objects, ego, path, stations, now):
        """Positive, bounded speed ceiling for new boxes just outside the stop gate.

        Hold from the last actual observation, not from a retained track's age.
        Missing frames/renumbering cannot remove the ceiling early or renew it
        forever. This layer never issues a zero target or overrides a stop.
        """
        self._edge_watches = {k: w for k, w in self._edge_watches.items()
                              if now-w['seen'] <= NEW_OBSTACLE_WATCH_HOLD_S}
        length, half = self._vehicle()
        speed = max(0., float(ego.velocity))
        distance = speed*NEW_OBSTACLE_REACTION_S + _new_obstacle_braking_distance(speed)
        for o in objects:
            track = self.map.motion.tracks[o['id']]
            if (o['age'] > TRACK_STALE_S or max(o['length'], o['width']) >= VEHICLE_MIN_LONG_M
                    or now-track.get('first_seen', now) > NEW_OBSTACLE_WINDOW_S or speed < .5):
                continue
            s, d, tangent = self._project(np.array([o['x']]), np.array([o['y']]),
                                           path[:, 0], path[:, 1], stations)
            rel = o['heading']-tangent[0]
            along = .5*(o['length']*abs(math.cos(rel))+o['width']*abs(math.sin(rel)))
            across = max(.5*math.hypot(o['length'], o['width']),
                         .5*(o['length']*abs(math.sin(rel))+o['width']*abs(math.cos(rel))))
            side = abs(d[0])-across-half
            forward = ((o['x']-ego.position.x)*math.cos(ego.yaw)
                       +(o['y']-ego.position.y)*math.sin(ego.yaw))
            if not (NEW_OBSTACLE_SIDE_MARGIN_M < side <= NEW_OBSTACLE_SIDE_MARGIN_M+NEW_OBSTACLE_WATCH_EXTRA_M
                    and s[0]-along-length <= distance and forward+along > 0.):
                continue
            state = self._edge_watches.get(o['id'])
            if state is None:
                # Spatial continuation only carries a speed ceiling, never a
                # clearance decision or a moving/stationary label.
                nearby = [(math.hypot(w['x']-o['x'], w['y']-o['y']), k, w)
                          for k, w in self._edge_watches.items() if w['seen'] < track['seen']]
                nearest = min(nearby, default=None, key=lambda row: row[0])
                if nearest is not None and nearest[0] <= 1.5:
                    _, old_id, state = nearest
                    self._edge_watches.pop(old_id)
                else:
                    state = dict(cap_mps=max(.5, min(speed, NEW_OBSTACLE_WATCH_SPEED_KPH/3.6)))
            state.update(seen=track['seen'], x=o['x'], y=o['y'])
            self._edge_watches[o['id']] = state
        return [dict(id=k, cap_mps=w['cap_mps'], age_s=round(max(0., now-w['seen']), 3))
                for k, w in self._edge_watches.items()]

    def _record_stopped_hazards(self, objects, ego, details, now):
        """Arm side-clearance release only after this traffic layer stopped Ego."""
        stopped = abs(float(ego.velocity)) <= .2
        blocked = {d['id'] for d in details['new_obstacle_guard']
                   if d['phase'] != 'released' and d['cap_mps'] <= .05}
        blocked.update(d['id'] for d in details['yields'] if d['cap'] <= .05)
        crossing = details['approach_crossing']
        if crossing.get('target_velocity', math.inf) <= .05:
            blocked.add(crossing['object_id'])
        for o in objects:
            track = self.map.motion.tracks[o['id']]
            state = track.get('tunnel_side_release')
            if stopped and o['id'] in blocked:
                if state is None:
                    state = track['tunnel_side_release'] = dict(since=now, armed=False)
                if state.get('since') is None:
                    state['since'] = now
                if now-state['since'] >= NEW_OBSTACLE_STOP_OBSERVE_S:
                    state['armed'] = True
            elif state is not None:
                state['since'] = None
            if state is not None and (math.hypot(*o['velocity']) > .2
                    or self._recent_approach_velocity(track.get('history', ()), now) is not None):
                state.update(armed=False, since=None)

    def _stationary_side_clear(self, objects, ego, path, stations, now):
        """Observed stopped pedestrians outside the body corridor may be passed slowly.

        A crossing's broad anticipation margin is not an occupied body. Require
        sustained fresh position evidence and check the entire commanded path;
        a label, timeout or a missing detection alone cannot grant clearance.
        Recompute each tick so resumed motion immediately restores prediction.
        """
        length, half = self._vehicle()
        clear = {}
        for o in objects:
            if o['age'] > TRACK_STALE_S or not (o['person'] or (o['mover'] and o['size'] != 'vehicle')):
                continue
            track = self.map.motion.tracks[o['id']]
            if not track.get('tunnel_side_release', {}).get('armed'):
                continue
            history = track.get('history', ())
            recent = np.array([h for h in history if now-h[0] <= NEW_OBSTACLE_OBSERVE_S+.1])
            if (len(recent) < 6 or recent[-1, 0]-recent[0, 0] < NEW_OBSTACLE_OBSERVE_S-.1
                    or np.max(np.linalg.norm(recent[:, 1:3]-np.median(recent[:, 1:3], axis=0), axis=1)) > .35
                    or math.hypot(*o['velocity']) > .2
                    or self._recent_approach_velocity(history, now) is not None):
                continue
            # Enclose every recent box; do not trust a tiny single-frame fragment
            # or a noisy heading while the person is standing next to the path.
            radius = max(.3, .5*float(np.max(np.hypot(recent[:, 3], recent[:, 4]))))
            s, d, _ = self._project(np.array([o['x']]), np.array([o['y']]),
                                     path[:, 0], path[:, 1], stations)
            if (s[0]+radius <= 0. or s[0]-radius > length+YIELD_RANGE_M
                    or abs(d[0])-radius-half <= YIELD_RESTART_SIDE_CLEARANCE_M):
                continue
            # Restrict the slow pass to the side band that could otherwise hold
            # a yield. Far-away and already passed objects do not cap cruise.
            forward = ((o['x']-ego.position.x)*math.cos(ego.yaw)
                       +(o['y']-ego.position.y)*math.sin(ego.yaw))
            crossing_radius = max(radius, .5*math.hypot(.8, .8))
            if abs(d[0])-crossing_radius-half <= YIELD_CLEARANCE_M:
                # Keep body-clear evidence even just behind the path start:
                # its clamped station is zero, which legacy yield otherwise
                # mistakes for a pedestrian still under the vehicle. Only the
                # part ahead/alongside warrants a slow-pass speed ceiling.
                clear[o['id']] = forward+radius > 0.
        return clear

    def _crossing_speed(self, objects, ego, path, now, desired_speed, side_clear=()):
        """Reuse ordinary-road crossing prediction before tunnel class confirmation.

        Feed observed world/map displacement, not box yaw or a guessed speed.
        This check owns no path and cannot raise any existing speed ceiling.
        The shared guard expires missing evidence and releases departing bodies.
        """
        if self._crossing_guard is None:
            length, half = self._vehicle()
            self._crossing_guard = DynamicCrossingGuard(
                vehicle_length=length, vehicle_width=2.*half,
                horizon=YIELD_HORIZON_S, max_distance=YIELD_RANGE_M,
                clearance=YIELD_CLEARANCE_M, stop_margin=YIELD_STOP_GAP_M,
                deceleration=YIELD_DECEL, reaction_time=NEW_OBSTACLE_REACTION_S,
                release_delay=NEW_OBSTACLE_CLEAR_S,
                approach_speed=NEW_OBSTACLE_CAUTION_SPEED_KPH/3.6)
        entries = []
        for o in objects:
            if o['size'] == 'vehicle' or o['age'] > TRACK_STALE_S or o['id'] in side_clear:
                continue
            history = self.map.motion.tracks[o['id']].get('history', ())
            recent = self._recent_approach_velocity(history, now)
            vx, vy = recent if recent is not None else o['velocity']
            # A missing vector remains zero; do not invent approach from class.
            age = max(0., o['age'])
            obj = SimpleNamespace(
                unique_id=o['id'], type=2,
                position=(o['x']+vx*age, o['y']+vy*age),
                velocity=math.hypot(vx, vy), heading_deg=math.degrees(math.atan2(vy, vx)),
                size_x=o['length'], size_y=o['width'],
                speed_unknown=False, motion_uncertain=not o['mover'])
            entries.append(dict(object_info=obj))
        pose = SimpleNamespace(position=(float(ego.position.x), float(ego.position.y)),
                               velocity=max(0., float(ego.velocity)))
        return self._crossing_guard.evaluate(path, pose, entries, desired_speed, now=now)

    def _new_obstacle_guard(self, objects, ego, path, stations, now):
        """Brake for a newly appearing close hazard before motion confirmation.

        Existing/far tracks retain normal planning. A triggered track latches
        until the path clears or observation resolves it; shrinking stopping
        distance while braking must not cause release/retrigger oscillations.
        State belongs to TrackMotion so spatial ID inheritance carries it too.
        """
        length, half = self._vehicle()
        speed = max(0., float(ego.velocity))
        distance = speed*NEW_OBSTACLE_REACTION_S + _new_obstacle_braking_distance(speed)
        decisions = []
        for o in objects:
            track = self.map.motion.tracks[o['id']]
            if o['size'] == 'vehicle':
                track.pop('new_obstacle_guard', None)
                continue
            state = track.get('new_obstacle_guard')
            if state is not None and state['phase'] == 'released' and not state.get('cautious'):
                continue
            history = list(track.get('history', ()))
            first_seen = track.get('first_seen', history[0][0] if history else now)
            if state is None and (now-first_seen > NEW_OBSTACLE_WINDOW_S or speed < .5):
                continue
            vx, vy = o['velocity']
            recent_velocity = self._recent_approach_velocity(history, now)
            if recent_velocity is not None:
                vx, vy = recent_velocity
            times = np.linspace(0., NEW_OBSTACLE_REACTION_S, 5) + max(0., o['age'])
            px, py = o['x']+vx*times, o['y']+vy*times
            s, d, tangent = self._project(px, py, path[:, 0], path[:, 1], stations)
            rel = o['heading']-tangent
            along = .5*(o['length']*np.abs(np.cos(rel))+o['width']*np.abs(np.sin(rel)))
            across = body_across = .5*(o['length']*np.abs(np.sin(rel))+o['width']*np.abs(np.cos(rel)))
            # Compact unclassified boxes can rotate as a walking person is
            # detected. Use their radius, rather than trusting initial yaw.
            if max(o['length'], o['width']) < VEHICLE_MIN_LONG_M:
                across = np.maximum(across, .5*math.hypot(o['length'], o['width']))
            gap = s-along-length
            forward = ((px-float(ego.position.x))*math.cos(float(ego.yaw))
                       +(py-float(ego.position.y))*math.sin(float(ego.yaw)))
            limit = distance if state is None else max(distance, state['distance_m'])
            danger = bool(o['age'] <= TRACK_STALE_S and np.any(
                (np.abs(d)-across <= half+NEW_OBSTACLE_SIDE_MARGIN_M)
                & (gap <= limit) & (forward+along > 0.)))
            if state is None:
                if not danger:
                    continue
                state = track['new_obstacle_guard'] = dict(phase='observing', started=now,
                                                          distance_m=distance, clear_since=None,
                                                          cautious=max(o['length'], o['width']) < VEHICLE_MIN_LONG_M,
                                                          stopped_since=None)
            # Released small objects still get approach checks; label flicker
            # or Ego simply approaching a stationary object does not re-arm.
            approaching = (recent_velocity is not None and
                           ((px[-1]-px[0])*math.cos(float(ego.yaw))
                            +(py[-1]-py[0])*math.sin(float(ego.yaw)) < -.15
                            or abs(d[-1]) < abs(d[0])-.15))
            approach_risk = bool(danger and approaching and np.any(
                (np.abs(d)-across <= half+YIELD_CLEARANCE_M)
                & (gap <= limit) & (forward+along > 0.)))
            if state['phase'] == 'released':
                if not approach_risk:
                    continue
                state.update(phase='observing', stopped_since=None, clear_since=None)
            if abs(float(ego.velocity)) <= .2:
                state['has_stopped'] = True
                if state.get('stopped_since') is None:
                    state['stopped_since'] = now
            else:
                state['stopped_since'] = None
            stopped_for = 0. if state.get('stopped_since') is None else now-state['stopped_since']
            recent = np.array([h[:3] for h in history if now-h[0] <= NEW_OBSTACLE_OBSERVE_S+.1])
            stationary = (len(recent) >= 6 and recent[-1, 0]-recent[0, 0] >= NEW_OBSTACLE_OBSERVE_S-.1
                          and np.max(np.linalg.norm(recent[:, 1:]-np.median(recent[:, 1:], axis=0), axis=1)) <= .35)
            release = None
            reason = 'new_close_obstacle'
            if state['cautious'] and o['age'] > TRACK_STALE_S:
                # Loss is not proof of clearance. Hand off to the accumulated
                # map at low speed, then expire with TrackMotion's bounded memory.
                if stopped_for >= NEW_OBSTACLE_STOP_OBSERVE_S or state['phase'] == 'caution':
                    state['phase'] = 'caution'
                reason = 'observation_lost'
            elif not danger:
                if state['clear_since'] is None:
                    state['clear_since'] = now
                if now-state['clear_since'] >= NEW_OBSTACLE_CLEAR_S:
                    release = 'path_clear'
            else:
                state['clear_since'] = None
                if state['cautious']:
                    if (state['phase'] == 'observing' and not state.get('has_stopped')
                            and (o['mover'] or o['person'])):
                        # Motion resolved before a stop: preserve normal moving
                        # traffic handling. Cautious restart is for a held Ego.
                        release = 'motion_confirmed'
                        state['cautious'] = False
                    elif (state['phase'] == 'observing' and stopped_for >= NEW_OBSTACLE_STOP_OBSERVE_S
                            and (stationary or o['mover'] or o['person']
                                 or stopped_for >= NEW_OBSTACLE_OBSERVE_S)):
                        state['phase'] = 'caution'
                    reason = 'approaching_before_confirmation' if approach_risk else 'watching_close_object'
                elif o['mover'] or o['person']:
                    # The normal yield/lead cap is calculated on this same
                    # tick. Releasing this guard never overrides that cap.
                    release = 'motion_confirmed'
                elif stationary and now-state['started'] >= NEW_OBSTACLE_OBSERVE_S:
                    release = 'stationary_observed'
            if release:
                state.update(phase='released', release_reason=release)
            cap = 0. if state['phase'] == 'observing' else NEW_OBSTACLE_CAUTION_SPEED_KPH/3.6
            if state['phase'] == 'caution':
                # Permit a validated static detour, but retain room to stop
                # before a footprint actually occupying the chosen path.
                # A stopped Ego creeps past a motionless box beside the validated path: only its measured
                # body (not the rotation-proof radius used for the watch zone) counts as occupying the path.
                width = across if o['mover'] or o['person'] or not stationary else body_across
                in_path = (np.abs(d)-width <= half+.2) & (forward+along > 0.)
                if o['age'] <= TRACK_STALE_S and in_path.any():
                    cap = min(cap, _stop_speed(float(np.min(gap[in_path]))-_new_obstacle_braking_distance(0.), YIELD_DECEL))
                if approach_risk:
                    cap = 0.
            decisions.append(dict(id=o['id'], phase=state['phase'], gap_m=round(float(np.min(gap)), 2),
                                  stop_distance_m=round(limit, 2), observation_s=round(now-first_seen, 3),
                                  stopped_s=round(stopped_for, 3), cautious=state['cautious'],
                                  cap_mps=cap if state['phase'] != 'released' else None,
                                  reason=release or reason))
        return decisions

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
        objects = self._objects(now)
        extension_objects, protected = self._extension_objects(objects)
        details['protected_vehicle_ids'] = sorted(protected)
        watches = self._edge_watch(extension_objects, ego, path, stations, now)
        details['edge_watch'] = watches
        caps.extend((w['cap_mps'], 'edge_watch') for w in watches)
        side_clear = (self._stationary_side_clear(extension_objects, ego, path, stations, now)
                      if result.debug_details.get('static_avoidance', {}).get('path_validated') else {})
        details['stationary_side_pass'] = sorted(side_clear)
        slow_pass = any(side_clear.values())
        # Caps set by a standing body that is neither a person nor a mover: the map planner's validated path already
        # keeps clear of it, so the planner's stall clock may release them (3 s rule, _release_standing_hold).
        standing_ids = {o['id'] for o in extension_objects if not (o['mover'] or o['person'])}
        standing_holds = []
        crossing_cap, crossing_details = self._crossing_speed(
            extension_objects, ego, path, now, result.acc_target_velocity, side_clear)
        details['approach_crossing'] = crossing_details
        if math.isfinite(crossing_cap):
            standing = crossing_details.get('object_id') in standing_ids
            (standing_holds if standing else caps).append((crossing_cap, 'crossing'))
        guard = self._new_obstacle_guard(extension_objects, ego, path, stations, now)
        details['new_obstacle_guard'] = guard
        if any(d['phase'] == 'observing' for d in guard):
            caps.append((0., 'new_obstacle'))
        caution = slow_pass or any(d['phase'] == 'caution' for d in guard)
        if slow_pass:
            caps.append((NEW_OBSTACLE_CAUTION_SPEED_KPH/3.6, 'caution'))
        for d in guard:
            if d['phase'] == 'caution':
                standing = d['id'] in standing_ids and d['reason'] == 'watching_close_object'
                (standing_holds if standing else caps).append((d['cap_mps'], 'caution'))
        details['cautious_restart'] = caution
        for o in objects:
            if o['age'] > TRACK_STALE_S or o['id'] in side_clear:
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
        self._record_stopped_hazards(extension_objects, ego, details, now)
        caps.extend(self._release_standing_hold(standing_holds, caps, result, ego, details))
        if caps:
            cap, kind = min(caps)
            details['cap'] = round(cap, 2)
            if cap < result.acc_target_velocity-1e-3:
                result.acc_target_velocity = cap
                result.reason = 'jamming_traffic_'+kind
                details['kind'] = kind
        if caution:
            # Rate-limit only increases. A new zero cap takes effect this tick;
            # a stopped Ego cannot accumulate a large release target behind a signal.
            dt = 0. if self._caution_time is None else min(.2, max(0., now-self._caution_time))
            previous = min(self._caution_target, max(0., float(ego.velocity))+.5)
            result.acc_target_velocity = min(result.acc_target_velocity,
                                             previous+NEW_OBSTACLE_CAUTION_ACCEL_MPS2*dt)
            self._caution_target = max(0., result.acc_target_velocity)
            self._caution_time = now
            details['caution_target_mps'] = self._caution_target
        else:
            self._caution_time, self._caution_target = None, max(0., float(ego.velocity))
        return result

    def _release_standing_hold(self, holds, caps, result, ego, details):
        """The map planner's 3 s rule (corridor_optimizer note_held) for a car held only by standing bodies.

        A stop (cap <= 0.05) on the validated path that neither another cap nor the planner's own speed shares runs
        the planner's stall clock, the same one a blocked path runs. Once it reaches FALLBACK_STALL_S the holds give
        way to FALLBACK_SPEED until a tick without them. People, movers, leads and the lane gate keep their caps."""
        speed = self.corridor.FALLBACK_SPEED
        hold = min(holds) if holds else (math.inf, '')
        others = min([cap for cap, _ in caps]+[result.acc_target_velocity])
        validated = result.debug_details.get('static_avoidance', {}).get('path_validated')
        if not (STANDING_HOLD_RELEASE and validated and others > .05
                and (hold[0] <= .05 or (self._standing_released and hold[0] < speed))):
            self._standing_released = False
            return holds
        self._standing_released = self.corridor.note_held(ego)
        if not self._standing_released:
            return holds
        details['standing_hold_release'] = dict(kind=hold[1], cap_mps=round(hold[0], 2), speed_cap_mps=speed,
                                                stall_s=self.corridor.FALLBACK_STALL_S)
        return [(speed, 'standing_release')]
