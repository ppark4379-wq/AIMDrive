"""Accumulated occupancy map of standing obstacles in the Jamming (tunnel) section.

The tunnel planner judged every frame's VE boxes pass/fail, so a box that grew, changed track ID or flipped to
``uncertain`` blocked a path found a moment earlier, and every flip reset the plan. Standing obstacles are accumulated
here as in the static-obstacle section (static_map.py), in the section's forward-Y frame, while moving traffic keeps
the existing moving-object handling.

What is a mover: the tracker labels standing objects ``uncertain`` or ``moving`` on single frames (box jitter, heading
flips), so the label alone is not used. A track is a mover once its centre has travelled MOVER_DISPLACEMENT from
where it was first seen within MOVER_RANGE of the vehicle (farther boxes jump by metres as the detector sees more of
the object), or the tracker called it moving for MOVER_FRAMES frames in a row. Movers do not enter the map; the cells
of a mover seen before it was recognised are cleared by the map's free-space evidence.

Which movers stand again depends on what they are (tunnel traffic, Expert/src/planning/jamming/traffic.py). The size
class comes from the recent boxes seen near the vehicle, not from one frame: far boxes are too small, and a box merged
with the tunnel wall jumps between 1 m and 7 m. A person-sized mover that walked PERSON_CONFIRM_DISPLACEMENT is a
pedestrian and stays a mover (the car waits for pedestrians, it never plans around them); a vehicle that stops stands
again after STALL_S (then it is avoided as a standing obstacle); anything else after STILL_S.
A vehicle that has never moved is a parked one only once it has been seen for PARK_OBSERVE_S near the vehicle.

Virtual boxes (``set_virtual``) are added to the distance field without entering the evidence: the stretch a slow car
being overtaken will drive through, and the stretch of the opposite lane an oncoming car may cover.
The packet is not rewritten; the map is read by the Jamming planner.
"""
from collections import deque
import contextlib
import math

import numpy as np

from AIM2026.Expert.static_map import StaticObstacleMap
from AIM2026.Expert.src.config.parameters.jamming import (
    SIZE_WINDOW_S, SIZE_RANGE_M, SIZE_MIN_FRAMES, PERSON_MAX_LONG_M, VEHICLE_MIN_LONG_M, VEHICLE_MIN_WIDTH_M,
    VEHICLE_SIZE_RATIO, PERSON_CONFIRM_DISPLACEMENT_M, PARK_OBSERVE_S, PARK_RANGE_M, PARK_STILL_M, STALL_S,
    VELOCITY_WINDOW_S,
)

MOVER_DISPLACEMENT = 2.0  # a track whose centre moved this far from its first position is a mover [m]
MOVER_RANGE = 20.         # ... counting only positions seen this close to the vehicle [m]
MOVER_FRAMES = 5          # ... or one labelled moving this many frames in a row
STILL_RADIUS = .5         # a mover whose centre stays this close to one spot ... [m]
STILL_S = 2.              # ... for this long, without a moving label, is standing again [s]
TRACK_TIMEOUT_S = 5.      # forget a track not seen for this long
REGION_LATERAL = (-8.5, 3.)  # section-frame lateral extent of the evidence region (road: -6.5 .. +0.5) [m]
REGION_BEFORE = 25.       # evidence starts this far before the section start line [m]
REGION_AFTER = 15.        # ... and ends this far after its end line [m]
HISTORY_S = max(SIZE_WINDOW_S, VELOCITY_WINDOW_S)
INHERIT_RADIUS_M = 1.5    # a new track ID this close to a recent track's centre (or half its size) continues it [m]
INHERIT_GAP_S = 1.        # ... if that track was seen this recently [s]
JUMP_M = 1.2              # a standing track's centre moving farther than this in one frame ... [m]
JUMP_GAP_S = .3           # ... (frames this close) is another object, not motion
MAP_MIN_FRAMES = 1        # a track enters the map once it has been seen this many times (1: at once)
FUSE_MIN_FRAMES = 5       # a standing car is mapped as the median of at least this many recent boxes
CROSSWISE_COS = math.cos(math.radians(45.))  # a car whose long axis is this far off the lane (section +y) is parked


class TrackMotion:
    """Tracks that have shown sustained motion, with their size class (map frame)."""
    def __init__(self):
        self.tracks = {}
        self.now = None

    def reset(self):
        self.tracks = {}
        self.now = None

    def update(self, boxes, now, ego=None):
        """``boxes`` and ``ego`` = (x, y) of the vehicle in one frame; without ``ego`` every position counts as near."""
        self.now = now
        for box in boxes:
            identity = box.get('track_id')
            if identity is None:
                continue
            x, y = box['center_world']
            if identity not in self.tracks:
                self.tracks[identity] = self._successor(x, y, now, max(box['length_m'], box['width_m']))
            track = self.tracks[identity]
            # Observation age survives short ID changes; it does not alter
            # the existing motion/size classification or packet contents.
            track.setdefault('first_seen', now)
            last = track.get('box')
            if (last is not None and now-track['seen'] <= JUMP_GAP_S and not track.get('mover')
                    and math.hypot(x-last['center_world'][0], y-last['center_world'][1]) > JUMP_M):
                # A jump no tunnel traffic makes in one frame: the tracker gave this ID to the box next to it (a row
                # of barriers). Count motion again from here instead of counting the jump as travel (its size and
                # standing evidence stay: the row is still there).
                track.update(first=None, origin=None, run=0)
                track['first_seen'] = now
                track.pop('new_obstacle_guard', None)
                track.pop('tunnel_side_release', None)
            moving = box.get('motion_state') == 'moving'
            track['run'] = track['run']+1 if moving else 0
            track['seen'], track['box'] = now, box
            track['frames'] = track.get('frames', 0)+1
            if math.hypot(x-track['anchor'][0], y-track['anchor'][1]) > STILL_RADIUS or moving:
                track['anchor'], track['still'] = (x, y), now
            distance = math.inf if ego is None else math.hypot(x-ego[0], y-ego[1])
            near = ego is None or distance <= MOVER_RANGE
            if near and track['first'] is None:
                track['first'] = (x, y)
            if near and track.get('origin') is None:
                track['origin'] = (x, y)
            if distance <= PARK_RANGE_M and track.get('near_since') is None:
                track['near_since'] = now
            history = track.setdefault('history', deque())
            axis = math.radians(box['heading_deg'])+(0. if box['length_m'] >= box['width_m'] else .5*math.pi)
            history.append((now, x, y, max(box['length_m'], box['width_m']), min(box['length_m'], box['width_m']),
                            distance, axis))
            while history and now-history[0][0] > HISTORY_S:
                history.popleft()
            track['size'] = self._size_class(track)
            if track['mover']:
                if track.get('person'):
                    pass                                    # a pedestrian is waited for, never planned around
                elif now-track['still'] >= (STALL_S if track['size'] == 'vehicle' else STILL_S):
                    track.update(mover=False, first=(x, y) if near else None, run=0)
            elif ((near and track['first'] is not None
                   and math.hypot(x-track['first'][0], y-track['first'][1]) >= MOVER_DISPLACEMENT)
                    or track['run'] >= MOVER_FRAMES):
                track['mover'] = track['observed_moving'] = True
            if (track['mover'] and track['size'] == 'person' and track.get('origin') is not None
                    and math.hypot(x-track['origin'][0], y-track['origin'][1]) >= PERSON_CONFIRM_DISPLACEMENT_M):
                track['person'] = True
        self.tracks = {k: v for k, v in self.tracks.items() if now-v['seen'] <= TRACK_TIMEOUT_S}

    def _successor(self, x, y, now, size):
        """A new track ID where a recent track was: the tracker often renumbers an object (a parked car took six IDs
        in one run), and every renumbering would restart the judgement of it. Carry over what was learnt."""
        radius = max(INHERIT_RADIUS_M, .5*size)
        best = None
        for track in self.tracks.values():
            if track.get('box') is None or now-track['seen'] > INHERIT_GAP_S or track['seen'] >= now:
                continue
            px, py = track['box']['center_world']
            distance = math.hypot(px-x, py-y)
            if distance <= radius and (best is None or distance < best[0]):
                best = (distance, track)
        # A car-sized box next to where a pedestrian was is another object, not the pedestrian renumbered.
        if best is None or (best[1].get('size') == 'person' and size > PERSON_MAX_LONG_M):
            return dict(first=None, run=0, mover=False, anchor=(x, y), still=now)
        old = best[1]
        keep = ('first', 'mover', 'anchor', 'still', 'observed_moving', 'person', 'size', 'origin', 'near_since', 'frames',
                'first_seen', 'new_obstacle_guard', 'tunnel_side_release')
        track = {k: old[k] for k in keep if k in old}
        if 'tunnel_side_release' in track:
            track['tunnel_side_release'] = dict(track['tunnel_side_release'])
        track.update(run=0, history=deque(old.get('history', ())), inherited=True)
        return track

    @staticmethod
    def _size_class(track):
        """'person', 'vehicle', 'unknown', or None while too few near boxes were seen. A car keeps its class against
        weak contrary evidence (seen alongside, it is cut to a fragment)."""
        frames = [h for h in track['history'] if track['seen']-h[0] <= SIZE_WINDOW_S and h[5] <= SIZE_RANGE_M]
        current = track.get('size')
        if len(frames) < SIZE_MIN_FRAMES:
            return current
        longs = np.array([h[3] for h in frames])
        p10, p20, p90 = np.percentile(longs, [10, 20, 90])
        if p90 <= PERSON_MAX_LONG_M:
            candidate = 'person'
        elif (p20 >= VEHICLE_MIN_LONG_M and float(np.median([h[4] for h in frames])) >= VEHICLE_MIN_WIDTH_M
              and p90/max(p10, .1) <= VEHICLE_SIZE_RATIO):
            candidate = 'vehicle'
        else:
            candidate = 'unknown'
        if current == 'vehicle' and candidate != 'person':
            return current
        return candidate

    def is_mover(self, identity):
        return bool(self.tracks.get(identity, {}).get('mover'))

    def standing(self, identity):
        """A standing obstacle the planner may drive around: not moving, and if never seen moving, seen long enough
        near the vehicle to be a parked one rather than one about to drive off."""
        track = self.tracks.get(identity)
        if track is None or track.get('mover'):
            return False
        if track.get('observed_moving') or not self.car_like(identity):
            return True
        recent = list(track['history'])[-5:]
        if len(recent) >= 3 and float(np.median([abs(math.cos(h[6])) for h in recent])) > CROSSWISE_COS:
            return True             # across the lane: parked, not traffic about to drive off along it
        if track.get('near_since') is None or track['seen']-track['near_since'] < PARK_OBSERVE_S:
            return False
        # Still over the recent boxes (medians: a box centre jitters by decimetres; the first, far boxes are compared
        # with nothing, their centres jump by metres as more of the car is seen): a car crawling slower than the
        # mover test can see has moved on by more than this.
        history = list(track['history'])
        if len(history) < 6 or history[-1][0]-history[0][0] < .5*PARK_OBSERVE_S:
            return False
        start = np.median(np.array([h[1:3] for h in history[:3]]), axis=0)
        recent = np.median(np.array([h[1:3] for h in history[-3:]]), axis=0)
        return float(np.hypot(*(recent-start))) < PARK_STILL_M

    def car_like(self, identity):
        """Classed as a car, or not classed yet (too few near boxes) while its box is car-sized."""
        track = self.tracks.get(identity, {})
        if track.get('size') is not None:
            return track['size'] == 'vehicle'
        box = track.get('box')
        return box is not None and max(box['length_m'], box['width_m']) >= VEHICLE_MIN_LONG_M

    def velocity(self, identity):
        """Line-fit velocity (vx, vy) [m/s] over the last VELOCITY_WINDOW_S; zero without enough history."""
        track = self.tracks.get(identity)
        if track is None or not track.get('history'):
            return 0., 0.
        rows = np.array([h[:3] for h in track['history'] if track['seen']-h[0] <= VELOCITY_WINDOW_S])
        if len(rows) < 4 or rows[-1, 0]-rows[0, 0] < .4:
            return 0., 0.
        t = rows[:, 0]-rows[:, 0].mean()
        denominator = float(t @ t)
        return (float(t @ (rows[:, 1]-rows[:, 1].mean()))/denominator,
                float(t @ (rows[:, 2]-rows[:, 2].mean()))/denominator)


class TunnelObstacleMap(StaticObstacleMap):
    def __init__(self, section=None):
        if section is None:
            from AIM2026.Expert.src.planning.jamming.section import JammingSection
            section = JammingSection()
        end = section.length+section.global_tail_m+REGION_AFTER
        super().__init__(x_range=REGION_LATERAL, y_range=(-REGION_BEFORE, end),
                         frame=(section.start[0], section.start[1], section.rotation))
        self.motion = TrackMotion()
        self.virtual, self.footprints, self._footprints_only = [], [], False

    def reset(self):
        super().reset()
        if hasattr(self, 'motion'):
            self.motion.reset()
        self.virtual, self.footprints = [], []

    def set_virtual(self, rectangles, footprints=()):
        """Boxes (x, y, heading [rad], length, width) in the map frame counted as occupied until the next call;
        ``footprints`` are the objects themselves (inside those boxes), used where only real bodies count."""
        self.virtual = [tuple(float(v) for v in r) for r in rectangles]
        self.footprints = [tuple(float(v) for v in r) for r in footprints]

    @contextlib.contextmanager
    def footprints_only(self):
        """distance() counts the objects' own footprints instead of the virtual boxes."""
        self._footprints_only = True
        try:
            yield
        finally:
            self._footprints_only = False

    def distance(self, x, y, exact_below=None):
        out = super().distance(x, y) if exact_below is None else super().distance(x, y, exact_below)
        boxes = self.footprints if self._footprints_only else self.virtual
        if not boxes:
            return out
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for cx, cy, heading, length, width in boxes:
            c, s = math.cos(heading), math.sin(heading)
            u, v = (x-cx)*c+(y-cy)*s, -(x-cx)*s+(y-cy)*c
            out = np.minimum(out, np.hypot(np.maximum(np.abs(u)-.5*length, 0.), np.maximum(np.abs(v)-.5*width, 0.)))
        return out

    def observe_packet(self, packet):
        """Accumulate the packet's standing obstacles inside the section (the packet is not changed)."""
        if packet.get('generation') != self.generation:
            self.reset()
            self.generation = packet.get('generation')
        boxes, local = [], []
        for box in packet['boxes']:
            if box.get('planning_speed_source') in ('static_memory', 'static_map'):
                continue
            x, y, heading = self.to_map(*box['center_world'], math.radians(box['heading_deg']))
            if self.contains(float(x), float(y)):
                boxes.append(box)
                local.append(dict(box, center_world=[float(x), float(y)], heading_deg=math.degrees(float(heading))))
        ego = packet.get('ego_pose')
        ego = None if ego is None else tuple(float(v) for v in self.to_map(ego['x'], ego['y'])[:2])
        self.motion.update(local, packet.get('sent_ns', 0)/1e9, ego)
        self.observe(packet, [self._fused(b) for b in boxes if self._standing_evidence(b.get('track_id'))])

    def _fused(self, box):
        """A standing car as the median of its recent boxes. Its VE heading and size swing from frame to frame
        (3.8x1.2, 2.8x2.6, 1.0x2.2, ... for one parked car), and the map keeps their union: 0.7 m wider than the car,
        enough to close a gap the car fits through."""
        identity = box.get('track_id')
        track = self.motion.tracks.get(identity)
        if track is None or not self.motion.car_like(identity) or track.get('mover'):
            return box
        frames = list(track.get('history', ()))
        if len(frames) < FUSE_MIN_FRAMES:
            return box
        axis = .5*math.atan2(sum(math.sin(2.*h[6]) for h in frames), sum(math.cos(2.*h[6]) for h in frames))
        cx, cy = float(np.median([h[1] for h in frames])), float(np.median([h[2] for h in frames]))
        ox, oy, rotation = self.frame
        c, s = math.cos(-rotation), math.sin(-rotation)          # map frame -> world frame
        return dict(box, center_world=[ox+c*cx-s*cy, oy+s*cx+c*cy], heading_deg=math.degrees(axis-rotation),
                    length_m=float(np.median([h[3] for h in frames])), width_m=float(np.median([h[4] for h in frames])))

    def _standing_evidence(self, identity):
        """Movers and cars not yet judged parked stay off the map: planned round too early, a car about to drive off
        (or a slow one not yet recognised as moving) would send the path into the opposite lane."""
        track = self.motion.tracks.get(identity)
        if track is None:
            return True
        if track.get('frames', 0) < MAP_MIN_FRAMES:
            return False            # one- or two-frame boxes far ahead are often not there; nothing clears them in a shadow
        return self.motion.standing(identity)
