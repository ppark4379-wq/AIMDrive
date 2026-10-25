"""Keep slowly or never moving obstacles that the detector sees only on some frames.

Thin or low objects (bars, small posts) are detected on a fraction of the frames at 10-20 m and, between hits, their box
size and centre jump by up to a metre. The tracker then creates a new ``uncertain`` track every second, which the planner
treats as a mover, so it brakes and releases in turns. A world position that was repeatedly reported without motion is
therefore remembered: for a short time after its last detection a stationary box is published at that position.

Conservative by design: a position needs several recent hits to be remembered, a mover cancels it, the memory ends a few
seconds after the last hit and as soon as the vehicle has passed the position.
"""
import math

RADIUS_M = 1.8            # detections closer than this to a remembered position are the same object
MIN_HITS = 3              # recent detections needed before a position is remembered
HIT_WINDOW_S = 6.0        # ... within this time
HOLD_S = 6.0              # a remembered position is published this long after its last detection
MAX_SIZE_M = 4.0          # only small objects: vehicles are tracked, not remembered
PASSED_M = -4.0           # drop a position once it is this far behind the vehicle (forward coordinate)
MAX_SPEED_MPS = 1.0       # planning speed below which a detection counts as not moving


class StaticMemory:
    def __init__(self, radius=RADIUS_M, min_hits=MIN_HITS, hit_window_s=HIT_WINDOW_S, hold_s=HOLD_S):
        self.radius, self.min_hits, self.hit_window_s, self.hold_s = radius, min_hits, hit_window_s, hold_s
        self.spots = []
        self.next_id = 0
        self.published = 0

    def _near(self, spot, point, radius=None):
        return math.hypot(spot['x'] - point[0], spot['y'] - point[1]) <= (radius or self.radius)

    def apply(self, packet):
        """Add remembered boxes to a tracked packet (in place) and return it."""
        now = packet['sent_ns'] / 1e9
        ego = packet['ego_pose']
        yaw = ego['yaw_grid_rad']
        c, s = math.cos(yaw), math.sin(yaw)
        boxes = packet['boxes']
        movers = [b for b in boxes if b.get('motion_state') == 'moving']
        seen = set()
        for box in boxes:
            if (box.get('motion_state') == 'moving' or max(box['length_m'], box['width_m']) > MAX_SIZE_M
                    or (box.get('planning_speed_mps') or 0.) > MAX_SPEED_MPS):
                continue
            x, y = box['center_world']
            spot = next((k for k in self.spots if self._near(k, (x, y))), None)
            if spot is None:
                spot = dict(id=self.next_id, x=x, y=y, hits=[], last=now, template=None)
                self.next_id += 1
                self.spots.append(spot)
            spot['hits'] = [t for t in spot['hits'] if now - t <= self.hit_window_s] + [now]
            spot['x'], spot['y'], spot['last'] = .7 * spot['x'] + .3 * x, .7 * spot['y'] + .3 * y, now
            area = box['length_m'] * box['width_m']
            if spot['template'] is None or area >= spot['template']['length_m'] * spot['template']['width_m']:
                spot['template'] = dict(box)
            seen.add(spot['id'])
        kept = []
        for spot in self.spots:
            age = now - spot['last']
            forward = c * (spot['x'] - ego['x']) + s * (spot['y'] - ego['y'])
            if age > self.hold_s or forward < PASSED_M or any(self._near(spot, m['center_world']) for m in movers):
                continue
            kept.append(spot)
            if spot['id'] in seen or age <= 0:
                continue
            recent = [t for t in spot['hits'] if now - t <= self.hit_window_s]
            if len(recent) < self.min_hits:
                continue
            ghost = dict(spot['template'])
            ghost.update(center_world=[spot['x'], spot['y']], motion_state='stationary', speed_mps=0.,
                         planning_speed_mps=0., observed_speed_mps=0., track_id=-1 - spot['id'],
                         planning_speed_source='static_memory', held_s=round(age, 2))
            boxes.append(ghost)
            self.published += 1
        self.spots = kept
        return packet
