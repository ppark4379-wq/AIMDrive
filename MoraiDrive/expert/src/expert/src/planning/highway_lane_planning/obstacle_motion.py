"""Highway-only motion history; never change the simulator's object labels."""

from . import parameters as p
from ...localization.point import Point


class ObstacleMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}
        self.states = {}
        self.time = 0.0
        self.serial = 0

    def update(self, entries, dt):
        self.time += dt
        self.tracks = {key: track for key, track in self.tracks.items()
                       if self.time - track['seen'] <= p.STOP_TRACK_HOLD_TIME}
        self.states = {}
        used = set()
        for entry in entries:
            obj = entry['object_info']
            uid = getattr(obj, 'unique_id', None)
            name = str(getattr(obj, 'name', '') or '')
            key = ('id', uid) if uid is not None else (('name', name) if name else None)
            if key is None:
                # Some transports omit IDs and recreate ObjectInfo every frame.
                candidates = [(track['position'].distance(obj.position), old_key)
                              for old_key, track in self.tracks.items()
                              if old_key[0] == 'anonymous' and old_key not in used
                              and track['type'] == obj.type]
                distance, old_key = min(candidates, default=(float('inf'), None))
                if distance <= p.STOP_ASSOCIATION_DISTANCE:
                    key = old_key
                else:
                    self.serial += 1
                    key = ('anonymous', self.serial)
            used.add(key)
            previous = self.tracks.get(key)
            speed = abs(float(obj.velocity))
            continuous = previous is not None and self.time - previous['seen'] <= 1.5 * dt + 1e-9
            anchor = previous['anchor'] if previous is not None else Point(obj.position.x, obj.position.y)
            displaced = anchor.distance(obj.position) > p.STOP_POSITION_TOLERANCE
            was_static = previous is not None and previous['state'].startswith('STATIC')
            moving = speed >= p.STOP_RELEASE_SPEED or displaced
            stopped = speed <= p.STOP_SPEED_THRESHOLD or (was_static and not moving)
            moving_seen = moving or (speed > p.STOP_SPEED_THRESHOLD and not was_static) or (
                previous is not None and previous['moving_seen'])
            seconds = previous['seconds'] if continuous and not moving else 0.0
            if stopped and continuous and not moving and previous['state'] != 'MOVING':
                seconds += dt
            if moving or not stopped:
                seconds = 0.0
                anchor = Point(obj.position.x, obj.position.y)
            native_static = bool(getattr(obj, 'is_static', False)) and stopped and not moving
            threshold = p.STOP_CONFIRM_TIME if moving_seen else p.INITIAL_STOP_CONFIRM_TIME
            if native_static:
                state = 'STATIC_EXPLICIT'
            elif stopped and not moving and (seconds + 1e-9 >= threshold or was_static):
                state = 'STATIC_CONFIRMED'
            else:
                state = 'STOPPING' if stopped and not moving else 'MOVING'
            record = dict(state=state, seconds=seconds, moving_seen=moving_seen,
                          position=Point(obj.position.x, obj.position.y), anchor=anchor,
                          seen=self.time, type=obj.type)
            self.tracks[key] = record
            self.states[id(obj)] = record

    def is_static(self, obj):
        return self.states.get(id(obj), {}).get('state', '').startswith('STATIC')
