"""Map-constrained VE tracking. Uses observed positions, never NPC/GT speeds."""
from collections import deque
from copy import deepcopy
import bisect
import math
import statistics


class Lane:
    def __init__(self, points):
        self.points = []
        for p in points:
            xy = tuple(p[:2])
            if not self.points or math.dist(xy, self.points[-1]) >= 1.:
                self.points.append(xy)
        if len(self.points) < 2:
            raise ValueError('Lane tracking needs a nonempty mapped lane')
        self.s = [0.]
        for a, b in zip(self.points, self.points[1:]):
            self.s.append(self.s[-1] + math.dist(a, b))

    def project(self, xy):
        best = None
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = self.s[i+1]-self.s[i]
            u = max(0., min(1., ((xy[0]-a[0])*dx+(xy[1]-a[1])*dy)/length**2))
            distance = math.hypot(xy[0]-a[0]-u*dx, xy[1]-a[1]-u*dy)
            candidate = (distance, self.s[i]+u*length)
            if best is None or candidate < best:
                best = candidate
        return best

    def at(self, s):
        s = max(0., min(s, self.s[-1]))
        i = max(0, min(len(self.points)-2, bisect.bisect_right(self.s, s)-1))
        a, b = self.points[i:i+2]
        u = (s-self.s[i])/(self.s[i+1]-self.s[i])
        return [a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])], math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))


class LaneTracker:
    CONFIRMED_HOLD_S = 2.0
    UNCONFIRMED_HOLD_S = .35
    MAX_UNCERTAINTY_M = 10.0
    MAX_BOX_MARGIN_M = 2.5  # Maximum total size increase: 5 m per dimension.

    def __init__(self, lanes):
        self.lanes = {name: Lane(points) for name, points in lanes.items()}
        if not self.lanes:
            raise ValueError('Lane tracking requires mapped highway lanes')
        self.tracks = {}
        self.next_id = 2000000
        self.epoch = None
        self.last_time = None

    @classmethod
    def from_context(cls, context):
        managed = context.cfg['highway_npc']['managed_traffic']
        definitions = (managed['competition']['lanes'] if managed.get('policy') == 'competition'
                       else managed['lanes'])
        lanes = {}
        for row in definitions:
            # Shared approach/forks are ambiguous: start only on the assigned lane.
            route = row['route']
            route = route[route.index(row.get('offset_origin_link', route[0])):]
            lanes[row['name']] = [p for link in route for p in context.map_loader.get_link_points(link)]
        return cls(lanes)

    @staticmethod
    def fit(history):
        if len(history) < 5 or history[-1][0]-history[0][0] < .6:
            return None
        slopes = [(b[1]-a[1])/(b[0]-a[0]) for i, a in enumerate(history)
                  for b in list(history)[i+1:] if b[0]-a[0] >= .3]
        if not slopes:
            return None
        speed = statistics.median(slopes)
        origin = statistics.median(s-speed*t for t, s in history)
        residual = statistics.median(abs(s-origin-speed*t) for t, s in history)
        spread = statistics.median(abs(v-speed) for v in slopes)
        if not 0 <= speed <= 35 or residual > .8 or spread > 2.5:
            return None
        return speed, max(2., spread*2), residual

    def apply(self, packet):
        now = packet.get('object_received_ns', packet['sensor_received_ns'])/1e9
        epoch = packet.get('generation')
        if epoch != self.epoch or (self.last_time is not None and now < self.last_time):
            self.tracks.clear()
        self.epoch, self.last_time = epoch, now
        # A spatial-only expiry cannot retire unknown-speed tracks: their reachable
        # interval grows forever. Expire before association as well as publication
        # so a stale identity cannot be revived by an unrelated passing vehicle.
        for identity, tr in list(self.tracks.items()):
            age = now-tr['seen']
            lifetime = self.CONFIRMED_HOLD_S if tr['confirmed'] else self.UNCONFIRMED_HOLD_S
            if age >= lifetime or 1.5+tr['sigma']*age > self.MAX_UNCERTAINTY_M:
                del self.tracks[identity]
        observations, passthrough = [], []
        for box in packet['boxes']:
            if (box.get('source') == 'lidar' or box.get('detection_held')
                    or box.get('planning_speed_source') in ('static_memory', 'static_map')
                    or not isinstance(box.get('track_id'), int) or box['track_id'] < 0):
                passthrough.append(box)
                continue
            projections = []
            for name, lane in self.lanes.items():
                distance, progress = lane.project(box['center_world'])
                projections.append((distance, name, progress))
            projections.sort()
            distance, name, s = projections[0]
            if distance > 2. or (len(projections)>1 and projections[1][0]-distance < .4):
                passthrough.append(box)
                continue
            observations.append((box, name, s))
        # One-to-one spatial association; incoming tracker IDs are diagnostic only.
        edges = []
        for i, (box, lane, s) in enumerate(observations):
            for identity, tr in self.tracks.items():
                if tr['lane'] != lane:
                    continue
                age = now-tr['seen']
                predicted = tr['s']+tr['speed']*age
                error = abs(s-predicted)
                gate = min(8., 2.5+tr['sigma']*age)
                if error <= gate:
                    edges.append((error, i, identity))
        assigned, used = {}, set()
        for error, i, identity in sorted(edges):
            if i not in assigned and identity not in used:
                assigned[i] = identity
                used.add(identity)
        output = list(passthrough)
        seen = set()
        for i, (box, lane, s) in enumerate(observations):
            identity = assigned.get(i)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
                self.tracks[identity] = dict(lane=lane, history=deque(), speed=0., sigma=15.,
                                             confirmed=False, seen=now, s=s)
            tr = self.tracks[identity]
            innovation = abs(s-(tr['s']+tr['speed']*(now-tr['seen'])))
            compatible = innovation <= max(2.5, tr['sigma']*(now-tr['seen']))
            history = tr['history']
            if not history or now > history[-1][0]:
                history.append((now, s))
            while history and now-history[0][0] > 1.5:
                history.popleft()
            fit = self.fit(history)
            if fit:
                tr['speed'], tr['sigma'], _ = fit
                tr['confirmed'] = True
            elif (len(history) >= 5 and history[-1][0]-history[0][0] >= .6) or not compatible:
                tr['confirmed'] = False
                tr['sigma'] = 15.
            # After a gap there may not yet be enough new samples to fit.
            # Preserve previously verified motion if the new position supports it.
            tr.update(seen=now, s=s, box=deepcopy(box))
            seen.add(identity)
            output.append(self.emit(identity, tr, now))
        for identity, tr in list(self.tracks.items()):
            if identity in seen:
                continue
            age = now-tr['seen']
            lane = self.lanes[tr['lane']]
            speed = tr['speed'] if tr['confirmed'] else 15.
            radius = 1.5+tr['sigma']*age
            predicted = tr['s']+speed*age
            ego = packet.get('ego_pose')
            ego_s = lane.project((ego['x'], ego['y']))[1] if ego else None
            # Spatial retirement supplements the finite lifetime above.
            if predicted-radius > lane.s[-1]+10 or (ego_s is not None and predicted+radius < ego_s-40):
                del self.tracks[identity]
                continue
            output.append(self.emit(identity, tr, now))
        return dict(packet, boxes=output, lane_tracking=True,
                    detection_held_count=sum(bool(b.get('detection_held')) for b in output))

    def emit(self, identity, tr, now):
        age = now-tr['seen']
        speed = tr['speed'] if tr['confirmed'] else 15.
        radius = 1.5+tr['sigma']*age
        lane = self.lanes[tr['lane']]
        center, heading = lane.at(tr['s']+speed*age)
        lateral = .5
        # Cover road curvature across the possible longitudinal positions as well.
        angle = math.radians(heading)
        if age:
            for fraction in (-1., -.5, 0., .5, 1.):
                point, _ = lane.at(tr['s']+speed*age+radius*fraction)
                lateral = max(lateral, abs(-(point[0]-center[0])*math.sin(angle)
                                           +(point[1]-center[1])*math.cos(angle))+.5)
        box = deepcopy(tr['box'])
        box.update(source_track_id=box['track_id'], track_id=identity,
                   raw_center_world=box['center_world'], center_world=center,
                   planning_heading_deg=heading, heading_deg=heading,
                   planning_speed_mps=tr['speed'] if tr['confirmed'] else None,
                   planning_speed_source='lane_position_history' if tr['confirmed'] else 'lane_unconfirmed',
                   motion_state='moving' if tr['confirmed'] and tr['speed']>.5 else 'uncertain',
                   lane_track_confirmed=tr['confirmed'], lane_track=tr['lane'],
                   detection_held=age>0, detection_hold_age_s=age,
                   longitudinal_uncertainty_m=min(radius, self.MAX_BOX_MARGIN_M),
                   lateral_uncertainty_m=min(lateral, self.MAX_BOX_MARGIN_M),
                   speed_uncertainty_mps=tr['sigma'],
                   detection_position_source='lane_motion_history')
        return box
