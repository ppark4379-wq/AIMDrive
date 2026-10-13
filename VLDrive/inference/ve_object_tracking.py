"""Short-lived world-space tracks from VE boxes, without simulator labels.

Only observed boxes are emitted. Missing detections may retain an association
for 0.75 s but never produce a phantom obstacle. Box geometry is not smoothed.
Motion is estimated from receipt-time samples, not synchronized sensor time.
"""
from collections import deque
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


class VeObjectTracker:
    def __init__(self, max_gap=.75):
        self.max_gap = max_gap
        self.tracks = {}
        self.serial = 0
        self.generation = None
        self.last_time = None

    def update(self, packet):
        now = packet['sent_ns'] / 1e9
        if packet['generation'] != self.generation or (self.last_time is not None and now <= self.last_time):
            self.tracks.clear()
        self.generation, self.last_time = packet['generation'], now
        self.tracks = {k: t for k, t in self.tracks.items() if now - t['seen'] <= self.max_gap}
        boxes, ids = packet['boxes'], list(self.tracks)
        assignments = {}
        if boxes and ids:
            # Private unmatched columns allow every box to start a fresh track.
            cost = np.full((len(boxes), len(ids) + len(boxes)), 1e6)
            for i, box in enumerate(boxes):
                cost[i, len(ids) + i] = 1.
                for j, identity in enumerate(ids):
                    track = self.tracks[identity]
                    dt = now - track['seen']
                    predicted = np.asarray(track['xy']) + np.asarray(track['velocity']) * dt
                    distance = float(np.linalg.norm(np.asarray(box['center_world']) - predicted))
                    previous = track['box']
                    ratio = max(max(box[k], previous[k]) / min(box[k], previous[k])
                                for k in ('length_m', 'width_m'))
                    gate = .8 + min(3., max(track['speed'], box['speed_mps'] or 0.) * dt)
                    if ratio <= 2. and distance <= gate:
                        cost[i, j] = .8 * distance / gate + .15 * (ratio - 1.)
            rows, columns = linear_sum_assignment(cost)
            assignments = {i: ids[j] for i, j in zip(rows, columns) if j < len(ids) and cost[i, j] < 1.}
        output = []
        for index, box in enumerate(boxes):
            identity = assignments.get(index)
            if identity is None:
                self.serial += 1
                identity = self.serial
                self.tracks[identity] = dict(history=deque(maxlen=20), state='uncertain',
                                             velocity=(0., 0.), speed=0.)
            track = self.tracks[identity]
            history = track['history']
            # Do not infer motion across a long detection gap.
            if history and now - history[-1][0] > .3:
                history.clear()
                track['state'] = 'uncertain'
            history.append((now, *box['center_world']))
            while len(history) > 2 and now - history[0][0] > .65:
                history.popleft()
            span = now - history[0][0]
            vx = vy = measured_speed = 0.
            displacement = math.dist(box['center_world'], history[0][1:])
            if span >= .25 and len(history) >= 3:
                values = np.asarray(history)
                times = values[:, 0] - values[:, 0].mean()
                velocity = (times[:, None] * (values[:, 1:] - values[:, 1:].mean(axis=0))).sum(axis=0) / (times @ times)
                vx, vy = map(float, velocity)
                measured_speed = math.hypot(vx, vy)
            spatial_moving = span >= .25 and displacement >= .35 and measured_speed >= .8
            raw_speed = box['speed_mps']
            speed = raw_speed if raw_speed is not None else (measured_speed if spatial_moving else 0.)
            if spatial_moving or (span >= .1 and raw_speed is not None and raw_speed >= .8):
                track['state'] = 'moving'
            elif span >= .5 and displacement < .3 and measured_speed < .6 and (raw_speed is None or raw_speed <= .25):
                track['state'] = 'stationary'
            elif track['state'] == 'stationary' and (speed > .4 or displacement >= .3):
                track['state'] = 'uncertain'
            if spatial_moving and (raw_speed is None or raw_speed < .8):
                speed = measured_speed
            if track['state'] == 'stationary':
                speed = 0.
            heading = box['heading_deg']
            if spatial_moving:
                # An OBB has a 180-degree ambiguity; resolve its forward sign
                # from observed motion without rotating/changing the footprint.
                angle = math.radians(heading)
                if vx * math.cos(angle) + vy * math.sin(angle) < 0.:
                    heading = (heading + 360.) % 360. - 180.
            angle = math.radians(heading)
            velocity = (vx, vy) if spatial_moving else (speed * math.cos(angle), speed * math.sin(angle))
            track.update(seen=now, xy=box['center_world'], box=box, velocity=velocity, speed=speed)
            output.append(dict(box, track_id=identity, motion_state=track['state'],
                               planning_speed_mps=speed, planning_heading_deg=heading,
                               motion_direction_observed=spatial_moving, track_span_s=span))
        return dict(packet, boxes=output, tracking_policy='world_position_receipt_time_v1')
