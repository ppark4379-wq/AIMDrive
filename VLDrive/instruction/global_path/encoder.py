"""Metric geometry -> [3] ego / [5] path features. No trainable weights."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


class PathEncoder:
    """Pose uses x/y metres, yaw_rad CCW, and the route's metric map frame.

    Positive lat_err = ego left of route; yaw_err = ego heading - route heading.
    Distances are signed arc distances; positive curvature bends left.
    Only past projection state may be supplied. No future ego GT is accepted.
    """
    def __init__(self, route, *, distance_scale_m=60., lateral_scale_m=5., curvature_scale=.1,
                 max_route_error_m=10., max_heading_error_deg=60.,
                 projection_backward_m=3., projection_forward_m=80., lookahead_m=60.):
        self.route = route
        self.options = dict(distance_scale_m=distance_scale_m, lateral_scale_m=lateral_scale_m,
            curvature_scale=curvature_scale, max_route_error_m=max_route_error_m,
            max_heading_error_deg=max_heading_error_deg, projection_backward_m=projection_backward_m,
            projection_forward_m=projection_forward_m, lookahead_m=lookahead_m)
        if route.get('schema') != 'route-geometry-v1' or any(not math.isfinite(v) or v <= 0 for v in self.options.values()):
            raise ValueError('Invalid processed route or encoder scales')
        self.points = np.asarray(route['points'], dtype=np.float64)
        self.arc = np.asarray(route['arc_m'], dtype=np.float64)
        heading = np.asarray(route['heading_rad'], dtype=np.float64)
        curvature = np.asarray(route['curvature'], dtype=np.float64)
        if (self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2
                or self.arc.shape != (len(self.points),) or heading.shape != (len(self.points) - 1,)
                or curvature.shape != heading.shape or not np.isfinite(self.points).all()
                or not np.isfinite(self.arc).all() or not np.isfinite(heading).all()
                or not np.isfinite(curvature).all() or self.arc[0] != 0
                or np.any(np.diff(self.arc) <= 0) or not math.isclose(route['length_m'], self.arc[-1])):
            raise ValueError('Invalid processed route geometry arrays')
        event_ids = set()
        for event in route['events']:
            values = [event[k] for k in ('start_m', 'end_m', 'curv_event')]
            if (event['id'] in event_ids or not all(math.isfinite(v) for v in values)
                    or not 0 <= event['start_m'] < event['end_m']
                    or event['direction'] not in ('left', 'right')):
                raise ValueError('Invalid or duplicate geometric event')
            event_ids.add(event['id'])
        self.vectors = np.diff(self.points, axis=0)
        self.lengths2 = (self.vectors ** 2).sum(1)
        if np.any(self.lengths2 <= 1e-12):
            raise ValueError('Processed path contains zero-length segments')

    def project(self, pose, previous_s=None):
        try:
            xy = np.array([pose['x'], pose['y']], dtype=np.float64)
            yaw = float(pose['yaw_rad'])
        except (KeyError, ValueError, TypeError):
            return None
        if not np.isfinite(xy).all() or not math.isfinite(yaw):
            return None
        fraction = np.clip(((xy - self.points[:-1]) * self.vectors).sum(1) / self.lengths2, 0, 1)
        offset = xy - (self.points[:-1] + fraction[:, None] * self.vectors)
        distance = np.linalg.norm(offset, axis=1)
        heading = np.asarray(self.route['heading_rad'])
        error = np.arctan2(np.sin(yaw - heading), np.cos(yaw - heading))
        progress = self.arc[:-1] + fraction * np.diff(self.arc)
        if previous_s is not None and self.route['closed']:
            length = self.route['length_m']
            progress += np.round((previous_s - progress) / length) * length
        valid = (distance <= self.options['max_route_error_m']) & (np.abs(error) <= math.radians(self.options['max_heading_error_deg']))
        if previous_s is not None:
            valid &= ((progress >= previous_s - self.options['projection_backward_m']) &
                      (progress <= previous_s + self.options['projection_forward_m']))
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return None
        scores = distance[candidates] + np.abs(error[candidates])
        if previous_s is not None:
            scores += .001 * np.abs(progress[candidates] - previous_s)
        i = int(candidates[np.argmin(scores)])
        lateral = math.cos(heading[i]) * offset[i, 1] - math.sin(heading[i]) * offset[i, 0]
        return dict(progress_m=float(progress[i]), lat_err=float(lateral), yaw_err=float(error[i]),
                    curv_now=float(self.route['curvature'][i]), segment=i)

    def next_event(self, progress, lookahead_m=None):
        if lookahead_m is None:
            lookahead_m = self.options['lookahead_m']
        candidates = []
        for event in self.route['events']:
            shifts = [0.]
            if self.route['closed']:
                length = self.route['length_m']
                lap = math.floor(progress / length)
                shifts = [(lap + k) * length for k in (-1, 0, 1)]
            for shift in shifts:
                start, end = event['start_m'] + shift, event['end_m'] + shift
                if end >= progress and start <= progress + lookahead_m:
                    candidates.append(dict(event, start_m=start, end_m=end,
                                           instance_id=event['id'] + '@' + str(shift)))
        return min(candidates, key=lambda e: e['start_m']) if candidates else None

    def encode(self, projection, event=None):
        if projection is None:
            return dict(ego_relation=[0.] * 3, path_context=[0.] * 5, path_valid=False)
        p = projection
        start = end = curvature = 0.
        if event is not None:
            start = (event['start_m'] - p['progress_m']) / self.options['distance_scale_m']
            end = (event['end_m'] - p['progress_m']) / self.options['distance_scale_m']
            curvature = event['curv_event'] / self.options['curvature_scale']
        return dict(ego_relation=[p['lat_err'] / self.options['lateral_scale_m'],
                                  math.sin(p['yaw_err']), math.cos(p['yaw_err'])],
                    path_context=[start, end, p['curv_now'] / self.options['curvature_scale'],
                                  curvature, float(event is not None)], path_valid=True)

    def encode_frame(self, pose, *, previous_s=None, event=None):
        """Live-call helper; retain the returned projection with its timestamp."""
        projection = self.project(pose, previous_s)
        if event is None and projection is not None:
            event = self.next_event(projection['progress_m'])
        return self.encode(projection, event), projection


class PathFeatureCache:
    """Verified prepared projections; join frames by (episode_id, timestamp)."""
    def __init__(self, path, labels_manifest_sha256):
        content = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        data = json.loads(content)
        if data.get('schema') != 'frame-path-cache-v1' or data.get('labels_manifest_sha256') != labels_manifest_sha256:
            raise ValueError('Path features and navigation/visual source bundle differ')
        self.encoders, self.frames = {}, {}
        self.instruction_options = data.get('instruction_options', {})
        from instruction.navigation.selector import INSTRUCTION_OPTION_KEYS
        if (not isinstance(self.instruction_options, dict)
                or set(self.instruction_options) - INSTRUCTION_OPTION_KEYS):
            raise ValueError('Invalid instruction options in path cache')
        self.max_gap_ns = data['max_sensor_gap_ms'] * 1e6
        if not math.isfinite(self.max_gap_ns) or self.max_gap_ns <= 0:
            raise ValueError('Invalid path cache time tolerance')
        for episode in data['episodes']:
            identity = episode['episode_id']
            if identity in self.encoders:
                raise ValueError('Duplicate path episode')
            self.encoders[identity] = PathEncoder(episode['route'], **data['encoder_options'])
            for frame in episode['frames']:
                key = (identity, frame['sensor_source_ns'])
                if key in self.frames:
                    raise ValueError('Duplicate path timestamp')
                projection = frame['projection']
                if (type(frame['index']) is not int or frame['index'] < 0
                        or type(frame['sensor_source_ns']) is not int or frame['sensor_source_ns'] < 0
                        or (projection is not None and any(not math.isfinite(projection[k]) for k in
                            ('progress_m', 'lat_err', 'yaw_err', 'curv_now')))):
                    raise ValueError('Invalid path cache frame/projection')
                self.frames[key] = frame

    def eligible(self, identity, timestamps):
        rows = [self.frames.get((identity, t)) for t in timestamps]
        return bool(rows) and all(r is not None and r['projection'] is not None for r in rows) and all(
            b['index'] == a['index'] + 1 and 0 < b['sensor_source_ns'] - a['sensor_source_ns'] <= self.max_gap_ns
            for a, b in zip(rows, rows[1:]))

    def sample(self, identity, timestamps):
        from instruction.navigation.selector import select_instruction
        encoder = self.encoders[identity]
        rows = [self.frames[(identity, t)]['projection'] for t in timestamps]
        selection = select_instruction(encoder, rows, **self.instruction_options)
        # A sustained turn-specific sentence uses one event; generic/compound
        # sentences update the current/next event at each timestamp.
        features = [encoder.encode(p, selection['event'] if selection['event'] is not None else
                    encoder.next_event(p['progress_m'])) for p in rows]
        return selection, {key: [f[key] for f in features] for key in features[0]}
