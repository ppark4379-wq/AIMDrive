"""EMA plus bounded missing-observation hold, for local-frame visualization.

Cells use fixed forward distances so partial segment loss does not erase a wall.
This is not ego-motion-compensated geometry and is not a collision boundary.
"""
import math
import numpy as np


class WallEMA:
    def __init__(self, time_constant=0.15, hold_seconds=0.3, sample_step=0.5,
                 reset_jump=1.0):
        if not all(np.isfinite(v) and v > 0 for v in
                   (time_constant, hold_seconds, sample_step, reset_jump)):
            raise ValueError('EMA parameters must be positive and finite')
        self.tau, self.hold, self.step, self.jump = time_constant, hold_seconds, sample_step, reset_jump
        self.cells = {'left': {}, 'right': {}}
        self.last_time = None
        self.frame_id = None

    def update(self, boundaries, now, observation_time=None):
        observed_at = now if observation_time is None else observation_time
        if not np.isfinite(now) or not np.isfinite(observed_at) or observed_at > now:
            raise ValueError('Expected finite monotonic observation time <= now')
        frame_id = boundaries.get('frame_id', 'base_link')
        if (self.last_time is not None and (now <= self.last_time or now-self.last_time > self.hold)
                or self.frame_id != frame_id):
            self.cells = {'left': {}, 'right': {}}
        self.last_time, self.frame_id = now, frame_id
        output = {k: boundaries[k] for k in ('schema_version', 'frame_id', 'frame_sequence',
                  'stamp_ns', 'sensor_timestamp_ns', 'stamp_source', 'complete') if k in boundaries}
        output.update(semantics='ema_display_only_no_ego_compensation',
                      time_constant_s=self.tau, hold_seconds=self.hold, valid={})
        for side, sign in (('left', 1), ('right', -1)):
            current = {}
            if now - observed_at <= self.hold:
                for segment in boundaries[side]:
                    points = np.asarray(segment['points'], dtype=float)
                    if len(points) < 2:
                        continue
                    for key in range(math.ceil(points[0, 0]/self.step),
                                     math.floor(points[-1, 0]/self.step)+1):
                        x = key*self.step
                        value = np.array([np.interp(x, points[:, 0], points[:, 1]),
                                          np.interp(x, points[:, 0], points[:, 2])])
                        # Choose vehicle-facing geometry if segments overlap.
                        if key not in current or sign*value[0] < sign*current[key][0]:
                            current[key] = value
            cells = {k: v for k, v in self.cells[side].items() if now-v[1] < self.hold}
            for key, value in current.items():
                previous = cells.get(key)
                if previous is not None and observed_at <= previous[1]:
                    continue
                if previous is not None and np.max(np.abs(value-previous[0])) <= self.jump:
                    alpha = -math.expm1(-(observed_at-previous[1])/self.tau)
                    value = previous[0] + alpha*(value-previous[0])
                cells[key] = (value.copy(), observed_at)
            self.cells[side] = cells
            # Join adjacent cells, never invent a span with no retained observation.
            keys = sorted(cells)
            runs = []
            for key in keys:
                if not runs or key != runs[-1][-1]+1 or abs(cells[key][0][0]-cells[runs[-1][-1]][0][0]) > self.jump:
                    runs.append([])
                runs[-1].append(key)
            output[side] = []
            for run in runs:
                if len(run) < 2:
                    continue
                xs = np.asarray(run)*self.step
                values = np.array([cells[k][0] for k in run])
                design = np.column_stack((xs, np.ones(len(xs))))
                # Keep each displayed run straight, rather than a cell-wise zigzag.
                coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
                fitted = design @ coefficients
                ages = [max(0.0, now-cells[k][1]) for k in run]
                held = [k not in current or cells[k][1] != observed_at for k in run]
                output[side].append(dict(
                    points=np.column_stack((xs, fitted)).tolist(), held=held,
                    observation_age_s=ages, contains_held=any(held),
                    max_age_s=max(ages), lifetime_s=max(0.001, self.hold-max(ages)),
                    curve_model='ema_display_line'))
            output['valid'][side] = (bool(output[side]) and boundaries.get('complete', False)
                                      and not any(s['contains_held'] for s in output[side]))
        return output
