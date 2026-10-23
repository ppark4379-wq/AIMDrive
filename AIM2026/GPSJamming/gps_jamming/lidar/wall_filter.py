"""Short-window boundary smoothing in local coordinates for RViz debugging.

No ego-motion compensation: never replace raw planner measurements silently.
Missing segments are not held or extrapolated.
"""
from copy import deepcopy
import math
import numpy as np


class WallDisplayFilter:
    def __init__(self, time_constant=0.15, max_jump=0.4, confirm_frames=3,
                 reset_gap=0.4):
        if time_constant <= 0 or max_jump <= 0 or confirm_frames < 1 or reset_gap <= 0:
            raise ValueError('Invalid wall filter parameters')
        self.tau, self.jump = time_constant, max_jump
        self.confirm, self.reset_gap = confirm_frames, reset_gap
        self.previous = {}
        self.pending = {}
        self.last_time = None

    def update(self, boundaries, now):
        result = deepcopy(boundaries)
        dt = now - self.last_time if self.last_time is not None else self.reset_gap + 1
        if dt <= 0 or dt > self.reset_gap:
            self.previous, self.pending = {}, {}
        self.last_time = now
        alpha = 1 - math.exp(-max(dt, 0) / self.tau)
        next_previous, next_pending = {}, {}
        for side in ('left', 'right'):
            result[side] = []
            for segment in boundaries[side]:
                output = deepcopy(segment)
                points = np.asarray(segment['points'], dtype=float)
                accepted = True
                values = []
                for point in points:
                    key = (side, round(float(point[0]), 6))
                    current = point[1:].copy()
                    previous = self.previous.get(key)
                    if previous is not None and np.max(np.abs(current - previous)) > self.jump:
                        candidate, count = self.pending.get(key, (current, 0))
                        count = count + 1 if np.max(np.abs(current - candidate)) <= self.jump / 2 else 1
                        next_pending[key] = (current.copy(), count)
                        if count < self.confirm:
                            accepted = False
                            next_previous[key] = previous
                            values.append(current)
                            continue
                    elif previous is not None:
                        current = previous + alpha * (current - previous)
                    next_previous[key] = current.copy()
                    values.append(current)
                # Reject the whole segment rather than drawing across rejected points.
                if accepted:
                    points[:, 1:] = values
                    output['points'] = points.tolist()
                    result[side].append(output)
        self.previous, self.pending = next_previous, next_pending
        result['filter'] = 'local_frame_display_only_no_ego_compensation'
        result['status'] = 'filtered' if result['left'] or result['right'] else 'unconfirmed_or_missing'
        return result
