"""Short-lived display fallback from measured boundary separation, not road truth."""
import copy
import math
import numpy as np


class WallWidthFallback:
    def __init__(self, max_age_s=2., confirm_frames=3, max_angle_deg=10.,
                 max_width_jump_m=.5, min_overlap_m=3.):
        if (not all(math.isfinite(v) and v > 0 for v in
                    (max_age_s, max_angle_deg, max_width_jump_m, min_overlap_m))
                or confirm_frames < 1):
            raise ValueError('Invalid width fallback parameters')
        self.age, self.confirm = max_age_s, confirm_frames
        self.angle, self.jump, self.overlap = max_angle_deg, max_width_jump_m, min_overlap_m
        self.state = None
        self.context = None
        self.last_time = None

    def update(self, boundaries, now):
        if not math.isfinite(now):
            raise ValueError('Finite observation time required')
        context = (boundaries.get('frame_id'), boundaries.get('clock_epoch'))
        if context != self.context or (self.last_time is not None and now <= self.last_time):
            self.state = None
        self.context, self.last_time = context, now
        output = copy.deepcopy(boundaries)
        output['semantics'] = 'wall_display_with_inferred_width_fallback'
        if self.state and now-self.state['time'] > self.age:
            self.state = None
        if not boundaries.get('complete', False):
            return output
        left, right = boundaries['left'], boundaries['right']
        if left and right:
            # Learn only from a single, overlapping, nearly parallel pair.
            if len(left) != 1 or len(right) != 1:
                return output
            l, r = left[0], right[0]
            ml, mr = l['line_slope'], r['line_slope']
            lo = max(l['points'][0][0], r['points'][0][0])
            hi = min(l['points'][-1][0], r['points'][-1][0])
            if hi-lo < self.overlap or abs(math.degrees(math.atan(ml)-math.atan(mr))) > self.angle:
                self.state = None
                return output
            m = (ml+mr)/2
            widths = [((ml-mr)*x+l['line_intercept']-r['line_intercept'])/math.hypot(1, m)
                      for x in (lo, hi)]
            if min(widths) <= 0 or abs(widths[1]-widths[0]) > self.jump:
                self.state = None
                return output
            width = sum(widths)/2
            consistent = self.state and abs(width-self.state['width']) <= self.jump
            self.state = dict(width=width, count=self.state['count']+1 if consistent else 1,
                              time=now, extent={s: (boundaries[s][0]['points'][0][0],
                                  boundaries[s][0]['points'][-1][0]) for s in ('left', 'right')})
            return output
        if bool(left) == bool(right) or not self.state or self.state['count'] < self.confirm:
            return output
        missing, source, sign = ('right', left, -1) if left else ('left', right, 1)
        lo, hi = self.state['extent'][missing]
        for segment in source:
            points = np.asarray(segment['points'], dtype=float)
            points = points[(points[:, 0] >= lo) & (points[:, 0] <= hi)].copy()
            if len(points) < 2:
                continue
            shift = sign*self.state['width']*math.hypot(1, segment['line_slope'])
            points[:, 1] += shift
            predicted = dict(points=points.tolist(), inferred=True,
                source='opposite_wall_measured_boundary_width',
                width_m=self.state['width'], width_age_s=now-self.state['time'],
                line_slope=segment['line_slope'], line_intercept=segment['line_intercept']+shift)
            output[missing].append(predicted)
        output.setdefault('valid', {})[missing] = False
        return output
