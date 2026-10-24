"""Vehicle-facing straight boundaries and measured-width fallback."""
import copy
import math
import numpy as np

def fit_boundary_curve(segment, inliers, plane, sign, sample_step,
                       bin_size=0.5, quantile=0.25, representatives_sink=None,
                       clearance_m=0.0):
    """Fit y(x) inside observed representative extent; never use frame history."""
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('clearance_m must be finite and nonnegative')
    ranges = segment['observed_ranges']
    selected = np.zeros(len(inliers), dtype=bool)
    for lo, hi in ranges:
        selected |= (inliers[:, 0] >= lo) & (inliers[:, 0] <= hi)
    points = inliers[selected]
    representatives = []
    bins = np.floor(points[:, 0] / bin_size).astype(np.int64)
    for key in np.unique(bins):
        group = points[bins == key]
        representatives.append([float(np.median(group[:, 0])),
                                float(sign * np.quantile(sign * group[:, 1], quantile))])
    # Expose pre-fit representatives even if too few remain for a curve.
    if representatives_sink is not None:
        a, b, c, d = plane
        representatives_sink.extend([[x, y, -(a*x + b*y + d)/c, 1.0]
                                     for x, y in representatives])
    if len(representatives) < 2:
        return None
    reps = np.asarray(representatives)
    # Cover the full accepted wall extent, including both observed endpoints.
    lo, hi = float(points[:, 0].min()), float(points[:, 0].max())
    xs = np.unique(np.concatenate(([lo], np.arange(
        np.ceil(lo / sample_step) * sample_step, hi, sample_step), [hi])))
    if len(xs) < 2:
        return None
    design = np.column_stack((reps[:, 0], np.ones(len(reps))))
    slope, intercept = np.linalg.lstsq(design, reps[:, 1], rcond=None)[0]
    fitted_intercept = float(intercept)
    # All accepted wall points must lie on the wall side of the boundary:
    # sign * (y - slope*x - intercept) >= 0.
    intercept = sign * np.min(sign * (points[:, 1] - slope * points[:, 0]))
    intercept -= sign * clearance_m * np.hypot(1, slope)
    ys = slope * xs + intercept
    # Side belongs to the seeded wall, not to each point's instantaneous Y.
    # An oblique wall may cross the forward axis within its observed extent.
    if not np.isfinite(ys).all():
        return None
    model = 'vehicle_facing_support_line'
    rmse = float(np.sqrt(np.mean((design @ np.array([slope, intercept]) - reps[:, 1]) ** 2)))
    a, b, c, d = plane
    zs = -(a * xs + b * ys + d) / c
    observed = np.zeros(len(xs), dtype=bool)
    for lo, hi in ranges:
        observed |= (xs >= lo) & (xs <= hi)
    output = dict(segment)
    output.update(points=np.column_stack((xs, ys, zs)).tolist(),
                  interpolated=(~observed).tolist(),
                  representatives=reps.tolist(), curve_model=model,
                  curve_rmse_m=rmse, facing_quantile=quantile,
                  line_slope=float(slope), line_intercept=float(intercept),
                  enclosing_point_count=len(points),
                  clearance_m=float(clearance_m),
                  support_offset_m=float(abs(intercept - fitted_intercept) / np.hypot(1, slope)))
    return output

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
