"""Straight least-squares boundary from vehicle-facing wall representatives."""
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
