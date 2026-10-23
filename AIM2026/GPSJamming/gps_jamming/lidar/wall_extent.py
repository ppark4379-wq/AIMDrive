"""Fit straight tunnel sides over the supplied cloud's observed extent.

Vertical evidence is required for the whole wall, rather than independently in
every 5 m slice. Angular continuity distinguishes sparse distant returns from
large unobserved openings. No output extends past measured wall support.
"""
import numpy as np

from .wall_curve import fit_boundary_curve


def detect_observed_walls(points, plane, *, x_min, x_max, max_height,
                          angular_gap_deg, **options):
    from .walls import detect_walls
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('Expected Nx4 XYZI')
    if not np.isfinite(angular_gap_deg) or not 0 < angular_gap_deg < 90:
        raise ValueError('angular_gap_deg must be in (0, 90)')
    points = points[np.isfinite(points).all(axis=1)]
    lo = float(points[:, 0].min()) if x_min is None and len(points) else x_min
    hi = float(points[:, 0].max()) if x_max is None and len(points) else x_max
    lo = 0. if lo is None else lo
    hi = lo + 1. if hi is None or hi <= lo else hi
    # The legacy segment selection has an exclusive upper edge.
    hi = float(np.nextafter(hi, np.inf))
    height_limit = max_height
    if height_limit is None:
        height_limit = options['min_height'] + 1.
        if len(points) and plane is not None:
            p = np.asarray(plane)
            if p.shape == (4,) and np.isfinite(p).all() and abs(p[2]) >= .5:
                heights = points[:, 2] + (p[0]*points[:, 0]+p[1]*points[:, 1]+p[3])/p[2]
                height_limit = max(height_limit, float(heights.max()) + 1.)
    fitting = dict(options, x_min=lo, x_max=hi, max_height=height_limit,
                   segment_length=hi-lo, max_observation_gap=hi-lo+1.,
                   connect_max_gap=0.)
    result, clouds = detect_walls(points, plane, **fitting)
    if result['status'] == 'ground_unavailable':
        return result, clouds
    p = np.asarray(plane)
    heights = points[:, 2]+(p[0]*points[:, 0]+p[1]*points[:, 1]+p[3])/p[2]
    eligible = ((points[:, 0] >= lo) & (points[:, 0] < hi)
                & (heights >= options['min_height']) & (heights <= height_limit))
    candidates = points[eligible]
    support_distance = options.get('boundary_support_distance')
    if support_distance is None:
        support_distance = options['distance_threshold']
    # Keep identity from the seed. Extrapolated intercept identifies which
    # side the wall occupies at the vehicle, even when far returns cross Y=0.
    seeds = [(side, model) for side, sign in (('left', 1), ('right', -1))
             for model in result[side] if sign*model['seed_line_intercept'] > 0]
    distances = np.asarray([np.abs(candidates[:, 1]-m['seed_line_slope']*candidates[:, 0]
                        -m['seed_line_intercept'])/np.hypot(1, m['seed_line_slope'])
                        for _, m in seeds])
    owners = np.argmin(distances, axis=0) if seeds else np.empty(0, dtype=int)
    for side, sign in (('left', 1), ('right', -1)):
        segments, representatives, accepted = [], [], []
        for index, (seed_side, model) in enumerate(seeds):
            if seed_side != side:
                continue
            support = candidates[(owners == index) & (distances[index] <= support_distance)]
            support = support[np.lexsort((support[:, 2], support[:, 1], support[:, 0]))]
            dx = np.diff(support[:, 0])
            angles = np.unwrap(np.arctan2(support[:, 1], support[:, 0]))
            # Large metric spacing is normal at grazing incidence. Require a
            # missing angular sector as well before splitting the observed wall.
            # Angular separation alone tends to zero at long range. Never
            # let a remote collinear return bridge an arbitrarily large hole.
            cuts = np.flatnonzero((dx > 2*options['max_observation_gap']) |
                ((dx > options['max_observation_gap']) &
                 (np.abs(np.diff(angles)) > np.radians(angular_gap_deg)))) + 1
            for group in np.split(support, cuts):
                if len(group) < 2 or np.ptp(group[:, 0]) <= 0:
                    continue
                extent = [float(group[:, 0].min()), float(group[:, 0].max())]
                segment = dict(model, observed_range=extent, observed_ranges=[extent],
                               interpolated_ranges=[], inliers=len(group))
                curve = fit_boundary_curve(segment, group, plane, sign,
                    options['sample_step'], options['curve_bin_size'],
                    options['curve_quantile'], representatives, options['clearance_m'])
                if curve is not None:
                    segments.append(curve)
                    accepted.append(group)
        result[side] = segments
        clouds[side] = np.concatenate(accepted) if accepted else np.empty((0, 4))
        clouds['representatives_' + side] = np.asarray(representatives).reshape(-1, 4)
    result['status'] = 'candidates_found' if result['left'] or result['right'] else 'no_supported_walls'
    result['extent_policy'] = 'seeded_wall_support_across_vehicle_axis'
    return result, clouds
