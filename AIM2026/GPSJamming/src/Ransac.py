"""Ground RANSAC and wall support/extent detection; numerical logic unchanged."""
import numpy as np
from .Boundary import fit_boundary_curve

def remove_ground(points, expected_z=-1.48, seed_band=0.45,
                  distance_threshold=0.12, max_tilt_deg=12.0,
                  iterations=80, min_inliers=40, seed=0, intermediates=None):
    """Optionally expose the final ground mask without repeating plane distances."""
    if (seed_band <= 0 or distance_threshold <= 0 or iterations < 1
            or min_inliers < 3 or not 0 <= max_tilt_deg < 90):
        raise ValueError("Invalid ground-removal parameters")
    xyz = points[:, :3].astype(np.float64)
    candidates = xyz[np.abs(xyz[:, 2] - expected_z) <= seed_band]
    if len(candidates) < min_inliers:
        if intermediates is not None:
            intermediates['ground_mask'] = np.zeros(len(points), dtype=bool)
        return points.copy(), {"ground_points": 0, "ground_status": "insufficient_seeds"}
    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    cos_tilt = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(iterations):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < cos_tilt:
            continue
        offset = -normal @ a
        if abs(-offset / normal[2] - expected_z) > seed_band:
            continue
        count = np.count_nonzero(np.abs(candidates @ normal + offset) <= distance_threshold)
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None or best_count < min_inliers:
        if intermediates is not None:
            intermediates['ground_mask'] = np.zeros(len(points), dtype=bool)
        return points.copy(), {"ground_points": 0, "ground_status": "no_supported_plane"}
    normal, offset = best
    # Refit the supported plane rather than keeping a noisy three-point sample.
    support = candidates[np.abs(candidates @ normal + offset) <= distance_threshold]
    center = support.mean(axis=0)
    _, _, basis = np.linalg.svd(support - center, full_matrices=False)
    refined = basis[-1]
    if refined[2] < 0:
        refined *= -1
    refined_offset = -refined @ center
    if (refined[2] >= cos_tilt and
            abs(-refined_offset / refined[2] - expected_z) <= seed_band):
        normal, offset = refined, refined_offset
    ground = np.abs(xyz @ normal + offset) <= distance_threshold
    if intermediates is not None:
        intermediates['ground_mask'] = ground
    return points[~ground].copy(), {"ground_points": int(ground.sum()),
        "ground_status": "plane_found", "ground_plane": [*normal.tolist(), float(offset)]}

def connect_segments(segments, sample_step, max_gap, max_offset, max_angle_deg):
    connected = []
    for segment in segments:
        current = dict(segment, observed_ranges=[segment['observed_range']],
                       interpolated=[False] * len(segment['points']))
        if connected:
            previous = connected[-1]
            p = np.asarray(previous['points'])
            q = np.asarray(current['points'])
            gap = q[0, 0] - p[-1, 0]
            slope_p = (p[-1, 1] - p[-2, 1]) / (p[-1, 0] - p[-2, 0])
            slope_q = (q[1, 1] - q[0, 1]) / (q[1, 0] - q[0, 0])
            offset = abs(q[0, 1] - (p[-1, 1] + slope_p * gap))
            angle = abs(np.degrees(np.arctan(slope_p) - np.arctan(slope_q)))
            if 0 < gap <= max_gap and offset <= max_offset and angle <= max_angle_deg:
                xs = np.arange(p[-1, 0] + sample_step, q[0, 0] - 1e-8, sample_step)
                t = (xs - p[-1, 0]) / gap
                bridge = p[-1] + t[:, None] * (q[0] - p[-1])
                previous['points'] += bridge.tolist() + current['points']
                previous['interpolated'] += [True] * len(bridge) + current['interpolated']
                previous.setdefault('interpolated_ranges', []).append([float(p[-1, 0]), float(q[0, 0])])
                previous['observed_ranges'] += current['observed_ranges']
                previous['observed_range'] = [previous['observed_range'][0], current['observed_range'][1]]
                previous['inliers'] += current['inliers']
                previous['confidence'] = min(previous['confidence'], current['confidence'])
                previous['support_ratio'] = min(previous['support_ratio'], current['support_ratio'])
                previous['rmse_m'] = max(previous['rmse_m'], current['rmse_m'])
                continue
        connected.append(current)
    return connected


def detect_walls(points, ground_plane, *, x_min=0.0, x_max=30.0,
                 segment_length=5.0, sample_step=0.5, min_lateral=1.5,
                 min_height=0.3, max_height=2.2, min_height_span=1.0,
                 min_length=3.0, distance_threshold=0.15,
                 max_slope=1.0, min_inliers=12, min_ratio=0.45,
                 max_observation_gap=1.5, iterations=60, seed=0,
                 connect_max_gap=2.5, connect_max_offset=0.5, connect_max_angle_deg=20.0,
                 curve_bin_size=0.5, curve_quantile=0.25, clearance_m=0.0,
                 observed_extent=False, angular_gap_deg=1.2,
                 boundary_support_distance=None):
    if observed_extent:
        return detect_observed_walls(points, ground_plane,
            x_min=x_min, x_max=x_max, max_height=max_height,
            angular_gap_deg=angular_gap_deg, sample_step=sample_step,
            min_lateral=min_lateral, min_height=min_height,
            min_height_span=min_height_span, min_length=min_length,
            distance_threshold=distance_threshold, max_slope=max_slope,
            min_inliers=min_inliers, min_ratio=min_ratio,
            max_observation_gap=max_observation_gap, iterations=iterations, seed=seed,
            curve_bin_size=curve_bin_size, curve_quantile=curve_quantile,
            clearance_m=clearance_m, boundary_support_distance=boundary_support_distance)
    if boundary_support_distance is None:
        boundary_support_distance = distance_threshold
    if not np.isfinite(boundary_support_distance) or boundary_support_distance < distance_threshold:
        raise ValueError('Boundary support distance must be at least the line fitting threshold')
    if not (x_max > x_min and segment_length > 0 and sample_step > 0
            and max_height > min_height >= 0 and min_lateral > 0
            and min_height_span > 0 and min_length > 0
            and distance_threshold > 0 and max_slope > 0
            and min_inliers >= 3 and 0 < min_ratio <= 1
            and max_observation_gap > 0 and iterations >= 1
            and connect_max_gap >= 0 and connect_max_offset >= 0
            and 0 <= connect_max_angle_deg <= 90
            and curve_bin_size > 0 and 0 <= curve_quantile <= 0.5
            and np.isfinite(clearance_m) and clearance_m >= 0):
        raise ValueError('Invalid wall detection parameters')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('Expected Nx4 XYZI')
    result = {'status': 'ground_unavailable', 'left': [], 'right': []}
    clouds = {side: np.empty((0, 4)) for side in ('left', 'right')}
    clouds.update({f'representatives_{side}': np.empty((0, 4)) for side in ('left', 'right')})
    if ground_plane is None:
        return result, clouds
    plane = np.asarray(ground_plane, dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or abs(plane[2]) < 0.5:
        return result, clouds
    a, b, c, d = plane
    ground_z = -(a * points[:, 0] + b * points[:, 1] + d) / c
    height = points[:, 2] - ground_z
    good = (np.isfinite(points).all(axis=1) & (height >= min_height)
            & (height <= max_height) & (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max))
    rng = np.random.default_rng(seed)
    for side, sign in (('left', 1), ('right', -1)):
        collected = []
        for start in np.arange(x_min, x_max, segment_length):
            end = min(start + segment_length, x_max)
            selection = (good & (sign * points[:, 1] >= min_lateral)
                         & (points[:, 0] >= start) & (points[:, 0] < end))
            local = points[selection]
            if len(local) < min_inliers:
                continue
            # Stable sampling under incoming packet/point permutations.
            local = local[np.lexsort((local[:, 2], local[:, 1], local[:, 0]))]
            xy = local[:, :2]
            best = None
            best_count = 0
            for _ in range(iterations):
                p, q = xy[rng.choice(len(xy), 2, replace=False)]
                if abs(q[0] - p[0]) < 0.5:
                    continue
                slope = (q[1] - p[1]) / (q[0] - p[0])
                if abs(slope) > max_slope:
                    continue
                intercept = p[1] - slope * p[0]
                mask = np.abs(xy[:, 1] - slope * xy[:, 0] - intercept) / np.hypot(1, slope) <= distance_threshold
                if mask.sum() > best_count:
                    candidate = local[mask]
                    candidate_heights = candidate[:, 2] + (a * candidate[:, 0] + b * candidate[:, 1] + d) / c
                    if np.ptp(candidate[:, 0]) < min_length or np.ptp(candidate_heights) < min_height_span:
                        continue
                    best, best_count = mask, int(mask.sum())
            if best is None or best_count < min_inliers:
                continue
            support = xy[best]
            slope, intercept = np.linalg.lstsq(
                np.column_stack((support[:, 0], np.ones(len(support)))), support[:, 1], rcond=None)[0]
            residual = np.abs(xy[:, 1] - slope * xy[:, 0] - intercept) / np.hypot(1, slope)
            mask = residual <= distance_threshold
            support = local[mask]
            ratio = len(support) / len(local)
            if len(support) < min_inliers or ratio < min_ratio or abs(slope) > max_slope:
                continue
            heights = support[:, 2] + (a * support[:, 0] + b * support[:, 1] + d) / c
            xs = np.sort(support[:, 0])
            if (np.ptp(heights) < min_height_span or np.ptp(xs) < min_length
                    or np.max(np.diff(xs)) > max_observation_gap):
                continue
            # Only sample inside the observed support. No extrapolation to ROI edges.
            samples = np.arange(np.ceil(xs[0] / sample_step) * sample_step,
                                xs[-1] + 1e-8, sample_step)
            ys = slope * samples + intercept
            if len(samples) < 2:
                continue
            zs = -(a * samples + b * ys + d) / c
            rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
            result[side].append({
                'points': np.column_stack((samples, ys, zs)).tolist(),
                'observed_range': [float(xs[0]), float(xs[-1])],
                'line_fit_range': [float(xs[0]), float(xs[-1])],
                'seed_line_slope': float(slope),
                'seed_line_intercept': float(intercept),
                'inliers': len(support), 'support_ratio': float(ratio),
                'rmse_m': rmse,
                'confidence': float(ratio * max(0, 1 - rmse / distance_threshold)),
            })
            # Strict inliers establish direction, not the wall's full extent.
            # Sparse/offset wall returns can lie outside that seed's x span.
            # Use actual boundary support endpoints; observed-extent processing
            # subsequently splits holes instead of extrapolating across them.
            boundary_support = local[residual <= boundary_support_distance]
            result[side][-1]['observed_range'] = [float(boundary_support[:, 0].min()),
                                                  float(boundary_support[:, 0].max())]
            result[side][-1]['boundary_support_count'] = len(boundary_support)
            collected.append(boundary_support)
        if collected:
            clouds[side] = np.concatenate(collected)
        result[side] = connect_segments(result[side], sample_step, connect_max_gap,
                                        connect_max_offset, connect_max_angle_deg)
        representatives = []
        curves = [fit_boundary_curve(segment, clouds[side], plane, sign, sample_step,
                                     curve_bin_size, curve_quantile,
                                     representatives, clearance_m) for segment in result[side]]
        clouds[f'representatives_{side}'] = np.asarray(representatives, dtype=float).reshape(-1, 4)
        result[side] = [curve for curve in curves if curve is not None]
    result['status'] = 'candidates_found' if result['left'] or result['right'] else 'no_supported_walls'
    return result, clouds

def detect_observed_walls(points, plane, *, x_min, x_max, max_height,
                          angular_gap_deg, **options):
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
