"""Local vertical-wall hypotheses and ground-referenced BEV boundaries.

Compatible neighboring segments are joined with explicitly marked interpolation.
Confidence is a geometric support score, not a calibrated probability.
"""
import numpy as np
from .wall_curve import fit_boundary_curve


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
        from .wall_extent import detect_observed_walls
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
