"""Rule-based geometry. Defaults are starting values, not calibrated turn labels."""
import numpy as np

from .load import load_path


def process_path(source, *, spacing_m=1., smoothing_m=3., turn_curvature=.015,
                 min_turn_length_m=3., segmentation='threshold',
                 turn_exit_curvature=.0075, merge_gap_m=5., min_turn_angle_deg=15., bend_refinements=()):
    """Resample by arc distance, smooth heading, and find signed bend regions.

    Events describe geometry, not intersection semantics or traffic rules.
    Closed-route events may have end_m > length_m to span the seam.
    """
    params = dict(spacing_m=spacing_m, smoothing_m=smoothing_m,
                  turn_curvature=turn_curvature, min_turn_length_m=min_turn_length_m)
    if not isinstance(bend_refinements, (list, tuple)):
        raise ValueError('Bend refinements must be a list')
    if any(not np.isfinite(v) or v <= 0 for v in params.values()):
        raise ValueError('Geometry parameters must be finite and positive')
    if segmentation not in ('threshold', 'hysteresis'):
        raise ValueError('segmentation must be threshold or hysteresis')
    route = load_path(source)
    points = np.asarray(route['points'])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = max(4, int(np.ceil(arc[-1] / spacing_m)) + 1)
    s = np.linspace(0., arc[-1], count)
    xy = np.column_stack([np.interp(s, arc, points[:, d]) for d in range(2)])
    vectors = np.diff(xy, axis=0)
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    # Smooth unit heading vectors to avoid +/- pi discontinuities.
    radius = min(len(headings) - 1, max(1, int(round(smoothing_m / (s[1] - s[0]) / 2))))
    mode = 'wrap' if route['closed'] else 'edge'
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smooth = np.arctan2(np.convolve(np.pad(np.sin(headings), radius, mode=mode), kernel, 'valid'),
                        np.convolve(np.pad(np.cos(headings), radius, mode=mode), kernel, 'valid'))
    if route['closed']:
        delta = np.angle(np.exp(1j * (np.roll(smooth, -1) - np.roll(smooth, 1))))
        curvature = delta / (2 * (s[1] - s[0]))
    else:
        curvature = np.gradient(np.unwrap(smooth), s[:-1])
    kinds = np.where(np.abs(curvature) >= turn_curvature, np.sign(curvature), 0).astype(int)
    runs = []
    start = 0
    for end in range(1, len(kinds) + 1):
        if end == len(kinds) or kinds[end] != kinds[start]:
            if kinds[start]:
                runs.append(dict(start_m=float(s[start]), end_m=float(s[end]), sign=int(kinds[start]),
                                 curv_event=float(curvature[start:end][np.argmax(np.abs(curvature[start:end]))])))
            start = end
    if (route['closed'] and len(runs) > 1 and runs[0]['start_m'] == 0
            and abs(runs[-1]['end_m'] - arc[-1]) < 1e-6 and runs[0]['sign'] == runs[-1]['sign']):
        first, last = runs.pop(0), runs.pop()
        last['end_m'] = first['end_m'] + float(arc[-1])
        last['curv_event'] = max((first['curv_event'], last['curv_event']), key=abs)
        runs.append(last)
    events = [dict(id='bend_%04d' % i, direction='left' if r['sign'] > 0 else 'right',
                   **{k: r[k] for k in ('start_m', 'end_m', 'curv_event')})
              for i, r in enumerate(runs) if r['end_m'] - r['start_m'] >= min_turn_length_m]
    if segmentation == 'hysteresis':
        from .segments import detect_bends
        events = detect_bends(s, curvature, closed=route['closed'], enter_curvature=turn_curvature,
            exit_curvature=turn_exit_curvature, merge_gap_m=merge_gap_m,
            min_length_m=min_turn_length_m, min_angle_deg=min_turn_angle_deg)
        params.update(segmentation=segmentation, turn_exit_curvature=turn_exit_curvature,
                      merge_gap_m=merge_gap_m, min_turn_angle_deg=min_turn_angle_deg)
    reviewed_sections = []
    if bend_refinements:
        from .refinements import refine_bends
        events, reviewed_sections = refine_bends(s, curvature, events, route_sha256=route['route_sha256'],
                                                closed=route['closed'], refinements=bend_refinements)
        params['bend_refinements'] = bend_refinements
    result = dict(schema='route-geometry-v1', route_sha256=route['route_sha256'], closed=route['closed'],
                points=xy.tolist(), arc_m=s.tolist(), heading_rad=headings.tolist(),
                curvature=curvature.tolist(), length_m=float(arc[-1]), events=events, preprocessing=params)
    if reviewed_sections:
        result['reviewed_sections'] = reviewed_sections
    return result
