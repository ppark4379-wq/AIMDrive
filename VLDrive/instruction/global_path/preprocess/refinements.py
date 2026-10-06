"""Route-specific, bounded bend re-detection after a manual geometry review."""
import math

import numpy as np

from .segments import detect_bends


def refine_bends(arc, curvature, events, *, route_sha256, closed, refinements):
    """Use smaller thresholds only in explicitly reviewed sections of one route.

    Original vertices, curvatures and existing event IDs remain unchanged. New
    events enter the real path cache, so their distances are not plot-only text.
    """
    if not isinstance(refinements, (list, tuple)):
        raise ValueError('Bend refinements must be a list')
    events, sections = list(events), []
    length = float(arc[-1])
    required = {'id', 'route_sha256', 'start_m', 'end_m'}
    optional = {'enter_curvature', 'exit_curvature', 'merge_gap_m', 'min_length_m', 'min_angle_deg'}
    seen = set()
    for config in refinements:
        if (not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional):
            raise ValueError('Invalid bend refinement fields')
        digest, identity = config['route_sha256'], config['id']
        start, end = config['start_m'], config['end_m']
        if (not isinstance(identity, str) or not identity or identity in seen or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or
                not 0 <= start < end):
            raise ValueError('Invalid bend refinement identity, route digest or range')
        seen.add(identity)
        if digest != route_sha256:
            continue
        if end > length or any(start < s['end_m'] and end > s['start_m'] for s in sections):
            raise ValueError('Bend refinement exceeds route or overlaps another reviewed section')
        lo = int(np.searchsorted(arc, start, side='left'))
        hi = int(np.searchsorted(arc, end, side='right') - 1)
        if hi <= lo:
            raise ValueError('Bend refinement range contains too few sampled points')
        candidates = detect_bends(arc[lo:hi + 1] - arc[lo], curvature[lo:hi], closed=False,
                                  **{key: config[key] for key in optional if key in config})
        new_ids = []
        for candidate in candidates:
            candidate = dict(candidate, id='reviewed_' + identity + '_' + candidate['id'],
                             start_m=float(arc[lo]) + candidate['start_m'],
                             end_m=float(arc[lo]) + candidate['end_m'], reviewed_section=identity)
            shifts = (-length, 0., length) if closed else (0.,)
            if any(candidate['start_m'] < e['end_m'] + shift and candidate['end_m'] > e['start_m'] + shift
                   for e in events for shift in shifts):
                raise ValueError('Refined bend would replace or overlap an existing event')
            if any(e['id'] == candidate['id'] for e in events):
                raise ValueError('Duplicate refined bend event ID')
            events.append(candidate)
            new_ids.append(candidate['id'])
        sections.append(dict(id=identity, start_m=start, end_m=end, event_ids=new_ids))
    return sorted(events, key=lambda e: e['start_m']), sections
