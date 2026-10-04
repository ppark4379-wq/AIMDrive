"""Choose a common qualitative instruction, never a numeric text label."""
import json
import math
from pathlib import Path

TEMPLATES = json.loads(Path(__file__).with_name('templates.json').read_text(encoding='utf-8'))
INSTRUCTION_OPTION_KEYS = {'scheme', 'gentle_curvature', 'sharp_curvature', 'straight_exclusions'}


def straight_exclusion_bounds(encoder, exclusions):
    """Reviewed route-specific ranges where no detected event must NOT mean straight.

    The planned route digest prevents applying one map's metre ranges to another
    route. Explicit options are recorded in the feature cache, not hidden in the
    visualization. Distances refer to one lap of the original metric path.
    """
    if not isinstance(exclusions, (list, tuple)):
        raise ValueError('Invalid straight exclusion list')
    bounds = []
    for item in exclusions:
        if (not isinstance(item, dict) or
                set(item) != {'route_sha256', 'start_m', 'end_m'}):
            raise ValueError('Invalid straight exclusion fields')
        digest = item['route_sha256']
        start, end = item['start_m'], item['end_m']
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or
                not 0 <= start < end):
            raise ValueError('Invalid straight exclusion range or route digest')
        if digest == encoder.route['route_sha256']:
            if end > encoder.route['length_m']:
                raise ValueError('Straight exclusion exceeds matching route length')
            bounds.append((start, end))
    return bounds


def select_instruction(encoder, projections, *, scheme='legacy', gentle_curvature=.025, sharp_curvature=.06,
                       straight_exclusions=()):
    """Use first-frame planned geometry only, never future ego progress.

    ``legacy`` preserves the original four-kind policy. ``geometry`` additionally
    distinguishes bend tightness and two opposite-sign planned bends. Both use
    only the first ego projection and already known planned route geometry.

    A generic sentence is deliberately used around upcoming transitions. A
    directional sentence is used only for a bend covering the whole configured
    lookahead; straight requires no bend in that same planned horizon and no
    matching reviewed straight exclusion. This is
    conservative, not a guarantee about arbitrary future vehicle motion.
    """
    if not projections or any(p is None for p in projections):
        raise ValueError('Instruction selection requires valid frame projections')
    if (scheme not in ('legacy', 'geometry')
            or not math.isfinite(gentle_curvature) or not math.isfinite(sharp_curvature)
            or not 0 < gentle_curvature < sharp_curvature):
        raise ValueError('Invalid navigation instruction scheme or curvature thresholds')
    progress = projections[0]['progress_m']
    exclusions = straight_exclusion_bounds(encoder, straight_exclusions)
    local_progress = progress % encoder.route['length_m'] if encoder.route['closed'] else progress
    horizon = encoder.options['lookahead_m']
    event = encoder.next_event(progress, horizon)
    kind, target = 'follow', None
    if event is None:
        kind = 'follow' if any(start <= local_progress < end for start, end in exclusions) else 'straight'
    elif scheme == 'geometry' and event.get('reviewed_section'):
        # Reviewed geometry is a planned bend target, including its approach
        # and exit. Positive/negative per-frame distances distinguish phases;
        # this sentence is not an instruction to turn immediately. Do not pin
        # a completed event onto later frames of the same window.
        curvature = abs(event['curv_event'])
        kind = ('gentle_' if curvature < gentle_curvature else 'sharp_' if curvature >= sharp_curvature else '') + event['direction']
    elif event['start_m'] <= progress and event['end_m'] >= progress + horizon:
        kind, target = event['direction'], event
        if scheme == 'geometry':
            curvature = abs(event['curv_event'])
            if curvature < gentle_curvature:
                kind = 'gentle_' + kind
            elif curvature >= sharp_curvature:
                kind = 'sharp_' + kind
    elif scheme == 'geometry':
        # A common compound sentence describes the planned sequence, not an
        # immediate turn at every frame. Keep the conservative horizon guard:
        # after the second bend ends, fall back to a generic common sentence.
        cursor = event['end_m'] + 1e-6
        limit = progress + horizon
        second = encoder.next_event(cursor, limit - cursor) if cursor < limit else None
        if (second is not None and second['direction'] != event['direction']
                and second['end_m'] >= limit):
            kind = event['direction'] + '_then_' + second['direction']
        # No fixed event target: numeric features must update the current/next
        # bend independently at each frame, never mix two distances in 8 fields.
    return dict(text=TEMPLATES[kind], kind=kind, event=target,
                event_id=target['instance_id'] if target else '')
