"""Local refinements add real path events without lowering global thresholds."""
import copy

import numpy as np
import pytest

from instruction.global_path.preprocess.refinements import refine_bends
from instruction.global_path.preprocess.process import process_path
from instruction.global_path.encoder import PathEncoder
from instruction.navigation.selector import select_instruction
from validation.navigation.path_report import projection_at


def fixture():
    arc = np.arange(201., dtype=float)
    curvature = np.zeros(200)
    curvature[40:120] = .006
    events = [dict(id='original', start_m=160., end_m=190., direction='right', curv_event=-.05)]
    options = dict(id='review', route_sha256='1' * 64, start_m=0., end_m=140.,
                   enter_curvature=.003, exit_curvature=.0015, merge_gap_m=5., min_length_m=20., min_angle_deg=15.)
    return arc, curvature, events, options


def test_refinement_adds_measured_gentle_bend_and_preserves_original():
    arc, curvature, original, options = fixture()
    before = copy.deepcopy(original)
    events, sections = refine_bends(arc, curvature, original, route_sha256='1' * 64,
                                   closed=False, refinements=[options])
    assert original == before and events[1] == before[0]
    assert events[0]['direction'] == 'left'
    assert (events[0]['start_m'], events[0]['end_m']) == (40., 120.)
    assert events[0]['curv_event'] == .006
    assert events[0]['angle_deg'] == pytest.approx(np.degrees(.006 * 80))
    assert events[0]['reviewed_section'] == 'review'
    assert sections[0]['event_ids'] == [events[0]['id']]


def test_unrelated_route_is_not_refined():
    arc, curvature, original, options = fixture()
    events, sections = refine_bends(arc, curvature, original, route_sha256='2' * 64,
                                   closed=False, refinements=[options])
    assert events == original and sections == []


@pytest.mark.parametrize('changes', [dict(start_m=-1.), dict(end_m=201.), dict(route_sha256='wrong'),
                                   dict(id=''), dict(enter_curvature=.0001), dict(extra=True)])
def test_invalid_refinement_is_rejected(changes):
    arc, curvature, original, options = fixture()
    options.update(changes)
    with pytest.raises(ValueError):
        refine_bends(arc, curvature, original, route_sha256='1' * 64, closed=False, refinements=[options])


def test_refinement_cannot_silently_replace_an_original_event():
    arc, curvature, _, options = fixture()
    original = [dict(id='protected', start_m=50., end_m=60., direction='left', curv_event=.04)]
    with pytest.raises(ValueError, match='replace or overlap'):
        refine_bends(arc, curvature, original, route_sha256='1' * 64, closed=False, refinements=[options])


def test_reviewed_bend_is_announced_ahead_and_numeric_target_is_not_pinned():
    route = process_path(dict(points=[[0, 0], [300, 0]]))
    route['events'] = [dict(id='gentle', start_m=100., end_m=140., direction='left',
                            curv_event=.006, reviewed_section='review')]
    encoder = PathEncoder(route)
    for position, expected in [(39.99, 'straight'), (40., 'gentle_left'),
                               (110., 'gentle_left'), (139.99, 'gentle_left'), (140.01, 'straight')]:
        result = select_instruction(encoder, [projection_at(route, position)], scheme='geometry')
        assert result['kind'] == expected
        assert result['event'] is None
    # An unreviewed event retains the existing conservative selection policy.
    route['events'][0].pop('reviewed_section')
    assert select_instruction(encoder, [projection_at(route, 110.)], scheme='geometry')['kind'] == 'follow'


def test_closed_lap_keeps_correct_future_and_current_event_distances():
    route = process_path(dict(points=[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], closed=True))
    route['events'] = [dict(id='gentle', start_m=100., end_m=180., direction='left',
                            curv_event=.006, reviewed_section='review')]
    encoder = PathEncoder(route)
    for position in (450., 520.):
        p = projection_at(route, position % route['length_m'])
        p['progress_m'] = position
        assert select_instruction(encoder, [p], scheme='geometry')['kind'] == 'gentle_left'
        event = encoder.next_event(position)
        features = encoder.encode(p, event)
        assert features['path_context'][0] == pytest.approx((500. - position) / 60.)
        assert features['path_context'][1] == pytest.approx((580. - position) / 60.)
