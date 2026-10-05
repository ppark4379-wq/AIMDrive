"""Expanded route-only templates preserve common-window selection causality."""
import hashlib

import pytest

from instruction.navigation.selector import select_instruction, TEMPLATES
from instruction.global_path.preprocess.process import process_path
from instruction.global_path.encoder import PathEncoder, PathFeatureCache
from data.src.processing.build_path_features import build_path_features
from test_navigation_cache import cache


def encoder(events):
    route = process_path(dict(points=[[0, 0], [300, 0]]))
    # Fixture events express arbitrary signed planned bend summaries.
    route['events'] = [dict(id='event_%d' % i, start_m=start, end_m=end,
                           direction=direction, curv_event=k) for i, (start, end, direction, k) in enumerate(events)]
    return PathEncoder(route)


@pytest.mark.parametrize('direction,curvature,kind', [
    ('left', .02, 'gentle_left'), ('right', -.02, 'gentle_right'),
    ('left', .04, 'left'), ('right', -.04, 'right'),
    ('left', .07, 'sharp_left'), ('right', -.07, 'sharp_right')])
def test_bend_categories(direction, curvature, kind):
    e = encoder([(10, 150, direction, curvature)])
    selected = select_instruction(e, [dict(progress_m=20)], scheme='geometry')
    assert selected['kind'] == kind
    assert selected['text'] == TEMPLATES[kind]
    assert selected['event'] is not None
    assert select_instruction(e, [dict(progress_m=20)])['kind'] == direction


@pytest.mark.parametrize('first,second,kind', [('left', 'right', 'left_then_right'),
                                            ('right', 'left', 'right_then_left')])
def test_compound_bends_update_numeric_event_per_frame(first, second, kind):
    e = encoder([(10, 30, first, .04 if first == 'left' else -.04),
                 (40, 100, second, .04 if second == 'left' else -.04)])
    selected = select_instruction(e, [dict(progress_m=5)], scheme='geometry')
    assert selected['kind'] == kind
    assert selected['event'] is None
    assert e.next_event(20)['direction'] == first
    assert e.next_event(50)['direction'] == second
    # Once the planned sequence no longer covers the lookahead, use follow.
    e.route['events'][-1]['end_m'] = 55
    assert select_instruction(e, [dict(progress_m=5)], scheme='geometry')['kind'] == 'follow'


def test_geometry_scheme_ignores_future_actual_ego_positions():
    e = encoder([(10, 30, 'left', .07), (40, 100, 'right', -.04)])
    first = dict(progress_m=5)
    a = select_instruction(e, [first, dict(progress_m=10)], scheme='geometry')
    b = select_instruction(e, [first, dict(progress_m=250)], scheme='geometry')
    assert a == b
    assert len(TEMPLATES) == 12


@pytest.mark.parametrize('direction,k,kind', [('left', .07, 'sharp_left'),
                                           ('right', -.02, 'gentle_right')])
def test_descriptive_names_upcoming_short_bend(direction, k, kind):
    e = encoder([(30, 45, direction, k)])
    assert select_instruction(e, [dict(progress_m=0)], scheme='geometry')['kind'] == 'follow'
    for progress in (0, 29, 35, 44):
        selected = select_instruction(e, [dict(progress_m=progress)], scheme='descriptive')
        assert selected['kind'] == kind
        assert selected['event'] is None
    assert select_instruction(e, [dict(progress_m=46)], scheme='descriptive')['kind'] == 'straight'


@pytest.mark.parametrize('first,second', [('left','right'), ('right','left'),
                                         ('left','left'), ('right','right')])
def test_descriptive_short_sequence_and_future_pose_independence(first, second):
    e = encoder([(10, 20, first, .04), (30, 40, second, -.04)])
    before = select_instruction(e, [dict(progress_m=0)], scheme='descriptive')
    after = select_instruction(e, [dict(progress_m=0),dict(progress_m=250)], scheme='descriptive')
    assert before == after
    assert before['kind'] == first + '_then_' + second
    assert before['event'] is None


def test_descriptive_does_not_invent_a_turn_beyond_horizon_or_in_exclusion():
    e = encoder([(80, 100, 'left', .04)])
    assert select_instruction(e, [dict(progress_m=0)], scheme='descriptive')['kind'] == 'straight'
    exclusion = dict(route_sha256=e.route['route_sha256'], start_m=0., end_m=10.)
    assert select_instruction(e, [dict(progress_m=0)], scheme='descriptive',
                              straight_exclusions=[exclusion])['kind'] == 'follow'


@pytest.mark.parametrize('options', [dict(scheme='unknown'), dict(gentle_curvature=.1),
                                    dict(sharp_curvature=float('nan'))])
def test_invalid_selection_options(options):
    with pytest.raises(ValueError, match='instruction scheme'):
        select_instruction(encoder([]), [dict(progress_m=0)], **options)


@pytest.mark.parametrize('scheme', ['geometry', 'descriptive'])
def test_instruction_policy_is_recorded_and_reused_in_cache(cache, tmp_path, scheme):
    _, bundle, _, _ = cache
    path = tmp_path / 'expanded.json'
    options = dict(scheme=scheme, gentle_curvature=.025, sharp_curvature=.06)
    build_path_features(bundle, path, instruction_options=options)
    digest = hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest()
    prepared = PathFeatureCache(path, digest)
    assert prepared.instruction_options == options
    selection, features = prepared.sample('train_episode', [0, 200000000, 400000000])
    assert selection['kind'] == 'straight'
    assert len(features['path_context']) == 3


def test_reviewed_straight_exclusion_only_changes_matching_route_and_range():
    e = encoder([])
    exclusion = dict(route_sha256=e.route['route_sha256'], start_m=20., end_m=80.)
    options = dict(scheme='geometry', straight_exclusions=[exclusion])
    for progress, expected in [(0., 'straight'), (19.99, 'straight'), (20., 'follow'),
                               (50., 'follow'), (79.99, 'follow'), (80., 'straight'), (150., 'straight')]:
        assert select_instruction(e, [dict(progress_m=progress)], **options)['kind'] == expected
    # No opt-in = original policy; an unrelated route digest never matches.
    assert select_instruction(e, [dict(progress_m=50)], scheme='geometry')['kind'] == 'straight'
    exclusion['route_sha256'] = '0' * 64
    assert select_instruction(e, [dict(progress_m=50)], **options)['kind'] == 'straight'


@pytest.mark.parametrize('changes', [dict(start_m=-1.), dict(end_m=301.), dict(end_m=float('nan')),
                                   dict(route_sha256='wrong'), dict(unreviewed_extra=True)])
def test_reviewed_straight_exclusion_validation(changes):
    e = encoder([])
    exclusion = dict(route_sha256=e.route['route_sha256'], start_m=20., end_m=80.)
    exclusion.update(changes)
    with pytest.raises(ValueError, match='exclusion'):
        select_instruction(e, [dict(progress_m=50)], straight_exclusions=[exclusion])


def test_straight_exclusions_are_preserved_in_path_cache(cache, tmp_path):
    _, bundle, _, _ = cache
    baseline = tmp_path / 'baseline.json'
    result = build_path_features(bundle, baseline)
    route = next(e['route'] for e in result['episodes'] if e['episode_id'] == 'train_episode')
    options = dict(scheme='geometry', straight_exclusions=[dict(route_sha256=route['route_sha256'],
                                                               start_m=0., end_m=route['length_m'])])
    corrected = tmp_path / 'corrected.json'
    build_path_features(bundle, corrected, instruction_options=options)
    digest = hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest()
    prepared = PathFeatureCache(corrected, digest)
    assert prepared.instruction_options == options
    selection, features = prepared.sample('train_episode', [0, 200000000, 400000000])
    assert selection['kind'] == 'follow'
    assert all(features['path_valid'])
