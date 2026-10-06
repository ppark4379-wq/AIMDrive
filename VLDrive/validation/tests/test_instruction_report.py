"""Selector-region plots must agree with the real selector, not bend labels."""
import numpy as np
import pytest
import json
from pathlib import Path

from instruction.global_path.encoder import PathEncoder
from instruction.global_path.preprocess.process import process_path
from instruction.navigation.selector import select_instruction
from validation.navigation.path_report import projection_at, frame_preview
from validation.navigation.instruction_report import instruction_regions


def test_current_route_refines_only_region_five_and_keeps_other_map_labels():
    from validation.navigation.instruction_report import build_instruction_report, DEFAULT_OPTIONS
    path = Path(__file__).resolve().parents[3] / 'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt'
    if not path.is_file():
        pytest.skip('Competition map is not present beside this checkout')
    options = json.loads(DEFAULT_OPTIONS.read_text(encoding='utf-8'))
    # This regression fixes the historical geometry-policy region numbering.
    options['instruction']['scheme'] = 'geometry'
    old_options = dict(options, geometry={key: value for key, value in options['geometry'].items()
                                         if key != 'bend_refinements'})
    baseline = build_instruction_report(path, old_options)
    corrected = build_instruction_report(path, options)
    assert len(baseline['instruction_regions']) == 29
    assert len(corrected['instruction_regions']) == 31
    for key in ('points', 'arc_m', 'heading_rad', 'curvature', 'length_m', 'route_sha256'):
        assert baseline['route'][key] == corrected['route'][key]
    added = [event for event in corrected['route']['events'] if event.get('reviewed_section')]
    assert len(added) == 1 and added[0]['direction'] == 'left'
    assert added[0]['start_m'] == pytest.approx(237., abs=.1)
    assert added[0]['end_m'] == pytest.approx(320., abs=.1)
    assert [e for e in corrected['route']['events'] if not e.get('reviewed_section')] == baseline['route']['events']
    pieces = [r for r in corrected['instruction_regions'] if r['display_id'].startswith('5')]
    assert [r['display_id'] for r in pieces] == ['5a', '5b', '5c']
    assert [r['kind'] for r in pieces] == ['straight', 'gentle_left', 'straight']
    assert pieces[0]['start_m'] == baseline['instruction_regions'][4]['start_m']
    assert pieces[-1]['end_m'] == baseline['instruction_regions'][4]['end_m']
    assert pieces[1]['start_m'] == pytest.approx(added[0]['start_m'] - 60.)
    for old in baseline['instruction_regions']:
        if old['region_id'] == 5:
            continue
        new = next(r for r in corrected['instruction_regions'] if r['display_id'] == str(old['region_id']))
        assert {k: v for k, v in old.items() if k != 'region_id'} == {
            k: v for k, v in new.items() if k not in ('region_id', 'display_id')}
    encoder = PathEncoder(corrected['route'], **options['encoder'])
    for position, sign in [(200., 1), (270., -1)]:
        preview = frame_preview(encoder, position, options['instruction'])
        assert preview['common_kind'] == 'gentle_left'
        assert preview['features']['path_context'][0] * sign > 0
        assert preview['features']['path_context'][1] > 0
        assert preview['features']['path_context'][4] == 1.
    assert frame_preview(encoder, 400., options['instruction'])['features']['path_context'][4] == 0.


def test_refined_bend_and_per_frame_distances_survive_feature_cache(tmp_path):
    """Synthetic on-path poses test cache flow; these are not driving records."""
    import hashlib
    from instruction.global_path.preprocess.load import load_path
    from instruction.global_path.encoder import PathFeatureCache
    from data.src.processing.build_path_features import build_path_features
    from validation.navigation.instruction_report import DEFAULT_OPTIONS
    source = Path(__file__).resolve().parents[3] / 'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt'
    if not source.is_file():
        pytest.skip('Competition map is not present beside this checkout')
    options = json.loads(DEFAULT_OPTIONS.read_text(encoding='utf-8'))
    planned = load_path(source)
    route = process_path(planned, **options['geometry'])
    frames = []
    for i, position in enumerate((200., 210., 220.)):
        index = int(np.searchsorted(route['arc_m'], position, side='right') - 1)
        xy = np.asarray(route['points'])
        frames.append(dict(index=i, sensor_source_ns=i * 100_000_000,
                           x=float(np.interp(position, route['arc_m'], xy[:, 0])),
                           y=float(np.interp(position, route['arc_m'], xy[:, 1])),
                           yaw_rad=route['heading_rad'][index]))
    bundle = tmp_path / 'fixture_bundle'
    bundle.mkdir()
    content = json.dumps(dict(schema='navigation-episode-v1', episode_id='synthetic',
                              route=planned, frames=frames)).encode()
    (bundle / 'episode.json').write_bytes(content)
    manifest = json.dumps(dict(schema='lmdrive-navigation-labels-v1', episodes=[
        dict(episode='episode.json', episode_id='synthetic', episode_sha256=hashlib.sha256(content).hexdigest())])).encode()
    (bundle / 'manifest.json').write_bytes(manifest)
    output = tmp_path / 'refined_features.json'
    build_path_features(bundle, output, geometry_options=options['geometry'],
                         encoder_options=options['encoder'], instruction_options=options['instruction'])
    cache = PathFeatureCache(output, hashlib.sha256(manifest).hexdigest())
    selection, features = cache.sample('synthetic', [0, 100_000_000, 200_000_000])
    assert selection['kind'] == 'gentle_left'
    assert all(features['path_valid'])
    distances = [row[0] * 60. for row in features['path_context']]
    assert distances == pytest.approx([236.9578848652851 - s for s in (200., 210., 220.)], abs=.01)
    assert all(row[4] == 1. for row in features['path_context'])
    assert cache.encoders['synthetic'].route['preprocessing']['bend_refinements'] == options['geometry']['bend_refinements']


def test_no_bends_select_straight_for_entire_route():
    encoder = PathEncoder(process_path(dict(points=[[0, 0], [200, 0]])))
    regions = instruction_regions(encoder, dict(scheme='geometry'))
    assert len(regions) == 1
    assert regions[0]['kind'] == 'straight'
    assert regions[0]['start_m'] == 0 and regions[0]['end_m'] == 200


@pytest.mark.parametrize('scheme', ['legacy', 'geometry'])
def test_regions_match_selector_including_adjacent_closed_laps(scheme):
    route = process_path(dict(points=[[0, 0], [150, 0], [150, 150], [0, 150], [0, 0]], closed=True))
    # Deliberate synthetic event catalog tests selector logic independently of
    # curvature extraction. These are never exported as real route events.
    route['events'] = [dict(id=str(i), start_m=a, end_m=b, direction=d, curv_event=k)
                       for i, (a, b, d, k) in enumerate([
                           (80., 180., 'left', .02), (210., 330., 'right', -.08),
                           (360., 500., 'left', .035), (550., 650., 'right', -.018)])]
    encoder = PathEncoder(route)
    options = dict(scheme=scheme)
    regions = instruction_regions(encoder, options)
    assert regions[0]['start_m'] == 0 and regions[-1]['end_m'] == route['length_m']
    assert sum(r['end_m'] - r['start_m'] for r in regions) == pytest.approx(route['length_m'])
    assert all(a['end_m'] == b['start_m'] and a['kind'] != b['kind'] for a, b in zip(regions, regions[1:]))
    for region in regions:
        for fraction in (.0001, .2, .5, .8, .9999):
            position = region['start_m'] + fraction * (region['end_m'] - region['start_m'])
            actual = select_instruction(encoder, [projection_at(route, position)], **options)
            assert actual['kind'] == region['kind'], (position, region, actual)
    for position in np.linspace(.123, route['length_m'] - .123, 1501):
        region = next(r for r in regions if r['start_m'] <= position < r['end_m'])
        assert select_instruction(encoder, [projection_at(route, position)], **options)['kind'] == region['kind']


def test_short_left_bend_is_not_mislabeled_as_left_sentence():
    route = process_path(dict(points=[[0, 0], [200, 0]]))
    route['events'] = [dict(id='short_left', start_m=80., end_m=100., direction='left', curv_event=.05)]
    encoder = PathEncoder(route)
    regions = instruction_regions(encoder, dict(scheme='geometry'))
    containing_turn = next(r for r in regions if r['start_m'] < 90 < r['end_m'])
    assert containing_turn['kind'] == 'follow'
    assert containing_turn['text'] == 'Follow the planned path.'
    preview = frame_preview(encoder, 90., dict(scheme='geometry'))
    assert preview['common_navigation'] == 'Follow the planned path.'
    assert 'Following the left bend' in preview['numeric_preview']
