"""Bend thresholds, gap merging, loop seams and review text are deterministic."""
import math

import numpy as np
import pytest

from instruction.global_path.preprocess.segments import detect_bends
from instruction.global_path.preprocess.process import process_path
from instruction.global_path.preprocess.load import load_path
from validation.navigation.path_report import build_report, route_segments
from data.src.processing.build_path_features import build_path_features
from instruction.global_path.encoder import PathFeatureCache
from .test_navigation_cache import cache


def detect(k, **kwargs):
    return detect_bends(np.arange(len(k) + 1, dtype=float), k, closed=False, **kwargs)


def test_hysteresis_extends_seed_and_rejoins_brief_dips():
    k = [0.] * 5 + [.01] * 3 + [.04] * 8 + [.005] * 3 + [.04] * 8 + [.01] * 3 + [0.] * 5
    events = detect(k)
    assert len(events) == 1
    assert events[0]['direction'] == 'left'
    assert (events[0]['start_m'], events[0]['end_m']) == (5., 30.)


def test_opposite_seed_is_never_merged_away():
    events = detect([.05] * 10 + [-.04] * 2 + [.05] * 10)
    assert len(events) == 2
    assert events[0]['end_m'] == 10
    assert events[1]['start_m'] == 12


def test_low_angle_and_unseeded_wiggles_are_filtered():
    assert not detect([.02] * 8)
    assert not detect([.01] * 100)
    with pytest.raises(ValueError, match='thresholds'):
        detect([.05] * 10, exit_curvature=.1)


def test_closed_seam_event_and_complete_segment_coverage():
    k = [.06] * 5 + [0.] * 15 + [.06] * 5
    events = detect_bends(np.arange(26.), k, closed=True)
    assert len(events) == 1
    assert events[0]['start_m'] == 20
    assert events[0]['end_m'] == 30
    segments = route_segments(dict(length_m=25., events=events))
    assert [row['kind'] for row in segments] == ['left', 'no_bend_event', 'left']
    assert segments[0]['start_m'] == 0 and segments[-1]['end_m'] == 25
    assert all(a['end_m'] == b['start_m'] for a, b in zip(segments, segments[1:]))


def test_straight_and_mirrored_quarter_circles():
    assert not process_path(dict(points=[[0, 0], [200, 0]]), segmentation='hysteresis')['events']
    angle = np.linspace(0, math.pi / 2, 101)
    points = np.column_stack((20 * np.sin(angle), 20 * (1 - np.cos(angle))))
    for sign, direction in [(1, 'left'), (-1, 'right')]:
        route = process_path(dict(points=(points * [1, sign]).tolist()), segmentation='hysteresis')
        assert len(route['events']) == 1
        assert route['events'][0]['direction'] == direction
        assert sign * route['events'][0]['angle_deg'] > 70


def test_text_loader_and_review_are_not_generated_numeric_sentences(tmp_path):
    path = tmp_path / 'route.txt'
    path.write_text('0 0 0\n100 0 0\n100 100 0\n')
    loaded = load_path(path)
    assert not loaded['closed']
    options = dict(geometry=dict(segmentation='hysteresis', smoothing_m=5), encoder={})
    report = build_report(path, options)
    assert report['source_points'] == 3
    assert report['route']['events'][0]['direction'] == 'left'
    assert 'NOT MLP/LLM output' in report['instruction_note']
    assert sum(s['end_m'] - s['start_m'] for s in report['segments']) == pytest.approx(report['length_m'])
    assert 'starts in 30.0 m' in report['examples'][0]['numeric_preview']


def test_new_cache_uses_explicit_segmentation_options(cache, tmp_path):
    import hashlib
    _, bundle, _, _ = cache
    output = tmp_path / 'mlp_features.json'
    build_path_features(bundle, output, geometry_options=dict(segmentation='hysteresis', smoothing_m=5))
    digest = hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest()
    prepared = PathFeatureCache(output, digest)
    assert prepared.encoders['train_episode'].route['preprocessing']['segmentation'] == 'hysteresis'
    selection, features = prepared.sample('train_episode', [0, 200000000, 400000000])
    assert selection['kind'] == 'straight'
    assert len(features['ego_relation']) == 3
