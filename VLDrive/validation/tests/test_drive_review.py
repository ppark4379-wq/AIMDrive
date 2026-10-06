"""Whole-drive viewer must not convert invalid windows into training inputs."""
import copy
import json

import pytest

from validation.navigation.drive_review import build_drive_review, write_drive_review
from validation.navigation.recording import load_recording
from validation.navigation.replay import build_window
from validation.tests.test_navigation_replay import OPTIONS, run, prepared


def test_trailing_window_matches_training_review_and_never_uses_future(run):
    recording, encoder, frames = prepared(run)
    report = build_drive_review(recording, OPTIONS)
    assert len(report['frames']) == 26
    assert all(row['model_features'] is None for row in report['frames'][:20])
    assert all(row['frame_navigation_source'] == 'current_pose' for row in report['frames'][:20])
    assert all(row['frame_features'] is not None for row in report['frames'][:20])
    row = report['frames'][22]
    expected = build_window(frames, encoder, 2, instruction_options=OPTIONS['instruction'])
    assert row['window_frame_indices'] == list(range(2, 23, 2))
    assert row['common_navigation'] == expected['common_navigation']
    assert row['model_features'] == expected['frames'][-1]['features']
    changed = copy.deepcopy(recording)
    changed['frames'][24]['x'] += 500
    after = build_drive_review(changed, OPTIONS)
    assert after['frames'][:24] == report['frames'][:24]


def test_skipped_invalid_source_frame_masks_whole_window(run):
    recording = load_recording(run)
    recording['frames'][3]['valid'] = False
    report = build_drive_review(recording, OPTIONS)
    row = report['frames'][22]
    assert row['path_valid']
    assert 3 not in row['window_frame_indices']
    assert not row['valid_window']
    assert row['model_features'] is None
    assert row['common_navigation'] is None
    assert row['frame_navigation_source'] == 'current_pose'
    assert row['frame_navigation']
    assert row['frame_features'] is not None
    missing = report['frames'][3]
    assert missing['frame_navigation_source'] == 'generic_fallback_no_valid_pose'
    assert missing['frame_navigation'] == 'Follow the planned path.'
    assert missing['frame_features'] is None
    assert missing['frame_numeric_preview'] is None
    assert not missing['path_valid']
    assert 'source_marked_invalid' in row['window_invalid_reasons']
    assert report['frames'][24]['valid_window']


def test_export_preserves_recording_and_is_self_contained(run, tmp_path):
    report = build_drive_review(load_recording(run), OPTIONS)
    with pytest.raises(ValueError, match='outside'):
        write_drive_review(report, run / 'review')
    output = write_drive_review(report, tmp_path / 'review')
    restored = json.loads((output / 'review.json').read_text(encoding='utf-8'))
    assert len(list((output / 'images').glob('*.jpg'))) == 26
    assert (output / restored['frames'][22]['preview_image']).is_file()
    assert '__REVIEW_DATA__' not in (output / 'viewer.html').read_text(encoding='utf-8')
    assert len((output / 'frames.csv').read_text(encoding='utf-8-sig').splitlines()) == 27
    with pytest.raises(FileExistsError):
        write_drive_review(report, output)
