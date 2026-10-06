"""Synthetic fixtures test the review tools; never claim to be real drive data."""
import json
import math
from pathlib import Path

from PIL import Image
import pytest

from instruction.global_path.encoder import PathEncoder
from instruction.global_path.preprocess.load import load_path
from instruction.global_path.preprocess.process import process_path
from validation.navigation.recording import load_recording, synchronization
from validation.navigation.replay import prepare_frames, build_window, write_replay


OPTIONS = dict(geometry=dict(segmentation='hysteresis', smoothing_m=5),
               encoder={}, instruction=dict(scheme='geometry'), max_sensor_gap_ms=150.)


@pytest.fixture
def run(tmp_path):
    root = tmp_path / 'synthetic_collection'
    (root / 'measurements_full').mkdir(parents=True)
    (root / 'rgb_front').mkdir()
    points = [[0, 0], [100, 0], [100, 200]]
    (root / 'route.json').write_text(json.dumps(dict(nav_waypoints=points)))
    for i in range(26):
        stamp = 1_000_000_000 + i * 100_000_000
        row = dict(frame_id=i, timestamp_ns=stamp, gps_x=40. + i * 2, gps_y=1., theta=0., speed=20.,
                   sensor_timestamps_ns=dict(cam_front=stamp, ego=stamp))
        (root / 'measurements_full' / ('%04d.json' % i)).write_text(json.dumps(row))
        Image.new('RGB', (320, 180), (i * 5, 70, 120)).save(root / 'rgb_front' / ('%04d.jpg' % i))
    return root


def prepared(run):
    recording = load_recording(run)
    encoder = PathEncoder(process_path(load_path(recording['route']), **OPTIONS['geometry']))
    frames = prepare_frames(recording, encoder, 'matches_recorded_planned_path')
    return recording, encoder, frames


def test_window_has_one_common_text_and_eleven_distinct_numeric_inputs(run):
    _, encoder, frames = prepared(run)
    window = build_window(frames, encoder, 0, instruction_options=OPTIONS['instruction'])
    assert window['valid_window']
    assert window['frame_indices'] == list(range(0, 21, 2))
    assert window['common_navigation'] == 'Follow the planned path.'
    first, last = window['frames'][0], window['frames'][-1]
    assert first['features']['ego_relation'] == pytest.approx([.2, 0., 1.])
    assert first['features']['path_context'][0] - last['features']['path_context'][0] == pytest.approx(40. / 60.)
    assert 'starts in' in first['numeric_preview']
    assert first['numeric_preview'] != last['numeric_preview']


def test_future_pose_does_not_choose_earlier_common_text(run):
    _, encoder, frames = prepared(run)
    before = build_window(frames, encoder, 0, instruction_options=OPTIONS['instruction'])
    changed = [dict(row) for row in frames]
    changed[20]['projection'] = dict(changed[20]['projection'], progress_m=84.)
    after = build_window(changed, encoder, 0, instruction_options=OPTIONS['instruction'])
    assert before['common_navigation'] == after['common_navigation']
    assert before['frames'][0]['features'] == after['frames'][0]['features']
    assert before['frames'][-1]['features'] != after['frames'][-1]['features']


def test_descriptive_window_flags_when_named_bend_has_been_passed(run):
    _, encoder, frames = prepared(run)
    # Synthetic event deliberately ends during the recorded window.
    encoder.route['events'] = [dict(id='short', start_m=45., end_m=55.,
                                     direction='left', curv_event=.07)]
    window = build_window(frames, encoder, 0, instruction_options=dict(scheme='descriptive'))
    assert window['common_navigation'] == 'Follow the sharp left bend.'
    assert window['frames'][0]['common_semantics_ok'] is True
    assert window['frames'][-1]['common_semantics_ok'] is False


@pytest.mark.parametrize('camera,pose,scene,status', [
    (100_000_000, 90_000_000, 100_000_000, 'declared_timestamps_match'),
    (100_000_000, 101_000_000, 100_000_000, 'future_pose_or_camera'),
    (100_000_000, 1_000_000, 100_000_000, 'stale_pose_or_camera'),
    (100_000_000, 100_000_000, 200_000_000, 'stale_pose_or_camera'),
    (None, 100_000_000, 100_000_000, 'missing_source_timestamps'),
])
def test_sync_uses_source_timestamps_not_file_times(camera, pose, scene, status):
    result = synchronization(dict(camera_source_ns=camera, pose_source_ns=pose, sensor_source_ns=scene))
    assert result['status'] == status
    assert result['valid'] == (status == 'declared_timestamps_match')


def test_native_audit_uses_configured_pose_age_limit():
    frame = dict(sensor_source_ns=100_000_000, pose_source_ns=70_000_000, audited_scene_association=True)
    assert synchronization(frame)['valid']
    assert not synchronization(frame, max_skew_ms=20.)['valid']


def test_no_timestamp_is_fabricated_for_legacy_recording(run, tmp_path):
    for path in (run / 'measurements_full').glob('*.json'):
        row = json.loads(path.read_text())
        row.pop('timestamp_ns')
        row.pop('sensor_timestamps_ns')
        path.write_text(json.dumps(row))
    recording = load_recording(run)
    assert all(row['sensor_source_ns'] is None for row in recording['frames'])
    target = tmp_path / 'review'
    with pytest.raises(ValueError, match='lacks synchronized'):
        write_replay(recording, target, OPTIONS)
    assert not target.exists()
    result = write_replay(recording, target, OPTIONS, allow_unverified_preview=True)
    assert result['valid_windows'] == 0
    assert all(not row['features']['path_valid'] and row['numeric_preview'] is None
               for row in result['windows'][0]['frames'])
    assert result['windows'][0]['common_navigation'] is None
    assert (target / 'viewer.html').is_file()
    assert (target / 'contact_sheet.png').is_file()
    assert len(list((target / 'frames').glob('*.png'))) == 11


def test_yaw_missing_is_not_replaced_with_planned_heading(run):
    recording, encoder, _ = prepared(run)
    recording['frames'][0].pop('yaw_rad')
    frames = prepare_frames(recording, encoder, 'matches_recorded_planned_path')
    assert not frames[0]['path_valid'] and frames[0]['pose']['yaw_rad'] is None
    assert frames[0]['candidate_projection'] is None


def test_nonfinite_pose_has_no_fake_numbers_and_serializes_as_null(run):
    recording, encoder, _ = prepared(run)
    recording['frames'][0]['yaw_rad'] = math.nan
    row = prepare_frames(recording, encoder, 'matches_recorded_planned_path')[0]
    assert row['pose']['yaw_rad'] is None and not row['path_valid']
    json.dumps(row, allow_nan=False)


def test_skipped_original_frame_is_checked_even_if_not_selected(run):
    _, encoder, frames = prepared(run)
    frames[1] = dict(frames[1], path_valid=False, projection=None)
    window = build_window(frames, encoder, 0)
    assert all(row['path_valid'] for row in window['frames'])
    assert not window['valid_window'] and window['common_navigation'] is None


def test_gap_and_insufficient_span_are_not_silently_repaired(run):
    _, encoder, frames = prepared(run)
    changed = [dict(row) for row in frames]
    changed[1]['sensor_source_ns'] += 500_000_000
    assert not build_window(changed, encoder, 0)['continuous_source_span']
    with pytest.raises(ValueError, match='Not enough'):
        build_window(frames, encoder, 10)


def test_wrong_or_absent_recorded_route_never_enables_numeric_features(run, tmp_path):
    recording, _, _ = prepared(run)
    external = tmp_path / 'other_route.txt'
    external.write_text('0 0\n100 0\n100 -200\n')
    result = write_replay(recording, tmp_path / 'wrong_path_review', OPTIONS,
                          path=external, allow_unverified_preview=True)
    assert result['valid_source_frames'] == 0
    assert 'different_recorded_planned_path' in result['invalid_source_reasons']
    recording['route'] = None
    with pytest.raises(ValueError, match='no planned path'):
        write_replay(recording, tmp_path / 'absent_path_review', OPTIONS)


def test_no_existing_review_or_source_is_overwritten(run, tmp_path):
    recording, _, _ = prepared(run)
    target = tmp_path / 'existing_review'
    target.mkdir()
    with pytest.raises(FileExistsError):
        write_replay(recording, target, OPTIONS)
    with pytest.raises(ValueError, match='outside the original'):
        write_replay(recording, run / 'review', OPTIONS)


def test_valid_replay_preserves_pose_features_provenance_and_common_text(run, tmp_path):
    recording, _, _ = prepared(run)
    result = write_replay(recording, tmp_path / 'valid_review', OPTIONS, windows=2)
    assert result['valid_windows'] == 2
    assert len(result['windows'][0]['frames']) == 11
    assert result['windows'][1]['frame_indices'] == list(range(5, 26, 2))
    assert result['windows'][0]['frames'][0]['image_sha256']
    assert result['source_provenance']
    saved = json.loads((tmp_path / 'valid_review/review.json').read_text())
    assert saved['windows'][0]['frames'][0]['features']['ego_relation'][0] == pytest.approx(.2)


def test_portable_assets_cannot_escape_image_root(tmp_path):
    image = tmp_path / 'outside.jpg'
    Image.new('RGB', (20, 20)).save(image)
    root = tmp_path / 'portable'
    root.mkdir()
    path = root / 'episode.json'
    path.write_text(json.dumps(dict(schema='navigation-episode-v1', episode_id='portable',
                                   frames=[dict(index=0, camera_files=dict(front='../outside.jpg'))])))
    with pytest.raises(ValueError, match='unsafe'):
        load_recording(path)


def test_portable_cannot_self_assert_native_audit(tmp_path):
    Image.new('RGB', (20, 20)).save(tmp_path / 'front.jpg')
    path = tmp_path / 'navigation_episode.json'
    path.write_text(json.dumps(dict(schema='navigation-episode-v1', episode_id='portable',
                                   frames=[dict(index=0, camera_files=dict(front='front.jpg'),
                                                audited_scene_association=True)])))
    frame = load_recording(path)['frames'][0]
    assert not synchronization(frame)['valid']


@pytest.mark.parametrize('pose_age_ms, valid', [(10, True), (60, False)])
def test_native_reuses_audited_scene_and_real_causal_heading(tmp_path, monkeypatch, pose_age_ms, valid):
    import navigation.adapters
    root = tmp_path / 'native_run/episode'
    (root / 'quality').mkdir(parents=True)
    (root / 'episode.json').write_text('{}')
    image = root.parent / 'front.jpg'
    Image.new('RGB', (20, 20)).save(image)
    stamp = 1_000_000_000
    signals = [dict(record_id=7, role='ego', source_timestamp_ns=stamp - pose_age_ms * 1_000_000)]
    (root / 'quality/active_signals.json').write_text(json.dumps(signals))
    document = dict(episode_id='native_episode', route=dict(points=[[0, 0], [100, 0]], closed=False),
                    sources={'audited_route': True},
                    frames=[dict(index=0, sensor_source_ns=stamp, x=10., y=0., yaw_rad=.2,
                                 camera_files=dict(camera_1='front.jpg'), ego_signal_record_id=7, valid=True)])
    calls = []

    def adapter(episode, config):
        calls.append((episode, config))
        return document

    monkeypatch.setattr(navigation.adapters, 'sensor_record', adapter)
    frame = load_recording(root)['frames'][0]
    assert calls[0][0] == root.resolve()
    assert frame['yaw_rad'] == .2
    assert frame['camera_source_ns'] is None  # No fictional hardware timestamp.
    assert frame['pose_source_ns'] == signals[0]['source_timestamp_ns']
    assert synchronization(frame)['valid'] == valid


def test_review_rejects_unknown_preprocessing_options(run, tmp_path):
    recording, _, _ = prepared(run)
    with pytest.raises(ValueError, match='Unknown'):
        write_replay(recording, tmp_path / 'bad_options', dict(OPTIONS, nonexistent_policy=True))


def test_nonchronological_source_timestamps_are_rejected(run):
    path = run / 'measurements_full/0001.json'
    row = json.loads(path.read_text())
    row['timestamp_ns'] = 1_000_000_000
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='strictly increase'):
        load_recording(run)


def test_straight_common_sentence_that_becomes_stale_is_flagged(run):
    _, encoder, frames = prepared(run)
    modified = [dict(row) for row in frames]
    modified[0]['projection'] = dict(modified[0]['projection'], progress_m=0.)
    window = build_window(modified, encoder, 0)
    assert window['common_kind'] == 'straight'
    assert window['semantic_warning_count'] > 0


def test_mp4_is_review_playback_not_claimed_recording_fps(run, tmp_path):
    cv2 = pytest.importorskip('cv2')
    recording, _, _ = prepared(run)
    target = tmp_path / 'video_review'
    result = write_replay(recording, target, OPTIONS, video_fps=5.)
    capture = cv2.VideoCapture(str(target / 'review.mp4'))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 11
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(5.)
    capture.release()
    assert 'NOT recovered recording timing' in result['video_note']
