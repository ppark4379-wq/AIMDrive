"""Raw past localization must not depend on traffic-label interpolation audits."""
import json

import pytest

from navigation.adapters import CausalEgoSamples, sensor_record


def sample(record_id, ms, x=3., yaw=.1):
    return dict(role='ego', record_id=record_id, source_timestamp_ns=int(ms*1e6),
                signal=dict(position=[x, 2., 0.], yaw=yaw, speed=4.))


def test_38ms_bracket_uses_20ms_past_pose_and_ignores_future():
    before, after = sample(1, 80), sample(2, 118, x=1000., yaw=2.)
    selected = CausalEgoSamples([after, before]).at(100_000_000)
    assert selected['valid']
    assert selected['pose_age_ms'] == 20.
    assert (selected['x'], selected['yaw_rad']) == (3., .1)
    after['signal']['position'][0] = -1000
    after['signal']['yaw'] = -2.
    assert CausalEgoSamples([before, after]).at(100_000_000) == selected


@pytest.mark.parametrize('sample_ms,valid,reason', [
    (100, True, None), (50, True, None), (49, False, 'stale_past_ego_sample'),
    (101, False, 'missing_past_ego_sample')])
def test_age_limit_and_no_future_fallback(sample_ms, valid, reason):
    result = CausalEgoSamples([sample(1, sample_ms)]).at(100_000_000)
    assert result['valid'] is valid
    assert result['reason'] == reason


def test_bad_latest_sample_is_not_silently_replaced_by_older_pose():
    result = CausalEgoSamples([sample(1, 80), sample(2, 90, yaw=float('nan'))]).at(100_000_000)
    assert not result['valid']
    assert result['reason'] == 'invalid_past_ego_pose'
    assert 'x' not in result


def test_missing_traffic_route_audit_does_not_hide_raw_pose(tmp_path, monkeypatch):
    from data.src.processing import build_instructions
    import navigation.adapters as adapters
    episode = tmp_path / 'episode_000'
    for name in ['quality', 'route_signal_audit_v1', 'gt_timeline_v1', 'native_associations']:
        (episode / name).mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(episode / 'route_signal_audit_v1/report.json', dict(route_key='full_loop'))
    write(episode / 'route_signal_audit_v1/frames.json', [dict(index=0,
          sensor_source_ns=100_000_000, reason='ego_not_source_bracketed')])
    write(episode / 'gt_timeline_v1/frames.json', [dict(index=0, lidar_source_ns=100_000_000,
          ego_xy=[999., 999.])])
    write(episode / 'native_associations/associations.json', [dict(index=0,
          scene_time_hypothesis_ns=100_000_000, camera_files={}, lidar_file='lidar.bin',
          native_filename_time_ns=100_000_000)])
    write(episode / 'quality/active_signals.json', [sample(1, 80), sample(2, 118)])
    write(tmp_path / 'definition.json', dict(full_loop=dict(route_links=['a'])))
    write(tmp_path / 'links.json', [dict(idx='a')])
    write(tmp_path / 'nodes.json', [])
    sources = {k:dict(path=str(tmp_path/v)) for k,v in
               [('route_definition','definition.json'),('link_set','links.json'),('node_set','nodes.json')]}
    sources['global_path'] = dict(recorded_sha256='a'*64)
    monkeypatch.setattr(build_instructions, 'sensor_record_route',
                        lambda *args: (dict(nav_waypoints=[[0,0],[100,0]]), ['a'], [0], None, sources))
    monkeypatch.setattr(adapters, 'route_from_mgeo', lambda *args: dict(points=[[0,0],[100,0]]))
    document = sensor_record(episode, {})
    frame = document['frames'][0]
    assert frame['valid']
    assert (frame['x'], frame['y'], frame['yaw_rad']) == (3., 2., .1)
    assert frame['ego_signal_record_id'] == 1
    assert frame['route_index'] is None  # downstream projects the actual causal pose
    assert frame['route_audit_reason'] == 'ego_not_source_bracketed'
    assert document['pose_policy']['future_samples_used'] is False

    # Missing audit hints cannot bypass downstream off-route rejection.
    from instruction.global_path.encoder import PathEncoder
    from instruction.global_path.preprocess.load import load_path
    from instruction.global_path.preprocess.process import process_path
    encoder = PathEncoder(process_path(load_path(document['route'])))
    assert encoder.project(dict(x=frame['x'], y=frame['y'], yaw_rad=frame['yaw_rad'])) is not None
    assert encoder.project(dict(x=3., y=1000., yaw_rad=.1)) is None
    write(episode / 'route_signal_audit_v1/frames.json', [dict(index=0,
          sensor_source_ns=100_000_000, route_index=0, route_link_id='wrong_map_link', route_link_index=0)])
    with pytest.raises(ValueError, match='Audited route link mismatch'):
        sensor_record(episode, {})
