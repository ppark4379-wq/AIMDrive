from training.connector_only.validate_mefa import episode_summary


def test_episode_weighting_and_cluster_bootstrap():
    # Ten correlated frames in one drive must not count as ten drives.
    rows = [dict(episode='a', category='dynamic', normal_lrec=1., zero_lrec=1.2, delta_lrec=.2)] * 10
    rows += [dict(episode='b', category='static', normal_lrec=1., zero_lrec=.8, delta_lrec=-.2)]
    result = episode_summary(rows, bootstrap_samples=100)['summary']
    assert abs(result['delta_lrec']) < 1e-12
    assert result['episodes'] == 2
    assert result['positive_episodes'] == 1
    assert result['frames'] == 11
    assert result['delta_ci95'] is None


def test_empty_reconstruction_is_excluded_not_scored_zero():
    rows = [dict(episode='a', category='dynamic', normal_lrec=1., zero_lrec=1.1, delta_lrec=.1),
            dict(episode='b', category='static', normal_lrec=None, zero_lrec=None, delta_lrec=None)]
    summary = episode_summary(rows, bootstrap_samples=100)['summary']
    assert summary['total_episodes'] == 2 and summary['episodes'] == 1
    assert summary['valid_frames'] == 1
    assert abs(summary['delta_lrec'] - .1) < 1e-12
