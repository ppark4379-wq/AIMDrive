import torch

from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from training.connector_only.native_inputs import causal_speed
from training.connector_only.run import paired_evaluation, select_windows, noise_scale_at_update


def test_noise_schedule_and_zero_noise_match_argmax():
    schedule = dict(start=1., end=0., decay_updates=256)
    assert noise_scale_at_update(schedule, 0) == 1.
    assert noise_scale_at_update(schedule, 128) == .5
    assert noise_scale_at_update(schedule, 256) == 0.
    assert noise_scale_at_update(schedule, 300) == 0.
    assert noise_scale_at_update(None, 100) == 1.
    from models.connector.ccdp import CCDPTokenSparsifier
    selector = CCDPTokenSparsifier(dim=8)
    selector.gumbel_noise_scale = 0.
    x = torch.randn(2, 3, 6, 8)
    a = selector(x, selection_mode='gumbel')
    b = selector(x, selection_mode='argmax')
    assert torch.equal(a['mask'], b['mask'])
    a['pruning_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in selector.parameters())


def test_ego_speed_never_uses_future_or_stale_values():
    rows = [(100_000_000, 3.), (200_000_000, 9.)]
    assert causal_speed(rows, 99_000_000) is None
    assert causal_speed(rows, 150_000_000) == 3.
    assert causal_speed(rows, 150_000_001) is None
    assert causal_speed(rows, 200_000_000) == 9.
    assert causal_speed([(0, float('nan'))], 0) is None


def test_windows_exclude_sensor_gaps_and_missing_speed():
    times = [0, 10, 20, 100, 110, 120, 130, 140]
    windows = select_windows(times, [1.] * 8, 3, 1, 2, 15)
    assert windows == [[0, 1, 2], [5, 6, 7]]
    windows = select_windows(times, [1., None, 1., 1., 1., 1., 1., 1.], 3, 1, 2, 15)
    assert all(1 not in ids for ids in windows)


def test_paired_evaluation_uses_last_frame_and_preserves_model():
    torch.manual_seed(607)
    model = QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=8, num_visual_tokens=6, language_dim=8, query_count=2,
        memory_size=2, qformer_hidden_size=16, qformer_num_layers=1,
        qformer_num_attention_heads=2, qformer_intermediate_size=32,
        reconstruction_layers=1, reconstruction_heads=2,
        reconstruction_reduction='mean', inference_selection_mode='topk'))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(1, 3, 6, 8)
    rows = [dict(tensor=x, episode='held_out', category='test', times_ns=[0, 1, 2])]
    result = paired_evaluation(model, rows, 'cpu')
    with torch.no_grad():
        expected = model(x, compute_reconstruction=True)['reconstruction_loss_per_frame'][-1].item()
    assert result['summary']['normal_lrec'] == expected
    assert result['summary']['valid_windows'] == 1
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_all_retained_frames_are_not_reported_as_perfect_reconstruction():
    model = QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=8, num_visual_tokens=6, language_dim=8, query_count=2, target_retention=1.,
        memory_size=2, qformer_hidden_size=16, qformer_num_layers=1,
        qformer_num_attention_heads=2, qformer_intermediate_size=32,
        reconstruction_layers=1, reconstruction_heads=2, inference_selection_mode='topk'))
    rows = [dict(tensor=torch.randn(1, 3, 6, 8), episode='held_out', category='test', times_ns=[0, 1, 2])]
    result = paired_evaluation(model, rows, 'cpu')
    assert result['summary']['valid_windows'] == 0
    assert result['summary']['empty_target_windows'] == 1
    assert result['summary']['normal_lrec'] is None
