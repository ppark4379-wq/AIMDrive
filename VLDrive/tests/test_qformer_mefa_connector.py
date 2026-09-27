import torch

from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector


def _connector():
    return QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=32,
        num_visual_tokens=12,
        language_dim=64,
        query_count=4,
        target_retention=0.3,
        memory_size=2,
        qformer_hidden_size=64,
        qformer_num_layers=2,
        qformer_num_attention_heads=8,
        qformer_intermediate_size=128,
        reconstruction_layers=1,
        reconstruction_heads=8,
    ))


def test_mefa_is_causal_and_connector_shapes_are_dynamic():
    torch.manual_seed(7)
    connector = _connector().eval()
    features = torch.randn(1, 3, 12, 32)
    baseline = connector(features, compute_reconstruction=False)
    changed = features.clone()
    changed[:, -1].add_(100.0)
    future_changed = connector(changed, compute_reconstruction=False)

    assert baseline["language_tokens"].shape[:2] == (1, 3)
    assert baseline["language_tokens"].shape[-1] == 64
    assert bool((baseline["selected_count_per_frame"] >= 1).all())
    assert bool((baseline["selected_count_per_frame"] <= 12).all())
    assert torch.equal(
        baseline["enhanced_memory"][:, :-1],
        future_changed["enhanced_memory"][:, :-1],
    )


def test_cycle_loss_backpropagates_into_ccdp():
    torch.manual_seed(11)
    connector = _connector().train()
    features = torch.randn(1, 3, 12, 32)
    output = connector(features, selection_mode="gumbel", compute_reconstruction=True)
    output["cycle_loss"].backward()

    gradients = [
        parameter.grad for name, parameter in connector.named_parameters()
        if name.startswith("ccdp.") and parameter.requires_grad
    ]
    assert output["reconstruction_computed"] is True
    assert gradients and any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
