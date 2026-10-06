import pytest
import torch

from instruction.global_path.tokenizer.path_tokenizer import PathTokenizer
from instruction.navigation.navigation_cross_attention import NavigationCrossAttention
from models.llm.multimodal_sequence import build_frame_multimodal_sequence
from models.llm.ddia_attention import build_ddia_attention_mask


def features(batch=2, frames=3):
    return dict(ego_relation=torch.randn(batch, frames, 3), path_context=torch.randn(batch, frames, 5),
                path_valid=torch.ones(batch, frames, dtype=torch.bool))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_batched_fusion_equals_independent_frames_and_gradients(dtype):
    torch.manual_seed(6)
    tokenizer, fusion = PathTokenizer(16), NavigationCrossAttention(32, 16, 4)
    tokenizer, fusion = tokenizer.to(dtype), fusion.to(dtype)
    data = features()
    text, mask = torch.randn(2, 5, 32).to(dtype), torch.ones(2, 5, dtype=torch.bool)
    output = fusion(text, tokenizer(data), mask, data['path_valid'])
    reference = torch.cat([fusion(text, tokenizer({k: v[:, i:i+1] for k, v in data.items()}),
                                 mask, data['path_valid'][:, i:i+1]) for i in range(3)], dim=1)
    torch.testing.assert_close(output, reference)
    assert output.shape == (2, 3, 5, 32)
    output.square().mean().backward()
    for module in (tokenizer, fusion):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


def test_invalid_paths_and_padding_are_exact_residual_fallback():
    data = features()
    data['path_valid'].zero_()
    tokenizer, fusion = PathTokenizer(16), NavigationCrossAttention(32, 16, 4)
    text = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    output = fusion(text, tokenizer(data), mask, data['path_valid'])
    torch.testing.assert_close(output, text[:, None].expand(-1, 3, -1, -1), rtol=0, atol=0)
    assert torch.isfinite(output).all()
    data['ego_relation'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        tokenizer(data)


def test_frame_mask_and_last_valid_readout():
    sequence = build_frame_multimodal_sequence(torch.randn(1, 2, 2, 8), torch.tensor([[1, 0]]),
        torch.randn(1, 2, 3, 8), torch.tensor([[[1, 1, 0], [1, 0, 0]]]))
    assert sequence.frame_readout_indices.tolist() == [[3, 7]]
    allowed = build_ddia_attention_mask(sequence.modality_ids, sequence.frame_ids)[0]
    assert allowed[0].nonzero().flatten().tolist() == [0]
    assert allowed[5].nonzero().flatten().tolist() == [5]
    assert allowed[2].nonzero().flatten().tolist() == [0]
    assert allowed[7].nonzero().flatten().tolist() == [2, 3, 5]
    assert not allowed[:5, 5:].any()
    assert not allowed[:, [1, 4, 6, 8, 9]].any()
