import torch

from lavis.models.vldrive.multimodal_sequence import PADDING, VISUAL, build_multimodal_sequence
from lavis.models.vldrive.visual_token_interface import VisualTokenBatch


def test_visual_interface_flat_and_temporal():
    flat = VisualTokenBatch(torch.randn(2, 6, 4)).flattened()
    temporal = VisualTokenBatch(torch.randn(2, 2, 3, 4)).flattened()
    assert flat.visual_tokens.shape == temporal.visual_tokens.shape == (2, 6, 4)


def test_variable_instruction_length_sequence():
    sequence = build_multimodal_sequence(
        torch.randn(2, 3, 4), torch.tensor([[1, 1, 1], [1, 0, 0]]),
        torch.randn(2, 2, 4), torch.tensor([[1, 1], [1, 0]])
    )
    assert sequence.modality_ids[1].tolist() == [0, PADDING, PADDING, VISUAL, PADDING]
    assert sequence.position_ids[1].tolist() == [0, 0, 0, 1, 1]
