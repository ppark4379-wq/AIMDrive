import os

import pytest
import torch

from lavis.models.vldrive.tinyllama_backbone import load_tinyllama, resolve_torch_dtype
from lavis.models.vldrive.ddia_attention import set_ddia_modality_ids
from lavis.models.vldrive.multimodal_sequence import INSTRUCTION


@pytest.mark.skipif(not os.getenv("TINYLLAMA_TEST_CHECKPOINT"), reason="set local checkpoint for integration test")
def test_tinyllama_local_checkpoint_forward_and_gradients():
    components = load_tinyllama(
        os.environ["TINYLLAMA_TEST_CHECKPOINT"],
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    config = components.model.config
    assert config.hidden_size > 0
    assert config.num_attention_heads >= config.num_key_value_heads
    tokens = components.tokenizer(["turn left"], return_tensors="pt")
    modality_ids = torch.full_like(tokens["input_ids"], INSTRUCTION)
    set_ddia_modality_ids(components.model, modality_ids)
    output = components.model(**tokens, use_cache=False)
    output.logits.sum().backward()
    first_attention = components.model.model.layers[0].self_attn
    assert first_attention.q_proj.weight.grad is not None
    assert first_attention.k_proj.weight.grad is not None
    assert first_attention.v_proj.weight.grad is not None


def test_precision_mapping():
    assert resolve_torch_dtype("fp32") == torch.float32
    assert resolve_torch_dtype("fp16") == torch.float16
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    with pytest.raises(ValueError):
        resolve_torch_dtype("not-a-dtype")
