from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from models.language_model import VLDriveModel
from models.vldrive import VLDriveTrainingModel


class VisualPassthrough(torch.nn.Module):
    """Only replace the expensive visual encoder/connector; run the actual text+LLM path."""
    def forward(self, features, **kwargs):
        b, t, k, _ = features.shape
        zero = features.new_tensor(0.)
        return dict(language_tokens=features, selected_valid_mask=torch.ones(b, t, k, dtype=torch.bool),
                    pruning_loss=zero, reconstruction_loss=zero, cycle_loss=zero,
                    selected_count_per_frame=torch.full((b, t), k), retention_logit_loss=zero,
                    selector=dict(retention_per_frame=torch.ones(b, t),
                                  soft_retention_per_frame=torch.ones(b, t),
                                  probabilities=torch.ones(b, t, k, 2) * .5))


def test_notice_reaches_tokenizer_ddia_embedding_and_waypoint_readout(tmp_path):
    torch.manual_seed(41)
    words = ["[UNK]", "[PAD]", "Follow", "the", "road", ".", "Watch", "stationary", "obstacles"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.save_pretrained(tmp_path)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128)).save_pretrained(tmp_path)
    language = VLDriveModel(str(tmp_path), local_files_only=True, precision="fp32",
                            visual_feature_dim=8, max_instruction_length=64)
    model = VLDriveTrainingModel.__new__(VLDriveTrainingModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(input_mode="cached")
    model.connector = VisualPassthrough()
    model.language_model = language
    model.eval()
    seen = []
    handle = language.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    features = torch.randn(1, 3, 2, 8)
    inputs = dict(visual_feature=features, instruction=["Follow the road."])
    baseline = model.predict(inputs)["trajectory"]
    empty = model.predict(dict(inputs, notice_text=[""]))["trajectory"]
    torch.testing.assert_close(baseline, empty, rtol=0, atol=0)
    result = model.predict(dict(inputs, notice_text=["Watch stationary obstacles."]))
    assert seen[-1] == ["Follow the road.\nWatch stationary obstacles."]
    assert result["trajectory"].shape == (1, 3, 5, 2)
    assert torch.isfinite(result["trajectory"]).all()
    assert not torch.allclose(baseline[:, -1], result["trajectory"][:, -1])
    loss = model.loss(result, dict(future_trajectory=torch.zeros_like(result["trajectory"]),
                                 frame_loss_mask=torch.tensor([[False, False, True]])))
    loss["loss"].backward()
    grad = language.tinyllama.get_input_embeddings().weight.grad
    assert grad[words.index("obstacles")].abs().sum() > 0
    handle.remove()
    with pytest.raises(ValueError, match="batch sizes"):
        model.predict(dict(inputs, notice_text=[]))
