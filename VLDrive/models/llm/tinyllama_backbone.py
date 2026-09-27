from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .ddia_attention import install_ddia_attention

DEFAULT_TINYLLAMA_CHECKPOINT = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


@dataclass
class TinyLlamaComponents:
    tokenizer: object
    model: nn.Module


def resolve_torch_dtype(precision: str):
    """Map the configured implementation precision to a PyTorch dtype."""
    normalized = precision.lower().replace("torch.", "")
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"auto", "bf16_or_fp16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError("precision must be one of: fp32, fp16, bf16, auto")


def load_tinyllama(
    model_name_or_path: str = DEFAULT_TINYLLAMA_CHECKPOINT,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    freeze_tinyllama: bool = False,
    use_ddia: bool = True,
    torch_dtype=None,
) -> TinyLlamaComponents:
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    transformers_version = version.parse(transformers.__version__)
    if use_ddia and not (
        version.parse("4.31.0") <= transformers_version < version.parse("4.32.0")
    ):
        raise RuntimeError(
            "DDIA currently requires Transformers 4.31.x: it provides TinyLLaMA GQA "
            "and the three-value LlamaAttention API used by this implementation. "
            f"Installed version: {transformers.__version__}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache_dir, local_files_only=local_files_only, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if use_ddia:
        install_ddia_attention(model, use_ddia=True)
    for parameter in model.parameters():
        parameter.requires_grad = not freeze_tinyllama
    return TinyLlamaComponents(tokenizer, model)


def parameter_report(module: nn.Module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    ddia = sum(parameter.numel() for name, parameter in module.named_parameters() if "self_attn" in name)
    tinyllama = getattr(module, "tinyllama", module)
    tinyllama_total = sum(parameter.numel() for parameter in tinyllama.parameters())
    tinyllama_trainable = sum(
        parameter.numel() for parameter in tinyllama.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "tinyllama": tinyllama_total,
        "tinyllama_trainable": tinyllama_trainable,
        "ddia_attention": ddia,
    }
