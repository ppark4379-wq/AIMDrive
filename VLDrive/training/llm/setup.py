"""LLM construction and shared training configuration without new defaults.

VLDriveTrainingConfig is shared with joint training. The joint model retains
its existing lm_head/end_predictor freeze and gradient-checkpointing policy.
"""
from models.vldrive import VLDriveTrainingConfig
from models.llm.language_model import VLDriveModel
from models.llm.tinyllama_backbone import load_tinyllama, resolve_torch_dtype

__all__ = ['VLDriveTrainingConfig', 'VLDriveModel', 'load_tinyllama', 'resolve_torch_dtype']
