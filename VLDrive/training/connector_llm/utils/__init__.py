"""Training diagnostics and model helpers used by connector-LLM joint training."""
from models.connector.visual_token_interface import VisualTokenBatch, last_valid_indices
from models.llm.tinyllama_backbone import parameter_report
from models.llm.multimodal_sequence import MultimodalSequence, build_multimodal_sequence
from training.common.logging import gradient_report

__all__ = [
    'VisualTokenBatch', 'last_valid_indices', 'parameter_report',
    'MultimodalSequence', 'build_multimodal_sequence', 'gradient_report',
]
