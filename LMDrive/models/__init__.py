"""LMDrive models.

vision_encoder  camera + LiDAR encoder (stage 1 pre-training, frozen in stage 2)
lmdrive         vision encoder + Q-Former + LLaMA driving model (stage 2)
resnet, transformer, pointpillars   building blocks of the vision encoder
qformer, llama  vendored BLIP-2 Q-Former and LLaMA implementations

models.lmdrive imports transformers, so it is not imported here; stage 1 only
needs PyTorch.
"""
from .vision_encoder import VISION_ENCODER_VARIANTS, VisionEncoder, create_vision_encoder

__all__ = ["VISION_ENCODER_VARIANTS", "VisionEncoder", "create_vision_encoder"]
