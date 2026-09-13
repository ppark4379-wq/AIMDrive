"""VLDrive models.

Stage 1 - the LMDrive-structure vision encoder trained on MORAI:
  vision_encoder            Memfuser camera + LiDAR encoder -> 106 visual tokens
  resnet, transformer, pointpillars   its building blocks

Stage 2 - VLDrive itself (frozen encoder -> CCDP -> MEFA -> TinyLlama -> waypoints):
  vldrive                   the full trainable model (VLDriveTrainingModel)
  frozen_vision_encoder     the stage-1 encoder, frozen, with its input contract
  ccdp                      Gumbel token pruning that keeps ~30% of the tokens
  qformer_mefa              Modified Q-Former with the Z=10 frame memory
  qformer                   vendored BLIP-2 Q-Former that MEFA is built on
  language_model            TinyLlama + DDIA + waypoint/end heads
  tinyllama_backbone        loads TinyLlama and installs DDIA attention
  ddia_attention            modality-aware attention bias
  multimodal_sequence       visual/instruction/padding sequence layout
  visual_token_interface    the [B,T,N,256] visual-token contract

models.vldrive pulls in transformers; stage 1 only needs PyTorch, so nothing is
imported here.
"""
