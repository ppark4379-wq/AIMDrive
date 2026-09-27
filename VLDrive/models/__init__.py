"""VLDrive model packages.

ve: camera/LiDAR encoder, its backbones and frozen Stage2 wrapper.
connector: CCDP, MEFA, Q-Former and the visual-token interface.
llm: TinyLlama, DDIA, multimodal sequence and waypoint/end heads.
vldrive: integration of the three components for Stage2 training/inference.

No eager imports: Stage1 does not require the language-model dependencies.
"""
