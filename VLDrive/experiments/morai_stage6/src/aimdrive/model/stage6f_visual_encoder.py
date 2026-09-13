"""Compatibility import for the original two visual policies."""
from aimdrive._lavis_compat import export
export(globals(), 'lavis.models.vldrive.visual_encoder', {'Stage6FVisualEncoder': '_VisualEncoderBase'})
