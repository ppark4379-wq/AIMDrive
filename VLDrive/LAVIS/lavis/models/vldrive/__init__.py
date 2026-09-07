"""VLDrive components; import only the component requested by the caller."""
from importlib import import_module

_EXPORTS = {
    "DDIALlamaAttention": ("ddia_attention", "DDIALlamaAttention"),
    "build_ddia_attention_mask": ("ddia_attention", "build_ddia_attention_mask"),
    "VLDriveModel": ("vldrive_model", "VLDriveModel"),
    "VLDriveOutput": ("vldrive_model", "VLDriveOutput"),
    "VisualTokenBatch": ("visual_token_interface", "VisualTokenBatch"),
    "MoraiDrivingModel": ("morai_model", "MoraiDrivingModel"),
    "MoraiDrivingConfig": ("morai_model", "MoraiDrivingConfig"),
    "MoraiVisualEncoder": ("visual_encoder", "MoraiVisualEncoder"),
    "VisualTrajectoryModel": ("visual_trajectory", "VisualTrajectoryModel"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
