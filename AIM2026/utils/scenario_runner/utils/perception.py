"""Runner views of the VE publisher profile, backed by the VLDrive originals.

The checkpoint table, map basis and publisher profile come from VLDrive/inference
(realtime_ve_objects and its published JSON profile); nothing is duplicated here.
"""
import json
import os
from pathlib import Path
import sys

from scenario_runner.project_paths import REPO_ROOT

_VLDRIVE = str(REPO_ROOT / "VLDrive")
if _VLDRIVE not in sys.path:
    sys.path.insert(0, _VLDRIVE)

PUBLISHER_PROFILE = REPO_ROOT / "VLDrive/configs/vision_encoder_realtime_best_20261003.json"
DEFAULT_SIGNAL_HEAD = REPO_ROOT / "MoraiDrive/.runtime/signal_state_p1_20261014/run_003/signal_head_final.pt"


def __getattr__(name):
    # Imported on first use so that importing the runner never needs the VE package.
    if name in ("CHECKPOINTS", "MAP_BASIS"):
        from inference import realtime_ve_objects
        return getattr(realtime_ve_objects, name)
    raise AttributeError(name)


def publisher_config(label=None):
    """The published realtime VE publisher profile; with a label, its checkpoint is selected."""
    profile = json.loads(PUBLISHER_PROFILE.read_text())
    if label is not None:
        selected = __getattr__("CHECKPOINTS")[label]
        profile.update(checkpoint=selected["path"], checkpoint_sha256=selected["sha256"])
    return profile


def signal_head():
    """The P1 signal-head sidecar, with the existing explicit override."""
    return Path(os.environ.get("VE_SIGNAL_HEAD") or DEFAULT_SIGNAL_HEAD)
