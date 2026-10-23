"""Reviewed VE weights in AIM2026/Weights/VE, pinned by SHA256.

From VLDrive inference/realtime_ve_objects.CHECKPOINTS and
utils/evaluation/morai_transfer_evaluation._load_model. A file whose digest
differs from its label is never loaded.
"""
import hashlib
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / 'Weights' / 'VE'
MODEL_NAME = 'memfuser_baseline_e1d3'

# label -> file in WEIGHTS_DIR and its SHA256 (from the VLDrive checkpoint table)
CHECKPOINTS = {
    # perception_g2_tunnel_20261014_114540/checkpoint-epoch-001.pth.tar: GPS full-route driving
    'P1': dict(file='ve_p1_20261014.pth.tar',
               sha256='054769a5462cd9684f0de6ba80c7b7fc28b234051a7b0490020a45be050e59f7'),
    # mixed_grid_fast_20260930_152712_134039/checkpoint-best-object.pth.tar
    'E3': dict(file='ve_e3_20260930.pth.tar',
               sha256='6824d3998ec4be85bf40f96de37a195d12ff00aa8181d6739ed75125a7bb5e5b'),
}
# Signal heads are bound to one base checkpoint (checked inside the sidecar).
SIGNAL_HEADS = {
    'P1': 'signal_head_p1_20261014.pt',  # signal_state_p1_20261014/run_003/signal_head_final.pt
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def resolve(label):
    """(checkpoint path, expected SHA256, signal-head path or None) of a reviewed label."""
    if label not in CHECKPOINTS:
        raise ValueError('Unknown VE checkpoint label %r; choose from %s' % (label, sorted(CHECKPOINTS)))
    entry = CHECKPOINTS[label]
    head = SIGNAL_HEADS.get(label)
    path = WEIGHTS_DIR / entry['file']
    if label == 'E3' and not path.is_file():
        path = Path(__file__).resolve().parents[3] / 'VLDrive/outputs/checkpoints/vision_encoder/checkpoint-best-object.pth.tar'
    return path, entry['sha256'], (WEIGHTS_DIR / head if head else None)


def load_encoder(path, expected_sha256, device):
    """(eval model on device, payload without being trusted further, actual SHA256)."""
    import torch
    from .model import create_vision_encoder
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('VE checkpoint not found: %s' % path)
    actual = digest(path)
    if actual != expected_sha256:
        raise ValueError('VE checkpoint SHA256 mismatch: %s' % path)
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), dict):
        raise ValueError('Expected a checkpoint containing state_dict')
    if payload.get('token_shape') not in (None, [106, 256]):
        raise ValueError('Checkpoint has an incompatible visual token contract')
    model = create_vision_encoder({'name': MODEL_NAME}, pretrained_backbone=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device), payload, actual


def load_signal(path, base_sha256, device):
    import torch
    from .model import load_signal_head
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Signal head not found: %s' % path)
    return load_signal_head(torch.load(str(path), map_location='cpu'), base_sha256).to(device)
