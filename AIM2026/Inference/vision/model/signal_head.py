"""Three-phase traffic-signal head on the encoder's native traffic-light token.

From VLDrive models/ve/signal_state.py. The head is a separate sidecar file
bound to one base checkpoint by SHA256; the base encoder weights are untouched.
Only the collected MORAI phases 1, 16 and 33 are supported, and a confident
prediction says nothing about which signal ID it belongs to.
"""
import math

import torch
from torch import nn

SIGNAL_CODES = (1, 16, 33)        # red, green, red + left arrow (MORAI status bits)
PHASE_CODES = (1, 4, 16, 33, 48)  # phase head: red, yellow (also red+yellow), green, red + left arrow, green + left arrow
PHASE_SCHEMA = 'morai_signal_state_head_v2'
NATIVE_SIGNAL_CODES = (1, 16)     # the encoder's own red/green output
SCHEMA = 'morai_signal_state_head_v1'
FEATURE_SOURCE = 'native_traffic_light_pred_head_input'
CONFIDENCE_THRESHOLD = 0.9


def check_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('Expected a lowercase SHA256 for the matching base encoder checkpoint')


class SignalStateHead(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128, codes=SIGNAL_CODES):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1 or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError('Positive integer head dimensions required')
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, len(codes)))

    def forward(self, feature):
        if feature.ndim != 2 or feature.shape[1] != self.feature_dim or not torch.isfinite(feature).all():
            raise ValueError('Expected finite batched native signal features')
        return self.network(feature)


def load_signal_head(payload, base_sha256):
    """Sidecar dict -> eval SignalStateHead; refuses a head trained on other weights."""
    check_sha256(base_sha256)
    if (payload.get('schema') != SCHEMA or payload.get('base_checkpoint_sha256') != base_sha256
            or payload.get('signal_codes') != list(SIGNAL_CODES)
            or payload.get('feature_source') != FEATURE_SOURCE):
        raise ValueError('Signal sidecar/base checkpoint contract mismatch')
    head = SignalStateHead(**payload['head_config'])
    state = payload['state_dict']
    if set(state) != set(head.state_dict()):
        raise ValueError('Unexpected sidecar tensors')
    for key, reference in head.state_dict().items():
        value = state[key]
        if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid signal tensor: ' + key)
    head.load_state_dict(state, strict=True)
    return head.eval()


def load_phase_head(payload, base_sha256):
    """Five-phase sidecar (adds yellow; trained on tokens collected with forced lamp colours) -> eval head."""
    check_sha256(base_sha256)
    if (payload.get('schema') != PHASE_SCHEMA or payload.get('base_checkpoint_sha256') != base_sha256
            or payload.get('signal_codes') != list(PHASE_CODES) or payload.get('feature_source') != FEATURE_SOURCE):
        raise ValueError('Phase sidecar/base checkpoint contract mismatch')
    head = SignalStateHead(codes=PHASE_CODES, **payload['head_config'])
    state = payload['state_dict']
    if set(state) != set(head.state_dict()):
        raise ValueError('Unexpected phase sidecar tensors')
    for key, reference in head.state_dict().items():
        value = state[key]
        if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid phase tensor: ' + key)
    head.load_state_dict(state, strict=True)
    return head.eval()


def decode_phase(logits, threshold=CONFIDENCE_THRESHOLD):
    """Phase-head logits -> [{'code': 1|4|16|33|48|None, 'confidence': p}]."""
    return _decode_codes(logits, PHASE_CODES, threshold)


def decode_signal(logits, threshold=CONFIDENCE_THRESHOLD):
    """Head logits -> [{'code': 1|16|33|None, 'confidence': p}]; None means abstain."""
    return _decode_codes(logits, SIGNAL_CODES, threshold)


def decode_native_signal(logits, threshold=CONFIDENCE_THRESHOLD):
    """Encoder red/green logits -> [{'code': 1|16|None, 'confidence': p}]."""
    return _decode_codes(logits, NATIVE_SIGNAL_CODES, threshold)


def _decode_codes(logits, codes, threshold):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid confidence threshold')
    if logits.ndim != 2 or logits.shape[1] != len(codes):
        raise ValueError('Unexpected signal logit dimensions')
    result = []
    for row in logits.detach():
        if not torch.isfinite(row).all():
            result.append(dict(code=None, confidence=None))
            continue
        confidence, index = row.softmax(0).max(0)
        result.append(dict(code=codes[int(index)] if float(confidence) >= threshold else None,
                           confidence=float(confidence)))
    return result
