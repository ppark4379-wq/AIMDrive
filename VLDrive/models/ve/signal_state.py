"""Optional three-phase signal sidecar; native VE weights/outputs stay intact.

Only the collected phases 1, 16 and 33 are supported. A confident prediction
is not a signal-ID association or evidence that unknown phases are rejected.
"""
import math
import torch
from torch import nn
from models.ve.route_permission import SENSOR_INPUT_KEYS, _check_sha256

SIGNAL_CODES = (1, 16, 33)
SCHEMA = 'morai_signal_state_head_v1'


class SignalStateHead(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1 or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError('Positive integer head dimensions required')
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, len(SIGNAL_CODES)))

    def forward(self, feature):
        if feature.ndim != 2 or feature.shape[1] != self.feature_dim or not torch.isfinite(feature).all():
            raise ValueError('Expected finite batched native signal features')
        return self.network(feature)


def extract_signal_feature(base, inputs, forward=None):
    """Capture the native token; perception-only forward may omit target_point.

    Full-model training/evaluation still requires the waypoint target input.
    All sensor keys remain mandatory and extra GT keys are always rejected.
    """
    if base.training or getattr(base, 'return_feature', None) is not False:
        raise ValueError('Native encoder must be eval with return_feature=False')
    allowed = (SENSOR_INPUT_KEYS,) if forward is None else (
        SENSOR_INPUT_KEYS, SENSOR_INPUT_KEYS - {'target_point'})
    if set(inputs) not in allowed:
        raise ValueError('Only native sensor/ego inputs and optional perception route target permitted; no signal GT')
    captured = []
    handle = base.traffic_light_pred_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            outputs = base(inputs) if forward is None else forward(base, inputs)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].ndim != 2:
        raise ValueError('Expected one batched native traffic-light token')
    return captured[0].detach(), outputs


def encode_checkpoint(head, base_sha256):
    _check_sha256(base_sha256)
    if not isinstance(head, SignalStateHead):
        raise TypeError('Expected SignalStateHead')
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Expected finite float32 sidecar')
    return dict(schema=SCHEMA, base_checkpoint_sha256=base_sha256,
                signal_codes=list(SIGNAL_CODES), feature_source='native_traffic_light_pred_head_input',
                head_config=dict(feature_dim=head.feature_dim, hidden_dim=head.hidden_dim), state_dict=state)


def decode_checkpoint(payload, base_sha256):
    _check_sha256(base_sha256)
    if (payload.get('schema') != SCHEMA or payload.get('base_checkpoint_sha256') != base_sha256
            or payload.get('signal_codes') != list(SIGNAL_CODES)
            or payload.get('feature_source') != 'native_traffic_light_pred_head_input'):
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


def decode_signal(logits, threshold=0.9):
    """None means abstain; never invent an MORAI phase for unavailable input."""
    return _decode_codes(logits, SIGNAL_CODES, threshold)


def decode_native_signal(logits, threshold=0.9):
    """Native binary red/green output, for mapped straight approaches only."""
    return _decode_codes(logits, (1, 16), threshold)


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
