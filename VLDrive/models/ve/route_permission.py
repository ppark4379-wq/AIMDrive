"""Opt-in route permission head beside the unchanged native visual encoder.

The model consumes the native traffic-light visual token and the *planned*
movement. Simulator signal bits, signal identifiers and teacher permissions are
supervision metadata, never inputs to this module. Deny/allow is distinct from
the native red/not-red output, which remains unchanged.
"""
import copy
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


MOVEMENT_INDEX = {'straight': 0, 'left': 1, 'right': 2}
OUTPUT_INDEX = {'deny': 0, 'allow': 1}
HEAD_SCHEMA = 'morai_route_permission_head_v1'
FEATURE_SOURCE = 'native_traffic_light_pred_head_input'
SENSOR_INPUT_KEYS = frozenset(('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear',
                             'rgb_center', 'lidar', 'num_points', 'velocity', 'target_point'))
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
PARAMETER_DTYPES = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16,
                                                 torch.float32, torch.float64)}


def movement_one_hot(movement, batch_size, *, device, dtype):
    """Validate known movement indices or exact one-hot vectors; never impute."""
    if not isinstance(movement, torch.Tensor):
        raise TypeError('Planned movement must be a tensor')
    if movement.device != device:
        raise ValueError('Movement and visual feature must share a device')
    if movement.shape == (batch_size,):
        if movement.dtype not in INTEGER_DTYPES:
            raise TypeError('Movement indices must use an integer dtype')
        if bool(((movement < 0) | (movement >= len(MOVEMENT_INDEX))).any()):
            raise ValueError('Unknown planned movement; expected straight=0, left=1 or right=2')
        return F.one_hot(movement.long(), num_classes=len(MOVEMENT_INDEX)).to(dtype=dtype)
    if movement.shape == (batch_size, len(MOVEMENT_INDEX)):
        if (not bool(torch.isfinite(movement).all())
                or not bool(((movement == 0) | (movement == 1)).all())
                or not bool((movement.sum(dim=1) == 1).all())):
            raise ValueError('Movement must be an exact one-hot vector for a known movement')
        return movement.to(dtype=dtype)
    raise ValueError('Movement must have shape (B,) or (B, 3)')


class RoutePermissionHead(nn.Module):
    """Condition a visual signal feature on the intended route movement."""
    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        if (type(feature_dim) is not int or feature_dim < 1
                or type(hidden_dim) is not int or hidden_dim < 1):
            raise ValueError('Head feature/hidden dimensions must be positive integers')
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim + len(MOVEMENT_INDEX), hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, len(OUTPUT_INDEX)))

    def forward(self, light_feature, movement):
        if (not isinstance(light_feature, torch.Tensor) or light_feature.ndim != 2
                or light_feature.shape[1] != self.feature_dim):
            raise ValueError('Light feature must have shape (B, feature_dim)')
        if not light_feature.is_floating_point() or not bool(torch.isfinite(light_feature).all()):
            raise ValueError('Light feature must be finite floating-point data')
        route = movement_one_hot(movement, len(light_feature), device=light_feature.device,
                                 dtype=light_feature.dtype)
        return self.network(torch.cat((light_feature, route), dim=1))


class RoutePermissionEncoder(nn.Module):
    """Capture the real light feature in one native forward and attach the head.

    ``freeze_encoder=True`` keeps the base in eval mode even while the head is
    trained and avoids recording its autograd graph. The explicit opt-in joint
    mode preserves the caller's base parameter freeze settings. The hook is
    local to each forward and removed even if native inference raises.
    """
    def __init__(self, base_encoder, freeze_encoder=True, head=None):
        super().__init__()
        if not isinstance(base_encoder, nn.Module):
            raise TypeError('Base encoder must be an nn.Module')
        if getattr(base_encoder, 'return_feature', None) is not False:
            raise ValueError('Base encoder must use native return_feature=False')
        light_head = getattr(base_encoder, 'traffic_light_pred_head', None)
        if not isinstance(light_head, nn.Module):
            raise ValueError('Base encoder has no native traffic_light_pred_head')
        if type(freeze_encoder) is not bool:
            raise TypeError('freeze_encoder must be boolean')
        self.base_encoder = base_encoder
        self.freeze_encoder = freeze_encoder
        self.permission_head = head if head is not None else RoutePermissionHead()
        if not isinstance(self.permission_head, RoutePermissionHead):
            raise TypeError('Permission head must be RoutePermissionHead')
        feature_dim = getattr(light_head, 'in_features', None)
        if feature_dim is not None and feature_dim != self.permission_head.feature_dim:
            raise ValueError('Permission head feature dimension differs from native light token')
        if self.freeze_encoder:
            self.base_encoder.requires_grad_(False)
            self.base_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.base_encoder.eval()
        return self

    def forward(self, sensor_inputs, movement):
        if not isinstance(sensor_inputs, dict) or set(sensor_inputs) != SENSOR_INPUT_KEYS:
            raise ValueError('Only the exact native sensor/ego/route inputs are permitted; no GT metadata')
        if self.base_encoder.return_feature is not False:
            raise ValueError('Base encoder must retain native return_feature=False')
        front = sensor_inputs['rgb_front']
        if not isinstance(front, torch.Tensor) or front.ndim < 1 or not front.is_floating_point():
            raise ValueError('Native front-camera input must be a batched floating-point tensor')
        movement_one_hot(movement, len(front), device=front.device, dtype=front.dtype)
        if self.freeze_encoder:
            self.base_encoder.eval()
        captured = []

        def capture_light_feature(module, arguments):
            if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
                raise ValueError('Unexpected native traffic-light feature call')
            captured.append(arguments[0])

        handle = self.base_encoder.traffic_light_pred_head.register_forward_pre_hook(capture_light_feature)
        try:
            with torch.no_grad() if self.freeze_encoder else nullcontext():
                base_outputs = self.base_encoder(sensor_inputs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('Expected exactly one native traffic-light feature per sensor forward')
        logits = self.permission_head(captured[0], movement)
        return {'base_outputs': base_outputs, 'route_permission': logits}


def _check_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('Expected a lowercase SHA256 for the matching base encoder checkpoint')


def build_route_permission_checkpoint(head, base_checkpoint_sha256):
    """Export a head-only sidecar tied to one immutable base checkpoint."""
    if not isinstance(head, RoutePermissionHead):
        raise TypeError('Expected RoutePermissionHead')
    _check_sha256(base_checkpoint_sha256)
    state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Cannot export nonfinite permission head weights')
    return dict(schema=HEAD_SCHEMA, base_checkpoint_sha256=base_checkpoint_sha256,
                feature_source=FEATURE_SOURCE, movement_index=dict(MOVEMENT_INDEX),
                output_index=dict(OUTPUT_INDEX),
                parameter_dtype=str(next(head.parameters()).dtype),
                head_config=dict(feature_dim=head.feature_dim, hidden_dim=head.hidden_dim),
                state_dict=state)


def load_route_permission_checkpoint(payload, expected_base_checkpoint_sha256):
    """Validate a sidecar and return its CPU head; never load base weights."""
    _check_sha256(expected_base_checkpoint_sha256)
    if (not isinstance(payload, dict) or payload.get('schema') != HEAD_SCHEMA
            or payload.get('base_checkpoint_sha256') != expected_base_checkpoint_sha256
            or payload.get('feature_source') != FEATURE_SOURCE
            or payload.get('movement_index') != MOVEMENT_INDEX
            or payload.get('output_index') != OUTPUT_INDEX):
        raise ValueError('Permission sidecar base checkpoint or feature/movement contract mismatch')
    config = payload.get('head_config')
    if not isinstance(config, dict) or set(config) != {'feature_dim', 'hidden_dim'}:
        raise ValueError('Invalid permission head configuration')
    head = RoutePermissionHead(**config)
    dtype = PARAMETER_DTYPES.get(payload.get('parameter_dtype'))
    if dtype is None:
        raise ValueError('Invalid permission head parameter dtype')
    head.to(dtype=dtype)
    state = payload.get('state_dict')
    if not isinstance(state, dict) or set(state) != set(head.state_dict()):
        raise ValueError('Sidecar must contain only the exact permission head parameters')
    for name, value in state.items():
        expected = head.state_dict()[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid permission head tensor: ' + name)
    head.load_state_dict(copy.deepcopy(state), strict=True)
    return head
