"""VE perception for the rule-based Expert: sensors -> boxes and signal phase.

    runtime = VisionRuntime('P1', lidar_to_base=lidar_to_base(pos, rot_deg))
    result = runtime.infer(images, sweep, speed_mps, pose)
    udp.send_detection(result.boxes, ego_pose, oldest_receipt,
                       signal=result.signal, native_signal=result.native_signal)

Only the perception heads run (the waypoint branch and any route target are
unused), as in VLDrive's ve_objects_udp.py. Boxes are placed with the pose the
inputs were captured at; the caller supplies it (UDP Pose GridProjector).
"""
from dataclasses import dataclass
import time

import torch

from . import checkpoint as checkpoints
from .contract import checkpoint_contract
from .model import decode_native_signal, decode_signal
from .postprocess import decode_world_boxes
from .preprocess import InputBuilder

# LiDAR mount of the competition sensor set (pos [1.0, 0, 1.48] m, no rotation).
DEFAULT_LIDAR_TO_BASE = dict(translation_xyz=(1.0, 0.0, 1.48), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))


@dataclass(frozen=True)
class Perception:
    boxes: tuple            # AIM2026.UDP.Box in map grid coordinates
    signal: dict            # {'code': 1|16|33|None, 'confidence'} or None without a signal head
    native_signal: dict     # encoder red/green {'code': 1|16|None, 'confidence'}
    counts: dict            # decode rejections (candidates, NMS, invalid speeds, ...)
    inference_ms: float


def _view_tokens(model, tokens, global_token, view_index, global_index):
    """VisionEncoder._camera_tokens after its patch embedding, unchanged."""
    if model.use_view_embed:
        tokens = tokens + model.view_embed[:, :, view_index:view_index + 1, :] + model.position_encoding(tokens)
    else:
        tokens = tokens + model.position_encoding(tokens)
    tokens = tokens.flatten(2).permute(2, 0, 1)
    global_token = (global_token + model.view_embed[:, :, view_index, :]
                    + model.global_embed[:, :, global_index:global_index + 1])
    return [tokens, global_token.permute(2, 0, 1)]


def forward_features_batched(model, inputs):
    """VisionEncoder.forward_features with the four 128 px views in one backbone call.

    Left, right, centre and rear share one shape, so they run as a batch of four
    instead of four batch-1 ResNet calls (the call overhead dominated on a GPU the
    simulator also renders on). Eval-mode BatchNorm uses stored statistics, so each
    view's features are those of a separate call; token order and view/global
    embedding slots are exactly those of forward_features.
    """
    if not (model.with_right_left_sensors and model.with_center_sensor and model.with_rear_sensor):
        return model.forward_features(inputs['rgb_front'], inputs['rgb_left'], inputs['rgb_right'],
                                      inputs['rgb_rear'], inputs['rgb_center'], inputs['lidar'], inputs['num_points'])
    batch = inputs['rgb_front'].shape[0]
    front_tokens, front_global = model.rgb_patch_embed(inputs['rgb_front'])
    side = torch.cat([inputs['rgb_left'], inputs['rgb_right'], inputs['rgb_center'], inputs['rgb_rear']], 0)
    side_tokens, side_global = model.rgb_patch_embed(side)
    features = _view_tokens(model, front_tokens, front_global, 0, 0)
    # (view_index, global_index) as in forward_features; the rear uses global slot 5.
    for k, (view, slot) in enumerate(((1, 1), (2, 2), (3, 3), (4, 5))):
        rows = slice(k * batch, (k + 1) * batch)
        features += _view_tokens(model, side_tokens[rows], side_global[rows], view, slot)
    lidar_token = model.lidar_backbone(inputs['lidar'], inputs['num_points'])
    lidar_token = lidar_token + model.position_encoding(lidar_token)
    lidar_token = lidar_token.flatten(2).permute(2, 0, 1)
    return torch.cat(features, 0), lidar_token


def forward_perception(model, inputs, *, batch_cameras=True):
    """Same trained feature/decoder/heads as VisionEncoder.forward; no waypoint branch."""
    if model.return_feature:
        raise ValueError('Perception needs the trained prediction heads enabled')
    velocity = inputs['velocity'].view(1, -1, 1)
    if batch_cameras:
        features, lidar_token = forward_features_batched(model, inputs)
    else:
        features, lidar_token = model.forward_features(
            inputs['rgb_front'], inputs['rgb_left'], inputs['rgb_right'], inputs['rgb_rear'],
            inputs['rgb_center'], inputs['lidar'], inputs['num_points'])
    batch = inputs['rgb_front'].shape[0]
    target = model.position_encoding(torch.ones((batch, 1, 50, 50), device=inputs['rgb_front'].device))
    target = torch.cat([target.flatten(2), model.query_pos_embed.repeat(batch, 1, 1)], 2).permute(2, 0, 1)
    memory = model.encoder(features, mask=model.attn_mask)
    query_embed = model.query_embed.repeat(1, batch, 1) + model.velocity_fc(velocity).repeat(6, 1, 1)
    query = torch.cat([lidar_token, query_embed], 0)
    hidden = model.decoder(query, memory, query_pos=target)[0].permute(1, 0, 2)
    traffic = model.traffic_pred_head(torch.cat([
        hidden[:, :2500], velocity.view(-1, 1, 1).repeat(1, 2500, 32)], dim=2))
    light = model.traffic_light_pred_head(hidden[:, 2500])
    return traffic, light, hidden[:, 2500]


class VisionRuntime:
    """Load one reviewed checkpoint (and its signal head) and run per-frame perception.

    ``checkpoint`` is a label of ``checkpoint.CHECKPOINTS`` (weights in
    AIM2026/Weights/VE). ``checkpoint_path``/``checkpoint_sha256`` override the
    file for a label-less run; the SHA256 is still enforced. ``signal_head`` is
    True for the label's registered head, a path, or None/False to skip it.
    """

    def __init__(self, checkpoint='P1', *, device='cuda', lidar_to_base=DEFAULT_LIDAR_TO_BASE,
                 signal_head=True, checkpoint_path=None, checkpoint_sha256=None, cpu_threads=4):
        if checkpoint_path is not None:
            if not checkpoint_sha256:
                raise ValueError('checkpoint_sha256 is required with checkpoint_path')
            path, sha, head_path = checkpoint_path, checkpoint_sha256, None
        else:
            path, sha, head_path = checkpoints.resolve(checkpoint)
        if signal_head not in (True, False, None):
            head_path = signal_head
        elif not signal_head:
            head_path = None
        elif head_path is None:
            raise ValueError('No signal head is registered for this checkpoint; pass signal_head=None')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.set_num_threads(int(cpu_threads))
        self.device = device
        self.model, payload, self.checkpoint_sha256 = checkpoints.load_encoder(path, sha, device)
        self.contract = checkpoint_contract(payload, self.checkpoint_sha256)
        self.inputs = InputBuilder(self.contract['adapter'], lidar_to_base)
        self.signal_head = (checkpoints.load_signal(head_path, self.checkpoint_sha256, device)
                            if head_path is not None else None)

    @torch.no_grad()
    def infer(self, images, lidar, speed_mps, pose):
        """One frame of perception.

        images    rgb_front/left/right/rear as JPEG bytes (UDP camera) or RGB arrays
        lidar     Nx4 sweep in the sensor frame (UDP LidarReceiver output)
        speed_mps vehicle speed; the model uses its magnitude
        pose      x, y, yaw_grid_rad on the map grid at capture time (UDP EgoPose or dict)
        """
        begin = time.monotonic()
        inputs = self.inputs.build(images, lidar, speed_mps, self.device,
                                   velocity_input=self.contract['velocity_input'])
        detection, light, light_token = forward_perception(self.model, inputs)
        signal = (decode_signal(self.signal_head(light_token))[0]
                  if self.signal_head is not None else None)
        get = pose.get if isinstance(pose, dict) else lambda key: getattr(pose, key)
        boxes, counts = decode_world_boxes(detection[0], float(get('x')), float(get('y')),
                                           float(get('yaw_grid_rad')))
        return Perception(tuple(boxes), signal, decode_native_signal(light)[0], counts,
                          (time.monotonic() - begin) * 1e3)
