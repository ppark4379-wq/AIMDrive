"""LMDrive vision encoder: multi-view cameras + LiDAR -> perception outputs or visual tokens.

This is the "Memfuser" model of LMDrive (timm/models/memfuser.py).  Parameter
names are unchanged, so the released LMDrive vision encoder checkpoint loads
with strict=True.

Inputs, a dict of tensors with batch size B
  rgb_front    [B, 3, 224, 224]  front camera
  rgb_left     [B, 3, 128, 128]  left camera
  rgb_right    [B, 3, 128, 128]  right camera
  rgb_rear     [B, 3, 128, 128]  rear camera
  rgb_center   [B, 3, 128, 128]  un-resized centre crop of the front camera
  lidar        [B, 40000, 4]     points from data.src.loaders.lidar.lidar_to_raw_features
  num_points   [B]               valid points per sample
  velocity     [B, 1]            ego speed in m/s
  target_point [B, 2]            next route point in the ego frame (pre-training only)

Outputs
  return_feature=False, stage 1 pre-training
    detection map  [B, 2500, 8]  per BEV cell: objectness, offset(2), heading(2), box(2), speed
    waypoints      [B, 5, 2]
    traffic light  [B, 2]        logits, class 0 means a red light affects the ego vehicle
    stop sign      [B, 2]        logits
    BEV features   [B, 2500, C]
  return_feature=True, stage 2 frozen encoder feeding the language model
    visual tokens  [B, 106, C]   100 pooled BEV + 1 traffic light + 5 waypoint tokens
"""
import torch
import torch.nn.functional as F
from torch import nn

from .pointpillars import PointPillarsLidarEncoder
from .resnet import create_resnet_features
from .transformer import (
    PositionEmbeddingSine,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

# Released LMDrive variants.  "memfuser_baseline_e1d3" is the published R50 encoder.
VISION_ENCODER_VARIANTS = {
    "memfuser_baseline": dict(enc_depth=2, dec_depth=4, rgb_backbone="resnet50d"),
    "memfuser_baseline_e3d3": dict(enc_depth=3, dec_depth=3, rgb_backbone="resnet50d"),
    "memfuser_baseline_e2d2": dict(enc_depth=2, dec_depth=2, rgb_backbone="resnet50d"),
    "memfuser_baseline_e1d3": dict(enc_depth=1, dec_depth=3, rgb_backbone="resnet50d"),
    "memfuser_baseline_e1d3_r26": dict(enc_depth=1, dec_depth=3, rgb_backbone="resnet26d"),
}


class CameraPatchEmbed(nn.Module):
    """CNN feature map -> 1x1 projected tokens plus one global (mean) token."""

    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Conv2d(backbone.out_channels, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.proj(self.backbone(x))
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x


class GRUWaypointsPredictor(nn.Module):
    """Decodes 5 waypoint queries into cumulative (x, y) offsets, seeded by the target point."""

    def __init__(self, input_dim, waypoints=5):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        return torch.cumsum(output, 1)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        enc_depth=1,
        dec_depth=3,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        normalize_before=False,
        rgb_backbone="resnet50d",
        pretrained_backbone=True,
        pretrained_backbone_path=None,
        with_right_left_sensors=True,
        with_rear_sensor=True,
        with_center_sensor=True,
        use_view_embed=True,
        return_feature=False,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.with_right_left_sensors = with_right_left_sensors
        self.with_rear_sensor = with_rear_sensor
        self.with_center_sensor = with_center_sensor
        self.use_view_embed = use_view_embed
        self.return_feature = return_feature
        self.attn_mask = None

        # Construction order matches the original so seeded initialisation is identical.
        self.rgb_backbone = create_resnet_features(
            rgb_backbone, pretrained=pretrained_backbone, weights_path=pretrained_backbone_path or None)
        self.lidar_backbone = PointPillarsLidarEncoder(
            num_input=9, num_features=(32, 32), min_x=-25, max_x=35, min_y=-30, max_y=30,
            pixels_per_meter=4, output_features=embed_dim,
        )
        self.rgb_patch_embed = CameraPatchEmbed(self.rgb_backbone, embed_dim)

        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 6))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 5, 1))
        self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 6))
        self.query_embed = nn.Parameter(torch.zeros(6, 1, embed_dim))

        self.waypoints_generator = GRUWaypointsPredictor(embed_dim)
        self.traffic_light_pred_head = nn.Linear(embed_dim, 2)
        self.stop_sign_head = nn.Linear(embed_dim, 2)
        self.traffic_pred_head = nn.Sequential(
            nn.Linear(embed_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, nn.GELU, normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)
        decoder_layer = TransformerDecoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, nn.GELU, normalize_before)
        self.decoder = TransformerDecoder(
            decoder_layer, dec_depth, nn.LayerNorm(embed_dim), return_intermediate=False)
        self.velocity_fc = nn.Linear(1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)
        nn.init.uniform_(self.query_pos_embed)

    def _camera_tokens(self, image, view_index, global_index):
        tokens, global_token = self.rgb_patch_embed(image)
        if self.use_view_embed:
            tokens = tokens + self.view_embed[:, :, view_index:view_index + 1, :] + self.position_encoding(tokens)
        else:
            tokens = tokens + self.position_encoding(tokens)
        tokens = tokens.flatten(2).permute(2, 0, 1)
        global_token = (global_token + self.view_embed[:, :, view_index, :]
                        + self.global_embed[:, :, global_index:global_index + 1])
        return [tokens, global_token.permute(2, 0, 1)]

    def forward_features(self, front_image, left_image, right_image, rear_image,
                         front_center_image, lidar, num_points):
        features = self._camera_tokens(front_image, 0, 0)
        if self.with_right_left_sensors:
            features += self._camera_tokens(left_image, 1, 1)
            features += self._camera_tokens(right_image, 2, 2)
        if self.with_center_sensor:
            features += self._camera_tokens(front_center_image, 3, 3)
        if self.with_rear_sensor:
            # The rear view uses global embedding slot 5, as in the original model.
            features += self._camera_tokens(rear_image, 4, 5)

        lidar_token = self.lidar_backbone(lidar, num_points)  # [B, C, 50, 50]
        lidar_token = lidar_token + self.position_encoding(lidar_token)
        lidar_token = lidar_token.flatten(2).permute(2, 0, 1)
        return torch.cat(features, 0), lidar_token

    def forward(self, x):
        velocity = x["velocity"].view(1, -1, 1)
        velocity_feature = self.velocity_fc(velocity).repeat(6, 1, 1)

        features, lidar_token = self.forward_features(
            x["rgb_front"], x["rgb_left"], x["rgb_right"], x["rgb_rear"],
            x["rgb_center"], x["lidar"], x["num_points"],
        )

        bs = x["rgb_front"].shape[0]
        tgt = self.position_encoding(torch.ones((bs, 1, 50, 50), device=x["rgb_front"].device))
        tgt = tgt.flatten(2)
        tgt = torch.cat([tgt, self.query_pos_embed.repeat(bs, 1, 1)], 2)
        tgt = tgt.permute(2, 0, 1)

        memory = self.encoder(features, mask=self.attn_mask)
        query_embed = self.query_embed.repeat(1, bs, 1) + velocity_feature
        query = torch.cat([lidar_token, query_embed], 0)
        hs = self.decoder(query, memory, query_pos=tgt)[0]
        hs = hs.permute(1, 0, 2)  # [B, 2500 + 6, C]

        traffic_feature = hs[:, :2500]
        traffic_light_state_feature = hs[:, 2500]
        stop_sign_feature = hs[:, 2500]
        waypoints_feature = hs[:, 2501:2506]

        if self.return_feature:
            traffic_feature = traffic_feature.reshape(bs, 50, 50, -1).permute(0, 3, 1, 2)
            traffic_feature = F.adaptive_avg_pool2d(traffic_feature, (10, 10)).view(bs, -1, 100).permute(0, 2, 1)
            return torch.cat([traffic_feature, traffic_light_state_feature.view(bs, 1, -1), waypoints_feature], 1)

        waypoints = self.waypoints_generator(waypoints_feature, x["target_point"])
        traffic_light_state = self.traffic_light_pred_head(traffic_light_state_feature)
        stop_sign = self.stop_sign_head(stop_sign_feature)

        velocity = velocity.view(-1, 1, 1).repeat(1, 2500, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
        traffic = self.traffic_pred_head(traffic_feature_with_vel)
        return traffic, waypoints, traffic_light_state, stop_sign, traffic_feature


def create_vision_encoder(model_config, **overrides):
    """Build the encoder from configs/vision_encoder_model.yaml (a dict) plus overrides.

    The optional "name" field must be one of VISION_ENCODER_VARIANTS; a trailing
    "_return_feature" (LMDrive naming) switches on token output.
    """
    settings = dict(model_config or {})
    settings.update(overrides)
    name = settings.pop("name", None)
    if name:
        if name.endswith("_return_feature"):
            name = name[: -len("_return_feature")]
            settings.setdefault("return_feature", True)
        if name not in VISION_ENCODER_VARIANTS:
            raise KeyError("Unknown vision encoder %r, choose from %s" % (name, sorted(VISION_ENCODER_VARIANTS)))
        for key, value in VISION_ENCODER_VARIANTS[name].items():
            if settings.setdefault(key, value) != value:
                raise ValueError("%s fixes %s=%r but the config sets %r" % (name, key, value, settings[key]))
    return VisionEncoder(**settings)
