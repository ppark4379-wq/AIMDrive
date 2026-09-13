"""Loss of the vision encoder pre-training stage (train_pretrain.py of LMDrive).

The stage 2 driving model computes its own waypoint and end-flag losses in
models.lmdrive.LMDriveModel.forward.
"""
from collections import OrderedDict

import torch
from torch import nn

VISION_ENCODER_LOSS_WEIGHTS = OrderedDict(
    loss_traffic=0.5,
    loss_waypoints=0.5,
    loss_velocity=0.05,
    loss_traffic_light_state=0.1,
    loss_stop_sign=0.01,
)


class DetectionMapLoss:
    """Loss on the [B, 2500, 8] BEV detection map (LAVLoss in LMDrive).

    Channel 0 is objectness (BCE weighted towards hard cells); the remaining
    channels are L1 regressions weighted by the target objectness.
    Returns (detection loss, speed loss).
    """

    def __init__(self):
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.loc_criterion = nn.L1Loss(reduction="none")
        self.ori_criterion = nn.L1Loss(reduction="none")
        self.box_criterion = nn.L1Loss(reduction="none")

    def __call__(self, output, target):
        prob = target[:, :, 0:1]
        prob_mean = prob.mean()
        prob_mean = torch.maximum(prob_mean, torch.ones_like(prob_mean) * 1e-7)
        prob_det = torch.sigmoid(output[:, :, 0] * (1 - 2 * target[:, :, 0]))

        det_loss = (prob_det * self.prob_criterion(output[:, :, 0], target[:, :, 0])).mean() / prob_det.mean()
        loc_loss = (prob * self.loc_criterion(output[:, :, 1:3], target[:, :, 1:3])).mean() / prob_mean
        box_loss = (prob * self.box_criterion(output[:, :, 3:5], target[:, :, 3:5])).mean() / prob_mean
        ori_loss = (prob * self.ori_criterion(output[:, :, 5:7], target[:, :, 5:7])).mean() / prob_mean
        spd_loss = (prob * self.ori_criterion(output[:, :, 7:8], target[:, :, 7:8])).mean() / prob_mean

        det_loss = 0.4 * det_loss + 0.2 * loc_loss + 0.2 * box_loss + 0.2 * ori_loss
        return det_loss, spd_loss


class VisionEncoderLoss:
    """Weighted sum of detection, waypoint, speed, traffic-light and stop-sign losses.

    outputs: (detection map, waypoints, traffic light logits, stop sign logits, BEV features)
    targets: see datasets.carla_perception_dataset
    """

    def __init__(self):
        self.detection = DetectionMapLoss()
        self.waypoints = nn.L1Loss()
        self.classification = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loss_traffic, loss_velocity = self.detection(outputs[0], targets[4])
        loss_waypoints = self.waypoints(outputs[1], targets[1])
        loss_traffic_light_state = self.classification(outputs[2], targets[3])
        loss_stop_sign = self.classification(outputs[3], targets[6])
        loss = (
            loss_traffic * 0.5
            + loss_waypoints * 0.5
            + loss_velocity * 0.05
            + loss_traffic_light_state * 0.1
            + loss_stop_sign * 0.01
        )
        return OrderedDict(
            loss=loss,
            loss_traffic=loss_traffic,
            loss_velocity=loss_velocity,
            loss_waypoints=loss_waypoints,
            loss_traffic_light_state=loss_traffic_light_state,
            loss_stop_sign=loss_stop_sign,
        )
