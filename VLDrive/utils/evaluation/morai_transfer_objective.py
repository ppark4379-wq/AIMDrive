"""LMDrive-compatible detection, waypoint and masked two-class light losses.

Weights retain the native task scale (.5 detection, .5 waypoint, .05 object
speed, .1 light). Waypoint L1 averages the five explicitly timed targets; the
upstream ten-element waypoint weighting does not match this five-query head.
Stop-sign labels are unavailable and its head is excluded.

The checked-in LMDrive training path selects ``LAVLoss``.  Earlier MORAI pilot
runs deliberately used the alternative ``MVTL1Loss`` formulation that remains
in that source file but is commented out at loss construction.  Keep the two
policies explicit so a checkpoint cannot silently change objective semantics.
"""
import torch
import torch.nn.functional as F

DETECTION_LOSS_POLICIES = ("mvt_l1_v1", "lmdrive_lav_v1")


def _mvt_l1_loss(prediction, target, zero):
    positive = target[:, :, 0] >= .01
    negative = target[:, :, 0] < .01
    pos_loss = (F.l1_loss(prediction[:, :, 0][positive], target[:, :, 0][positive])
                if positive.any() else zero)
    neg_loss = (F.l1_loss(prediction[:, :, 0][negative], target[:, :, 0][negative])
                if negative.any() else zero)
    box_loss = (F.l1_loss(prediction[positive][:, 1:7], target[positive][:, 1:7])
                if positive.any() else zero)
    detection = .25 * (pos_loss + neg_loss) + .5 * box_loss
    speed = (F.l1_loss(prediction[positive][:, 7], target[positive][:, 7])
             if positive.any() else zero)
    return detection, speed


def _lmdrive_lav_loss(prediction, target, zero):
    """The LAVLoss selected by the checked-in LMDrive ``train_pretrain.py``."""
    probability = target[:, :, 0:1]
    probability_mean = probability.mean().clamp_min(1e-7)
    hard_example_weight = torch.sigmoid(
        prediction[:, :, 0] * (1 - 2 * target[:, :, 0]))
    probability_loss = (
        hard_example_weight
        * F.binary_cross_entropy_with_logits(
            prediction[:, :, 0], target[:, :, 0], reduction="none")
    ).mean() / hard_example_weight.mean().clamp_min(1e-7)

    def positive_l1(first, last):
        error = F.l1_loss(
            prediction[:, :, first:last], target[:, :, first:last], reduction="none")
        return (probability * error).mean() / probability_mean

    location_loss = positive_l1(1, 3)
    box_loss = positive_l1(3, 5)
    orientation_loss = positive_l1(5, 7)
    speed_loss = positive_l1(7, 8)
    detection = (.4 * probability_loss + .2 * location_loss
                 + .2 * box_loss + .2 * orientation_loss)
    return detection, speed_loss


def multitask_loss(outputs, targets, include_traffic_light=True, include_objects=True,
                   detection_loss_policy="mvt_l1_v1"):
    if detection_loss_policy not in DETECTION_LOSS_POLICIES:
        raise ValueError("Unknown detection loss policy: " + str(detection_loss_policy))
    det, waypoints, light = outputs[:3]
    valid = targets["object_valid"].bool() & include_objects
    prediction, target = det[valid], targets["detection"][valid]
    zero = det.sum() * 0
    det_loss = speed_loss = zero
    if prediction.numel():
        if detection_loss_policy == "lmdrive_lav_v1":
            det_loss, speed_loss = _lmdrive_lav_loss(prediction, target, zero)
        else:
            det_loss, speed_loss = _mvt_l1_loss(prediction, target, zero)
    valid_wp = targets.get("waypoint_valid", torch.ones(len(waypoints), dtype=torch.bool, device=waypoints.device))
    wp_loss = (F.l1_loss(waypoints[valid_wp], targets["waypoints"][valid_wp])
               if valid_wp.any() else waypoints.sum() * 0)
    valid_light = targets["traffic_light"] >= 0
    light_loss = (F.cross_entropy(light[valid_light], targets["traffic_light"][valid_light])
                  if include_traffic_light and valid_light.any() else light.sum() * 0)
    total = .5 * det_loss + .5 * wp_loss + .05 * speed_loss + .1 * light_loss
    return total, {"detection": det_loss, "waypoint_l1": wp_loss,
                   "object_speed": speed_loss, "traffic_light_ce": light_loss}
