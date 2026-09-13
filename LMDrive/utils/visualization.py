"""TensorBoard images for vision encoder training: camera views, detection maps and waypoints."""
from collections import OrderedDict

import cv2
import numpy as np
import torch

from datasets.perception_labels import add_rect, convert_grid_to_xy

# Undo the per-channel scaling of the detection map targets.
DETECTION_SCALE = np.array([1.0, 3.5, 3.5, 1.0, 1.0, 3.5, 2.0, 8.0])


def denormalize_image(image):
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    return image * std + mean


def find_peak_cells(det_data):
    padded = np.zeros((52, 52, 8))
    padded[1:51, 1:51] = det_data
    peaks = []
    for i in range(1, 51):
        for j in range(1, 51):
            prob = padded[i, j, 0]
            if prob > 0.9 or (prob > 0.4 and prob > padded[i, j - 1, 0] and prob > padded[i, j + 1, 0]
                              and prob > padded[i - 1, j, 0] and prob > padded[i + 1, j, 0]):
                peaks.append((i - 1, j - 1))
    return peaks


def render_detection_map(det_data, pixels_per_meter=5, max_distance=30, t=0):
    """[50, 50, 8] detection map -> 300x300 image with one box per detected object."""
    det_data = det_data * DETECTION_SCALE
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for i, j in find_peak_cells(det_data):
        center_x, center_y = convert_grid_to_xy(i, j)
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        ori = np.array([det_data[i, j, 3], det_data[i, j, 4]])
        loc_x = center_x + det_data[i, j, 1] + t * det_data[i, j, 7] * ori[0]
        loc_y = center_y + det_data[i, j, 2] - t * det_data[i, j, 7] * ori[1]
        box = np.array(det_data[i, j, 5:7])
        if box[0] < 1.5:
            box = box * 1.5
        add_rect(act_img, np.array([loc_x, -loc_y]), ori, box, 255, pixels_per_meter, max_distance,
                 np.array([1, 1, 1]))
        img = img + np.clip(act_img, 0, 255)
    return np.clip(img, 0, 255)[:, :, 0].astype(np.uint8)


def render_waypoints(waypoints, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size), np.uint8)
    for waypoint in waypoints:
        location = tuple(np.around(waypoint * pixels_per_meter + pixels_per_meter * max_distance).astype(int))
        img = cv2.circle(img, location, 3, 255, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def vision_encoder_images(inputs, outputs, targets):
    """First sample of a batch as CHW images for SummaryWriter.add_image."""
    def rendered(array):
        return torch.clip(torch.tensor(array[:250, 25:275]), 0, 255).view(1, 250, 250)

    detection = outputs[0][0].detach().float()
    return OrderedDict([
        ("front_view", denormalize_image(inputs["rgb_front"][0].float())),
        ("left_view", denormalize_image(inputs["rgb_left"][0].float())),
        ("right_view", denormalize_image(inputs["rgb_right"][0].float())),
        ("rear_view", denormalize_image(inputs["rgb_rear"][0].float())),
        ("front_center_view", denormalize_image(inputs["rgb_center"][0].float())),
        ("pred_traffic", torch.clip(detection, 0, 1).view(1, 50, 50, 8)[:, :, :, 0]),
        ("pred_traffic_render", rendered(render_detection_map(detection.view(50, 50, 8).cpu().numpy()))),
        ("gt_traffic", torch.clip(targets[4][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0]),
        ("gt_highres_traffic", torch.clip(targets[0][0], 0, 1)),
        ("pred_waypoints", rendered(render_waypoints(outputs[1][0].detach().float().cpu().numpy()))),
        ("gt_waypoints", torch.clip(targets[5][0], 0, 1)),
    ])
