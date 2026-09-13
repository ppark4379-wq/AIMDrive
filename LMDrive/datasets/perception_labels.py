"""Perception targets for vision encoder pre-training, built from CARLA measurements.

generate_heatmap           300x300 BEV image of surrounding actors (5 px/m, 30 m range)
generate_det_data          50x50x8 detection map: objectness, offset, heading, box size, speed
generate_future_waypoints  300x300 BEV image of the expert's future route

Ported from timm/data/heatmap_utils.py and det_utils.py of the original LMDrive code.
"""
import math

import cv2
import numpy as np
from skimage.measure import block_reduce

VALUES = [255]
EXTENT = [0]


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = list(np.around(left_up).astype(int))
    left_down = list(np.around(left_down).astype(int))
    right_down = list(np.around(right_down).astype(int))
    right_up = list(np.around(right_up).astype(int))
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img


def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw


def generate_future_waypoints(measurements, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size), np.uint8)
    ego_x = measurements["gps_x"]
    ego_y = measurements["gps_y"]
    ego_theta = measurements["theta"] + np.pi / 2
    R = np.array([
        [np.cos(ego_theta), -np.sin(ego_theta)],
        [np.sin(ego_theta), np.cos(ego_theta)],
    ])
    for waypoint in measurements["future_waypoints"]:
        new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
        if new_loc[0] ** 2 + new_loc[1] ** 2 > (max_distance + 3) ** 2 * 2:
            break
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = tuple(np.around(new_loc).astype(int))
        img = cv2.circle(img, new_loc, 3, 255, -1)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def generate_heatmap(measurements, actors_data, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), int)
    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    R = np.array([
        [np.cos(ego_theta), -np.sin(ego_theta)],
        [np.sin(ego_theta), np.cos(ego_theta)],
    ])
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(actors_data[_id]["loc"][:2]) + TR.T.dot(
                    np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 1:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # do not draw the ego vehicle
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  # traffic lights are not drawn
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * (1.5 / box[0])  # enlarge pedestrians and bikes
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(act_img, loc, ori, box + EXTENT[i], VALUES[i],
                               pixels_per_meter, max_distance, color)
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img[:, :, 0]


def convert_grid_to_xy(i, j):
    return j - 24.5, 29.5 - i


def generate_det_data(heatmap, measurements, actors_data, pixels_per_meter=5, max_distance=30):
    traffic_heatmap = block_reduce(heatmap, block_size=(5, 5), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:50, 5:55]
    det_data = np.zeros((50, 50, 8))

    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    R = np.array([
        [np.cos(ego_theta), -np.sin(ego_theta)],
        [np.sin(ego_theta), np.cos(ego_theta)],
    ])
    need_deleted_ids = []
    for _id in actors_data:
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if dis <= 1 or dis >= (max_distance + 3) ** 2 * 2 or "box" not in actors_data[_id]:
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(50):  # vertical
        for j in range(50):  # horizontal
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id
            if min_id is None:
                det_data[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                continue
            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            speed = np.linalg.norm(actors_data[min_id]["vel"])
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dis) - np.sqrt(0.5)), 0.5)
            det_data[i][j] = np.array([
                prob,
                (loc[0] - center_x) / 3.5,
                (loc[1] - center_y) / 3.5,
                ori[0],
                ori[1],
                box[0] / 3.5,
                box[1] / 2.0,
                speed / 8.0,
            ])
    return det_data
