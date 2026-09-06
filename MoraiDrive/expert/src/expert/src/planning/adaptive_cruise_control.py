#!/usr/bin/env python
# -*- coding: utf-8 -*-


class AdaptiveCruiseControl:
    def __init__(
        self,
        velocity_gain,
        distance_gain,
        time_gap,
        vehicle_length,
        pedestrian_lateral_threshold=6.0,
        vehicle_lateral_threshold=2.5,
        traffic_light_lateral_threshold=4.0,
        pedestrian_longitudinal_max=35.0,
    ):
        self.velocity_gain = velocity_gain
        self.distance_gain = distance_gain
        self.time_gap = time_gap
        self.vehicle_length = vehicle_length
        self.pedestrian_lateral_threshold = pedestrian_lateral_threshold
        self.vehicle_lateral_threshold = vehicle_lateral_threshold
        self.traffic_light_lateral_threshold = traffic_light_lateral_threshold
        self.pedestrian_longitudinal_max = pedestrian_longitudinal_max

        self.object_type = None
        self.object_distance = 0
        self.object_velocity = 0

    @staticmethod
    def _distance_to_path_segments(point, path):
        if not path:
            return float('inf')
        if len(path) == 1:
            return point.distance(path[0])

        min_dist = float('inf')
        px, py = point.x, point.y
        for start, end in zip(path[:-1], path[1:]):
            sx, sy = start.x, start.y
            ex, ey = end.x, end.y
            dx = ex - sx
            dy = ey - sy
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                dist = point.distance(start)
            else:
                t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                proj_x = sx + t * dx
                proj_y = sy + t * dy
                dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def check_object(self, local_path, object_info_dic_list, current_traffic_light):
        self.object_type = None
        min_relative_distance = float('inf')
        for object_info_dic in object_info_dic_list:
            object_info = object_info_dic['object_info']
            local_position = object_info_dic['local_position']

            object_type = object_info.type
            if object_type == 0:
                distance_threshold = self.pedestrian_lateral_threshold
                if local_position.x > self.pedestrian_longitudinal_max:
                    continue
            elif object_type in [1, 2]:
                distance_threshold = self.vehicle_lateral_threshold
            elif object_type == 3:
                if current_traffic_light and object_info.name == current_traffic_light[0] and not current_traffic_light[1] in [16, 48]:
                    distance_threshold = self.traffic_light_lateral_threshold
                else:
                    continue
            else:
                continue

            distance_from_path = self._distance_to_path_segments(object_info.position, local_path)
            if distance_from_path < distance_threshold:
                relative_distance = max(0.0, local_position.x)
                if relative_distance < min_relative_distance:
                    min_relative_distance = relative_distance
                    self.object_type = object_type
                    self.object_distance = relative_distance - self.vehicle_length
                    self.object_velocity = object_info.velocity

    def get_target_velocity(self, ego_vel, target_vel):
        out_vel = target_vel

        if self.object_type == 0:
            print("ACC ON_Person")
            default_space = 8
        elif self.object_type in [1, 2]:
            print("ACC ON_Vehicle")
            default_space = 5
        elif self.object_type == 3:
            print("ACC ON_Traffic Light")
            default_space = 1
        else:
            return out_vel

        velocity_error = ego_vel - self.object_velocity

        safe_distance = ego_vel * self.time_gap + default_space
        distance_error = safe_distance - self.object_distance

        acceleration = -(self.velocity_gain * velocity_error + self.distance_gain * distance_error)
        out_vel = min(ego_vel + acceleration, target_vel)

        if self.object_type == 0 and (distance_error > 0):
            out_vel = out_vel - 5.

        if self.object_distance < default_space:
            out_vel = 0.

        return out_vel
