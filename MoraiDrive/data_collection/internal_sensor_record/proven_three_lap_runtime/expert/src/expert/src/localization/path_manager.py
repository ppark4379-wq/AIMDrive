#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .point import Point
import numpy as np
from math import sqrt,pi 


class PathManager:
    # 이전 waypoint 기준 탐색 범위 (뒤/앞). 루프/스파이럴 구간처럼 경로가
    # 물리적으로 가까이 지나가는 곳에서 전체탐색을 하면 진행 방향과 무관한
    # 먼 인덱스로 순간 이동(aliasing)하는 문제가 있어 진행 범위로 제한한다.
    _SEARCH_WINDOW_BACK = 20
    _SEARCH_WINDOW_FORWARD = 120

    def __init__(self, path, is_closed_path, local_path_size):
        self.path = path
        self.is_closed_path = is_closed_path
        self.local_path_size = local_path_size
        self.velocity_profile = []
        self.current_waypoint = 0
        self._last_waypoint_idx = None
        self._distance_to_end = [0.0] * len(path)
        for i in range(len(path) - 2, -1, -1):
            dx = path[i + 1].x - path[i].x
            dy = path[i + 1].y - path[i].y
            self._distance_to_end[i] = (
                self._distance_to_end[i + 1] + sqrt(dx*dx + dy*dy)
            )
        self.terminal_deceleration = 2.0

    def set_velocity_profile(
        self,
        max_velocity,
        road_friction,
        window_size,
        speed_limit_zones=None,
        speed_limit_deceleration=3.0,
        terminal_deceleration=2.0,
    ):
        self.terminal_deceleration = max(0.1, float(terminal_deceleration))
        default_max_velocity = float(max_velocity) / 3.6
        path_size = len(self.path)
        if path_size == 0:
            self.velocity_profile = []
            self.speed_limit_profile = []
            return

        window_size = max(1, min(int(window_size), (path_size - 1) // 2))
        x = np.fromiter((point.x for point in self.path), dtype=float, count=path_size)
        y = np.fromiter((point.y for point in self.path), dtype=float, count=path_size)
        center_idx = np.arange(path_size)

        if self.is_closed_path:
            start_idx = (center_idx - window_size) % path_size
            end_idx = (center_idx + window_size) % path_size
        else:
            radius = np.minimum.reduce((np.full(path_size, window_size), center_idx, path_size - 1 - center_idx))
            start_idx = center_idx - radius
            end_idx = center_idx + radius

        d_st_x = x[start_idx] - x
        d_st_y = y[start_idx] - y
        d_ed_x = x[end_idx] - x
        d_ed_y = y[end_idx] - y

        dcom = 2.0 * (d_st_x * d_ed_y - d_st_y * d_ed_x)
        d_st2 = d_st_x * d_st_x + d_st_y * d_st_y
        d_ed2 = d_ed_x * d_ed_x + d_ed_y * d_ed_y

        with np.errstate(divide='ignore', invalid='ignore'):
            u1 = (d_ed_y * d_st2 - d_st_y * d_ed2) / dcom
            u2 = (d_st_x * d_ed2 - d_ed_x * d_st2) / dcom
            radius = np.sqrt(u1 * u1 + u2 * u2)
            velocity_profile = np.sqrt(radius * 9.8 * road_friction)

        velocity_profile = np.where(
            np.isfinite(velocity_profile), velocity_profile, np.inf
        )
        speed_limits = np.full(path_size, default_max_velocity, dtype=float)
        self.speed_zone_indices = []
        for zone in speed_limit_zones or []:
            start_index = self._nearest_path_index(zone["start_point"])
            end_index = self._nearest_path_index(zone["end_point"])
            zone_limit = float(zone["max_velocity"]) / 3.6

            if start_index <= end_index:
                speed_limits[start_index:end_index + 1] = zone_limit
            elif self.is_closed_path:
                speed_limits[start_index:] = zone_limit
                speed_limits[:end_index + 1] = zone_limit
            else:
                raise ValueError(
                    "Open-path speed zone ends before it starts: {} -> {}".format(
                        start_index, end_index
                    )
                )

            self.speed_zone_indices.append({
                "name": zone.get("name", "unnamed"),
                "start_link_id": zone.get("start_link_id"),
                "end_link_id": zone.get("end_link_id"),
                "start_index": start_index,
                "end_index": end_index,
                "max_velocity": float(zone["max_velocity"]),
            })
            print(
                "[VelocityProfile] zone {}: path_index {} -> {}, limit={}km/h".format(
                    zone.get("name", "unnamed"),
                    start_index,
                    end_index,
                    float(zone["max_velocity"]),
                )
            )

        velocity_profile = np.minimum(velocity_profile, speed_limits)
        velocity_profile = self._apply_deceleration_limit(
            velocity_profile,
            float(speed_limit_deceleration),
        )
        self.speed_limit_profile = speed_limits.tolist()

        self.velocity_profile = velocity_profile.tolist()

    def _nearest_path_index(self, point):
        px, py = float(point[0]), float(point[1])
        min_distance = float('inf')
        nearest_index = 0
        for index, path_point in enumerate(self.path):
            dx = float(path_point.x) - px
            dy = float(path_point.y) - py
            distance = dx * dx + dy * dy
            if distance < min_distance:
                min_distance = distance
                nearest_index = index
        return nearest_index

    def _apply_deceleration_limit(self, velocity_profile, deceleration):
        if deceleration <= 0.0 or len(velocity_profile) < 2:
            return velocity_profile

        profile = velocity_profile.copy()
        path_size = len(profile)
        passes = 2 if self.is_closed_path else 1
        for _ in range(passes):
            start_index = path_size - 1 if self.is_closed_path else path_size - 2
            for index in range(start_index, -1, -1):
                next_index = (index + 1) % path_size
                dx = float(self.path[next_index].x) - float(self.path[index].x)
                dy = float(self.path[next_index].y) - float(self.path[index].y)
                segment_length = sqrt(dx * dx + dy * dy)
                allowed_velocity = sqrt(
                    profile[next_index] * profile[next_index]
                    + 2.0 * deceleration * segment_length
                )
                if profile[index] > allowed_velocity:
                    profile[index] = allowed_velocity
        return profile

    def _search_indices(self, path_size):
        # 이전 waypoint를 모르면(최초 호출 등) 전체 탐색으로 초기 위치를 잡는다.
        if self._last_waypoint_idx is None:
            return range(path_size)

        start = self._last_waypoint_idx - self._SEARCH_WINDOW_BACK
        end = self._last_waypoint_idx + self._SEARCH_WINDOW_FORWARD

        if self.is_closed_path:
            return (i % path_size for i in range(start, end + 1))

        start = max(0, start)
        end = min(path_size - 1, end)
        return range(start, end + 1)

    def get_local_path(self, vehicle_state):
        path_size = len(self.path)
        min_distance = float('inf')
        current_waypoint = self._last_waypoint_idx or 0
        for i in self._search_indices(path_size):
            point = self.path[i]
            dx = point.x - vehicle_state.position.x
            dy = point.y - vehicle_state.position.y
            distance = dx*dx + dy*dy
            if distance < min_distance:
                min_distance = distance
                current_waypoint = i

        self.current_waypoint = current_waypoint
        self._last_waypoint_idx = current_waypoint

        if current_waypoint + self.local_path_size < len(self.path):
            local_path = self.path[current_waypoint:current_waypoint + self.local_path_size]
        else:
            local_path = self.path[current_waypoint:]
            # 연결된 경로 (closed path) 일 경우, 경로 끝과 처음을 이어준다.
            if self.is_closed_path:
                local_path += self.path[:self.local_path_size + len(self.path) - current_waypoint]

        # planned_velocity를 "현재 지점"의 목표속도 1개만 보면, 급커브 진입 직전에야
        # 감속 필요를 알게 돼서 제동거리가 모자라 커브를 못 넘긴다(코너 컷팅 -> 이탈).
        # local_path와 같은 구간(전방 look-ahead) 내 최소 목표속도를 미리 반영해서
        # 급커브 진입 전에 충분히 감속할 시간/거리를 확보한다.
        window_indices = range(current_waypoint, current_waypoint + len(local_path))
        planned_velocity = min(
            self.velocity_profile[i % path_size] for i in window_indices
        )
        if not self.is_closed_path:
            # Do not put zero-speed waypoints inside the look-ahead window: that
            # made the controller stop as soon as the END entered the 50-point
            # local path (about 20 m early). Instead, cap speed using the actual
            # remaining route distance and reach zero only at the final point.
            remaining_distance = self._distance_to_end[current_waypoint]
            terminal_velocity = sqrt(
                2.0 * self.terminal_deceleration * remaining_distance
            )
            planned_velocity = min(planned_velocity, terminal_velocity)

        return local_path, planned_velocity
