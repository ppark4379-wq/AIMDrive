#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..vehicle_state import VehicleState
import numpy as np


class PurePursuit(object):
    def __init__(self, lfd_gain, wheelbase, min_lfd, max_lfd):
        """Pure Pursuit 알고리즘을 이용한 Steering 계산"""
        self.lfd_gain = lfd_gain
        self.wheelbase = wheelbase
        self.min_lfd = min_lfd
        self.max_lfd = max_lfd

        self._path = []
        self._vehicle_state = VehicleState()
        self._debug_counter = 0

    @property
    def path(self):
        return self._path

    @property
    def vehicle_state(self):
        return self._vehicle_state

    @path.setter
    def path(self, path):
        self._path = path

    @vehicle_state.setter
    def vehicle_state(self, vehicle_state):
        self._vehicle_state = vehicle_state

    def calculate_steering_angle(self):
        lfd = self.lfd_gain * self._vehicle_state.velocity
        lfd = np.clip(lfd, self.min_lfd, self.max_lfd)

        steering_angle = 0.
        lookahead_pt = None
        lookahead_rotated = None
        theta = None
        for point in self._path:
            diff = point - self._vehicle_state.position
            rotated_diff = diff.rotate(-self.vehicle_state.yaw)
            if rotated_diff.x > 0:
                dis = rotated_diff.distance()
                if dis >= lfd:
                    lookahead_pt = point
                    lookahead_rotated = rotated_diff
                    theta = rotated_diff.angle
                    steering_angle = np.arctan2(2*self.wheelbase*np.sin(theta), lfd)
                    break

        # 디버그 로그 (sampling_rate=30Hz 가정, ~1초마다 한 번 출력)
        self._debug_counter += 1
        if self._debug_counter >= 30:
            self._debug_counter = 0
            pos = self._vehicle_state.position
            yaw_deg = np.degrees(self._vehicle_state.yaw)
            vel = self._vehicle_state.velocity
            print(f'[PP] pos=({pos.x:.2f},{pos.y:.2f}) yaw={yaw_deg:+.2f}deg vel={vel:.2f} lfd={lfd:.2f} path_len={len(self._path)}')
            if self._path:
                p0 = self._path[0]; pN = self._path[-1]
                print(f'[PP]   local_path first=({p0.x:.2f},{p0.y:.2f}) last=({pN.x:.2f},{pN.y:.2f})')
            if lookahead_pt is not None:
                print(f'[PP]   lookahead=({lookahead_pt.x:.2f},{lookahead_pt.y:.2f}) '
                      f'veh_frame=({lookahead_rotated.x:+.2f},{lookahead_rotated.y:+.2f}) '
                      f'theta={np.degrees(theta):+.2f}deg steering={np.degrees(steering_angle):+.2f}deg')
            else:
                print(f'[PP]   ⚠️ NO LOOKAHEAD: 모든 path 점이 차량 후방(x<=0) 또는 lfd 안쪽 → steering=0')

        return steering_angle