#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from ...interfaces.perception_data import is_ve_tracked_unknown
from ..jamming.motion import stationary as jamming_stationary
from ...interfaces.geometry import Point


# A re-plan (planning/blocked) between two validated static paths must not hand a stopped obstacle back to the
# path-distance ACC: that stops the vehicle and restarts its speed ramp every cycle.
STATIC_ACC_BYPASS_GRACE_S = 2.0


class ForwardObjectDetector(object):
    def __init__(self, static_object_list=[]):
        """전방 장애물 유무 확인 (static_object_list: 정지선 신호 / dynamic_object_list: 차량, 사람)"""
        self._static_object_list = static_object_list
        self._dynamic_object_list = []

    @property
    def dynamic_object_list(self):
        return self._dynamic_object_list

    @dynamic_object_list.setter
    def dynamic_object_list(self, dynamic_object_list):
        self._dynamic_object_list = dynamic_object_list

    def detect_object(self, vehicle_state, forward_only=True):
        tmp_translation = vehicle_state.position
        tmp_t = np.array([[np.cos(vehicle_state.yaw), -np.sin(vehicle_state.yaw), tmp_translation.x],
                          [np.sin(vehicle_state.yaw), np.cos(vehicle_state.yaw), tmp_translation.y],
                          [0, 0, 1]])
        tmp_det_t = np.array([[tmp_t[0][0], tmp_t[1][0], -(tmp_t[0][0]*tmp_translation.x+tmp_t[1][0]*tmp_translation.y)],
                              [tmp_t[0][1], tmp_t[1][1], -(tmp_t[0][1]*tmp_translation.x+tmp_t[1][1]*tmp_translation.y)],
                              [0, 0, 1]])

        object_info_dic_list = []
        for object_info in self._dynamic_object_list + self._static_object_list:
            global_position_vector = np.array([[object_info.position.x], [object_info.position.y], [1]])
            local_position_vector = tmp_det_t.dot(global_position_vector)
            local_position = Point(local_position_vector[0][0], local_position_vector[1][0])
            if not forward_only or local_position.x > 0:
                object_info_dic_list.append({"object_info": object_info, "local_position": local_position})

        return object_info_dic_list


def select_objects(self, vehicle_state, dynamic_object_list):
    self.forward_object_detector._dynamic_object_list = dynamic_object_list
    forward_object_info_dic_list = self.forward_object_detector.detect_object(
        vehicle_state, forward_only=True
    )
    # 목표 차선 뒤에서 접근하는 차량도 미래 OBB 검사에 포함합니다.
    planning_object_info_dic_list = self.forward_object_detector.detect_object(
        vehicle_state, forward_only=False
    )
    return forward_object_info_dic_list, planning_object_info_dic_list


def _legacy_acc_entries(self, vehicle_state, lattice_result, forward_entries):
    """Avoid applying path-distance vehicle ACC twice on mapped highways.

    The highway lane planner already associates vehicles with lane links,
    checks fallback collisions, and returns an ACC speed ceiling.  The
    legacy path-distance detector can mistake an adjacent/old-lane vehicle
    for a lead vehicle immediately after a lane change because the two
    paths remain close.  Keep pedestrians and traffic lights in the legacy
    controller, but let the lane planner be the sole vehicle authority.
    """
    static_details = getattr(lattice_result, "debug_details", {}).get("static_avoidance", {})
    jamming_details = getattr(lattice_result, "debug_details", {})
    if jamming_details.get('jamming_dynamic', {}).get('speed_authority'):
        # The moving-object overlay evaluates the actually commanded path
        # and both target-side/front traffic. Keep people and signals in
        # legacy ACC; avoid a second vehicle stop on the old path.
        return [entry for entry in forward_entries if entry['object_info'].type not in (1,2)
                and not is_ve_tracked_unknown(entry['object_info'])]
    if (jamming_details.get('jamming_corridor', {}).get('standing_only')
            and static_details.get("path_validated", False)):
        # Jamming section without movers: the accumulated-map planner already keeps clear of every VE object, which
        # the tracker may still call 'uncertain' (unknown speed). Keep people and signals in legacy ACC.
        return [entry for entry in forward_entries if not is_ve_tracked_unknown(entry['object_info'])
                and not (getattr(entry['object_info'], 'type', -1) in (1, 2)
                         and abs(getattr(entry['object_info'], 'velocity', 0.)) < .2)]
    now = time.monotonic()
    static_active = (getattr(lattice_result, "mode", "") == "static_obstacle"
                     and getattr(lattice_result, "active", False))
    if static_active and static_details.get("path_validated", False):
        self._static_acc_bypass_until = now + STATIC_ACC_BYPASS_GRACE_S
    elif not static_active:
        self._static_acc_bypass_until = 0.
    # The grace only covers a re-plan in which the static planner itself still allows motion; a path that
    # became blocked (acc_target_velocity 0) hands the objects back to ACC at once.
    if static_active and (static_details.get("path_validated", False)
                          or (now <= getattr(self, "_static_acc_bypass_until", 0.)
                              and getattr(lattice_result, "acc_target_velocity", 0.) > 0.)):
        # Only bypass static vehicle ACC after validating the complete swept
        # path. Moving vehicles, people and traffic signals keep their stops.
        return [entry for entry in forward_entries if not (
            (getattr(entry["object_info"], "is_static", False)
             or (jamming_details.get('jamming') and jamming_stationary(entry['object_info'])))
            and (getattr(entry["object_info"], "type", -1) in (1, 2)
                 or is_ve_tracked_unknown(entry["object_info"]))
            and abs(getattr(entry["object_info"], "velocity", 0.0)) < 0.2
        )]
    if getattr(lattice_result, "mode", "") != "highway_lane":
        return [
            entry
            for entry in forward_entries
            if not self.lattice_planning_process.roundabout_entry_should_ignore_legacy_acc(
                vehicle_state, entry["object_info"]
            )
        ]
    return [
        entry
        for entry in forward_entries
        if getattr(entry["object_info"], "type", -1) not in (1, 2)
        and not is_ve_tracked_unknown(entry["object_info"])
    ]

