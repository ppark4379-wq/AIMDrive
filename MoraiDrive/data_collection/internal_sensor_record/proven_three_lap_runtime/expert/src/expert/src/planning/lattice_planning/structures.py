#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class PolynomialCoefficients:
    """Fifth-order lateral polynomial coefficients."""

    def __init__(self, a0=0.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0):
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a3 = float(a3)
        self.a4 = float(a4)
        self.a5 = float(a5)


class CandidatePath:
    """One sampled lattice path and its individual cost terms."""

    def __init__(self, offset=0.0, layer="short"):
        self.points = []
        self.offset = float(offset)
        self.layer = layer
        self.obstacle_cost = 0.0
        self.curvature_cost = 0.0
        self.offset_cost = 0.0
        self.offset_change_cost = 0.0
        self.right_preference_cost = 0.0
        self.cost = 0.0
        self.valid = True


class LatticePlanningResult:
    """Output passed from the planning process to AutonomousDriving."""

    def __init__(
        self,
        path,
        active=False,
        obstacle_detected=False,
        long_obstacle_detected=False,
        longitudinal_speed_scale=1.0,
        mode="inactive",
        acc_override=False,
        nearest_vehicle_distance=float("inf"),
        ttc=float("inf"),
        highway_state="OFF",
        current_link_id="",
        target_link_id="",
        highway_lane_links=None,
        object_lane_matches=None,
        predicted_object_paths=None,
        selected_offset=0.0,
        candidates=None,
        reason="inactive",
    ):
        self.path = path
        self.active = bool(active)
        self.obstacle_detected = bool(obstacle_detected)
        self.long_obstacle_detected = bool(long_obstacle_detected)
        self.longitudinal_speed_scale = float(longitudinal_speed_scale)
        self.mode = mode
        self.acc_override = bool(acc_override)
        self.nearest_vehicle_distance = float(nearest_vehicle_distance)
        self.ttc = float(ttc)
        self.highway_state = highway_state
        self.current_link_id = current_link_id
        self.target_link_id = target_link_id
        self.highway_lane_links = list(highway_lane_links or [])
        self.object_lane_matches = list(object_lane_matches or [])
        self.predicted_object_paths = list(predicted_object_paths or [])
        self.selected_offset = float(selected_offset)
        self.candidates = list(candidates or [])
        self.reason = reason
