"""Model path adapter: unchanged expert longitudinal output + LMDrive lateral."""
import math
from types import SimpleNamespace
import numpy as np

from control.expert_source import expert_module
from control.lmdrive_lateral import LMDriveLateralPID


class ExpertWaypointController:
    def __init__(self, cfg):
        self.config = cfg['expert_config']
        self.lateral_config = cfg['lateral']
        self.max_speed_mps = None if cfg.get('max_speed_kmh') is None else cfg['max_speed_kmh']/3.6
        self.PathController = expert_module('control.path_controller').PathController
        self.Point = expert_module('localization.point').Point
        self.VehicleState = expert_module('vehicle_state').VehicleState
        self.reset()

    def reset(self):
        self.core = self.PathController(self.config)
        self.lateral = LMDriveLateralPID(self.lateral_config)
        self.elapsed = 0.

    def step(self, points, speed, horizons, dt=None, *, now=None, force_lateral=False):
        points = np.asarray(points, dtype=np.float64)
        horizons = np.asarray(horizons, dtype=np.float64)
        if (points.shape != (5, 2) or horizons.shape != (5,)
                or not np.isfinite(points).all() or not np.isfinite(horizons).all()
                or not np.all(np.diff(horizons)>0) or not math.isfinite(speed) or speed<-.1):
            raise ValueError('invalid_expert_control_input')
        # Network output replaces the expert planner's path and target velocity.
        # Keep the expert call and its longitudinal state/output untouched.
        # Its Pure Pursuit result is discarded; only LMDrive supplies steering.
        planning_path = [self.Point(forward,left) for left,forward in points]
        vehicle_state = self.VehicleState(0.,0.,0.,float(speed))
        predicted_velocity = float(np.linalg.norm(points[1]-points[0])/(horizons[1]-horizons[0]))
        target_velocity = (predicted_velocity if self.max_speed_mps is None
                           else min(predicted_velocity,self.max_speed_mps))
        context = SimpleNamespace(mode='model_waypoints',active=False,
                                  reason='model_waypoints',debug_details={},highway_state='OFF')
        lateral = self.lateral.step(points, speed, self.elapsed if now is None else now,
                                    force=force_lateral)
        command = self.core.control_path(planning_path,vehicle_state,target_velocity,
                                         context,sync_mode=False,control_dt=dt)
        self.elapsed += self.core.control_dt
        # Preserve this installation's observed UDP polarity (positive left).
        steer = float(-self.core.steering_sign*lateral['steer_pid_left_positive'])
        requested_deg = -steer*math.degrees(self.core.max_steering)
        raw_deg = (self.core.steering_sign*lateral['steering_pid_raw']
                   *self.lateral_config['steering_scale']*math.degrees(self.core.max_steering))
        return dict(reason='tracking',controller='lmdrive_lateral_expert_longitudinal',
                    accel=float(command.accel),brake=float(command.brake),steer=steer,
                    steering_requested_deg=requested_deg,
                    steering_unclipped_deg=raw_deg,
                    steering_angle_convention='expert_front_steer_and_morai_status',
                    desired_speed_mps=target_velocity,predicted_speed_mps=predicted_velocity,
                    speed_mps=float(speed),aim_distance_m=float(np.linalg.norm(lateral['aim_left_forward'])),
                    tracking_polyline_left_forward=points.tolist(),
                    tracking_includes_anchor_origin=False,reference_to_rear_axle_m=0.,
                    expert_sampling_time_s=self.core.control_dt,
                    expert_steering_sign=self.core.steering_sign,**lateral)
