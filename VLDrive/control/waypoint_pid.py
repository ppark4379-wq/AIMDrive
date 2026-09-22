"""LMDrive control_pid adapted to MORAI; upstream Apache-2.0, see third_party/."""
import math
import numpy as np

from control.third_party.lmdrive.pid_controller import PIDController


def stop_command(reason):
    return dict(accel=0.,brake=1.,steer=0.,reason=reason)


def current_waypoints(result,pose,age,min_future_s=.05,max_pose_jump_m=10.,*,selection='temporal'):
    """Convert anchor [left,-forward] to current ego [left,forward].

    Legacy temporal mode discards elapsed horizons. Spatial mode retains the
    predicted polyline and its anchor origin; the tracker chooses an unpassed
    target by position. The runtime enforces prediction freshness in both modes.
    Global path never replaces a predicted waypoint.
    """
    if not math.isfinite(age) or age<0:
        raise ValueError('invalid_prediction_age')
    if result.get('coordinate_convention')!='lmdrive_left_negative_forward':
        raise ValueError('unsupported_waypoint_coordinates')
    raw=np.asarray(result['waypoints_raw'],dtype=np.float64)
    horizons=np.asarray(result['horizons_s'],dtype=np.float64)
    if raw.shape!=(5,2) or horizons.shape!=(5,) or not np.isfinite(raw).all() or not np.isfinite(horizons).all():
        raise ValueError('invalid_waypoints')
    if not np.allclose(horizons,[.5,1,1.5,2,2.5],atol=1e-8,rtol=0):
        raise ValueError('unsupported_horizons')
    anchor=result['navigation']['pose']
    for value in (anchor,pose):
        if not all(math.isfinite(float(value[k])) for k in ('x','y','yaw_rad')):
            raise ValueError('invalid_pose')
    displacement=np.array([anchor['x']-pose['x'],anchor['y']-pose['y']])
    if np.linalg.norm(displacement)>max_pose_jump_m:
        raise ValueError('pose_jump')
    a,c=anchor['yaw_rad'],pose['yaw_rad']
    forward=-raw[:,1];left=raw[:,0]
    world=np.stack([np.cos(a)*forward-np.sin(a)*left,
                    np.sin(a)*forward+np.cos(a)*left],axis=-1)+displacement
    local=np.stack([-np.sin(c)*world[:,0]+np.cos(c)*world[:,1],
                    np.cos(c)*world[:,0]+np.sin(c)*world[:,1]],axis=-1)
    if selection=='expert':
        return local,horizons,[0,1,2,3,4]
    if selection=='spatial':
        origin=[-np.sin(c)*displacement[0]+np.cos(c)*displacement[1],
                np.cos(c)*displacement[0]+np.sin(c)*displacement[1]]
        return np.vstack([origin,local]),np.r_[0.,horizons],[-1,0,1,2,3,4]
    if selection!='temporal':
        raise ValueError('invalid_waypoint_selection')
    indices=np.flatnonzero(horizons-age>min_future_s)
    if len(indices)<2 or np.any(local[indices[:2],1]<-.5):
        raise ValueError('insufficient_forward_horizon')
    return local[indices],horizons[indices],indices.tolist()


class WaypointPID:
    def __init__(self,options):
        self.options=options
        self.reset()

    def reset(self):
        self.turn=PIDController(**self.options['turn'])
        self.speed=PIDController(**self.options['speed'])

    def step(self,points,speed,spacing_s=.5):
        points=np.asarray(points,dtype=np.float64)
        if points.ndim!=2 or points.shape[0]<2 or points.shape[1]!=2 or not np.isfinite(points).all():
            raise ValueError('invalid_pid_waypoints')
        if not math.isfinite(speed) or speed<-.1 or not math.isfinite(spacing_s) or spacing_s<=0:
            raise ValueError('invalid_speed_or_horizon')
        speed=max(0.,float(speed))
        # Original LMDrive: norm(wp0-wp1)*2, for its 0.5-second spacing.
        predicted_speed=float(np.linalg.norm(points[0]-points[1])/spacing_s)
        desired=min(predicted_speed,self.options['max_speed_mps'])
        brake=desired<self.options['brake_speed'] or speed>desired*self.options['brake_ratio']
        aim=(points[0]+points[1])/2.
        angle=float(np.degrees(np.pi/2-np.arctan2(aim[1],aim[0]))/90.)
        if speed<.01:
            angle=0.
        steer=float(np.clip(self.turn.step(angle),-1.,1.))
        delta=float(np.clip(desired-speed,0.,self.options['clip_delta']))
        throttle=float(np.clip(self.speed.step(delta),0.,self.options['max_throttle']))
        if brake:
            throttle=0.
        # PID error is LEFT-positive. This installation's EgoCtrlCmd was
        # observed to turn right with negative commands (2026-09-20 logs).
        # Keep polarity configurable; do not infer it from a different MORAI
        # controller/protocol example. UDP magnitude remains normalized, not rad.
        steer_pid=steer*self.options['steering_scale']
        steer=steer_pid*self.options['morai_steering_sign']
        return dict(accel=throttle,brake=float(brake),steer=steer,reason='tracking',
                    steer_pid_left_positive=steer_pid,steering_sign=self.options['morai_steering_sign'],
                    desired_speed_mps=desired,predicted_speed_mps=predicted_speed,
                    speed_mps=speed,angle_error=angle,aim_left_forward=aim.tolist())
