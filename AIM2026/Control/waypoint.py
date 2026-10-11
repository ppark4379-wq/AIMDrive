"""LMDrive waypoint control, with explicit Stage-2 and UDP axis conversion."""
from dataclasses import dataclass
import math
import numpy as np

from .pid import PID


@dataclass(frozen=True)
class Control:
    accel: float
    brake: float
    steer: float
    desired_speed_mps: float = 0.
    angle_error: float = 0.
    reason: str = ''


class WaypointController:
    """Five waypoints at 0.5 s intervals -> normalized pedal/steering commands.

    Input coordinates are the Joint target axes: [left, -forward] metres.
    ``positive_steer`` declares the simulator command sign, never guessed from
    the VE auxiliary waypoint head (which uses a different coordinate system).
    """
    def __init__(self, *, positive_steer, max_speed_mps=None):
        if positive_steer not in ('left', 'right'):
            raise ValueError('Set Control.positive_steer to the verified UDP sign: left or right')
        if max_speed_mps is not None and (not math.isfinite(max_speed_mps) or max_speed_mps <= 0):
            raise ValueError('max_speed_mps must be positive or null')
        self.steer_sign = 1. if positive_steer == 'left' else -1.
        self.max_speed_mps = max_speed_mps
        self.turn = PID(1.25, .75, .3)
        self.speed = PID(5., .5, 1.)

    def reset(self):
        self.turn.reset()
        self.speed.reset()

    def stop(self, reason):
        self.reset()
        return Control(0., 1., 0., reason=reason)

    def step(self, waypoints, speed_mps):
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.shape != (5, 2) or not np.isfinite(wp).all() or not math.isfinite(speed_mps):
            return self.stop('invalid_waypoints_or_speed')
        if speed_mps < -.01:
            return self.stop('reverse_motion')
        speed_mps = max(0., speed_mps)
        desired = float(np.linalg.norm(wp[1] - wp[0]) * 2.)
        if self.max_speed_mps is not None:
            desired = min(desired, self.max_speed_mps)
        brake = desired < .1 or speed_mps > desired * 1.1
        # LMDrive uses atan2(forward, lateral), then divides angle by 90 deg.
        # For our left-positive targets this yields a left-positive error.
        left, negative_forward = wp[:2].mean(axis=0)
        angle = (math.pi / 2 - math.atan2(-negative_forward, left)) / (math.pi / 2)
        if speed_mps < .01:
            angle = 0.
        steer = float(np.clip(self.turn.step(angle), -1., 1.)) * .8 * self.steer_sign
        delta = float(np.clip(desired - speed_mps, 0., .35))
        throttle = float(np.clip(self.speed.step(delta), 0., .75))
        return Control(0. if brake else throttle, float(brake), steer, desired, angle)


def to_world(waypoints, pose):
    """Anchor a prediction to the GPS/IMU pose at sensor acquisition."""
    wp = np.asarray(waypoints, dtype=np.float64)
    c, s = math.cos(pose['yaw_rad']), math.sin(pose['yaw_rad'])
    return np.column_stack((-c * wp[:, 1] - s * wp[:, 0] + pose['x'],
                            -s * wp[:, 1] + c * wp[:, 0] + pose['y']))


def from_world(points, pose):
    """Express the same prediction relative to the latest ego pose."""
    delta = np.asarray(points, dtype=np.float64) - [pose['x'], pose['y']]
    c, s = math.cos(pose['yaw_rad']), math.sin(pose['yaw_rad'])
    return np.column_stack((-s * delta[:, 0] + c * delta[:, 1],
                            -c * delta[:, 0] - s * delta[:, 1]))


def remaining_waypoints(world_points, pose, age_s):
    """Advance the timestamped prediction, then rebase it to the current ego.

    LMDrive stalls CARLA while computing; UDP MORAI keeps moving. Merely
    replaying old local coordinates would move their reference with the car.
    Two complete future samples must still lie within the 2.5 s horizon.
    """
    if not math.isfinite(age_s) or not 0 <= age_s <= 1.5:
        raise ValueError('Prediction no longer has two complete future waypoints')
    times = np.arange(1, 6) * .5
    query = np.minimum(times + age_s, times[-1])
    points = np.asarray(world_points, dtype=float)
    sampled = np.column_stack([np.interp(query, times, points[:, axis]) for axis in (0, 1)])
    return from_world(sampled, pose)
