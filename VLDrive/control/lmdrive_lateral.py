"""LMDrive lateral PID only; longitudinal control belongs to MORAI expert.

Reference: leaderboard/team_code/{lmdriver_agent,lmdriver_config}.py.
The unmodified upstream PID and license live in third_party/lmdrive.
"""
import math

import numpy as np

from control.third_party.lmdrive.pid_controller import PIDController
from control.steering_target import select_target


class LMDriveLateralPID:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.pid = PIDController(K_P=self.cfg['turn_KP'], K_I=self.cfg['turn_KI'],
                                 K_D=self.cfg['turn_KD'], n=self.cfg['turn_n'])
        self.next_update = None
        self.last = None

    def step(self, points, speed, now, *, force=False):
        if not math.isfinite(now):
            raise ValueError('invalid_lateral_clock')
        # Validate usable remaining geometry on every 30 Hz control tick,
        # including ticks on which the lateral PID output would be held.
        selection = select_target(points,speed,self.cfg['target'])
        due = (force or self.next_update is None or now+1e-9 >= self.next_update
               or now < self.last['lateral_calculated_at'])
        if due:
            # Points already use [left, positive-forward] after pose conversion.
            aim = np.asarray(selection['aim_left_forward'])
            error = float(np.degrees(np.pi/2-np.arctan2(aim[1], aim[0]))/90.)
            if speed < .01:
                error = 0.  # Upstream does not reset the integral at rest.
            previous = self.pid._window[-1]
            raw = float(self.pid.step(error))
            self.last = dict(
                steer_pid_left_positive=float(np.clip(raw, -1., 1.)*self.cfg['steering_scale']),
                steering_pid_raw=raw, steering_saturated=abs(raw)>1.,
                angle_error=error, angle_error_deg=error*90.,
                steering_p=self.cfg['turn_KP']*error,
                steering_i=self.cfg['turn_KI']*float(np.mean(self.pid._window)),
                steering_d=self.cfg['turn_KD']*(error-previous),
                lateral_calculated_at=now,**selection)
            period = 1./self.cfg['update_hz']
            if self.next_update is None or force or now < self.next_update-period:
                self.next_update = now+period
            else:
                # Maintain phase despite small scheduling jitter; never catch up
                # by inserting multiple PID samples from the same observation.
                missed = max(1, math.floor((now-self.next_update+1e-9)/period)+1)
                self.next_update += missed*period
        return dict(self.last, lateral_updated=due)
