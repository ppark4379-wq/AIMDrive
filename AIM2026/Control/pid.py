"""LMDrive's finite-window PID (error samples, not a time integral).

Reference: opendilab/LMDrive, leaderboard/team_code/pid_controller.py.
Call at a fixed control cadence; reset after a stopped/invalid-input interval.
"""
from collections import deque
import math


class PID:
    def __init__(self, kp, ki, kd, window=40):
        if type(window) is not int or window < 2:
            raise ValueError('PID window must be at least two samples')
        if not all(math.isfinite(v) and v >= 0 for v in (kp, ki, kd)):
            raise ValueError('Invalid PID gains')
        self.kp, self.ki, self.kd = kp, ki, kd
        self.errors = deque(maxlen=window)
        self.reset()

    def reset(self):
        self.errors.clear()
        self.errors.extend([0.] * self.errors.maxlen)

    def step(self, error):
        if not math.isfinite(error):
            raise ValueError('Nonfinite PID error')
        self.errors.append(float(error))
        return (self.kp * error + self.ki * sum(self.errors) / len(self.errors)
                + self.kd * (self.errors[-1] - self.errors[-2]))
