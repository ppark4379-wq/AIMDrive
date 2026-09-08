#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class Pid:
    def __init__(self, p_gain, i_gain, d_gain, sampling_time):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.sampling_time = sampling_time

        self.previous_error = 0
        self.integral_error = 0

    def reset(self):
        """Discard accumulated control error after an abrupt mode change."""
        self.previous_error = 0
        self.integral_error = 0

    def set_sampling_time(self, sampling_time):
        """Update controller dt without discarding the current error history."""
        sampling_time = float(sampling_time)
        if sampling_time <= 0.0:
            raise ValueError("sampling_time must be positive")
        self.sampling_time = sampling_time

    def get_output(self, target_value, current_value):
        error = target_value-current_value
        candidate_integral = self.integral_error + error*self.sampling_time
        derivative_error = (error-self.previous_error)/self.sampling_time

        output = self.p_gain*error + self.i_gain*candidate_integral + self.d_gain*derivative_error
        # Do not wind the integrator farther into actuator saturation.  This is
        # especially important when a long acceleration phase is followed by a
        # merge-yield or emergency-stop target.
        if not (
            (output > 1.0 and error > 0.0)
            or (output < -1.0 and error < 0.0)
        ):
            self.integral_error = candidate_integral
        self.previous_error = error
        return float(np.clip(output, -1.0, 1.0))
