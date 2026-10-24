class ControlInput:
    def __init__(self, acc, steering):
        command = max(-1.0, min(1.0, float(acc)))
        if command > 0:
            self.accel = command
            self.brake = 0.
        else:
            self.accel = 0.
            self.brake = -command
        self.front_steer = steering
        self.rear_steer = 0.
        self.longlCmdType = 1
