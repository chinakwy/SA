import numpy as np


class GuidanceDDriveSquare:
    """ Steer a differential drive robot on a square path.
    For correct operation, this block has to use the same wheelRadius and
    wheelDistance parameters as the robot """
    def __init__(self):
        self.length = 2
        self.cw = True
        self.transVelocity = 0.74
        self.rotVelocity = 45 * np.pi / 180
        self.wheelRadius = [0.03, 0.03]
        self.wheelDistance = 0.25

    def guide(self, t, ts, states):
        mode = states[-1][0]
        tStart = states[-1][1]

        if mode == 1:
            if (t - tStart + 1e-6) >= round(self.length / self.transVelocity, 4):
                mode = 2
                tStart = t
        else:
            if (t - tStart + 1e-6) >= round(np.pi / (2 * self.rotVelocity), 4):
                mode = 1
                tStart = t
        state = [mode, tStart]

        if t > ts[-1]:
            # Four same states are calculated every 0.1 seconds.
            # This setting allows the states to add only one state every 0.1 seconds,
            # which is convenient for doStepBack to run.
            ts.append(t)
            states.append(state)

        if mode == 1:  # straight
            vOmega = [self.transVelocity, 0]
        else:  # mode == 2 Rotate
            vOmega = [0, -(2 * self.cw -1) * self.rotVelocity]
            # vOmega = [0, -self.rotVelocity]
        rad_right = self.wheelRadius[0]
        rad_left = self.wheelRadius[1]
        u = np.dot(([1 / rad_right, 0.5 * self.wheelDistance / rad_right],
                    [1 / rad_left, -0.5 * self.wheelDistance / rad_left]), vOmega)
        return u
