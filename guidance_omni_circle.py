import numpy as np


class GuidanceOmniCircle:
    """ Steer an omnidirectional robot on a circular path
        By setting the parameter rotVelocity unequal zero, the robot changes
        and moves simultaneously
        """
    def __init__(self):
        self.radius = 2
        self.transVelocity = 1
        self.rotVelocity = 20 * np.pi / 180
        # self.t = t
        self.cw = True

    def guide(self, t):
        if self.cw:
            v = -abs(self.transVelocity)
        else:
            v = abs(self.transVelocity)

        trig_arg = t / self.radius * v - t * self.rotVelocity
        u = np.array([self.transVelocity * np.cos(trig_arg), self.transVelocity * np.sin(trig_arg), self.rotVelocity])
        return u
