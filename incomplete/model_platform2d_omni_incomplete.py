import numpy as np
from scipy.integrate import odeint


class ModelPlatform2dOmni:
    """
    Implementation of a kinematic model of an omnidirectional robot
    Expected input format:
    - vector [vx, vy, omega]
    Output format:
    - struct with field .pose = [r_x, r_y, phi]
    """

    def __init__(self, u):
        self.u = u

    def model_equations(self, X, t_odeint):  # 为了odeint,这里必须要有个时间的形参
        phi = X[2] % (2 * np.pi)
        # u = np.array([vx, vy, omega])
        # u = controller.guide(t)

        # TODO
        #  Transformation from body-fixed velocities to inertial velocities
        # np.cos()  np.sin()

        J = ([0, 0, 0],  # TODO
             [0, 0, 0],  # TODO
             [0, 0, 1])

        #  see MR01 -> compute state derivatives
        dX = np.dot(J, self.u.T)
        return dX

    def continuous_integration(self, Xs, t):
        X = odeint(self.model_equations, Xs[-1], [t, t + 0.1])
        X = X[-1]
        Xs.append(X)
        return Xs
