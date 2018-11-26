import numpy as np
from scipy.integrate import odeint


class ModelPlatform2dVomega:
    """ Implementation of a kinematic model of a robot with v/omega % steering input.
     Expected input format:
     - column vector [v; omega]

     Output format:
     - struct with field .pose = [r_x; r_y; phi]"""

    def __init__(self, u):
        self.u = u

    def model_equations(self, X, t_odeint):
        # Input format:
        # X = [x, y, phi]'
        # u = [v, omega]'
        phi = X[2] % (2 * np.pi)

        # TODO
        # Transformation from body-fixed velocities to inertial velocities
        # np.cos()  np.sin()
        J = ([np.cos(phi), - np.sin(phi), 0],
             [np.sin(phi), np.cos(phi),   0],
             [0,           0,             1])

        # Transformation of the inputs into(generalized)body - fixed velocities
        W = ([1, 0],
             [0, 0],
             [0, 1])
        # see MR01 -> compute state derivatives

        dX = np.dot(J, np.dot(W, self.u.T))
        return dX

    def continuous_integration(self, t, Xs):
        X = odeint(self.model_equations, Xs[-1], [t, t + 0.1])
        X = X[-1]
        Xs.append(X)
        return Xs
