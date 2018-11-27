import numpy as np
from scipy.integrate import odeint


class ModelPlatform2dDDrive:
    """ model_platform2d_ddrive.

        Implementation of a kinematic model of a ground robot with differential
        drive locomotion system

        Expected input format:
        - vector [theta_r_dot, theta_l_dot] - wheel rotational rates

        Output format:
        - struct with field .pose = [r_x, r_y, phi]
    """
    def __init__(self, u):
        self.wheelRadius = [0.03, 0.03]
        self.wheelDistance = 0.25
        self.u = u  # u = controller.guide(t, ts, states)

    def model_equations(self, X, t_odeint):
        # Input format:
        # X = [x, y, phi]'
        # u = [omega_r, omega_l]'
        phi = X[2] % (2 * np.pi)

        # TODO
        # Transformation from body-fixed velocities to inertial velocities
        # np.cos()  np.sin()
        J = ([np.cos(phi), - np.sin(phi), 0],
             [np.sin(phi), np.cos(phi),   0],
             [0,           0,             1])

        # Transformation of the inputs into(generalized)body - fixed velocities
        rad_right_wheel = self.wheelRadius[0]
        rad_left_wheel = self.wheelRadius[1]
        # TODO
        V = ([rad_right_wheel / 2, rad_left_wheel / 2],
             [0,                   0],
             [rad_right_wheel / self.wheelDistance, -rad_left_wheel / self.wheelDistance])

        # see MR01 -> compute state derivatives

        dX = np.dot(J, np.dot(V, self.u.T))
        return dX

    def continuous_integration(self, t, Xs, t_interval):
        """Solver"""
        X_odeint = odeint(self.model_equations, Xs[-1], [t, t + t_interval])
        X = X_odeint[-1]
        Xs.append(X)
        return Xs