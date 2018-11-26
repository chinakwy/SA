import numpy as np
from scipy.integrate import odeint
import math
import numpy.matlib  # np.matlib.repmat
from scipy.linalg import expm, block_diag
import pandas as pd


class FilterDDriveEKFSlAM:  # Simultaneous Localization And Mapping with  Extended Kalman Filter
    """ Localization algorithm for a Differential Drive Mobile Robot based
       on the Extended Kalman Filter (EKF)"""
    def __init__(self, odometer_output, sensor_output):

        self.sensor = sensor_output
        self.landmarks = sensor_output.landmarks
        self.odometer = odometer_output  # SensorOdometerWheelspeed

        # self.initialPose = [0, 0, 0]  # [0 0 0]'
        self.initialPoseCov = np.zeros((3, 3))

        # self.odometryError = 5 * np.pi / 180  # sigma of assumed odometer uncertainty in rad/s
        self.odometryError = self.odometer.odometryError
        self.wheelRadiusError = 1e-3  # sigma of assumed wheel diameter uncertainty in m
        self.wheelDistanceError = 5e-3  # sigma of assumed wheel distance uncertainty in m
        self.bearingError = 5 * np.pi / 180  # sigma of assumed bearing error in rad
        self.rangeError = 2 / 100  # sigma of assumed range error in percent of the aquired distance value  ️

        self.useBearing = True  # enable update from bearing measurements
        self.useRange = True  # enable update from range measurements

        self.useNumericPrediction = False  # do mean prediction using ode45
        self.useExactDiscretization = False
        # if true: do exact discretization, followed by linearization for covariance propagation
        # if false (the default), linearize first and then discretize using the matrix exponential

        self.wheelRadius = [0.03, 0.03]
        self.wheelDistance = 0.25

        # varianbles of functions for the extra visualization window
        self.states = []
        self.state_cov = []
        self.xhatlog = []
        self.yhatlog = []
        self.phihatlog = []
        self.xlog = []
        self.ylog = []
        self.philog = []
        self.sigmaX = []
        self.sigmaY = []
        self.sigmaPhi = []

        self.pose = []
        self.cov = []
        self.featurePositions = []
        self.landmarkIds = []
        self.featureCovariances = []

    def filter_step(self, t, X, current_poses, control_output):
        global state
        odometer_all = self.odometer.sample(t, control_output)
        sensor_all = self.sensor.sample(t, current_poses[-1])
        if t == 0:
            # if not len(state):
            state = {}
            state['x'] = X
            # state['pose'] = [path[0][0], path[0][1], 90 * np.pi / 180]
            state['cov'] = self.initialPoseCov
            state['lastInput'] = [0, 0]  # initial speed is zero
            state['t'] = 0

            state['features'] = []

        # use shorter symbols
        x = state['x']
        P = state['cov']
        u = state['lastInput']
        tNow = state['t']

        sensor = sensor_all[sensor_all.t >= t]
        sensor.index = range(len(sensor))
        for i in range(len(sensor)):
            iPredict = 1
            iUpdate = 1
            # while tNow < t:
            while tNow < sensor.t[i]:
                # determine, which measurement to proceed next
                # if iPredict <= len(sensor.t[i]):
                if iPredict <= 1:
                    tNextUpdate = sensor.t[i]
                else:
                    tNextUpdate = t

                odometer = odometer_all[odometer_all.t >= tNow]
                odometer = odometer[odometer.t <= tNextUpdate]
                odometer.index = range(len(odometer))

                while tNow < tNextUpdate:
                    if iPredict <= len(odometer):
                        tNext = odometer.t[iPredict - 1]
                        if tNext <= tNextUpdate:
                            x, P = self.do_prediction(x, P, u, tNext - tNow)
                            tNow = tNext
                            u = odometer.speed[iPredict - 1]
                            iPredict = iPredict + 1
                        else:
                            break
                    else:
                        break

                if tNow < tNextUpdate:
                    x, P = self.do_prediction(x, P, u, tNextUpdate - tNow)
                    tNow = tNextUpdate

                # if iUpdate <= len(sensor.t[i]):
                if iUpdate <= 1:
                    sensor_data = {'ranges': sensor.ranges[i], 'bearings': sensor.bearings[i], 'lmIds': sensor.lmIds[i]}
                    x, P, state['features'] = self.do_update(x, P, state['features'], sensor_data)
                    iUpdate = iUpdate + 1

            # put short-named intermediate variables back into the filter state
            state['x'] = x
            state['pose'] = x[0:3]
            state['cov'] = P
            state['lastInput'] = u
            state['t'] = tNow
            state['realPose'] = X

            out_pose = np.concatenate((state['x'][0:2], (state['x'][2]+np.pi) % (2*np.pi)-np.pi), axis=None)
            out_cov = state['cov'][0:3, 0:3]
            out_featurePositions = np.array(state['x'])[3:].reshape(-1, 2)
            out_landmarkIds = state['features']
            autoCorrs = np.diag(state['cov'][3:, 3:])
            crossCorrs = np.diag(state['cov'][3:, 3:], 1)
            out_featureCovariances = np.hstack((autoCorrs[np.arange(0, len(autoCorrs), 2)].reshape(-1, 1),
                                               crossCorrs[np.arange(0, len(crossCorrs), 2)].reshape(-1, 1),
                                                autoCorrs[np.arange(1, len(autoCorrs), 2)].reshape(-1, 1)))

            self.pose.append(out_pose)
            self.cov.append(out_cov)
            self.featurePositions.append(out_featurePositions)
            self.landmarkIds.append(out_landmarkIds)
            self.featureCovariances.append(out_featureCovariances)
        # =========================create_error_figure
        self.states.append(state)
        self.state_cov.append(state['cov'])

        self.xhatlog.append(state['x'][0])
        self.yhatlog.append(state['x'][1])
        self.phihatlog.append(state['x'][2])

        self.xlog.append(state['realPose'][0])
        self.ylog.append(state['realPose'][1])
        self.philog.append(state['realPose'][2])

        self.sigmaX.append(np.power(state['cov'][0][0], 0.5))
        self.sigmaY.append(np.power(state['cov'][1][1], 0.5))
        self.sigmaPhi.append(np.power(state['cov'][2][2], 0.5))
        # =================================================
        data = {'pose': self.pose, 'cov': self.cov, 'featurePositions': self.featurePositions,
                'landmarkIds': self.landmarkIds, 'featureCovariances': self.featureCovariances}
        out = pd.DataFrame(data)
        return out

    def do_prediction(self, x, P, u, T):
        global v, omega
        # Implementation of the prediction step

        # get the model parameters

        R_R = self.wheelRadius[0]
        R_L = self.wheelRadius[1]

        a = self.wheelDistance / 2

        dtheta_R = u[0]
        dtheta_L = u[1]

        # TODO: implement the prediction step

        # to simplify the equations, we assume a v/omega controlled
        # differential drive and convert our 'real' inputs to v/omega

        v = (R_R * dtheta_R + R_L * dtheta_L) / 2
        omega = (R_R * dtheta_R - R_L * dtheta_L) / (2 * a)

        # some more abbreviations (see slides of exercise 2)
        phi = x[2]
        sk = np.sin(phi)
        ck = np.cos(phi)
        skp1 = np.sin(phi + T * omega)
        ckp1 = np.cos(phi + T * omega)

        # do state prediction
        if self.useNumericPrediction:
            # Use numeric integration.This is applicable to any nonlinear system   # relative/absolute tolerance
            prediction_x = odeint(self.vomega_model, x[0:3], [0, T], rtol=1e-13, atol=np.finfo(float).eps)
            x[0:3] = np.array(prediction_x[-1])

        else:
            # exact solution
            if np.abs(omega) < 1e-12:
                x[0:3] = np.array(x)[0:3] + np.array([0,  # TODO
                                                      0,  # TODO
                                                      0])
            else:
                x[0:3] = np.array(x)[0:3] + np.array([0,  # TODO
                                                      0,  # TODO
                                                      T * omega])

        # do covariance prediction

        # input error covariance
        N = np.diag([self.odometryError ** 2, self.odometryError ** 2])

        if not self.useExactDiscretization:
            # linearize first
            A = np.array([[0, 0, 0],  # TODO
                          [0, 0,  0],  # TODO
                          [0, 0,       0]])

            B = np.array([[0, 0],  # TODO
                          [0, 0],  # TODO
                          [R_R / (2 * a), -R_L / (2 * a)]])

            # ...then discretize using the matrix exponential
            F = expm(np.dot(A, T))

            I = np.eye(len(F))
            S = np.dot(T, I)

            i = 2
            while True:
                D = np.dot(T ** i / math.factorial(i), A ** (i - 1))
                S = S + D
                i = i + 1
                if (np.abs(D) <= 2.220446049250313e-16).all():
                    # np.finfo(float).eps = 2.220446049250313e-16
                    break

            H = np.dot(S, B)

            F = block_diag(F, np.eye(len(P)-3))
            P = np.dot(np.dot(F, P), F.T)
            P[0:3, 0:3] = np.array(P[0:3, 0:3]) + np.dot(np.dot(H, N), H.T)

        else:
            # discretize first (only applicable, if ODEs can be solved
            # analytically), then linearize the discrete model
            if np.abs(omega) < 1e-12:
                A_dis = np.array([[0, 0, 0],  # TODO
                                  [0, 0,  0],  # TODO
                                  [0, 0, 1]])
                B_dis = np.array([[0, 0],  # TODO
                                  [0, 0],  # TODO
                                  [0,                0]])
            else:
                A_dis = np.array([[0, 0, 0],  # TODO
                                  [0, 0, 0],  # TODO
                                  [0, 0, 1]])
                B_dis = np.array([[0,  # TODO
                                  0],  # TODO

                                  [0,  # TODO
                                  0],  # TODO

                                  [R_R * T / (2 * a), -R_L * T / (2 * a)]])

            A_dis = block_diag(A_dis, np.eye(len(P) - 3))
            P = np.dot(np.dot(A_dis, P), A_dis.T)
            P[0:3, 0:3] = P[0:3, 0:3] + np.dot(np.dot(B_dis, N), B_dis.T)
        return x, P

    def do_update(self, x, P, features, meas):
        # Implementation of the update step
        visIdx = meas['lmIds']  # assume we can associate each measurement with a known landmark
        # if not len(visIdx):❗️
        #     return

        # determine, which measurements belong to features already part
        # of the state vector and which we see for the first time (map
        # management)
        _, fidx_old, midx_old = intersect_mtlb(features, visIdx)
        _, midx_new = setdiff_mtlb(np.arange(len(visIdx)), midx_old)

        # % interpretation of the index arrays
        # midx_old -> indices of measurements of known landmarks...
        # fidx_old -> ...and their associated indices in the state vector
        # midx_new -> indices of measurments of landmarks we see for the first time

        if len(midx_new):

            # feature initialization for first-time measurements.
            # Regardless of the block.useBearing/block.useRange
            # settings, we always have to use both measurements here to
            # uniquely determine the initial feature estimate!
            length = len(x)

            # length of state vector before adding new features
            bearings = np.array(meas['bearings'])[midx_new]
            ranges = np.array(meas['ranges'])[midx_new]

            pose = np.array(x)[0:3]
            c = np.cos(pose[2] + bearings)
            s = np.sin(pose[2] + bearings)

            # extend state vector with initial x/y estimates of new features
            x_helper1 = []
            for i in range(len(ranges)):
                x_helper1.append((pose[0] + ranges * c)[i])
                x_helper1.append((pose[1] + ranges * s)[i])
            x = np.concatenate((np.array(x), np.array(x_helper1)))

            # intermediate covariance matrix for old state combined with new measurements
            bearingErrors = np.matlib.repmat(self.bearingError**2, 1, len(midx_new))
            rangeErrors = (self.rangeError * ranges)**2

            p_helper1 = []
            for i in range(len(rangeErrors)):
                p_helper1.append(bearingErrors[0][i])
                p_helper1.append(rangeErrors[i])
            P_helper = block_diag(P, np.diag(p_helper1))

            # matrix to transform P_helper into covariance for extended state
            J = np.zeros((len(x), length + 2 * len(midx_new)))
            # ❗️J(sub2ind(size(J), 1:len, 1:len)) = 1  # keep old state variables -> top-left block is identity matrix
            J[np.arange(0, length), np.arange(0, length)] = 1
            x_rows = length + np.arange(0, 2*len(midx_new), 2)
            y_rows = length + np.arange(1, 2*len(midx_new), 2)
            J[x_rows, 0] = 1
            J[y_rows, 1] = 1
            J[x_rows, 2] = -ranges * s
            J[y_rows, 2] = ranges * c
            bearing_cols = length + np.arange(0, 2*len(midx_new), 2)
            range_cols = length + np.arange(1, 2*len(midx_new), 2)
            J[x_rows, bearing_cols] = J[x_rows, 2]
            J[y_rows, bearing_cols] = J[y_rows, 2]
            J[x_rows, range_cols] = c
            J[y_rows, range_cols] = s

            # extend state covariance (using the transformation J), process noise Q and landmark association data
            P = np.dot(np.dot(J, P_helper), J.T)
            features = np.concatenate((features, np.array(visIdx)[midx_new]), axis=None)

        if len(midx_old):
            # process measurements of tracked landmarks

            # prepare innovation vector, output jacobi matrix and measurement noise covariance
            delta_y = np.zeros((0, 1))
            C = np.zeros((0, len(x)))
            W = np.zeros((0, 0))

            # indices of the feature coordinates in the state vector
            x_idx = 3 + 2 * np.array(fidx_old)
            y_idx = 4 + 2 * np.array(fidx_old)

            deltas = np.hstack((np.array(x[x_idx] - x[0]).reshape(-1, 1), np.array(x[y_idx] - x[1]).reshape(-1, 1)))

            if self.useBearing:
                # predict bearing measurements
                # z = []
                # for i in range(len(deltas)):
                #     z_value = math.atan2(deltas[i, 1], deltas[i, 0]) - x[2]
                #     z.append(z_value)
                # z = np.array(z)
                z = np.arctan2(deltas[:, 1], deltas[:, 0]) - x[2]
                # determine innovation from bearing measurements

                delta_y = np.concatenate(
                    (delta_y, np.mod(np.array(meas['bearings'])[midx_old] - z + np.pi, 2 * np.pi) - np.pi), axis=None)

                # fill in corresponding entries in the jacobi matrix
                denoms = np.sum(deltas**2, 1)
                C_b = np.zeros((len(midx_old), len(x)))
                C_b[:, [0, 1]] = np.hstack(
                    ((deltas[:, 1] / denoms).reshape(-1, 1), (-deltas[:, 0] / denoms).reshape(-1, 1)))
                C_b[:, 2] = -1
                C_b[np.arange(len(midx_old)), x_idx] = - deltas[:, 1] / denoms
                C_b[np.arange(len(midx_old)), y_idx] = deltas[:, 0] / denoms
                C = np.vstack((C, C_b))

                # covariance of measurement noise -> for bearing measurements independent of sensor output
                W = block_diag(W, np.dot(self.bearingError**2, np.eye(len(midx_old))))

            if self.useRange:

                # predict range measurements
                # z = np.power(np.sum(deltas[:, 0]**2 + deltas[:, 1]**2, 1), 0.5)
                z = np.power(deltas[:, 0] ** 2 + deltas[:, 1] ** 2, 0.5)
                # 'real' sensor output
                ranges = np.array(meas['ranges'])[midx_old]
                # compute difference = innovation
                delta_y = np.concatenate((delta_y, ranges - z), axis=None)

                # fill in corresponding entries in the jacobi matrix
                C_r = np.zeros((len(midx_old), len(x)))
                C_r[:, [0, 1]] = np.hstack(((-deltas[:, 0] / z).reshape(-1, 1), (-deltas[:, 1] / z).reshape(-1, 1)))
                C_r[np.arange(len(midx_old)), x_idx] = deltas[:, 0] / z
                C_r[np.arange(len(midx_old)), y_idx] = deltas[:, 1] / z
                C = np.vstack((C, C_r))

                # covariance of measurement noise (noise scales with distance)
                W = block_diag(W, np.diag(np.dot(self.rangeError, ranges)**2))

            # compute Kalman gain K
            k1 = np.dot(P, C.T)
            k2 = np.dot(np.dot(C, P), C.T) + W
            K = np.dot(k1, np.linalg.inv(k2))

            # EKF update
            x = x + np.dot(K, delta_y)
            P = np.dot((np.eye(len(P)) - np.dot(K, C)), P)
        return x, P, features

    def vomega_model(self, x, t_odeint):
        phi = x[2]
        # np.cos()  np.sin()
        dx = np.array([0, 0, omega])  # TODO
        return dx


def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def setdiff_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    c = list(set(a).difference(set(b)))
    return c, ia[np.isin(a1, c)]
