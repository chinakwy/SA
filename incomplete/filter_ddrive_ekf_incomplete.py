import numpy as np
from scipy.integrate import odeint
import math
import numpy.matlib  # np.matlib.repmat
from scipy.linalg import expm, block_diag
import pandas as pd


class FilterDDriveEKF:
    """% Localization algorithm for a Differential Drive Mobile Robot based
        % on the Extended Kalman Filter (EKF)"""
    def __init__(self, odometer_output, sensor_output):

        self.sensor = sensor_output
        self.landmarks = sensor_output.landmarks
        self.odometer = odometer_output  # SensorOdometerWheelspeed

        # self.initialPose = np.array([0, 0, 0]).T
        self.initialPoseCov = np.zeros((3, 3))

        self.odometryError = self.odometer.odometryError
        # self.odometryError = 5 * np.pi / 180  # sigma of assumed odometer uncertainty in rad/s
        self.wheelRadiusError = 1e-3  # sigma of assumed wheel diameter uncertainty in m  1e-3
        self.wheelDistanceError = 5e-3  # sigma of assumed wheel distance uncertainty in m  5e-3
        self.bearingError = 5 * np.pi / 180  # sigma of assumed bearing error in rad
        self.rangeError = 2 / 100    # sigma of assumed range error in percent of the aquired distance value   2 / 100

        self.useCartesianSensor = False
        # if true, compute landmark positions (m_ix, m_iy) from range & bearing and use then as output measurements
        # otherwise (the default), use range & bearing directly in the output equations
        # (which allows to disable them individually)

        self.useBearing = True  # enable update from bearing measurements
        self.useRange = False  # enable update from range measurements

        self.useNumericPrediction = False  # do mean prediction using ode45
        self.useExactDiscretization = False
        # if true: do exact discretization, followed by linearization for covariance propagation
        # if false (the default), linearize first and then discretize using the matrix exponential

        self.wheelRadius = [0.03, 0.03]
        self.wheelDistance = 0.25

        self.xs = []  #用于 filter_step pd.DataFrame
        self.Ps = []

    def filter_step(self, t, X, control_output):
        global state
        odometer_all = self.odometer.sample(t, control_output)
        sensor_all = self.sensor.sample(t, X)
        if t == 0:
            # if not len(state):
            state = {}
            state['pose'] = X
            # state['pose'] = [path[0][0], path[0][1], 90 * np.pi / 180]
            state['cov'] = self.initialPoseCov
            state['lastInput'] = [0, 0]  # initial speed is zero
            state['t'] = 0
        # use shorter symbols
        x = state['pose']
        P = state['cov']
        u = state['lastInput']
        tNow = state['t']

        sensor = sensor_all[sensor_all.t >= t]  # 1/10,1/100,1/15❗️
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
                        tNext = odometer.t[iPredict-1]
                        if tNext <= tNextUpdate:
                            x, P = self.do_prediction(x, P, u, tNext-tNow)
                            tNow = tNext
                            u = odometer.speed[iPredict-1]
                            iPredict = iPredict + 1
                        else:
                            break
                    else:
                        break

                if tNow < tNextUpdate:
                    x, P = self.do_prediction(x, P, u, tNextUpdate-tNow)
                    tNow = tNextUpdate

                # if iUpdate <= len(sensor.t[i]):
                if iUpdate <= 1:
                    meas = {'ranges': sensor.ranges[i], 'bearings': sensor.bearings[i], 'lmIds': sensor.lmIds[i]}
                    x, P = self.do_update(x, P, meas, self.landmarks)
                    iUpdate = iUpdate + 1

        # put short-named intermediate variables back into the filter state
            state['pose'] = x
            state['cov'] = P
            state['lastInput'] = u
            state['t'] = tNow

        # the output of the localization filter is the estimated state (=pose) vector

            self.xs.append(x)
            self.Ps.append(P)
        data = {'pose': self.xs, 'cov': self.Ps}
        out = pd.DataFrame(data)
        return out

    def do_prediction(self, x, P, u, T):
        # Mittelwert ̂𝒙𝒌 sowie einer Kovarianzmatrix 𝑷̂𝒌 zum Zeitpunkt 𝒕𝒌,
        # anhand von Messungen 𝒖̃𝒌 der Systemeingänge 𝒖 über das Zeitintervall [𝒕𝒌, 𝒕𝒌+𝟏]
        # Eingang： werden hier durch das Odometer bereitgestellt
        # Ausgang：aktualisierte Zustandsschätzung ̂𝒙𝒌+𝟏 sowie die zugehörige Kovarianzmatrix 𝑷̂𝒌+𝟏 zum Zeitpunkt 𝒕𝒌+𝟏
        global v, omega
        # Implementation of the prediction step

        # get the model parameters

        R_R = self.wheelRadius[0]
        R_L = self.wheelRadius[1]

        a = self.wheelDistance / 2

        dtheta_R = u[0]
        dtheta_L = u[1]

        # to simplify the equations, we convert differential drive and convert our 'real' inputs to v/omega
        v = (R_R * dtheta_R + R_L * dtheta_L) / 2
        omega = (R_R * dtheta_R - R_L * dtheta_L) / (2 * a)

        # some more abbreviations (see slides of exercise 2)
        phi = x[2]
        sk = np.sin(phi)
        ck = np.cos(phi)
        skp1 = np.sin(phi+T*omega)
        ckp1 = np.cos(phi+T*omega)

        # (a)(e) Die Zeitdiskretisierung für die Zustands-Prädiktion kann auch durch
        # numerische Integration zur Laufzeit durchgeführt werden.
        # Hierfür bietet sich das schrittweitengesteuerte Runge-Kutta-Verfahren der Ordnung 4/5 an
        # do state prediction
        if self.useNumericPrediction:
            # Use numeric integration.
            # This is applicable to any nonlinear system   # relative/absolute tolerance
            prediction_x = odeint(self.vomega_model, x, [0, T], rtol=1e-13, atol=np.finfo(float).eps)
            x = np.array(prediction_x[-1])

        else:
            # exact solution
            if np.abs(omega) < 1e-12:
                x = np.array(x) + np.array([0 * 0 * 0,  # TODO
                                            0 * 0 * 0,  # TODO
                                            0])
            else:
                x = np.array(x) + np.array([0,  # TODO
                                            0,  # TODO
                                            T * omega])

        # (b) Für die Kovarianzpropagation wird ein lineares diskretes Model des Differentialantriebs benötigt.
        # do covariance prediction

        # input error covariance
        N = np.diag([self.odometryError**2, self.odometryError**2])

        if not self.useExactDiscretization:
            # linearize first
            # Bestimmen Sie hierzu die entsprechenden Jacobimatrizen durch Ableitung von 𝒇̂(̂𝒙,𝒖̃)
            # bzgl. des Zustandsvektors ̂𝒙 = [𝑟̂ x, 𝑟̂ y, 𝜑̂]T ---> A
            # sowie des Eingangsvektors 𝒖̃ = [𝜃̇ R , 𝜃̇ L]T ---> B
            A = np.array([[0, 0, 0],  # TODO
                          [0, 0, 0],  # TODO
                          [0, 0, 0]])

            B = np.array([[0, 0],  # TODO
                         [0, 0],  # TODO
                         [R_R / (2 * a), -R_L / (2 * a)]])

            # (c)...then discretize using the matrix exponential
            # die diskrete Systemmatrix 𝑭 und Eingangsmatrix 𝑯
            F = expm(np.dot(A, T))
            I = np.eye(len(F))
            S = np.dot(T, I)

            i = 2
            while True:
                D = np.dot(T ** i / math.factorial(i), A**(i-1))
                S = S + D
                i = i + 1
                if (np.abs(D) <= 2.220446049250313e-16).all():
                    # np.finfo(float).eps = 2.220446049250313e-16
                    break

            H = np.dot(S, B)
            # (d) Implementieren Sie unter Verwendung der Ergebnisse aus den Teilaufgaben (a) – (c)
            # den Prädiktionsschritt des erweiterten Kalman-Filters
            # gemäß der Vorlesungsfolie MR02II/34 in die Funktion doPropagation()
            # P = F * P * F.' + H * N * H.'
            P = np.dot(np.dot(F, P), F.T) + np.dot(np.dot(H, N), H.T)

        # (f) Könnte man eine linear diskrete Systembeschreibung auch
        # durch Linearisierung der Funktion 𝒇̂dis(̂𝒙𝑘, 𝒖̃𝑘) aus Teilaufgabe (a) gewinnen?
        else:
            # discretize first (only applicable, if ODEs can be solved
            # analytically), then linearize the discrete model
            if np.abs(omega) < 1e-12:
                A_dis = np.array([[0, 0, 0],  # TODO
                                  [0, 0, 0],  # TODO
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

            # P = A_dis * P * A_dis.' + B_dis * N * B_dis.'
            P = np.dot(np.dot(A_dis, P), A_dis.T) + np.dot(np.dot(B_dis, N), B_dis.T)
        return x, P

    def do_update(self, x, P, meas, landmarks):

        # Implementation of the update step
        # ranges, bearings, lmIds = self.sensor.sample(x)
        visIdx = meas['lmIds']  # assume we can associate each measurement with a known landmark
        # if not len(visIdx):❗️‼️
        #     return
        # (a) Das Landmarken-Messsystem ermittelt gemäß der folgenden Skizze Abstände 𝑑̃𝑖 und Relativ-
        # winkel 𝛽̃𝑖 von der aktuellen Roboter-Pose zu bekannten Landmarken (𝑚x,𝑖,𝑚y,𝑖).
        beta = meas['bearings']  # Relativwinkel 𝛽̃𝑖
        d = meas['ranges']  # Abstände 𝑑̃𝑖
        m = landmarks[[visIdx]]  # bekannten Landmarken (𝑚x,𝑖,𝑚y,𝑖)

        # (b) Bestimmen Sie die Jacobimatrix der Ausgangsgleichung für eine Landmarke 𝑖
        C = np.zeros([0, 3])  # Jacobimatrix
        W = np.zeros([0, 0])  # vollständige Kovarianzmatrix des Messrauschens 𝑾
        delta_y = np.zeros([0, 1])  # vollständige Ausgangsgleichung 𝒈(̂𝒙,𝒎̂),

        b = m - np.matlib.repmat((x[0], x[1]), len(visIdx), 1)

        # (c) Implementieren Sie in der Funktion doUpdate()für alle sichtbaren Landmarken
        # die vollständige Ausgangsgleichung 𝒈(̂𝒙,𝒎̂), die vollständige Jacobimatrix 𝑪
        # sowie die vollständige Kovarianzmatrix des Messrauschens 𝑾.
        if not self.useCartesianSensor:
            if self.useBearing:
                # compute output vector z = h(x_k|k-1) (= model-based prediction of measurement)
                y_pred = np.arctan2(b[:, 1], b[:, 0]) - x[2]

                # innovation vector (measurement - output prediction)
                # force into interval +-pi
                delta_y = np.vstack((delta_y, np.array(np.mod(beta - y_pred + np.pi, 2 * np.pi) - np.pi).
                                     reshape(-1, 1))).reshape(-1)
                # H = jacobi matrix of output function w.r.t. state (dh/dx)
                denoms = b[:, 0] ** 2 + b[:, 1] ** 2
                c1 = np.array([b[:, 1] / denoms]).T
                c2 = np.array([-b[:, 0] / denoms]).T
                c3 = np.matlib.repmat(-1, 1, len(visIdx)).T
                c4 = np.hstack((c1, c2, c3))
                C = np.vstack((C, c4))

                # W = covariance matrix of measurement noise
                W = block_diag(W, np.dot(self.bearingError ** 2, np.eye(len(visIdx))))

            if self.useRange:
                y_pred = np.power(b[:, 0] ** 2 + b[:, 1] ** 2, 0.5)

                delta_y = np.hstack((delta_y, d - y_pred))
                c1 = np.array([-b[:, 0] / y_pred]).T
                c2 = np.array([-b[:, 1] / y_pred]).T
                c3 = np.zeros((len(visIdx), 1))
                c4 = np.hstack((c1, c2, c3))
                C = np.vstack((C, c4))

                W = block_diag(W, np.diag(np.dot(self.rangeError, d) ** 2))

        else:
            delta_y_list = []
            for i in range(len(visIdx)):
                # transform d/beta uncertainty into m_ix/m_iy uncertainty
                J_y = np.array([[np.cos(beta[i]), -d[i] * np.sin(beta[i])],
                                [np.sin(beta[i]), d[i] * np.cos(beta[i])]])

                w1 = np.dot(np.dot(J_y, np.diag(((self.rangeError * d[0]) ** 2, self.bearingError ** 2))), J_y.T)
                W = block_diag(W, w1)

                y_meas = np.dot(d[i], np.array([np.cos(beta[i]), np.sin(beta[i])]))
                y_pred = np.dot(np.array([[np.cos(x[2]), -np.sin(x[2])],
                                          [np.sin(x[2]), np.cos(x[2])]]).T, np.array(b[i, :]).T)
                delta_y_list.extend(y_meas - y_pred)
                # np.cos() np.sin()
                c1 = np.array([[0],  # TODO
                               [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * b[i, 0] - np.sin(x[2]) * b[i, 1]]])
                C = np.vstack((C, c1))
            delta_y = np.array(delta_y_list)

        # (d) compute Kalman gain K
        k1 = np.dot(P, C.T)
        k2 = np.dot(np.dot(C, P), C.T) + W
        K = np.dot(k1, np.linalg.inv(k2))
        # update state x and covariance matrix P
        x = x + np.dot(K, delta_y)
        P = np.dot((np.eye(3) - np.dot(K, C)), P)

        return x, P

    def vomega_model(self, x, t_odeint):
        phi = x[2]
        dx = (0, 0, omega)  # TODO
        return dx
