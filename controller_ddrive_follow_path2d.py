import numpy as np
from visual_translated_y import translated_y
import pandas as pd


scale = 100


class ControllerDDriveFollowPath2d:  # sampling_time = 1／50
    def __init__(self, path_data, canvas):
        self.velocity = 0.5  # translational speed in m/s
        self.omega = 90 * np.pi / 180  # rotational speed in rad/s
        self.firstTargetPoint = 1
        self.startBackwards = False
        self.k_rot = 10

        self.sampling_time = 1 / 50

        self.path = path_data

        self.wheelRadius = [0.03, 0.03]  # ❗️
        self.wheelDistance = 0.25  # ❗

        self.canvas = canvas
        self.next_point = self.canvas.create_rectangle(0, 0, 0, 0, outline="blue", width=1)
        self.pathPt = []

        self.us = []  # pd.DataFrame
        self.states = []
        self.ts = []

    def control(self, t, Xs):
        global state
        # --------control
        # initialize state on the arrival of the first pose input

        if t == 0:
            state = {}
            state['pose'] = [self.path[0][0], self.path[0][1], 90 * np.pi / 180]
            # state['pose'] = Xs[-1]
            state['targetPointIndex'] = np.minimum(len(self.path), self.firstTargetPoint)
            targetPt = self.path[state['targetPointIndex'] - 1]
            state['targetPose'] = np.array([targetPt[0], targetPt[1],
                                            (np.arctan2(targetPt[1] - state['pose'][1],
                                                        targetPt[0] - state['pose'][0]) + np.pi)
                                            % (2 * np.pi) - np.pi]).T
            state['backwards'] = self.startBackwards
        else:
            state = self.states[-1]  # for doback
            state['pose'] = Xs[-1]
            # state['pose'] = X
        control_t = t
        while control_t < t+0.0999:
            pose = state['pose']
            targetPose = state['targetPose']
            targetVec = np.array([np.cos(targetPose[2]), np.sin(targetPose[2])])
            # determine distance of projected position on line-to-target from next target point
            posOnLine = np.dot(targetVec.T, [pose[0] - targetPose[0], pose[1] - targetPose[1]])

            if posOnLine >= -0.01:
                # Distance > 0 --> we are beyond the target point (the condition also applies
                # if the target point is still slightly ahead of the robot)  Switch to next target point
                if not state['backwards']:
                    state['targetPointIndex'] = state['targetPointIndex'] + 1
                    if state['targetPointIndex'] > len(self.path):
                        if self.path[0][0] == self.path[-1][0] and self.path[0][1] == self.path[-1][1]:
                            state['targetPointIndex'] = 2
                        else:
                            state['targetPointIndex'] = len(self.path) - 1
                            state['backwards'] = True
                else:
                    state['targetPointIndex'] = state['targetPointIndex'] - 1
                    if state['targetPointIndex'] == 0:
                        if self.path[0][0] == self.path[-1][0] and self.path[0][1] == self.path[-1][1]:
                            state['targetPointIndex'] = len(self.path) - 1
                        else:
                            state['targetPointIndex'] = 2
                            state['backwards'] = False

                targetPt = self.path[state['targetPointIndex'] - 1]  # Index minus 1
                targetPose = np.array([targetPt[0], targetPt[1],
                                       (np.arctan2(targetPt[1] - targetPose[1], targetPt[0] - targetPose[0]) + np.pi)
                                       % (2 * np.pi) - np.pi]).T
                state['targetPose'] = targetPose

            # Determine orientation difference between robot and line-of-sight to next target point
            diffAngle = (np.arctan2(targetPose[1] - pose[1], targetPose[0] - pose[0]) - pose[2] + np.pi) % (
                        2 * np.pi) - np.pi
            if np.abs(diffAngle) > 5 * np.pi / 180:
                # too much orientation difference --> turn on the spot
                vOmega = [0, np.sign(diffAngle) * self.omega]
            else:
                # approach target point, maybe with a slight curvature to reduce the remaining orientation error
                # (proportional controller)
                vOmega = [self.velocity, diffAngle * self.k_rot]

            # limit v / self.omega outputs
            if np.abs(vOmega[1]) > self.omega:
                vOmega[1] = np.sign(vOmega[1]) * self.omega
            vOmega[0] = np.minimum(vOmega[0], self.velocity)

            # convert v/self.omega to [theta_R, theta_L]'

            R_right = self.wheelRadius[0]
            R_left = self.wheelRadius[1]
            u = np.dot(
                ([1 / R_right, 0.5 * self.wheelDistance / R_right], [1 / R_left, -0.5 * self.wheelDistance / R_left]),
                vOmega)
            control_t += self.sampling_time

            self.us.append(u)
            self.states.append(state)
            self.ts.append(control_t)
            data = {'u': self.us, 'state': self.states, 't': self.ts}
            self.out = pd.DataFrame(data)
        return self.out

    def draw_path(self, t):
        path = np.dot(self.path, scale)

        if t <= 0.1:
            for i in range(len(path) - 1):
                self.canvas.create_line(path[i][0], translated_y(path[i][1]),
                                        path[i + 1][0], translated_y(path[i + 1][1]), fill='blue', width=1)

            if path[0][0] != path[-1][0] or path[-1][1] != path[-1][1]:
                #  draw start point
                p = 10
                a = 3
                x = path[0][0]
                y = translated_y(path[0][1])
                star_points = []
                for i in (1, -1):
                    star_points.extend((x, y + i * p))
                    star_points.extend((x + i * a, y + i * a))
                    star_points.extend((x + i * p, y))
                    star_points.extend((x + i * a, y - i * a))
                self.canvas.create_polygon(star_points, fill='blue')

                #  draw goal point
                x = path[-1][0]
                y = translated_y(path[-1][1])
                goal_points = []
                for i in (1, -1):
                    goal_points.extend((x, y + i * p))
                    goal_points.extend((x + i * a, y + i * a))
                    goal_points.extend((x + i * p, y))
                    goal_points.extend((x + i * a, y - i * a))
                self.canvas.create_polygon(goal_points, fill='gold')

    def draw_nextpoint(self):
        path = np.dot(self.path, scale)
        l = 4
        # if len(state):
        if state:
            if state['targetPointIndex'] <= len(path):
                self.pathPt = path[state['targetPointIndex'] - 1]
        if len(self.pathPt):
            self.canvas.coords(self.next_point, self.pathPt[0] - l, translated_y(self.pathPt[1] - l),
                               self.pathPt[0] + l, translated_y(self.pathPt[1] + l))
