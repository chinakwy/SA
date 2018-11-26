import numpy as np
import numpy.matlib  # np.matlib.repmat
from visual_translated_y import translated_y
scale = 100


def model_platform2d_on_path(path, t, Xs):
    global state
    v = -0.5  # translational speed in m/s
    omega = 90 * np.pi / 180
    initialPose = np.array([path[0][0], path[0][1], np.arctan2(path[1][1]-path[0][1], path[1][0]-path[0][0])])
    firstTargetPoint = 2
    startBackwards = False
    odometryError = np.array([0.005, 0.005, (0.5 * np.pi / 180)])  # stddev for odometry [m, m, rad]

# def move()
    if t <= 0.1:
        state = {}
        state['targetPoint'] = np.minimum(len(path), firstTargetPoint)
        state['backwards'] = startBackwards
        state['doTurn'] = True
        state['lastT'] = 0.0
        state['pose'] = initialPose

    if state['backwards']:
        di_r = -1
    else:
        di_r = 1
    deltaT = t - state['lastT']  # deltaT = t - state.lastT
    state['lastT'] = t
    v = abs(v)
    omega = abs(omega)
    pose = np.array(state['pose'])

    while deltaT > 0:
        delta = path[state['targetPoint']-1] - pose[0:2]
        if all(delta == 0):
            # determine next point
            if all(np.array(path[0]) == np.array(path[-1])):
                # path closed -> circle
                if di_r == 1:
                    if state['targetPoint'] < len(path):
                        state['targetPoint'] = state['targetPoint'] + 1
                    else:
                        state['targetPoint'] = 2
                else:
                    if state['targetPoint'] > 1:
                        state['targetPoint'] = state['targetPoint'] - 1
                    else:
                        state['targetPoint'] = len(path) - 1
            else:
                # path not closed -> toggle between start and end point
                if di_r == 1:
                    if state['targetPoint'] < len(path):
                        state['targetPoint'] = state['targetPoint'] + 1
                    else:
                        state['backwards'] = not state['backwards']
                        state['targetPoint'] = len(path) - 1
                else:
                    if state['targetPoint'] > 1:
                        state['targetPoint'] = state['targetPoint'] - 1
                    else:
                        state['backwards'] = not state['backwards']
                        state['targetPoint'] = 2
            state['doTurn'] = True
            delta = path[state['targetPoint']-1] - pose[0:2]
        if state['doTurn']:
            # adjust heading towards target point,
            # compute angle difference \in (-pi, pi]
            destAngle = np.arctan2(delta[1], delta[0])
            deltaAngle = (destAngle - pose[2]) % (2*np.pi)
            if deltaAngle > np.pi:
                deltaAngle = deltaAngle - 2*np.pi

            tWholeTurn = abs(deltaAngle)/omega
            if tWholeTurn > deltaT:
                pose[2] = pose[2] + deltaT*omega*np.sign(deltaAngle)
                deltaT = 0
            else:
                # remaining angle difference can be removed completely within this time step
                pose[2] = destAngle
                deltaT = deltaT - tWholeTurn
                state['doTurn'] = False
        else:
            # approach target point
            distance = np.linalg.norm(np.array(delta))  # find norm
            tWholeDistance = distance / v
            if tWholeDistance > deltaT:
                # remaining distance cannot be travelled completely in deltaT
                pose[0:2] = pose[0:2] + deltaT * v / distance * np.array(delta)
                deltaT = 0
            else:
                # remaining distance can be completely travelled within this time step
                pose[0:2] = path[state['targetPoint'] - 1]
                deltaT = deltaT - tWholeDistance

    delta_noisy = pose - state['pose'] + odometryError*numpy.random.randn(1, 3)
    state['pose'] = [pose[0], pose[1], (pose[2]+np.pi) % (2*np.pi)-np.pi]

    X = np.array(state['pose'])
    Xs.append(X)
    return Xs


class Path2d:
    def __init__(self, canvas):
        self.canvas = canvas

    def draw_path(self, t, path_data):
        path = np.dot(path_data, scale)
        if t <= 0.1:
            for i in range(len(path)-1):

                self.canvas.create_line(path[i][0], translated_y(path[i][1]),
                                                path[i+1][0], translated_y(path[i+1][1]), fill='blue', width=1)

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
