import numpy as np
import pandas as pd
import numpy.matlib  # np.matlib.repmat
from scipy.linalg import block_diag
from visual_translated_y import translated_y
from sensor_rangefinder2d import SensorRangefinder2d

scale = 100


class GuidanceDWA2d:
    def __init__(self, canvas):
        self.vMin = 0  # min. translational velocity [m / s]
        self.vMax = 1  # max. translational velocity [m / s]
        self.omegaMin = -90 * np.pi / 180  # min. angular velocity [rad / s]
        self.omegaMax = 90 * np.pi / 180  # max. angular velocity [rad / s]
        self.accV = 2  # translational acceleration/deceleration [m / s^2]
        self.accOmega = 180 * np.pi / 180  # angular acceleration/deceleration [rad / s^2]
        # weight factors: [heading, obstacle distance, velocity, approach]
        self.weights = np.array([5, 1, 0.2, 1])
        self.maxDistanceCare = 0.7  # upper limit for obstacle distance utility [m]

        # obstacle line field related parameters
        self.safetyMargin = 0.05  # additional clearance between robot & obstacles [m]

        self.radius = 0.1  # robot radius (usually overwritten in experiment file [m]

        self.showAllCollisionPoints = True
        # set to True, to visualize all collision distances (e. g. to verify your calculation of the collision distance)
        # by default (False) the collision distances are only visualized for non-admissible candidates

        self.debugOuts = []

        self.canvas = canvas
        sensor_range_finder2d = SensorRangefinder2d(canvas)
        self.numRays = sensor_range_finder2d.numRays
        self.ObstacleLines = {}  # obstacleLines, Ge:Linien-Hindernis
        for i in range(self.numRays):
            self.ObstacleLines[i] = self.canvas.create_line(0, 0, 0, 0, fill='#FF00FF')  # Pink

        self.maxlen_arcs = 225

        self.TrajCandidates_curves = {}
        self.maxN_TrajCandidates_curves = 8000
        for _ in range(self.maxN_TrajCandidates_curves):
            self.TrajCandidates_curves[_] = self.canvas.create_line(0, 0, 0, 0, fill='#32CD32')  # LimeGreen
        self.TrajCandidates_minDist = {}
        self.maxN_TrajCandidates_minDist = 1000
        for _ in range(self.maxN_TrajCandidates_minDist):
            self.TrajCandidates_minDist[_] = self.canvas.create_line(0, 0, 0, 0, fill='#FF8C00', width=3)  # DarkOrange

        self.SelectedTrajectory = {}
        for _ in range(self.maxlen_arcs):
            self.SelectedTrajectory[_] = self.canvas.create_line(0, 0, 0, 0, fill='#008000', width=3)  # Green
# ######################################################
# visualization functions for the main window
#######################################################

    def draw_obstacle_lines(self, Xs):  # efficient analytical collision check of trajectory candidates
        debugOut = self.debugOuts[-1]
        lineData = debugOut['obstacleLines']
        pose = Xs[-1]

        N_obst = len(lineData)
        R_mat = np.mat([[np.cos(pose[2]), -np.sin(pose[2])],
                        [np.sin(pose[2]), np.cos(pose[2])]])

        transformedLineData = (np.matlib.repmat([pose[0], pose[1]], N_obst, 2) +
                               np.dot(lineData, block_diag(R_mat.T, R_mat.T))) * scale

        for i in range(len(debugOut['obstacleLines'])):
            self.canvas.coords(self.ObstacleLines[i],
                               transformedLineData[i][0], translated_y(transformedLineData[i][1]),
                               transformedLineData[i][2], translated_y(transformedLineData[i][3]))
        rest_ids = list(set(np.arange(self.numRays)).difference(set(np.arange(len(debugOut['obstacleLines'])))))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.ObstacleLines[rest_ids[_]], 0, 0, 0, 0)

    def draw_traj_candidates(self, Xs):  # Green line
        debugOut = self.debugOuts[-1]
        pose = Xs[-1]
        R_mat = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])

        N_rows = np.size(debugOut['omegas'], axis=0)
        N_cloumns = np.size(debugOut['omegas'], axis=1)
        curvePts = np.zeros((0, 2))
        for i in range(N_rows):
            for j in range(N_cloumns):
                v = np.array(debugOut['velocities'])[i][j]
                omega = np.array(debugOut['omegas'])[i][j]

                if i+j > 0:
                    curvePts = np.vstack((curvePts, np.array([np.inf, np.inf])))

                if omega != 0:  # curve
                    R = v / omega
                    arcs = np.linspace(0, 5 * np.pi / 4,
                                       np.min((self.maxlen_arcs, np.max((10, 12.5 * np.pi * np.abs(R)))))).reshape(-1, 1)
                    curvePts = np.vstack((curvePts,
                                          np.hstack((np.sign(v) * np.abs(R) * np.sin(arcs), R * (1 - np.cos(arcs))))))
                else:  # straight line
                    curveLength = v * 2
                    curvePts = np.vstack((curvePts, np.array([[0, 0], [curveLength, 0]])))

        curvePts = np.array(np.matlib.repmat(pose[0:2], len(curvePts), 1) + np.dot(curvePts, R_mat.T)) * scale
        where_are_nan = np.isnan(curvePts)
        curvePts[where_are_nan] = np.inf
        tcc_number = len(curvePts) - 1
        for _ in range(tcc_number):
            self.canvas.coords(self.TrajCandidates_curves[_], curvePts[_][0], translated_y(curvePts[_][1]),
                               curvePts[_+1][0], translated_y(curvePts[_+1][1]))
        rest_ids = list(set(np.arange(self.maxN_TrajCandidates_curves)).difference(set(np.arange(tcc_number))))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.TrajCandidates_curves[rest_ids[_]], 0, 0, 0, 0)

        tcm_curvePts = np.zeros((0, 2))
        for i in range(N_rows):
            for j in range(N_cloumns):
                if self.showAllCollisionPoints or not debugOut['admissibleCandidates'][i][j]:
                    d_coll = debugOut['minDists'][i][j]
                    v = debugOut['velocities'][i][j]
                    omega = debugOut['omegas'][i][j]
                    if np.isfinite(d_coll):
                        if len(tcm_curvePts) > 0:
                            tcm_curvePts = np.vstack((tcm_curvePts, np.array([np.inf, np.inf])))
                        if omega == 0:
                            lineLength = np.sign(v) * d_coll
                            tcm_curvePts = np.vstack((tcm_curvePts, np.array([[0, 0], [lineLength, 0]])))
                        else:
                            R = v / omega
                            arcs = np.linspace(0, d_coll / np.abs(R), np.min((225, np.max((10, d_coll / 0.1)))))\
                                .reshape(-1, 1)
                            tcm_curvePts = np.vstack((tcm_curvePts,
                                          np.hstack((np.sign(v) * np.abs(R) * np.sin(arcs), R * (1 - np.cos(arcs))))))

        tcm_curvePts = np.array(np.matlib.repmat(pose[0:2], len(tcm_curvePts), 1) + np.dot(tcm_curvePts, R_mat.T)) * scale
        where_are_nan = np.isnan(tcm_curvePts)
        tcm_curvePts[where_are_nan] = np.inf
        tcm_number = len(tcm_curvePts) - 1
        for _ in range(tcm_number):
            self.canvas.coords(self.TrajCandidates_minDist[_], tcm_curvePts[_][0], translated_y(tcm_curvePts[_][1]),
                               tcm_curvePts[_ + 1][0], translated_y(tcm_curvePts[_ + 1][1]))
        rest_ids = list(set(np.arange(self.maxN_TrajCandidates_minDist)).difference(set(np.arange(tcm_number))))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.TrajCandidates_minDist[rest_ids[_]], 0, 0, 0, 0)

    def draw_selected_trajectory(self, Xs):  # Front direction line
        pose = Xs[-1]
        R_mat = np.mat([[np.cos(pose[2]), -np.sin(pose[2])],
                        [np.sin(pose[2]), np.cos(pose[2])]])

        v = out[0]
        omega = out[1]

        if omega != 0:  # curve
            R = v / omega
            arcs = np.linspace(0, 5 * np.pi / 4, np.min((self.maxlen_arcs, np.max((10, 12.5 * np.pi * np.abs(R))))))\
                .reshape(-1, 1)
            curvePts = np.hstack((np.sign(v) * np.abs(R) * np.sin(arcs), R * (1 - np.cos(arcs))))
        else:  # straight line
            curveLength = v * 2
            curvePts = np.array([[0, 0], [curveLength, 0]])

        curvePts = np.array(np.matlib.repmat(pose[0:2], len(curvePts), 1) + np.dot(curvePts, R_mat.T)) * scale
        st_number = len(curvePts) - 1
        for _ in range(st_number):
            self.canvas.coords(self.SelectedTrajectory[_], curvePts[_][0], translated_y(curvePts[_][1]),
                               curvePts[_+1][0], translated_y(curvePts[_+1][1]))

        rest_ids = list(set(np.arange(self.maxlen_arcs)).difference(set(np.arange(st_number))))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.SelectedTrajectory[rest_ids[_]], 0, 0, 0, 0)

#######################################################
# Dynamic Window Approach implementation
#######################################################

    def dwa_step(self, t, relativeGoal, rangefinder_data):
        global out, state
        rangefinder_bearings = np.array(rangefinder_data['bearings'])[-1]
        rangefinder_ranges = np.array(rangefinder_data['ranges'])[-1]
        if t <= 0.1:
            # state is [t, v, omega]
            state = np.array([0, 0, 0])
        debugOut = {}

        debugOut['prevState'] = state
        # if ~isempty(rangefinder) && ~isempty(relativeGoal)
        # there might be a circular references between this module and the platform,
        # which can cause DWA to be compute before an actual pose has been initialized
        T = t - state[0]  # timestep
        velocity = state[1]  # current translational velocity
        omega = state[2]  # current angular velocity

        # ===== Task 1 ================================================
        # Compute Obstacles Line Field from laser range data.
        #
        # Laser Range data is provided as two column vectors
        # - bearings are the ray angles, measured relative to the
        # robot's orientation. The elements are monotonically
        # increasing, i. e. rays are stored from right to left.
        bearings = rangefinder_bearings
        # ranges are the measured obstacle distances. We immediately
        # subtract the desired minimum obstacle distance
        minDistance = self.radius + self.safetyMargin
        ranges = np.array(rangefinder_ranges) - minDistance
        # if a ray did not hit an obstacle within the measurement
        # range, the corresponding entry is set to inf. These rays will
        # not become an obstacle line later.
        invalidRays = (np.isinf(ranges) == 0)

        deltaBearings = bearings[1:] - bearings[0:-1]
        rayExtentRight = np.dot(0.5, ranges) * np.tan(np.concatenate((deltaBearings[0], deltaBearings), axis=None))
        rayExtentLeft = np.dot(0.5, ranges) * np.tan(np.concatenate((deltaBearings, deltaBearings[-1]), axis=None))

        ranges = ranges[invalidRays]
        bearings = bearings[invalidRays]
        rayExtentLeft = rayExtentLeft[invalidRays]
        rayExtentRight = rayExtentRight[invalidRays]

        rayVectors = np.hstack((np.array(ranges * np.cos(bearings)).reshape(-1, 1),
                                np.array(ranges * np.sin(bearings)).reshape(-1, 1)))
        rayNormals = np.hstack((-rayVectors[:, 1].reshape(-1, 1), rayVectors[:, 0].reshape(-1, 1))) \
                     / np.matlib.repmat(np.power(np.sum(rayVectors**2, 1), 0.5).reshape(-1, 1), 1, 2)

        # Obstacle lines should be stored in a Nx4 matrix, where each
        # row contains the coordinates of the start and end point of
        # the obstacle line, i. e. [px py qx qy]
        # TODO:
        obstacleLines = np.hstack((rayVectors + np.matlib.repmat(rayExtentLeft.reshape(-1, 1) + minDistance, 1, 2) * rayNormals,
                                   rayVectors - np.matlib.repmat(rayExtentRight.reshape(-1, 1) + minDistance, 1, 2) * rayNormals))

        debugOut['obstacleLines'] = obstacleLines  # Store obstacle Lines for visualization

        # ===== Task 2 ================================================
        # Determine the v/omega candidates

        # compute the limits of the dynamic window
        # TODO
        minV = np.max((velocity - T * self.accV, self.vMin))
        maxV = np.min((velocity + T * self.accV, self.vMax))
        minOmega = np.max((omega - T * self.accOmega, self.omegaMin))
        maxOmega = np.min((omega + T * self.accOmega, self.omegaMax))
        # Generate the candidates as two matrices
        # - velocities and
        # - omegas
        # Hint: use the 'meshgrid' function
        # TODO
        omegas, velocities = np.meshgrid(np.linspace(minOmega, maxOmega, 5), np.linspace(minV, maxV, 7))

        # prevent insanely large arcs by snapping tiny angular rates to
        # zero; store results for visualization
        omegas[np.abs(omegas) < 1e-4] = 0
        debugOut['velocities'] = velocities
        debugOut['omegas'] = omegas

        # ===== Task 3 ================================================
        # Determine minimal collision distance for each v/omega pair
        # Note: You have to complete the function lineFieldMinDist at
        # the end of this file!
        szCandidates = np.shape(velocities)
        d_coll = np.zeros(szCandidates)
        for i in range(len(velocities[0])):
            for j in range(len(velocities)):
                d_coll[j][i] = line_field_min_dist(velocities[j][i], omegas[j][i], obstacleLines)

        # ===== Task 4 ================================================
        # Rule out non-admissible v/omega candidates

        # Assume the candidate v/omega pair is applied for one timestep
        # and compute the distance traveled.
        # TODO
        dNextStep = T * np.abs(velocities)

        # After one timestep, determine the required distance to
        # completely stop the robot (i. e. v -> 0 AND omega -> 0). For
        # simplification it is assumed that the arc radius stays
        # constant during the deceleration phase.
        # TODO
        dDeceleration = max_mtlb(velocities**2 / (2 * self.accV), np.abs(velocities * omegas)/(2 * self.accOmega))

        # admissible candidates require a collision distance larger
        # than the combined distance from the next timestep and the
        # deceleration phase.
        admissible = d_coll > (np.array(dNextStep) + np.array(dDeceleration))

        debugOut['minDists'] = d_coll
        debugOut['admissibleCandidates'] = admissible

        # ===== Task 5 ================================================
        # Compute the components of the objective function.
        # Because of the visualization, this is done for all candidates
        # (admissible and non-admissible ones)
        #
        # ----- (a): Heading Utility ----------------------------------
        # predict the location of the robot after
        # - using v/omega for one timestep and then
        # - reducing omega to zero
        # while keeping the arc radius constant.
        # Hint: Reuse dNextStep and dDeceleration from above!

        dPred = np.array(dNextStep) + np.array(dDeceleration)

        # TODO: compute predicted positions for all v/omega candidates
        phis = omegas / np.abs(velocities) * dPred

        phis[np.isfinite(phis) == 0] = 0
        radii = velocities / omegas
        x_data = radii * np.sin(phis)
        y_data = radii * (1 - np.cos(phis))

        # for omega = 0 (i.e.straight line motion)
        straight = (omegas == 0)
        x_data[straight] = np.sign(velocities[straight]) * dPred[straight]
        y_data[straight] = 0

        # compute relative target angle
        # TODO: compute relative target angle and the heading utility
        goal = np.array(relativeGoal)[-1][0]
        thetas = np.abs(np.remainder(np.arctan2(goal[1] - y_data, goal[0] - x_data) - phis + np.pi,
                                     2 * np.pi) - np.pi)
        backwards = (velocities < 0)
        thetas[backwards] = np.pi - thetas[backwards]
        utility_heading = 1 - (thetas / np.pi)

        # utility_heading = 1 - (thetas / pi)
        debugOut['utility_heading'] = utility_heading

        # ----- (b): Obstacle Clearance Utility -----------------------
        #  Use predicted obstacle distance after one timestep and the
        #  deceleration phase
        # TODO
        utility_dist = min_mtlb(1, max_mtlb(0, d_coll - dNextStep - dDeceleration) / self.maxDistanceCare)

        # store for visualization
        debugOut['utility_dist'] = utility_dist

        # ----- (c): Velocity Utility ---------------------------------
        # The faster the robot, the better
        # TODO
        utility_velocity = np.abs(velocities) / self.vMax

        # store for visualization
        debugOut['utility_velocity'] = utility_velocity

        # ----- (-): Approach Utility ---------------------------------
        # % Use the predicted target pose after one timestep and the
        # % deceleration phase (reuse X, Y) from 5a) to compute a
        # % predicted goal distance.
        # % Then compute the reduction in goal distance for each v/omega
        # % candidate and form a utility value in the interval [0, 1]
        utility_approach = np.power((goal[0]**2 + goal[1]**2), 0.5) - \
                           np.power(((x_data - goal[0])**2 + (y_data - goal[1])**2), 0.5)
        minApproach = np.min((np.min((utility_approach))))
        maxApproach = np.max((np.max((utility_approach))))
        if minApproach < maxApproach:
            utility_approach = 0.5 + 0.5 * utility_approach / np.max((np.abs(minApproach), np.abs(maxApproach)))
        else:
            utility_approach[:] = 0

        # store for visualization
        debugOut['utility_approach'] = utility_approach

        # create summed utility function
        utility_sum = self.weights[0] * np.array(utility_heading) + self.weights[1] * np.array(utility_dist) + \
                      self.weights[2] * np.array(utility_velocity) + self.weights[3] * np.array(utility_approach)
        # optionally smooth the result
        # %utility_sum = imfilter(utility_sum, fspecial('gaussian',[2 2],2), 'replicate', 'same');
        # % Drop non-admissible velocities
        utility_sum[admissible == 0] = 0
        debugOut['utility_sum'] = utility_sum

        # select the v/omega pair with the highest utility value
        bestUtility = np.max((np.array(utility_sum).reshape(-1, 1)))
        utility_sum_list = np.array(utility_sum).reshape(-1, 1).tolist()
        bestIndex = utility_sum_list.index(np.max(utility_sum_list))
        if bestUtility > 0:
            velocity = np.array(velocities).reshape(-1, 1).tolist()[bestIndex][0]
            omega = np.array(omegas).reshape(-1, 1).tolist()[bestIndex][0]
        else:
            print('No admissible velocity - applying maximum deceleration\n')
            velocity = np.sign(velocity) * np.max((0, np.abs(velocity) - self.accV * T))
            omega = np.sign(omega) * np.max((0, np.abs(omega) - self.accOmega * T))

        state = [t, velocity, omega]
        self.debugOuts.append(debugOut)

        out = np.array(state[1:3])  # the block's output is [v, omega]
        self.out = out  # For the later figure
        return out


################################################################################
# Determine the collision distance d_coll w.r.t. the obstacle line field on
# a circular arc with radius v/omega or on a straight line (if omega = 0).
# Hint: set showAllCollisionPoints to true to verify your results visually!

def line_field_min_dist(v, omega, obstacles):  # lineFieldMinDist
    d_coll = np.inf
    # collision impossible, if the robot is not moving
    if v == 0:
        return d_coll  # ❗️

    p = np.array(obstacles)[:, [0, 1]]
    D = np.array(obstacles)[:, [2, 3]] - p

    if omega == 0:
        # Robot will move on a straight line
        # TODO: implement intersection line (obstacle) vs. line (trajectory) intersection test

        slnObstacle = -p[:, 1] / D[:, 1]
        dists = np.sign(v) * (p[:, 0] + slnObstacle * D[:, 0])
        isects = ((slnObstacle >= 0) & (slnObstacle <= 1) & (dists >= 0))
        if isects.any():
            d_coll = np.min((dists[isects]))
    else:
        # Robot will move on an arc of radius v/omega
        # Note: both v and omega may be negative, therefore R may be negative, too!
        R = v / omega

        # TODO: implement line (obstacle) vs. circle (trajectory) intersection test

        # The following solution covers all 4 combinations of positive or
        # negative v and positive or negative omega
        p_times_p = np.sum(p**2, 1)
        D_times_D = np.sum(D**2, 1)
        p_times_D = np.sum(p * D, 1)

        # we are solving a quadratic equation here, which is only solvable,
        #  if the expression under the square root is >= 0
        rootTerm = (p_times_D - R * D[:, 1])**2 - D_times_D * (p_times_p - 2 * R * p[:, 1])
        solvable = (rootTerm >= 0)

        # throw away all rows without a non-complex solution
        p = p[solvable, :]
        D = D[solvable, :]
        p_times_D = p_times_D.reshape(-1, 1)[solvable]
        D_times_D = D_times_D.reshape(-1, 1)[solvable]

        rootTerm = np.power((rootTerm.reshape(-1, 1)[solvable]), 0.5)

        # get the solutions (both solutions concatenated in 3rd dimension) ❗️
        sln1 = (R * D[:, 1].reshape(-1, 1) - p_times_D - rootTerm) / D_times_D
        sln2 = (R * D[:, 1].reshape(-1, 1) - p_times_D + rootTerm) / D_times_D
        sln = np.array([sln1, sln2])

        # corresponding coordinates as N (number of obstacles) x 2 (x/y coordinates) x 2 (two solutions) array
        xy1 = p + np.matlib.repmat(sln[0, :, :], 1, 2) * D
        xy2 = p + np.matlib.repmat(sln[1, :, :], 1, 2) * D
        xy = np.array([xy1, xy2])

        # resulting 'position' (angle) on the circular arc as N (number of obstacles) x 1 x 2 (to solutions) array
        # We need the arc angle in the range [0, 2pi)
        phi = np.mod(np.arctan2(np.sign(v) * xy[:, :, 0].reshape(2, -1, 1),
                                np.sign(R) * (R - xy[:, :, 1].reshape(2, -1, 1))),  2 * np.pi)
        # mask pseudo-solutions outside the limits of the corresponding obstacle line
        phi[sln < 0] = np.inf
        phi[sln > 1] = np.inf
        # find the arc angle of the closer solution per obstacle line
        # find the arc closest solution across all obstacle lines
        if not phi.any():
            minPhi = []
        else:
            minPhi = np.min(phi)
        if np.isfinite(minPhi):
            # if the resulting arc angle is finite, an actual intersection
            # % was found, the distance d_coll is the corresponding arc length
            d_coll = np.abs(R) * minPhi
    return d_coll


def max_mtlb(a, b):
    if isinstance(a, int):
        c = [[np.max((a, b[i][j])) for j in range(len(b[i]))] for i in range(len(b))]
    else:
        c = [[np.max((a[i][j], b[i][j])) for j in range(len(a[i]))] for i in range(len(a))]
    return np.array(c)


def min_mtlb(a, b):
    if isinstance(a, int):
        c = [[np.min((a, b[i][j])) for j in range(len(b[i]))] for i in range(len(b))]
    else:
        c = [[np.min((a[i][j], b[i][j])) for j in range(len(a[i]))] for i in range(len(a))]
    return np.array(c)
