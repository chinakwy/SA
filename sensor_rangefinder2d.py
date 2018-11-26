import numpy as np
import pandas as pd
import numpy.matlib  # np.matlib.repmat
from isect_gridmap_rays import ISectGridMapRays
from visual_translated_y import translated_y
scale = 100


class SensorRangefinder2d:
    def __init__(self, canvas):
        self.canvas = canvas
        self.color = '#A52A2A'
        self.fieldOfView = np.array([-90, 90]) * np.pi / 180  # [rad], relative to robot orientation
        self.increment = 1 * np.pi / 180  # angular difference between adjacent rays in rad
        self.maxRange = 4.5  # maximum detection distance in m
        self.error = 0.5 / 100  # stddev of range error, error scales with distance
        self.map_obstacles = np.load("./map/office.npy")
        # output data format:
        # .range = Nx1 vector
        # .bearing = Nx1 vector
        # numRays...number of rays

        self.numRays = len(np.arange(self.fieldOfView[0], self.fieldOfView[1] + self.increment, self.increment))
        self.drawRays = {}
        for a in range(self.numRays):
            self.drawRays[a] = self.canvas.create_line(0, 0, 0, 0, fill=self.color)

        self.bearings = []  # for sample_out
        self.ranges = []

    def draw_rays(self, X):
        out = self.sample(X)
        bearing = np.array(out['bearings'])[-1]
        ranges = np.array(out['ranges'])[-1]
        data = np.array([X[0], X[1], X[2]])
        pose = np.array(data)
        angles = np.array(bearing) + pose[2]
        ranges = np.array(ranges)
        ranges[np.isinf(ranges)] = self.maxRange
        xy = np.array([ranges * np.cos(angles), ranges * np.sin(angles)]).T
        # numRays = len(ranges)

        x_data = (pose[0] + np.array([np.zeros(self.numRays), xy[:, 0]])) * scale
        y_data = (pose[1] + np.array([np.zeros(self.numRays), xy[:, 1]])) * scale

        for a in range(self.numRays):
            self.canvas.coords(self.drawRays[a], x_data[0][a], translated_y(y_data[0][a]),
                                            x_data[1][a], translated_y(y_data[1][a]))

    def sample(self, X):
        # data = np.array([X[0] / 100, X[1] / 100, X[2]])
        data = np.array(X)
        bearing = np.arange(self.fieldOfView[0], self.fieldOfView[1] + self.increment, self.increment)
        arcs = (bearing + data[2] + np.pi) % (2 * np.pi) - np.pi
        map_scale = 0.01
        map_offset = [0, 0]
        rayStart = np.matlib.repmat((data[0:2] - map_offset) / map_scale, len(arcs), 1)
        rayEnd = rayStart + (self.maxRange / map_scale) * np.array([np.cos(arcs), np.sin(arcs)]).T

        isect_detector = ISectGridMapRays()
        isect, ranges = isect_detector.intersection_gridmap_rays(self.map_obstacles, rayStart, rayEnd)

        scale = 0.01
        ranges[isect == 0] = float('inf')
        ranges = ranges * scale

        # apply distance-dependent gaussian noise
        ranges[isect == 1] = ranges[isect == 1] + self.error * (ranges[isect == 1] * np.random.rand(np.sum(isect)))  #❗️
        self.bearings.append(bearing)
        self.ranges.append(ranges)
        data = {'bearings': self.bearings, 'ranges': self.ranges}
        out = pd.DataFrame(data)

        return out
