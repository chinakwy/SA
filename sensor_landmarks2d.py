import numpy as np
import numpy.matlib  # np.matlib.repmat
import pandas as pd  # pd.DataFrame
from isect_gridmap_rays import ISectGridMapRays
from visual_translated_y import translated_y

scale = 100
map_scale = 0.01  # 用在rayStarts
map_offset = [0, 0]


class SensorLandmarks2d:  # sampling_time = 1/15
    def __init__(self, canvas, useEmptyRoom):
        self.range = 7
        self.fieldOfView = np.array([-65, 65]) * np.pi / 180
        self.bearingError = 5 * np.pi / 180
        self.rangeError = 1 / 100
        self.color = 'blue'
        self.sampling_time = 1 / 15  # Abtastzeit

        self.ranges = []  # for pd.DataFrame
        self.bearings = []
        self.lmIds = []
        self.ts = []

        if useEmptyRoom:
            self.landmarks = np.array([0.175000000000000, 0.17500000000000,
                                       0.175000000000000, 5.16500000000000,
                                       4.04500000000000, 5.16500000000000,
                                       4.04500000000000, 5.84500000000000,
                                       5.25500000000000, 5.84500000000000,
                                       6.64500000000000, 5.84500000000000,
                                       7.49500000000000, 0.17500000000000,
                                       7.84500000000000, 0.17500000000000,
                                       7.84500000000000, 1.04500000000000,
                                       7.84500000000000, 5.24500000000000,
                                       7.84500000000000, 5.51500000000000,
                                       7.84500000000000, 5.84500000000000]).reshape(-1, 2)
            self.map_obstacles = np.load("./map/emptyroom.npy")

        else:
            self.landmarks = np.array(
                [0.175000000000000, 0.175000000000000, 0.175000000000000, 0.685000000000000, 0.175000000000000,
                 1.21500000000000, 0.175000000000000, 1.98500000000000, 0.175000000000000, 2.64500000000000,
                 0.175000000000000, 3.49500000000000, 0.175000000000000, 3.67500000000000, 0.175000000000000,
                 3.88500000000000, 0.175000000000000, 5.66500000000000, 0.175000000000000, 5.84500000000000,
                 0.305000000000000, 3.89500000000000, 0.305000000000000, 4.80500000000000, 0.305000000000000,
                 5.66500000000000, 0.385000000000000, 3.50500000000000, 0.385000000000000, 3.67500000000000,
                 0.655000000000000, 1.21500000000000, 0.665000000000000, 0.695000000000000, 0.755000000000000,
                 0.175000000000000, 1.25500000000000, 5.84500000000000, 1.31500000000000, 3.50500000000000,
                 1.31500000000000, 3.67500000000000, 1.38500000000000, 3.67500000000000, 1.38500000000000,
                 3.79500000000000, 1.49500000000000, 2.00500000000000, 1.49500000000000, 2.79500000000000,
                 1.49500000000000, 3.49500000000000, 1.67500000000000, 2.17500000000000, 1.67500000000000,
                 2.78500000000000, 1.67500000000000, 3.49500000000000, 1.93500000000000, 0.175000000000000,
                 1.93500000000000, 0.365000000000000, 1.93500000000000, 1.80500000000000, 1.93500000000000,
                 1.99500000000000, 2.08500000000000, 3.67500000000000, 2.08500000000000, 3.79500000000000,
                 2.10500000000000, 1.79500000000000, 2.11500000000000, 0.175000000000000, 2.11500000000000,
                 0.365000000000000, 2.11500000000000, 1.99500000000000, 2.17500000000000, 3.49500000000000,
                 2.17500000000000, 3.67500000000000, 2.28500000000000, 2.17500000000000, 2.49500000000000,
                 5.84500000000000, 2.85500000000000, 1.99500000000000, 3.14500000000000, 3.49500000000000,
                 3.14500000000000, 3.67500000000000, 3.33500000000000, 0.175000000000000, 3.34500000000000,
                 2.17500000000000, 3.49500000000000, 3.67500000000000, 3.49500000000000, 4.82500000000000,
                 3.49500000000000, 5.84500000000000, 3.67500000000000, 3.67500000000000, 3.67500000000000,
                 4.79500000000000, 3.67500000000000, 5.84500000000000, 3.81500000000000, 1.99500000000000,
                 4.19500000000000, 3.49500000000000, 4.34500000000000, 0.175000000000000, 4.51500000000000,
                 2.17500000000000, 4.51500000000000, 2.37500000000000, 4.51500000000000, 3.67500000000000,
                 4.75500000000000, 5.84500000000000, 5.16500000000000, 1.99500000000000, 5.18500000000000,
                 2.37500000000000, 5.37500000000000, 0.175000000000000, 5.37500000000000, 0.415000000000000,
                 5.37500000000000, 3.49500000000000, 5.37500000000000, 3.67500000000000, 5.84500000000000,
                 5.54500000000000, 5.84500000000000, 5.84500000000000, 6.04500000000000, 0.425000000000000,
                 6.04500000000000, 2.17500000000000, 6.04500000000000, 2.37500000000000, 6.34500000000000,
                 3.50500000000000, 6.34500000000000, 4.09500000000000, 6.34500000000000, 4.65500000000000,
                 6.47500000000000, 1.30500000000000, 6.47500000000000, 1.99500000000000, 6.52500000000000,
                 3.67500000000000, 6.52500000000000, 4.65500000000000, 6.59500000000000, 0.425000000000000,
                 6.60500000000000, 0.175000000000000, 6.85500000000000, 1.29500000000000, 6.85500000000000,
                 1.99500000000000, 6.88500000000000, 5.54500000000000, 6.92500000000000, 1.99500000000000,
                 6.93500000000000, 2.17500000000000, 7.16500000000000, 3.49500000000000, 7.84500000000000,
                 0.175000000000000, 7.84500000000000, 1.11500000000000, 7.84500000000000, 2.16500000000000,
                 7.84500000000000, 3.49500000000000, 7.84500000000000, 3.67500000000000, 7.84500000000000,
                 4.61500000000000, 7.84500000000000, 5.54500000000000]).reshape(-1, 2)
            self.map_obstacles = np.load("./map/office.npy")

        self.canvas = canvas
        self.fov = canvas.create_polygon(0, 0, 0, 0, fill='', outline='gold', width=7)
        self.connections = {}
        for i in range(len(self.landmarks)):
            self.connections[i] = self.canvas.create_line(0, 0, 0, 0, fill='blue', dash=(7, 1, 2, 1), width=2)

    def sample(self, t, X):
        global landmarks_t  # landmarks_t += 1/15
        if t == 0:
            landmarks_t = 0

        while landmarks_t < t+0.0999:
            pose = X
            # map_scale = 0.01
            # map_offset = [0, 0]
            lmPositions = self.landmarks

            bearings = np.arctan2(lmPositions[:, 1] - pose[1], lmPositions[:, 0] - pose[0]) - pose[2]
            bearings = np.mod(bearings + np.pi, 2 * np.pi) - np.pi
            lmIds = np.arange(len(lmPositions))  # 0到11

            inFovIdx = np.where((bearings >= self.fieldOfView[0]) & (bearings <= self.fieldOfView[1]))[0]
            bearings = bearings[inFovIdx]
            lmIds = lmIds[inFovIdx]

            ranges = np.power(np.sum((lmPositions[list(inFovIdx)] - np.matlib.repmat(pose[0:2], len(bearings), 1))
                                     ** 2, 1), 0.5)
            inRangeIdx = np.where(ranges < self.range)[0]
            ranges = ranges[inRangeIdx]
            bearings = bearings[inRangeIdx]
            lmIds = lmIds[inRangeIdx]

            # detect visible landmarks

            rayStarts = np.matlib.repmat((pose[0:2] - map_offset) / map_scale, len(lmIds), 1)
            rayEnds = (lmPositions[lmIds] - np.matlib.repmat(map_offset, len(lmIds), 1)) / map_scale

            isect_detector = ISectGridMapRays()
            isect, _ = isect_detector.intersection_gridmap_rays(self.map_obstacles, rayStarts, rayEnds)

            visIdx = (isect == 0).nonzero()[0]
            ranges = ranges[visIdx]
            bearings = bearings[visIdx]
            lmIds = lmIds[visIdx]

            bearings = np.mod((bearings + np.dot(self.bearingError, np.random.randn(bearings.size))) + np.pi,
                              2*np.pi) - np.pi
            ranges = ranges + (np.dot(self.rangeError, ranges) * np.random.randn(ranges.size))

            if round(landmarks_t, 4) not in self.ts and round(landmarks_t, 4) > 0:
                # I don't know why sometimes the same result will appear three times, so I added this one.
                self.ranges.append(ranges)
                self.bearings.append(bearings)
                self.lmIds.append(lmIds)
                self.ts.append(round(landmarks_t, 4))

            landmarks_t += self.sampling_time
        data = {'t': self.ts, 'ranges': self.ranges, 'bearings': self.bearings, 'lmIds': self.lmIds}
        self.out = pd.DataFrame(data)
        return self.out

    def draw_fov(self, X_data):
        pose = np.array([X_data[0]*scale, X_data[1]*scale, X_data[2]])
        ranges = self.range * scale

        fov_ts = pose[2] + np.linspace(self.fieldOfView[0], self.fieldOfView[1], 20)
        xydata = [pose[0], translated_y(pose[1])]
        for i in range(20):
            xdata = pose[0] + ranges * np.cos(fov_ts[i])
            ydata = translated_y(pose[1] + ranges * np.sin(fov_ts[i]))
            xydata.append(xdata)
            xydata.append(ydata)
        self.canvas.coords(self.fov, xydata)

    def draw_connections(self, t, X_data):
        pose = np.array([X_data[0]*scale, X_data[1]*scale, X_data[2]])
        landmarks = np.dot(self.landmarks, scale)

        # sample_outpt = self.sample(t, pose)
        sample_outpt = self.out
        lmIds = np.array(sample_outpt['lmIds'])[-1]

        rest_ids = list(set(np.arange(len(landmarks))).difference(set(lmIds)))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.connections[rest_ids[_]], 0, 0, 0, 0)

        lmCount = len(lmIds)  # lmCount = len(out.lmIds)
        for i in range(lmCount):
            self.canvas.coords(self.connections[lmIds[i]], pose[0], translated_y(pose[1]),
                               landmarks[lmIds[i]][0], translated_y(landmarks[lmIds[i]][1]))

    def draw_landmarks(self, t):
        landmarks = np.dot(self.landmarks, scale)

        if t <= 0.1:
            p = 4
            a = 2
            for _ in range(len(landmarks)):
                x = landmarks[_][0]
                y = translated_y(landmarks[_][1])
                star_points = []
                for i in (1, -1):
                    star_points.extend((x, y + i * p))
                    star_points.extend((x + i * a, y + i * a))
                    star_points.extend((x + i * p, y))
                    star_points.extend((x + i * a, y - i * a))
                self.canvas.create_polygon(star_points, fill='#3CB371', outline='black')
