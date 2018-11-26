import numpy as np
import pandas as pd
from visual_translated_y import translated_y


scale = 100


class GuidanceWayPoints2d:
    def __init__(self, path, canvas):
        self.canvas = canvas
        self.positionTolerance = 0.3
        self.relative = False
        self.path = path

        self.relativeGoal = []
        self.ts = []

        self.target_point = self.canvas.create_oval(0, 0, 0, 0, outline="blue", width=1)

    def draw_target_point(self):
        r = 5
        if state_targetPointIndex >= 1 & state_targetPointIndex <= len(self.path):
            target_point = self.path[state_targetPointIndex-1] * scale
            self.canvas.coords(self.target_point, target_point[0]-r, translated_y(target_point[1]-r),
                               target_point[0]+r, translated_y(target_point[1]+r))

    def control(self, t, poseProvider):  # poseProvider =Xs
        global state_targetPointIndex
        out = np.array([0, 0])  # default output
        if t <= 0.1:

            state_targetPointIndex = 1
        pose = poseProvider[-1]

        if state_targetPointIndex < len(self.path):
            dist = np.power((np.sum(np.array([self.path[state_targetPointIndex-1][0] - pose[0],
                                              self.path[state_targetPointIndex-1][1] - pose[1]]) ** 2)), 0.5)
            if dist <= self.positionTolerance:
                state_targetPointIndex = state_targetPointIndex + 1

        if state_targetPointIndex <= len(self.path):
            out = np.array([self.path[state_targetPointIndex-1]])
            if self.relative:
                out = np.dot(np.array([[np.cos(pose[2]), np.sin(pose[2])],
                                       [-np.sin(pose[2]), np.cos(pose[2])]]),
                             (out - pose[0:2]).reshape(2, 1)).reshape(1, 2)
        self.relativeGoal.append(out)
        self.ts.append(t)
        data = {'ts': self.ts, 'relativeGoal': self.relativeGoal}
        control_out = pd.DataFrame(data)
        return control_out
        # self.relative = True, 第一个元素不知道代表什么，第二个代表角度
        # poseProvider不知道是否为真实位置
