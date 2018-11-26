import numpy as np
from sensor_rangefinder2d import SensorRangefinder2d
from visual_translated_y import translated_y

scale = 100


class VisualMapScans2d:
    """
      This block collects a scan map.
      Visualization of the scan map is done by storing a Mx2 array of absolute
      ray endpoint coordinates for every scan in the state.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.rangefinder = SensorRangefinder2d(canvas)

    def draw_map(self, X):
        state_data = self.add_scan(X)

        state = np.dot(state_data, scale)
        for i in range(len(state[0])):
            self.canvas.create_oval(state[0][i], translated_y(state[1][i]), state[0][i]+1, translated_y(state[1][i]+1),
                                    fill='', outline="blue", width=1)

    def add_scan(self, X):
        out = self.rangefinder.sample(X)
        ranges = np.array(out['ranges'])[-1]
        bearing = np.array(out['bearings'])[-1]

        validIdx = (np.isinf(ranges) == False)
        pose = X
        abs_bearings = bearing[validIdx] + pose[2]
        state = [ranges[validIdx] * np.cos(abs_bearings) + pose[0],
                 ranges[validIdx] * np.sin(abs_bearings) + pose[1]]
        return state

