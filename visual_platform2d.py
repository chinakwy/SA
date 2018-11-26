import numpy as np
from visual_translated_y import translated_y

scale = 100


class VisualPlatform2d:
    def __init__(self, canvas1):
        self.robot_radius = 0.1
        self.robot_color = 'blue'
        self.canvas = canvas1
        self.body = self.canvas.create_oval(0, 0, 0, 0, fill=self.robot_color, outline='white')
        self.direction = self.canvas.create_line(0, 0, 0, 0, fill='white', width=3)

    def draw_bot(self, X_data):
        r = self.robot_radius * scale
        X = np.array([X_data[0]*scale, X_data[1]*scale, X_data[2]])
        self.canvas.coords(self.direction, (X[0], translated_y(X[1]),
                                       X[0] + r * np.cos(X[2]), translated_y(X[1] + r * np.sin(X[2]))))
        self.canvas.coords(self.body, (X[0] - r, translated_y(X[1] - r), X[0] + r, translated_y(X[1] + r)))

    def draw_track(self, X_data, X_last_data):
        X = np.array([X_data[0] * scale, X_data[1] * scale, X_data[2]])
        X_last = np.array([X_last_data[0] * scale, X_last_data[1] * scale, X_last_data[2]])
        self.canvas.create_line(X_last[0], translated_y(X_last[1]), X[0], translated_y(X[1]), fill='blue')


