import numpy as np
from visual_translated_y import translated_y
scale = 100


class VisualConstPoints2d:
    def __init__(self, canvas, path_points):
        self.canvas = canvas
        self.color = 'blue'
        self.pathPoints = np.dot(path_points, scale)

    def draw_const_points(self, t):
        if t <= 0.1:
            p = 4
            a = 1
            for _ in range(len(self.pathPoints)):
                x = self.pathPoints[_][0]
                y = translated_y(self.pathPoints[_][1])
                star_points = []
                for i in (1, -1):
                    star_points.extend((x, y + i * p))
                    star_points.extend((x + i * a, y + i * a))
                    star_points.extend((x + i * p, y))
                    star_points.extend((x + i * a, y - i * a))
                self.canvas.create_polygon(star_points, fill=self.color)
