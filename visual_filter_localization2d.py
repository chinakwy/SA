import numpy as np
import numpy.matlib  # np.matlib.repmat
from visual_platform2d import VisualPlatform2d  # platform.robot_radius
from visual_translated_y import translated_y


scale = 100


class VisualFilterLocalization2d:
    """ This block implements the common behaviour of a localization algorithm
        for a ground robot, i. e. common parameters and the visualization, but
        not the actual algorithm. The block should be used as a "base class" by
        instantiating it and passing the localization algorithm as a function
        handle.

        The expected format are block inputs are
        - first input: a struct with field .pose = [r_x; r_x; phi]
        - number and type of further inputs is irrelevant

        The expected output format is a struct with fields
        - .pose = [r_x; r_y; phi] (mean of pose probability distribution)
        - .cov = 3x3 covariance matrix of .pose
    """
    def __init__(self, canvas):
        # self.color = '#6495ED'
        # filter.graphicElements(end).useLogs = true;
        self.sigmaScale = 3
        # self.radius = 10
        platform = VisualPlatform2d(canvas)
        self.radius = platform.robot_radius

        self.useBearing = True
        self.useRange = True

        self.initialPose = []
        self.initialPoseError = [0.005, 0.005, 1 * np.pi / 180]
        self.odometryError = [0.005, 0.005, 0.5 * np.pi / 180]
        self.bearingError = 5 * np.pi / 180
        self.rangeError = 1 / 100

        self.canvas = canvas

        self.ellipse = self.canvas.create_polygon(0, 0, 0, 0, fill='', outline='#6495ED', width=2)  # '#6495ED'
        self.pie = self.canvas.create_polygon(0, 0, 0, 0, fill='', outline='#00FFFF', width=2)  # '#00FFFF'

    def draw_track(self, t, xs):
        # xs = np.array(local_filter.filter_step(t, Xs[-1], controller.control(t, Xs)).pose)
        if t > 0.1:
            self.canvas.create_line(xs[-2][0] * scale, translated_y(xs[-2][1] * scale),
                                    xs[-1][0] * scale, translated_y(xs[-1][1] * scale),
                                    fill='#6495ED', dash=(7, 3))  # dash: Line segment

    def draw_pose(self, cov, pose):
        # error ellipse

        eigval, eigvec = np.linalg.eig(cov[0:2, 0:2])
        ellipse_t = np.linspace(0, 2 * np.pi, 100)
        xy1 = np.diag(self.radius + np.dot(self.sigmaScale, np.power(eigval, 0.5)))
        xy2 = np.hstack((np.cos(ellipse_t).reshape((len(ellipse_t), 1)),
                         np.sin(ellipse_t).reshape((len(ellipse_t), 1))))
        xy3 = np.dot(np.dot(xy2, xy1), eigvec.T)
        xy4 = np.matlib.repmat([pose[0], pose[1]], len(ellipse_t), 1)
        xy = xy3 + xy4
        xydata = []
        for i in range(len(ellipse_t)):
            xdata = xy[i][0] * scale
            ydata = translated_y(xy[i][1] * scale)
            xydata.append(xdata)
            xydata.append(ydata)
        self.canvas.coords(self.ellipse, xydata)

        # pie slice for visualizing orientation error
        ranges = pose[2] + np.dot(self.sigmaScale, np.power(cov[2][2], 0.5)) * np.array([-1, 1])
        pie_t = np.linspace(ranges[0], ranges[1], 10)
        pielength = self.radius * 3
        pie_xydata = [pose[0] * scale, translated_y(pose[1] * scale)]
        for i in range(len(pie_t)):
            pie_xdata = (pose[0] + pielength * np.cos(pie_t[i])) * scale
            pie_ydata = translated_y((pose[1] + pielength * np.sin(pie_t[i])) * scale)
            pie_xydata.append(pie_xdata)
            pie_xydata.append(pie_ydata)
        self.canvas.coords(self.pie, pie_xydata)

