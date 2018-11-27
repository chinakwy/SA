import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.patches as patches
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D


class VisulaDWAFigure:
    def __init__(self, controller_dwa):
        self.controller = controller_dwa

        self.omegaMin = self.controller.omegaMin  # axVelocity
        self.omegaMax = self.controller.omegaMax
        self.vMin = self.controller.vMin
        self.vMax = self.controller.vMax
        self.xy_bottom_left = (self.omegaMin * 180 / np.pi, self.vMin)
        self.width_rect = (self.omegaMax - self.omegaMin) * 180 / np.pi
        self.heigt_rect = self.vMax - self.vMin

        self.fig = Figure(figsize=(8, 8), dpi=90)
        # self.axUtilitySum = Axes3D(self.fig)

        self.axVelocity = self.fig.add_subplot(2, 2, 1)
        self.axHeadingUtility = self.fig.add_subplot(2, 4, 5, projection='3d')
        self.axDistUtility = self.fig.add_subplot(2, 4, 6, projection='3d')
        self.axVelocityUtility = self.fig.add_subplot(2, 4, 7, projection='3d')
        self.axApproachUtility = self.fig.add_subplot(2, 4, 8, projection='3d')
        self.axUtilitySum = self.fig.add_subplot(2, 2, 2, projection='3d')

    def create_figure(self):
        self.axVelocity.plot([], [])
        self.axVelocity.add_patch(patches.Rectangle(self.xy_bottom_left, self.width_rect, self.heigt_rect, color='gray', alpha=0.2))
        self.axVelocity.grid(True, color='#DCDCDC')
        self.axVelocity.set_xlim(self.omegaMin * 180 / np.pi - 10, self.omegaMax * 180 / np.pi + 10)
        self.axVelocity.set_ylim(self.vMin - 0.1, self.vMax + 0.1)
        self.axVelocity.set_xlabel('omega [°/s]')
        self.axVelocity.set_ylabel('velocity [m/s]')
        self.axVelocity.set_title('Dynamic Window')
        self.axHeadingUtility.set_title('Heading Utility')
        self.axDistUtility.set_title('Obstacle Clearance Utility')
        self.axVelocityUtility.set_title('Velocity Utility')
        self.axApproachUtility.set_title('Approach Utility')
        self.axUtilitySum.set_title('Summed Utility')

    def update_figure(self, i):
        debugOut = self.controller.debugOuts[-1]
        out = self.controller.out  # [v, omega]
        weights = self.controller.weights
        if len(debugOut):
            minOmega = debugOut['omegas'][0, 0]
            maxOmega = debugOut['omegas'][0, -1]
            minV = debugOut['velocities'][0, 0]
            maxV = debugOut['velocities'][-1, 0]
            xy_dynWindowRect = (minOmega * 180 / np.pi, minV)
            width_dynWindowRect = (maxOmega - minOmega) * 180 / np.pi + np.finfo(float).eps
            height_dynWindowRect = maxV - minV + np.finfo(float).eps

            x_invalidCandidateMarkers = debugOut['omegas'][debugOut['admissibleCandidates'] == 0] * 180 / np.pi
            y_invalidCandidateMarkers = debugOut['velocities'][debugOut['admissibleCandidates'] == 0]
            x_candidateMarkers = debugOut['omegas'][debugOut['admissibleCandidates']] * 180 / np.pi
            y_candidateMarkers = debugOut['velocities'][debugOut['admissibleCandidates']]
            x_velocityMarker = debugOut['prevState'][2] * 180 / np.pi
            y_velocityMarker = debugOut['prevState'][1]
            x_selectedVelocityMarker = out[1] * 180 / np.pi
            y_selectedVelocityMarker = out[0]

            x_data = debugOut['omegas'] * 180 / np.pi
            y_data = debugOut['velocities']
            z_headingSurf = debugOut['utility_heading']
            z_distSurf = debugOut['utility_dist']
            z_velocitySurf = debugOut['utility_velocity']
            z_approachSurf = debugOut['utility_approach']
            z_sumSurf = debugOut['utility_sum']

            x_maxUtilityMarker = np.dot(out[1] * 180 / np.pi, [1, 1])
            y_maxUtilityMarker = np.dot(out[0], [1, 1])
            z_maxUtilityMarker = [0, np.sum(weights)]

            self.axVelocity.clear()
            self.axHeadingUtility.clear()
            self.axDistUtility.clear()
            self.axVelocityUtility.clear()
            self.axApproachUtility.clear()
            self.axUtilitySum.clear()

            self.axVelocity.add_patch(patches.Rectangle(xy_dynWindowRect, width_dynWindowRect, height_dynWindowRect, alpha=0.3))
            self.axVelocity.plot(x_candidateMarkers, y_candidateMarkers, 'g.', label='candidate')
            self.axVelocity.plot(x_velocityMarker, y_velocityMarker, 'bx', linewidth=2, label='velocity')
            self.axVelocity.plot(x_invalidCandidateMarkers, y_invalidCandidateMarkers, 'r.', label='invalid Candidate')
            self.axVelocity.plot(x_selectedVelocityMarker, y_selectedVelocityMarker, 'bo', label='selected Velocity')
            self.axVelocity.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=1.5)
            self.axVelocity.add_patch(patches.Rectangle(self.xy_bottom_left, self.width_rect, self.heigt_rect, color='gray', alpha=0.2))
            self.axVelocity.grid(True, color='#DCDCDC')
            self.axVelocity.set_xlim(self.omegaMin * 180 / np.pi - 10, self.omegaMax * 180 / np.pi + 10)
            self.axVelocity.set_ylim(self.vMin - 0.1, self.vMax + 0.1)
            self.axVelocity.set_xlabel('omega [°/s]')
            self.axVelocity.set_ylabel('velocity [m/s]')
            self.axVelocity.set_title('Dynamic Window')

            self.axHeadingUtility.plot_surface(x_data, y_data, z_headingSurf, cmap='rainbow')
            self.axHeadingUtility.set_title('Heading Utility')
            self.axHeadingUtility.set_xlabel('omega[°/s]')
            self.axHeadingUtility.set_ylabel('velocities[m/s]')

            self.axDistUtility.plot_surface(x_data, y_data, z_distSurf, cmap='rainbow')
            self.axDistUtility.set_title('Obstacle Clearance Utility')
            self.axDistUtility.set_xlabel('omega[°/s]')
            self.axDistUtility.set_ylabel('velocities[m/s]')

            self.axVelocityUtility.plot_surface(x_data, y_data, z_velocitySurf, cmap='rainbow')
            self.axVelocityUtility.set_title('Velocity Utility')
            self.axVelocityUtility.set_xlabel('omega[°/s]')
            self.axVelocityUtility.set_ylabel('velocities[m/s]')

            self.axApproachUtility.plot_surface(x_data, y_data, z_approachSurf, cmap='rainbow')
            self.axApproachUtility.set_title('Approach Utility')
            self.axApproachUtility.set_xlabel('omega[°/s]')
            self.axApproachUtility.set_ylabel('velocities[m/s]')

            self.axUtilitySum.plot_surface(x_data, y_data, z_sumSurf, cmap='rainbow')
            self.axUtilitySum.plot(x_maxUtilityMarker, y_maxUtilityMarker, z_maxUtilityMarker, 'r', label='max Utility')
            self.axUtilitySum.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
            self.axUtilitySum.set_title('Summed Utility')
            self.axUtilitySum.set_xlabel('omega[°/s]')
            self.axUtilitySum.set_ylabel('velocities[m/s]')
        else:
            self.axVelocity.plot([], [])

    def visual_figure(self):
        root = tk.Tk()
        root.wm_title("DWA Internals for robot/controller")
        root.wm_attributes('-topmost', 1)
        windowWidth = 900
        windowHeight = 680
        screenWidth, screenHeight = root.maxsize()
        geometryParam = '%dx%d+%d+%d' % (windowWidth, windowHeight,
                                         (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
        root.geometry(geometryParam)

        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ani = animation.FuncAnimation(self.fig, self.update_figure, init_func=self.create_figure,
                                      frames=1000, interval=1000, blit=False)
        tk.mainloop()
