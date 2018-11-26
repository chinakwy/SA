import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import tkinter as tk


class VisulaErrorFigure:
    def __init__(self, filter_slam):
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.axErrX = self.fig.add_subplot(311)
        self.axErrY = self.fig.add_subplot(312)
        self.axErrPhi = self.fig.add_subplot(313)
        # self.axNees = self.fig.add_subplot(414)

        self.sensor = filter_slam.sensor

        self.sigmaX = filter_slam.sigmaX
        self.xhatlog = filter_slam.xhatlog
        self.xlog = filter_slam.xlog

        self.sigmaY = filter_slam.sigmaY
        self.yhatlog = filter_slam.yhatlog
        self.ylog = filter_slam.ylog

        self.sigmaPhi = filter_slam.sigmaPhi
        self.phihatlog = filter_slam.phihatlog
        self.philog = filter_slam.philog

    def create_error_figure(self):

        self.axErrX.plot([], [])
        self.axErrX.grid(True, color='#DCDCDC')
        self.axErrX.set_ylabel('error_X in m')

        self.axErrY.plot([], [])
        self.axErrY.grid(True, color='#DCDCDC')
        self.axErrY.set_ylabel('error_Y in m')

        self.axErrPhi.plot([], [])
        self.axErrPhi.grid(True, color='#DCDCDC')
        self.axErrPhi.set_ylabel('error_Phi in °')

        # self.axNees.plot([], [])
        # self.axNees.grid(True, color='#DCDCDC')
        # self.axNees.set_ylabel('NEES')
        # self.axNees.set_xlabel('Time [s]')

    def update_error_figure(self, i):

        t = self.sensor.ts[0:len(self.sigmaX)]

        hSigmaX_1 = np.dot(self.sigmaX, 3)
        hSigmaX_2 = np.dot(self.sigmaX, -3)
        hErrX = np.array(self.xhatlog) - np.array(self.xlog)

        hSigmaY_1 = np.dot(self.sigmaY, 3)
        hSigmaY_2 = np.dot(self.sigmaY, -3)
        hErrY = np.array(self.yhatlog) - np.array(self.ylog)

        hSigmaPhi_1 = np.dot(self.sigmaPhi, 3) * 180 / np.pi
        hSigmaPhi_2 = np.dot(self.sigmaPhi, -3) * 180 / np.pi
        hErrPhi = (np.array(self.phihatlog) - np.array(self.philog)) * 180 / np.pi

        self.axErrX.clear()
        self.axErrY.clear()
        self.axErrPhi.clear()

        self.axErrX.plot(t, hSigmaX_1, "r--")
        self.axErrX.plot(t, hSigmaX_2, "r--")
        self.axErrX.plot(t, hErrX)
        self.axErrX.grid(True, color='#DCDCDC')
        self.axErrX.set_ylabel('error_X in m')

        self.axErrY.plot(t, hSigmaY_1, "r--")
        self.axErrY.plot(t, hSigmaY_2, "r--")
        self.axErrY.plot(t, hErrY)
        self.axErrY.grid(True, color='#DCDCDC')
        self.axErrY.set_ylabel('error_Y in m')

        self.axErrPhi.plot(t, hSigmaPhi_1, "r--")
        self.axErrPhi.plot(t, hSigmaPhi_2, "r--")
        self.axErrPhi.plot(t, hErrPhi)
        self.axErrPhi.grid(True, color='#DCDCDC')
        self.axErrPhi.set_ylabel('error_Phi in °')
        self.axErrPhi.set_xlabel('Time [s]')

        # self.axNees.plot(t, [])
        # self.axNees.grid(True, color='#DCDCDC')
        # self.axNees.set_ylabel('NEES')
        # self.axNees.set_xlabel('Time [s]')

    def visual_error_figure(self):
        root = tk.Tk()
        root.wm_title("Error plots for robot/slam")
        root.wm_attributes('-topmost', 1)  # Window top
        windowWidth = 700
        windowHeight = 500
        screenWidth, screenHeight = root.maxsize()  # Get screen width and height
        geometryParam = '%dx%d+%d+%d' % (windowWidth, windowHeight,
                                         (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
        root.geometry(geometryParam)  # Set window size and offset coordinates

        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ani = animation.FuncAnimation(self.fig, self.update_error_figure, init_func=self.create_error_figure,
                                      frames=1000, interval=1000, blit=False)
        tk.mainloop()
