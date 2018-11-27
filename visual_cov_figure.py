import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import tkinter as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable


class VisualCovFigure:
    def __init__(self, filter_slam):
        self.filter_slam = filter_slam

        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes('right', '5%', '5%')

    def update_cov_figure(self, i):
        cov = self.filter_slam.state_cov

        hCovImg = np.array(cov[-1])

        self.im = self.ax.imshow(hCovImg, interpolation='nearest', cmap='gray', origin='upper')
        self.cax.cla()
        self.fig.colorbar(self.im, cax=self.cax, shrink=0.9)

    def visual_cov_figure(self):
        root = tk.Tk()
        root.wm_title("Covariance for robot/slam")
        windowWidth = 500
        windowHeight = 500
        screenWidth, screenHeight = root.maxsize()
        geometryParam = '%dx%d+%d+%d' % (windowWidth, windowHeight,
                                         (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
        root.geometry(geometryParam)
        root.wm_attributes('-topmost', 1)
        root.geometry(geometryParam)

        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ani = animation.FuncAnimation(self.fig, self.update_cov_figure, frames=1000, interval=1000, blit=False)

        tk.mainloop()

