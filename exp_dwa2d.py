import tkinter as tk
import numpy as np
import time
import sys
from sensor_rangefinder2d import SensorRangefinder2d
from visual_platform2d import VisualPlatform2d
from visual_const_points2d import VisualConstPoints2d
from model_platform2d_vomega import ModelPlatform2dVomega
from guidance_waypoints2d import GuidanceWayPoints2d
from guidance_dwa2d import GuidanceDWA2d
from visual_dwa_figure import VisulaDWAFigure

window = tk.Tk()
window.title('Simulation: dwa2d')
window.geometry('850x750')
height = 600
width = 800

canvas = tk.Canvas(window, bg='white', height=height, width=width)

canvas.place(x=0, y=60)

var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, bg='green', font=('Arial', 12), width=15, height=2)
l1.pack()

var2 = tk.StringVar()
l2 = tk.Label(window, textvariable=var2, bg='white', font=('Arial', 12))
l2.pack()
var2.set('Dynamic Window Approach (DWA)')

# =================== main part for simulation ====================
map_image_file = tk.PhotoImage(file='./map/office.gif')
image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)   # Add map to the canvas


t = 0
ts = [0]

state = [1, 0]  # state = [mode, tStart]  controller.guide
states = [state]

# select the scenario
scenario = 1

if scenario == 1:
    pathPoints = np.array([1.00, 0.50,
                          1.00, 4.10,
                          2.60, 4.30,
                          2.60, 2.70,
                          5.50, 2.70,
                          6.00, 5.10,
                          7.00, 5.10,
                          6.00, 5.10,
                          6.00, 3.30,
                          7.50, 2.70,
                          7.50, 0.75,
                          5.00, 1.50,
                          2.00, 1.50,
                          2.00, 0.50]).reshape(-1, 2)
    initialPose = np.array([pathPoints[0][0], pathPoints[0][1], 90 * np.pi / 180])

# if scenario == 2:
else:
    pathPoints = np.array([[2.00, 1.14],
                           [5.00, 1.14]])

    initialPose = np.array([pathPoints[0][0], pathPoints[0][1], 0 * np.pi / 180])
Xs = [initialPose]


path = VisualConstPoints2d(canvas, pathPoints)  # Draw all the target points

# instantiate the platform (v/omega drive) and set its parameters
platform = VisualPlatform2d(canvas)
platform.robot_radius = 0.14

# add the sensors
rangefinder = SensorRangefinder2d(canvas)
rangefinder.maxRange = 3
rangefinder.fieldOfView = np.array([-90, 90]) * np.pi / 180
# Sample time of the laser rangefinder (propagates to the DWA module via the trigger mechanism)

# the robot is controlled by the Dynamic Window Approach
controller = GuidanceDWA2d(canvas)
controller.radius = platform.robot_radius
visual_figure = VisulaDWAFigure(controller)

# Another control 'layer' provides intermediate goals for the DWA to
# prevent it from getting stuck while travelling through the office environment
targetProvider = GuidanceWayPoints2d(pathPoints, canvas)
targetProvider.relative = True


def run():
    global t, Xs
    if t == 0:
        X_last = Xs[-1]
    else:
        X_last = Xs[-2]
    X = Xs[-1]

    relative_goal = targetProvider.control(t, Xs)['relativeGoal']
    controller_output = controller.dwa_step(t, relative_goal, rangefinder.sample(X))
    model = ModelPlatform2dVomega(controller_output)
    Xs = model.continuous_integration(t, Xs)

    rangefinder.draw_rays(X)
    platform.draw_bot(X)
    platform.draw_track(X, X_last)

    path.draw_const_points(t)

    controller.draw_obstacle_lines(Xs)
    controller.draw_traj_candidates(Xs)
    controller.draw_selected_trajectory(Xs)
    targetProvider.draw_target_point()

    t += 0.1
    t = round(t, 4)
    window.update_idletasks()
    window.update()
# ===============================================================


def dorun():
    while True:
        if not do_run:
            break
        var1.set('do Run')
        run()
        time.sleep(0.01)


def dopause():
    var1.set('do Pause')


do_run = False


def toggle_run_pause():
    """执行运行与暂停"""
    global do_run
    if not do_run:
        do_run = True
        dorun()
    else:
        do_run = False
        dopause()


def dostep():
    global do_run
    do_run = False
    var1.set('do Step')
    run()


def dostep_back():
    global t, do_run, Xs
    do_run = False

    if t > 0.11:
        var1.set('do StepBack')
        del (Xs[-1])

        X = Xs[-1]

        rangefinder.draw_rays(X)
        platform.draw_bot(X)
        path.draw_const_points(t)

        # controller.draw_obstacle_lines(Xs)
        controller.draw_traj_candidates(Xs)
        controller.draw_selected_trajectory(Xs)
        targetProvider.draw_target_point()
        t -= 0.1
        window.update_idletasks()
        window.update()


button_img_play_green_gif = tk.PhotoImage(file='./res/play_green.gif')
button_img_pause_green_gif = tk.PhotoImage(file='./res/pause_green.gif')
button_img_next_green_gif = tk.PhotoImage(file='./res/next_green.gif')
button_img_prev_green_gif = tk.PhotoImage(file='./res/prev_green.gif')

button_img_radar_gif = tk.PhotoImage(file='./res/radar.gif')

button_doRun = tk.Button(window, image=button_img_play_green_gif, command=toggle_run_pause).place(x=40, y=10)  # run
button_doPause = tk.Button(window, image=button_img_pause_green_gif, command=toggle_run_pause).place(x=60, y=10)
button_doStep = tk.Button(window, image=button_img_next_green_gif, command=dostep).place(x=80, y=10)  # doStep
button_doStepBack = tk.Button(window, image=button_img_prev_green_gif, command=dostep_back).place(x=100, y=10)  # doBack

button_radar = tk.Button(window, image=button_img_radar_gif, command=visual_figure.visual_figure).place(x=120, y=10)

button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)

# s = tk.Scale(window, label='timeline', from_=0, to=10, orient=tk.HORIZONTAL, length=500, showvalue=1,
#              tickinterval=5, resolution=0.01).place(x=150, y=670)

if __name__ == '__main__':
    window.mainloop()
