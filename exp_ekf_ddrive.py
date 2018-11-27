import tkinter as tk
import numpy as np
import time
import sys
from model_platform2d_ddrive import ModelPlatform2dDDrive
from visual_platform2d import VisualPlatform2d
from controller_ddrive_follow_path2d import ControllerDDriveFollowPath2d  # 1／50
from sensor_landmarks2d import SensorLandmarks2d  # 1/15
from sensor_odometer_wheelspeed import SensorOdometerWheelspeed
from filter_ddrive_ekf import FilterDDriveEKF
from visual_filter_localization2d import VisualFilterLocalization2d

window = tk.Tk()
window.title('Simulation: exp_ekf_ddrive')
window.geometry('900x700')  # Window size
height = 600  # y
width = 800  # x
canvas = tk.Canvas(window, bg='white', height=height, width=width)  # Build a canvas

var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, font=('Verdana bold', 18))
l1.pack()

var2 = tk.StringVar()
l2 = tk.Label(window, textvariable=var2, font=('Arial bold', 12))
l2.pack()
var2.set('Differential Drive with EKF-based fusion of dead reckoning and a landmark sensor')

useEmptyRoom = False
useFilterPose = True


if useEmptyRoom:
    map_image_file = tk.PhotoImage(file='./map/emptyroom.gif')  # map
    image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)  # Add map to the canvas
    canvas.place(x=0, y=60)  # Canvas placement
    pathPoints = ([2.00, 0.75],
                  [2.00, 4.75],
                  [6.00, 4.75],
                  [6.00, 0.75],
                  [2.00, 0.75])
    path = np.array(pathPoints)

else:
    map_image_file = tk.PhotoImage(file='./map/office.gif')  # map
    image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)  # Add map to the canvas
    canvas.place(x=0, y=60)  # Canvas placement
    pathPoints = [[1.00, 0.50],
                  [1.00, 4.30],
                  [2.60, 4.30],
                  [2.60, 2.70],
                  [6.00, 2.70],
                  [6.00, 5.10],
                  [7.00, 5.10],
                  [6.00, 5.10],
                  [6.00, 3.30],
                  [7.50, 2.70],
                  [7.50, 0.75],
                  [5.00, 1.50],
                  [2.00, 1.50],
                  [2.00, 0.50]]
    path = np.array(pathPoints)

# The first path point is used as initial pose
X_0 = np.array([path[0][0], path[0][1], 90*np.pi/180])  # initial_pose
Xs = [X_0]
current_poses = [X_0]
X = Xs[-1]

t = 0
ts = [0]  # 1／100

# add the controller (which uses the 'platform' as input by default)
controller = ControllerDDriveFollowPath2d(path, canvas)
controller.wheelRadius = [0.03, 0.03]
controller.wheelDistance = 0.25

# add the sensors
odometer = SensorOdometerWheelspeed()
odometer.odometryError = 10 * np.pi / 180

landmarks_detector = SensorLandmarks2d(canvas, useEmptyRoom)
landmarks_detector.range = 7
landmarks_detector.fieldOfView = 70 * np.array([-1, 1]) * np.pi / 180
landmarks_detector.bearingError = 5 * np.pi / 180
landmarks_detector.rangeError = 2 / 100

# ...and finally the localization filter
local_filter = FilterDDriveEKF(odometer, landmarks_detector)
local_filter.useNumericPrediction = False
local_filter.useExactDiscretization = False
local_filter.useCartesianSensor = False
local_filter.useRange = True
local_filter.useBearing = True
local_filter2d = VisualFilterLocalization2d(canvas)

platform = VisualPlatform2d(canvas)
platform.robot_radius = 0.1


def run():
    global t, ts, Xs, xs, covs, control_outputs, estimated_values
    X = Xs[-1]
    if t == 0:
        control_outputs = controller.control(t, Xs)

    model = ModelPlatform2dDDrive(np.array(control_outputs['u'])[-1])  # u = [omega_r, omega_l]
    Xs = model.continuous_integration(t, Xs, 1/15)

    estimated_values = local_filter.filter_step(t, X, np.array(control_outputs['u'])[-1])

    covs = np.array(estimated_values['cov'])
    xs = np.array(estimated_values['pose'])

    if useFilterPose:
        current_poses = xs
    else:
        current_poses = Xs

    control_outputs = controller.control(t, current_poses)

    platform.draw_bot(current_poses[-1])
    controller.draw_path(t)
    controller.draw_nextpoint()

    landmarks_detector.draw_landmarks(t)
    landmarks_detector.draw_fov(current_poses[-1])  # Scanning range
    landmarks_detector.draw_connections(t, current_poses[-1])

    local_filter2d.draw_pose(covs[-1], xs[-1])
    local_filter2d.draw_track(t, xs)

    t += 1/15
    t = round(t, 4)
    ts.append(t)

    window.update_idletasks()
    window.update()
# ===============================================================


def dorun():
    while True:
        if not do_run:
            break
        var1.set('Running')
        run()
        time.sleep(0.1)


def dopause():
    var1.set('Pause')


do_run = False


def toggle_run_pause():
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
    var1.set('Step Forward')
    run()


def dostep_back():
    global t, do_run, X, Xs, ts, xs, covs, control_outputs, estimated_values
    do_run = False

    if t > 0.11:
        var1.set('Step Back')
        del (Xs[-1])  # Delete last one row of list
        del (ts[-1])
        xs = np.delete(xs, -1, axis=0)  # Delete last one row of array
        covs = np.delete(covs, -1, axis=0)
        X = Xs[-1]
        odometer.out = odometer.out[odometer.out['t'] <= t]
        landmarks_detector.out = landmarks_detector.out[landmarks_detector.out['t'] <= t]
        local_filter.out = local_filter.out[local_filter.out['t'] <= t]
        controller.out = controller.out[controller.out['t'] <= t]
        control_outputs = controller.out
        estimated_values = local_filter.out

        if useFilterPose:
            current_poses = xs
        else:
            current_poses = Xs

        platform.draw_bot(current_poses[-1])
        controller.draw_path(t)
        controller.draw_nextpoint()

        landmarks_detector.draw_fov(current_poses[-1])  # Scanning range
        landmarks_detector.draw_connections(t, current_poses[-1])

        local_filter2d.draw_pose(covs[-1], xs[-1])
        t -= 1/15
        window.update_idletasks()
        window.update()


button_img_play_green_gif = tk.PhotoImage(file='./res/play_green.gif')
button_img_pause_green_gif = tk.PhotoImage(file='./res/pause_green.gif')
button_img_next_green_gif = tk.PhotoImage(file='./res/next_green.gif')
button_img_prev_green_gif = tk.PhotoImage(file='./res/prev_green.gif')


button_doRun = tk.Button(window, image=button_img_play_green_gif, command=toggle_run_pause).place(x=60, y=10)  # run
button_doPause = tk.Button(window, image=button_img_pause_green_gif, command=toggle_run_pause).place(x=90, y=10)
button_doStep = tk.Button(window, image=button_img_next_green_gif, command=dostep).place(x=120, y=10)  # doStep
button_doStepBack = tk.Button(window, image=button_img_prev_green_gif, command=dostep_back).place(x=150, y=10)  # doBack

button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)

if __name__ == '__main__':
    window.mainloop()
